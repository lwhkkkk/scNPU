"""Full-cell PCA of legacy 10x HDF5 with bounded-memory exact covariance PCA.

Normalize all genes to 1e4 counts/cell; select Seurat dispersion HVGs among
genes present in >=3 cells; log1p; center and unit-scale HVGs without clipping.
Only zero-count cells are removed. No doublet, mitochondrial or batch filtering.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time

import anndata as ad
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import linalg, sparse
from scanpy.preprocessing._highly_variable_genes import _get_mean_bins, _get_disp_stats
from threadpoolctl import threadpool_info


def log(message):
    print(f'[{datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', flush=True)


def hvg_stats(sums, squares, n):
    mean = sums / n
    var = np.maximum((squares - sums * mean) / (n - 1), 0)
    mean = np.maximum(mean, 1e-12)
    dispersion = var / mean
    dispersion[dispersion == 0] = np.nan
    frame = pd.DataFrame({'means': np.log1p(mean), 'dispersions': np.log(dispersion)})
    frame['mean_bin'] = _get_mean_bins(frame['means'], 'seurat', 20)
    stats = _get_disp_stats(frame, 'seurat')
    frame['dispersions_norm'] = (frame['dispersions'] - stats['avg']) / stats['dev']
    return frame.drop(columns='mean_bin')


def verify_methods(raw):
    """Compare streaming HVG moments and covariance PCA with library references."""
    from sklearn.decomposition import PCA
    raw = raw[:2000].copy()
    valid = np.asarray(raw.sum(axis=1)).ravel() > 0
    raw = raw[valid]
    raw = raw[:, np.asarray((raw > 0).sum(axis=0)).ravel() >= 3]
    norm = raw.multiply(1e4 / np.asarray(raw.sum(axis=1))).tocsr()
    own = hvg_stats(np.asarray(norm.sum(axis=0)).ravel(),
                    np.asarray(norm.power(2).sum(axis=0)).ravel(), norm.shape[0])
    ref = ad.AnnData(norm.copy())
    sc.pp.log1p(ref)
    sc.pp.highly_variable_genes(ref, n_top_genes=min(200, ref.n_vars), flavor='seurat')
    for key in ['means', 'dispersions', 'dispersions_norm']:
        np.testing.assert_allclose(own[key], ref.var[key], rtol=2e-5, atol=2e-5, equal_nan=True)
    test = ref.X[:300, :128].toarray()
    std = test.std(axis=0, ddof=1)
    test = test[:, std > 0] / std[std > 0]
    mean = test.mean(axis=0)
    cov = (test.T @ test - len(test) * np.outer(mean, mean)) / (len(test) - 1)
    val, vec = linalg.eigh(cov)
    reference = PCA(n_components=20, svd_solver='full').fit(test)
    np.testing.assert_allclose(val[-20:][::-1], reference.explained_variance_, rtol=1e-9, atol=1e-9)
    overlap = np.linalg.svd(vec[:, -20:].T @ reference.components_.T, compute_uv=False)
    np.testing.assert_allclose(overlap, 1, atol=1e-8)
    return {'hvg_vs_scanpy': 'passed', 'covariance_pca_vs_sklearn': 'passed'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='brain_1m.h5')
    parser.add_argument('--output', default='results')
    parser.add_argument('--n-hvg', type=int, default=2000)
    parser.add_argument('--n-pcs', type=int, default=50)
    parser.add_argument('--chunk-size', type=int, default=8192)
    parser.add_argument('--device', choices=['cpu', 'npu'], default='npu')
    parser.add_argument('--npu-index', type=int, default=0)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    backend = None
    if args.device == 'npu':
        from npu_backend import NPUBackend, TOLERANCES
        backend = NPUBackend(args.npu_index)
        log(f'NPU initialized: {backend.info}')
    log(f'Start {vars(args)}; BLAS {threadpool_info()}')
    with h5py.File(args.input, 'r') as handle:
        groups = [name for name in handle if isinstance(handle[name], h5py.Group) and 'data' in handle[name]]
        if len(groups) != 1:
            raise ValueError(f'Expected one genome group, found {groups}')
        group = handle[groups[0]]
        n_genes, n_cells = map(int, group['shape'][:])
        ptr = group['indptr'][:].astype(np.int64)
        assert len(ptr) == n_cells + 1 and ptr[-1] == len(group['data'])
        ids = group['genes'].asstr()[:]
        names = group['gene_names'].asstr()[:]
        barcodes = group['barcodes'].asstr()[:]
        mito = np.char.startswith(names.astype(str), ('mt-')) | np.char.startswith(names.astype(str), 'MT-')
        log(f'Input: {n_cells:,} cells x {n_genes:,} genes; {ptr[-1]:,} stored values; genome={groups[0]}')

        def blocks():
            for start in range(0, n_cells, args.chunk_size):
                stop = min(start + args.chunk_size, n_cells)
                lo, hi = int(ptr[start]), int(ptr[stop])
                values = group['data'][lo:hi].astype(np.float64)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Invalid count values')
                mat = sparse.csr_matrix((values, group['indices'][lo:hi], ptr[start:stop + 1] - lo),
                                        shape=(stop - start, n_genes))
                mat.sum_duplicates()
                mat.eliminate_zeros()
                yield start, stop, mat

        sums, squares = np.zeros(n_genes), np.zeros(n_genes)
        detection = np.zeros(n_genes, dtype=np.int64)
        totals = np.zeros(n_cells)
        detected = np.zeros(n_cells, dtype=np.int32)
        mito_counts = np.zeros(n_cells)
        validation = None
        for start, stop, mat in blocks():
            if validation is None:
                validation = verify_methods(mat)
                log(f'Reference verification: {validation}')
            total = np.asarray(mat.sum(axis=1)).ravel()
            totals[start:stop] = total
            detected[start:stop] = np.diff(mat.indptr)
            mito_counts[start:stop] = np.asarray(mat[:, mito].sum(axis=1)).ravel()
            detection += np.bincount(mat.indices, minlength=n_genes)
            scale = np.divide(1e4, total, out=np.zeros_like(total), where=total > 0)
            mat.data *= np.repeat(scale, np.diff(mat.indptr))
            sums += np.bincount(mat.indices, weights=mat.data, minlength=n_genes)
            squares += np.bincount(mat.indices, weights=mat.data ** 2, minlength=n_genes)
            if start // args.chunk_size % 10 == 0 or stop == n_cells:
                log(f'Pass 1 statistics: {stop:,}/{n_cells:,}')
        keep = totals > 0
        n = int(keep.sum())
        eligible = detection >= 3
        gene_frame = pd.DataFrame({'gene_id': ids, 'gene_name': names, 'n_cells_by_counts': detection})
        stats = hvg_stats(sums[eligible], squares[eligible], n)
        for col in stats:
            gene_frame.loc[eligible, col] = stats[col].to_numpy()
        rank = gene_frame['dispersions_norm'].to_numpy()
        candidates = np.flatnonzero(eligible & np.isfinite(rank))
        chosen = candidates[np.argsort(-rank[candidates], kind='stable')[:args.n_hvg]]
        selected = np.sort(chosen)
        h = len(selected)
        if h < args.n_hvg or min(n, h) <= args.n_pcs:
            raise ValueError(f'Insufficient HVGs/cells: {n=}, {h=}')
        gene_frame['highly_variable'] = False
        gene_frame.loc[selected, 'highly_variable'] = True
        gene_frame.to_csv(out / 'gene_statistics.csv.gz', index=False)
        obs = pd.DataFrame({'total_counts': totals[keep], 'n_genes_by_counts': detected[keep],
                            'pct_counts_mt': 100 * mito_counts[keep] / totals[keep]},
                           index=pd.Index(barcodes[keep], name='barcode'))
        obs.to_csv(out / 'cell_qc.csv.gz')
        log(f'HVG selection complete: {n:,} cells, {h} genes; zero-count cells removed={n_cells-n}')
        expr = np.lib.format.open_memmap(out / 'log1p_hvg.npy', mode='w+', dtype='float32', shape=(n, h))
        sum_log = np.zeros(h)
        gram = np.zeros((h, h))
        row = 0
        for start, stop, mat in blocks():
            mask = keep[start:stop]
            subset = mat[mask][:, selected].copy()
            subset.data *= np.repeat(1e4 / totals[start:stop][mask], np.diff(subset.indptr))
            np.log1p(subset.data, out=subset.data)
            dense = subset.toarray().astype(np.float32).astype(np.float64)
            expr[row:row + len(dense)] = dense
            sum_log += dense.sum(axis=0)
            gram += backend.gram(dense) if backend else dense.T @ dense
            row += len(dense)
            if start // args.chunk_size % 10 == 0 or stop == n_cells:
                log(f'Pass 2 HVG covariance: {stop:,}/{n_cells:,}')
        assert row == n
        expr.flush()
    mean = sum_log / n
    covariance = (gram - n * np.outer(mean, mean)) / (n - 1)
    sd = np.sqrt(np.diag(covariance))
    assert np.isfinite(sd).all() and (sd > 0).all()
    covariance /= np.outer(sd, sd)
    covariance = (covariance + covariance.T) / 2
    log(f'Solving centered, unit-variance covariance on {args.device}')
    if backend:
        variance, loadings = backend.eigh_top(covariance, args.n_pcs)
    else:
        variance, loadings = linalg.eigh(covariance, subset_by_index=[h - args.n_pcs, h - 1], driver='evr')
        variance, loadings = variance[::-1].copy(), loadings[:, ::-1].copy()
    signs = np.sign(loadings[np.abs(loadings).argmax(axis=0), np.arange(args.n_pcs)])
    loadings *= signs
    ratio = variance / np.trace(covariance)
    residual = np.linalg.norm(covariance @ loadings - loadings * variance) / np.linalg.norm(variance)
    assert residual < (TOLERANCES['eigen_residual'] if backend else 1e-9)
    np.testing.assert_allclose(loadings.T @ loadings, np.eye(args.n_pcs),
                               atol=TOLERANCES['orthogonality_abs'] if backend else 1e-10)
    scores = np.lib.format.open_memmap(out / 'pca_scores.npy', mode='w+', dtype='float32', shape=(n, args.n_pcs))
    score_sum = np.zeros(args.n_pcs)
    score_gram = np.zeros((args.n_pcs, args.n_pcs))
    projection = loadings / sd[:, None]
    offset = mean @ projection
    for start in range(0, n, args.chunk_size):
        stop = min(start + args.chunk_size, n)
        if backend:
            score = backend.project(expr[start:stop], projection, offset)
        else:
            score = (np.asarray(expr[start:stop], dtype=np.float64) @ projection - offset).astype(np.float32)
        scores[start:stop] = score
        score = score.astype(np.float64)
        score_sum += score.sum(axis=0)
        score_gram += score.T @ score
    scores.flush()
    score_cov = (score_gram - np.outer(score_sum, score_sum) / n) / (n - 1)
    np.testing.assert_allclose(score_sum / n, 0, atol=TOLERANCES['score_mean_abs'] if backend else 1e-6)
    np.testing.assert_allclose(score_cov, np.diag(variance),
                               rtol=TOLERANCES['score_covariance_relative'] if backend else 1e-5,
                               atol=TOLERANCES['score_covariance_abs'] if backend else 1e-6)
    validation.update({'full_score_covariance': 'passed', 'eigen_relative_residual': float(residual),
                       'score_mean_max_abs': float(np.abs(score_sum / n).max())})
    var = gene_frame.iloc[selected].copy().set_index('gene_id')
    var['pca_mean_log1p'] = mean
    var['pca_std_log1p'] = sd
    result = ad.AnnData(X=None, obs=obs, var=var, shape=(n, h))
    result.obsm['X_pca'] = np.asarray(scores)
    result.varm['PCs'] = loadings.astype(np.float32)
    result.uns['pca'] = {'variance': variance, 'variance_ratio': ratio,
                          'params': {'zero_center': True, 'unit_variance': True,
                                     'n_comps': args.n_pcs, 'device': args.device,
                                     'solver': 'streamed_covariance_npu_svd' if backend else 'exact_streamed_covariance_eigh'}}
    result.uns['preprocessing'] = {'target_sum': 1e4, 'log1p': True, 'hvg_flavor': 'seurat',
                                  'hvg_tie_break': 'original gene order', 'min_cells_per_gene': 3,
                                  'expression_file': 'log1p_hvg.npy', 'expression_in_X': False,
                                  'cell_filter': 'total_counts > 0', 'clipping': False,
                                  'batch_correction': False, 'doublet_removal': False}
    result.write_h5ad(out / 'brain_1m_pca.h5ad', compression='lzf')
    pc_names = [f'PC{i+1}' for i in range(args.n_pcs)]
    pd.DataFrame(loadings, index=var.index, columns=pc_names).to_csv(out / 'pca_loadings.csv.gz')
    pd.DataFrame({'PC': pc_names, 'variance': variance, 'variance_ratio': ratio,
                  'cumulative_variance_ratio': np.cumsum(ratio)}).to_csv(out / 'explained_variance.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(np.arange(1, args.n_pcs + 1), 100 * ratio, '.-')
    axes[0].set(xlabel='Principal component', ylabel='Explained variance (%)')
    axes[1].plot(np.arange(1, args.n_pcs + 1), 100 * np.cumsum(ratio), '.-')
    axes[1].set(xlabel='Number of components', ylabel='Cumulative explained variance (%)')
    fig.savefig(out / 'explained_variance.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    density = ax.hexbin(scores[:, 0], scores[:, 1], gridsize=180, bins='log', mincnt=1, cmap='viridis')
    fig.colorbar(density, ax=ax, label='Cells per hexagon (log scale)')
    ax.set(xlabel=f'PC1 ({100*ratio[0]:.2f}%)', ylabel=f'PC2 ({100*ratio[1]:.2f}%)',
           title=f'10x mouse brain: {n:,} cells')
    fig.savefig(out / 'pca_pc1_pc2.png', dpi=180)
    plt.close(fig)
    digest = hashlib.sha256()
    with open(args.input, 'rb') as inp:
        while block := inp.read(8 * 1024 * 1024):
            digest.update(block)
    summary = {'input_cells': n_cells, 'input_genes': n_genes, 'retained_cells': n,
               'eligible_genes': int(eligible.sum()), 'n_hvg': h, 'n_pcs': args.n_pcs,
               'input_sha256': digest.hexdigest(), 'explained_variance_ratio': ratio.tolist(),
               'cumulative_explained_variance_ratio': float(ratio.sum()), 'validation': validation,
               'elapsed_seconds': time.perf_counter() - started,
               'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
               'compute_backend': backend.info if backend else {'device': 'cpu'},
               'packages': {p: importlib.metadata.version(p) for p in
                            ['numpy', 'scipy', 'pandas', 'h5py', 'scanpy', 'anndata', 'scikit-learn']}}
    # Reopen the exported artifact and verify its dimensions before marking success.
    with h5py.File(out / 'brain_1m_pca.h5ad', 'r') as check:
        assert check['obsm/X_pca'].shape == (n, args.n_pcs)
        assert check['varm/PCs'].shape == (h, args.n_pcs)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'SUCCESS').write_text(datetime.now(timezone.utc).isoformat() + '\n')
    log(f'COMPLETE: {json.dumps(summary)}')


if __name__ == '__main__':
    main()
