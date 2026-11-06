"""Bounded-memory 10x PCA with a canonical CSR cache and explicit CPU/NPU split.

Normalize all genes to 1e4 counts/cell; select Seurat dispersion HVGs among
genes present in >=3 cells; log1p; center and unit-scale HVGs without clipping.
Only zero-count cells are removed. No doublet, mitochondrial or batch filtering.
"""
import argparse
from datetime import datetime, timezone
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

from .io import TenXReader, CSRCache, sha256_file
from .backend import DenseBackend, TOLERANCES


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
    if raw.shape[0] < 3 or raw.shape[1] < 2:
        return {'sample_reference': 'not evaluated: first block too small; full score validation still required'}
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
    if test.shape[1] < 2:
        return {'hvg_vs_scanpy': 'passed', 'sample_pca': 'not evaluated: constant sample'}
    mean = test.mean(axis=0)
    cov = (test.T @ test - len(test) * np.outer(mean, mean)) / (len(test) - 1)
    val, vec = linalg.eigh(cov)
    k = min(20, test.shape[0]-1, test.shape[1])
    reference = PCA(n_components=k, svd_solver='full').fit(test)
    np.testing.assert_allclose(val[-k:][::-1], reference.explained_variance_, rtol=1e-9, atol=1e-9)
    overlap = np.linalg.svd(vec[:, -k:].T @ reference.components_.T, compute_uv=False)
    np.testing.assert_allclose(overlap, 1, atol=1e-8)
    return {'hvg_vs_scanpy': 'passed', 'covariance_pca_vs_sklearn': 'passed'}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='brain_1m.h5')
    parser.add_argument('--output', default='results')
    parser.add_argument('--n-hvg', type=int, default=2000)
    parser.add_argument('--n-pcs', type=int, default=50)
    parser.add_argument('--chunk-size', type=int, default=8192)
    parser.add_argument('--device', choices=['cpu', 'npu'], default='npu')
    parser.add_argument('--npu-index', type=int, default=0)
    parser.add_argument('--cache', help='Immutable canonical CSR cache path; first run creates it')
    parser.add_argument('--max-block-nnz', type=int, default=32_000_000)
    parser.add_argument('--hvg-backend', choices=['host', 'tilelang'], default='host')
    parser.add_argument('--max-covariance-gib', type=float, default=1.)
    args = parser.parse_args(argv)
    if args.n_pcs < 2 or args.n_hvg <= args.n_pcs or args.chunk_size < 1:
        parser.error('Require 2 <= n_pcs < n_hvg, positive chunk-size')
    if (not np.isfinite(args.max_covariance_gib) or args.max_covariance_gib <= 0
            or 6 * args.n_hvg**2 * 8 > args.max_covariance_gib * 1024**3):
        parser.error('Covariance and solver workspace exceed the explicit memory budget')
    if args.hvg_backend != 'host' and args.device != 'npu':
        parser.error('tilelang HVG backend requires --device npu')
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    with DenseBackend(f'npu:{args.npu_index}' if args.device == 'npu' else 'cpu') as backend:
        return _run(args, out, backend, started)


def _run(args, out, backend, started):
    log(f'Compute backend: {backend.info}')
    stages = {}
    stage_start = time.perf_counter()
    source_hash = sha256_file(args.input)
    stages['source_hash_seconds'] = time.perf_counter() - stage_start
    log(f'Start {vars(args)}; BLAS {threadpool_info()}')
    reader = TenXReader(args.input, rows_per_block=args.chunk_size, max_block_nnz=args.max_block_nnz)
    n_cells, n_genes = reader.n_cells, reader.n_genes
    ids, names, barcodes = reader.ids, reader.names, reader.barcodes
    mito = np.char.startswith(names.astype(str), 'mt-') | np.char.startswith(names.astype(str), 'MT-')
    cache = CSRCache(args.cache, reader, source_hash) if args.cache else None
    cache_reused = bool(cache is not None and cache.ready)
    log(f'Input: {n_cells:,} cells x {n_genes:,} genes; {reader.ptr[-1]:,} nonzeros; cache reused={cache_reused}')
    stage_start = time.perf_counter()
    # Preserve bounded-memory traversal and the original experiment's semantics.
    sums, squares = np.zeros(n_genes), np.zeros(n_genes)
    detection = np.zeros(n_genes, dtype=np.int64)
    totals = np.zeros(n_cells)
    detected = np.zeros(n_cells, dtype=np.int32)
    mito_counts = np.zeros(n_cells)
    validation = None
    source = cache.blocks() if cache_reused else reader.blocks()
    for start, stop, mat in source:
        if cache is not None and not cache_reused:
            cache.append(start, stop, mat)
        mat = mat.astype(np.float64, copy=not mat.data.flags.writeable)
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
    if cache is not None and not cache_reused:
        cache.finish()
    stages['pass1_qc_moments_cache_seconds'] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    keep = totals > 0
    n = int(keep.sum())
    if n < 2:
        raise ValueError('PCA requires at least two nonzero cells')
    eligible = detection >= 3
    if not eligible.any():
        raise ValueError('No genes pass min_cells=3')
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
    stages['hvg_selection_and_qc_export_seconds'] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    log(f'HVG selection complete: {n:,} cells, {h} genes; zero-count cells removed={n_cells-n}')
    expr = np.lib.format.open_memmap(out / 'log1p_hvg.npy', mode='w+', dtype='float32', shape=(n, h))
    sum_log = np.zeros(h)
    gram = np.zeros((h, h))
    row = 0
    for start, stop, mat in (cache.blocks() if cache is not None else reader.blocks()):
        mask = keep[start:stop]
        if not mask.any():
            continue
        subset = mat[mask][:, selected].copy()
        if args.hvg_backend == 'host':
            subset = subset.astype(np.float64)
            subset.data *= np.repeat(1e4 / totals[start:stop][mask], np.diff(subset.indptr))
            np.log1p(subset.data, out=subset.data)
            # One FP32 materialization; FP64 only for host accumulation.
            dense = subset.astype(np.float32).toarray()
            device_dense = dense
        else:
            from .kernels import hvg_to_dense
            with np.errstate(over='ignore', under='ignore', divide='ignore'):
                values32 = subset.data.astype(np.float32)
                totals32 = totals[start:stop][mask].astype(np.float32)
                factors32 = np.float32(1e4) / totals32
            if (not np.isfinite(values32).all() or np.any(values32 <= 0)
                    or not np.isfinite(factors32).all() or np.any(totals32 <= 0)
                    or not np.isfinite(totals32).all()):
                raise ValueError('Counts/totals exceed the FP32 HVG kernel range; use --hvg-backend host')
            device_dense = hvg_to_dense(
                backend.upload(values32), backend.upload(subset.indices, np.int32),
                backend.upload(subset.indptr, np.int32), backend.upload(totals32),
                n_genes=h, max_row_nnz=int(np.diff(subset.indptr).max(initial=0)))
            dense = backend.host(device_dense)
        expr[row:row + len(dense)] = dense
        sum_log += dense.sum(axis=0, dtype=np.float64)
        gram += backend.gram(device_dense)
        row += len(dense)
        if start // args.chunk_size % 10 == 0 or stop == n_cells:
            log(f'Pass 2 HVG covariance: {stop:,}/{n_cells:,}')
    if row != n:
        raise RuntimeError('Streamed expression row count does not match retained cells')
    expr.flush()
    stages['pass2_transform_gram_scratch_seconds'] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    mean = sum_log / n
    covariance = (gram - n * np.outer(mean, mean)) / (n - 1)
    sd = np.sqrt(np.diag(covariance))
    if not np.isfinite(sd).all() or not (sd > 0).all():
        raise ArithmeticError('Nonpositive HVG variance; covariance PCA is unsuitable for this input')
    covariance /= np.outer(sd, sd)
    covariance = (covariance + covariance.T) / 2
    log('Solving centered, unit-variance covariance with CPU SciPy eigh')
    variance, loadings = backend.eigh_top(covariance, args.n_pcs)
    signs = np.sign(loadings[np.abs(loadings).argmax(axis=0), np.arange(args.n_pcs)])
    loadings *= signs
    ratio = variance / np.trace(covariance)
    residual = np.linalg.norm(covariance @ loadings - loadings * variance) / np.linalg.norm(variance)
    if not residual < (TOLERANCES['eigen_residual'] if args.device == 'npu' else 1e-9):
        raise ArithmeticError(f'Eigen residual precision gate failed: {residual}')
    np.testing.assert_allclose(loadings.T @ loadings, np.eye(args.n_pcs),
                               atol=TOLERANCES['orthogonality_abs'] if args.device == 'npu' else 1e-10)
    stages['covariance_eigh_seconds'] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    scores = np.lib.format.open_memmap(out / 'pca_scores.npy', mode='w+', dtype='float32', shape=(n, args.n_pcs))
    score_sum = np.zeros(args.n_pcs)
    score_gram = np.zeros((args.n_pcs, args.n_pcs))
    projection = loadings / sd[:, None]
    offset = mean @ projection
    for start in range(0, n, args.chunk_size):
        stop = min(start + args.chunk_size, n)
        score = backend.project(expr[start:stop], projection, offset)
        scores[start:stop] = score
        score = score.astype(np.float64)
        score_sum += score.sum(axis=0)
        score_gram += score.T @ score
    scores.flush()
    score_cov = (score_gram - np.outer(score_sum, score_sum) / n) / (n - 1)
    np.testing.assert_allclose(score_sum / n, 0, atol=TOLERANCES['score_mean_abs'] if args.device == 'npu' else 1e-6)
    np.testing.assert_allclose(score_cov, np.diag(variance),
                               rtol=TOLERANCES['score_covariance_relative'] if args.device == 'npu' else 1e-5,
                               atol=TOLERANCES['score_covariance_abs'] if args.device == 'npu' else 1e-6)
    validation.update({'full_score_covariance': 'passed', 'eigen_relative_residual': float(residual),
                       'score_mean_max_abs': float(np.abs(score_sum / n).max())})
    stages['projection_and_full_score_validation_seconds'] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    var = gene_frame.iloc[selected].copy().set_index('gene_id')
    var['pca_mean_log1p'] = mean
    var['pca_std_log1p'] = sd
    result = ad.AnnData(X=None, obs=obs, var=var, shape=(n, h))
    result.obsm['X_pca'] = np.asarray(scores)
    result.varm['PCs'] = loadings.astype(np.float32)
    result.uns['pca'] = {'variance': variance, 'variance_ratio': ratio,
                          'params': {'zero_center': True, 'unit_variance': True,
                                     'n_comps': args.n_pcs, 'device': args.device, 'hvg_backend': args.hvg_backend,
                                     'solver': 'streamed_covariance_cpu_eigh'}}
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
    stages['artifact_export_and_plot_seconds'] = time.perf_counter() - stage_start
    summary = {'input_cells': n_cells, 'input_genes': n_genes, 'retained_cells': n,
               'eligible_genes': int(eligible.sum()), 'n_hvg': h, 'n_pcs': args.n_pcs,
               'input_sha256': source_hash, 'explained_variance_ratio': ratio.tolist(),
               'cumulative_explained_variance_ratio': float(ratio.sum()), 'validation': validation,
               'elapsed_seconds': time.perf_counter() - started,
               'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
               'compute_backend': backend.info,
               'stages': stages, 'cache_reused': cache_reused,
               'cache_path': str(cache.path) if cache is not None else None,
               'hvg_backend': args.hvg_backend,
               'packages': {p: importlib.metadata.version(p) for p in
                            ['numpy', 'scipy', 'pandas', 'h5py', 'scanpy', 'anndata', 'scikit-learn']}}
    # Reopen the exported artifact and verify its dimensions before marking success.
    with h5py.File(out / 'brain_1m_pca.h5ad', 'r') as check:
        if (check['obsm/X_pca'].shape != (n, args.n_pcs)
                or check['varm/PCs'].shape != (h, args.n_pcs)):
            raise RuntimeError('Exported PCA artifact has inconsistent shapes')
    summary['elapsed_seconds'] = time.perf_counter() - started
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'SUCCESS').write_text(datetime.now(timezone.utc).isoformat() + '\n')
    log(f'COMPLETE: {json.dumps(summary)}')
    return summary


if __name__ == '__main__':
    main()
