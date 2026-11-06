"""Compare full workflow outputs and sampled neighbor stability, without rerunning PCA."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import linalg
from sklearn.neighbors import NearestNeighbors

# Declared before looking at new full-data outputs; signs/rotations are not biology.
LIMITS = dict(expression_max_abs=2e-4, eigenvalue_max_relative=2e-4,
              loading_subspace_min_cosine=.9999, aligned_score_relative_frobenius=2e-4,
              aligned_score_max_abs=2e-3, sampled_neighbor_overlap=.99)


def compare(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    a = ad.read_h5ad(reference / "brain_1m_pca.h5ad", backed="r")
    b = ad.read_h5ad(candidate / "brain_1m_pca.h5ad", backed="r")
    if not a.obs_names.equals(b.obs_names) or not a.var_names.equals(b.var_names):
        raise AssertionError("Retained cells or selected HVG identities/order differ")
    for column in ("total_counts", "n_genes_by_counts", "pct_counts_mt"):
        np.testing.assert_allclose(a.obs[column], b.obs[column], atol=1e-9, rtol=0)
    av, bv = a.uns["pca"]["variance"], b.uns["pca"]["variance"]
    load_a, load_b = np.asarray(a.varm["PCs"], dtype=np.float64), np.asarray(b.varm["PCs"], dtype=np.float64)
    if not all(np.isfinite(v).all() for v in (av, bv, load_a, load_b)):
        raise AssertionError("Non-finite eigenvalues/loadings")
    # QR removes tiny FP32 orthogonality drift before measuring principal angles.
    qa, _ = linalg.qr(load_a, mode="economic")
    qb, _ = linalg.qr(load_b, mode="economic")
    minimum_cosine = linalg.svdvals(qa.T @ qb).min()
    rotation, _ = linalg.orthogonal_procrustes(load_b, load_a)
    x = np.load(reference / "log1p_hvg.npy", mmap_mode="r")
    y = np.load(candidate / "log1p_hvg.npy", mmap_mode="r")
    sa = np.load(reference / "pca_scores.npy", mmap_mode="r")
    sb = np.load(candidate / "pca_scores.npy", mmap_mode="r")
    if x.shape != y.shape or sa.shape != sb.shape:
        raise AssertionError("Expression or score shape mismatch")
    expression_error = score_error = sum_error = sum_reference = 0.
    for start in range(0, len(x), 8192):
        sl = slice(start, start+8192)
        if not all(np.isfinite(v).all() for v in (x[sl], y[sl], sa[sl], sb[sl])):
            raise AssertionError(f"Non-finite expression/scores in block starting at {start}")
        expression_error = max(expression_error, float(np.abs(x[sl]-y[sl]).max(initial=0)))
        aligned = np.asarray(sb[sl], dtype=np.float64) @ rotation
        delta = aligned - sa[sl]
        score_error = max(score_error, float(np.abs(delta).max(initial=0)))
        sum_error += float(np.sum(delta**2))
        sum_reference += float(np.sum(np.asarray(sa[sl], dtype=np.float64)**2))
    chosen = np.sort(np.random.default_rng(2026).choice(len(sa), min(2048, len(sa)), replace=False))
    k = min(15, len(chosen)-1)
    first, second = sa[chosen], sb[chosen]
    def neighbors(data):
        found = NearestNeighbors(n_neighbors=k+1).fit(data).kneighbors(data, return_distance=False)
        return np.array([row[row != i][:k] for i, row in enumerate(found)])
    na, nb = neighbors(first), neighbors(second)
    overlap = float(np.mean([len(set(i) & set(j))/k for i, j in zip(na, nb)]))
    metrics = dict(expression_max_abs=expression_error,
                   eigenvalue_max_relative=float(np.max(np.abs(av-bv)/np.maximum(np.abs(av), 1e-12))),
                   loading_subspace_min_cosine=float(minimum_cosine),
                   aligned_score_relative_frobenius=float(np.sqrt(sum_error/max(sum_reference, 1e-30))),
                   aligned_score_max_abs=score_error, sampled_neighbor_overlap=overlap)
    passed = {key: (value >= LIMITS[key] if key in ("loading_subspace_min_cosine", "sampled_neighbor_overlap") else value <= LIMITS[key])
              for key, value in metrics.items()}
    a.file.close()
    b.file.close()
    return dict(reference=str(reference), candidate=str(candidate), limits=LIMITS,
                metrics=metrics, passed=passed, all_passed=all(passed.values()),
                neighbor_scope="2048 fixed-seed sampled cells, k=15 within sample; not a full million-cell graph")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_passed"] else 1)
