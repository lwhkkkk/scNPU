"""Small explicit Scanpy-compatible preprocessing surface.

QC and HVG currently use Scanpy's CPU implementations. NPU normalization and
log1p operate on sparse values, while PCA uses block Gram/projection on NPU and
a host FP64 eigensolver. No implicit global Scanpy monkeypatching is performed.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
import warnings

import numpy as np
from scipy import linalg, sparse

from .session import DeviceSession, _matrix


def _backend(backend):
    if backend not in {"cpu", "npu"}:
        raise ValueError("backend must be 'cpu' or 'npu'.")


def _target(adata, copy):
    if not copy and adata.is_view:
        raise ValueError("In-place operations on AnnData views are unsupported; use copy=True.")
    return adata.copy() if copy else adata


def calculate_qc_metrics(adata, **kwargs):
    """Use Scanpy's CPU QC implementation, including its return/inplace contract."""
    import scanpy as sc
    return sc.pp.calculate_qc_metrics(adata, **kwargs)


def highly_variable_genes(adata, **kwargs):
    """Use Scanpy's CPU HVG implementation; no flavor is silently approximated."""
    import scanpy as sc
    return sc.pp.highly_variable_genes(adata, **kwargs)


def normalize_total(adata, *, target_sum=1e4, layer=None, copy=False, backend="cpu", device="npu:0"):
    """Normalize all genes of each row. This subset always writes the selected layer.

    ``exclude_highly_expressed``, ``key_added`` and ``inplace=False`` are not
    accepted; use Scanpy for those options. NPU mode requires explicit target_sum.
    """
    _backend(backend)
    result = _target(adata, copy)
    if backend == "cpu":
        import scanpy as sc
        sc.pp.normalize_total(result, target_sum=target_sum, layer=layer)
    else:
        with DeviceSession(result, layer=layer, device=device) as session:
            session.normalize_total(target_sum).writeback(result)
    return result if copy else None


def log1p(adata, *, base=None, layer=None, copy=False, backend="cpu", device="npu:0"):
    _backend(backend)
    result = _target(adata, copy)
    if backend == "cpu":
        import scanpy as sc
        sc.pp.log1p(result, base=base, layer=layer)
    else:
        with DeviceSession(result, layer=layer, device=device) as session:
            session.log1p(base).writeback(result)
        result.uns["log1p"] = {"base": base}
    return result if copy else None


@contextmanager
def _npu_precision(device):
    import torch
    import torch_npu
    previous_hf32 = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = False
    try:
        with torch.npu.device(device), torch.no_grad():
            yield torch
    finally:
        try:
            torch.npu.synchronize(device)
        finally:
            torch.npu.matmul.allow_hf32 = previous_hf32


def pca(adata, n_comps=None, *, zero_center=True, scale=False, mask_var=None,
        layer=None, copy=False, backend="cpu", device="npu:0", block_rows=8192,
        max_covariance_bytes=512 * 1024**2):
    """Covariance PCA, with optional unit variance scaling and no clipping.

    ``mask_var=None`` uses all genes, explicitly. Pass ``'highly_variable'`` to
    select that mask. Unlike Scanpy's solver auto-selection this always computes
    a covariance/second-moment matrix. Select HVGs first for high-dimensional data.
    CPU Gram is FP64; NPU Gram/projection is strict FP32 with FP64 host accumulation.
    """
    _backend(backend)
    if not isinstance(block_rows, int) or block_rows <= 0:
        raise ValueError("block_rows must be a positive integer.")
    if not zero_center and scale:
        raise ValueError("scale=True requires zero_center=True.")
    result = _target(adata, copy)
    x = _matrix(result, layer)
    if isinstance(mask_var, str):
        mask = np.asarray(result.var[mask_var])
    elif mask_var is None:
        mask = np.ones(result.n_vars, dtype=bool)
    else:
        mask = np.asarray(mask_var)
    if mask.shape != (result.n_vars,) or mask.dtype != bool:
        raise ValueError("mask_var must be a boolean vector with one entry per gene.")
    n, p = result.n_obs, int(mask.sum())
    if n_comps is None:
        n_comps = min(50, n - 1, p)
    if not isinstance(n_comps, int) or not 1 <= n_comps <= min(n - 1, p):
        raise ValueError("Require 1 <= n_comps <= min(n_obs - 1, selected n_vars).")
    # Budget includes a conservative eigensolver/Gram workspace, not only one matrix.
    if 6 * p * p * 8 > max_covariance_bytes:
        raise MemoryError("Covariance workspace exceeds max_covariance_bytes; select fewer genes.")
    if sparse.issparse(x):
        x = sparse.csr_matrix(x)
    start = time.perf_counter()
    gram = np.zeros((p, p), dtype=np.float64)
    sums = np.zeros(p, dtype=np.float64)
    squares = np.zeros(p, dtype=np.float64)

    def blocks(dtype):
        for lo in range(0, n, block_rows):
            block = x[lo:lo + block_rows, :][:, mask]
            if sparse.issparse(block):
                block = block.toarray()
            block = np.asarray(block, dtype=dtype, order="C")
            if not np.isfinite(block).all():
                raise ValueError("PCA input contains non-finite values or overflows working dtype.")
            yield lo, block

    def accumulate(torch=None):
        for _, block in blocks(np.float32 if torch else np.float64):
            sums[:] += block.sum(axis=0, dtype=np.float64)
            squares[:] += np.einsum("ij,ij->j", block, block, dtype=np.float64)
            if torch:
                tensor = torch.from_numpy(block).to(device)
                gram[:] += (tensor.T @ tensor).cpu().numpy()
            else:
                gram[:] += block.T @ block

    if backend == "npu":
        with _npu_precision(device) as torch:
            accumulate(torch)
    else:
        accumulate()
    means = sums / n
    variances = np.maximum((squares - n * means**2) / (n - 1), 0)
    if not np.any(variances > 0):
        raise ValueError("PCA requires nonzero variance.")
    if zero_center:
        gram -= n * np.outer(means, means)
    gram /= n - 1
    std = np.sqrt(variances) if scale else np.ones(p)
    std[std == 0] = 1
    gram /= std[:, None]
    gram /= std[None, :]
    gram = (gram + gram.T) * 0.5
    # Cancellation is intrinsic to covariance PCA. Warn rather than silently
    # treating strongly negative variances as meaningful eigenvalues.
    if zero_center and np.min(np.diag(gram)) < -1e-5 * max(float(np.max(variances)), 1):
        warnings.warn("Covariance subtraction lost precision; use a centered SVD solver for this input.", RuntimeWarning)
    values, vectors = linalg.eigh(gram, subset_by_index=[p - n_comps, p - 1], driver="evr")
    values, vectors = values[::-1].copy(), vectors[:, ::-1].copy()
    if values[0] <= 0:
        raise ValueError("PCA requires nonzero variance.")
    signs = np.sign(vectors[np.abs(vectors).argmax(axis=0), np.arange(n_comps)])
    vectors *= np.where(signs == 0, 1, signs)
    projection = vectors / std[:, None]
    offset = means @ projection if zero_center else np.zeros(n_comps)
    scores = np.empty((n, n_comps), dtype=np.float32)

    def project(torch=None):
        if torch:
            weights = torch.from_numpy(np.asarray(projection, dtype=np.float32)).to(device)
            bias = torch.from_numpy(np.asarray(offset, dtype=np.float32)).to(device)
        for lo, block in blocks(np.float32 if torch else np.float64):
            if torch:
                values = (torch.from_numpy(block).to(device) @ weights - bias).cpu().numpy()
            else:
                values = block @ projection - offset
            scores[lo:lo + len(block)] = values

    if backend == "npu":
        with _npu_precision(device) as torch:
            project(torch)
    else:
        project()
    if not zero_center:
        # Truncated SVD's directions use the uncentered second moment, but
        # explained variance measures the centered variance of projected data.
        values -= n / (n - 1) * (means @ vectors)**2
    values = np.maximum(values, 0)
    total_variance = float((variances / std**2).sum())
    if not zero_center:
        # sklearn/Scanpy TruncatedSVD reports population variance (ddof=0).
        values *= (n - 1) / n
        total_variance *= (n - 1) / n
    loadings = np.zeros((result.n_vars, n_comps), dtype=np.float64)
    loadings[mask] = vectors
    result.obsm["X_pca"] = scores
    result.varm["PCs"] = loadings
    result.uns["pca"] = {
        "params": {"zero_center": zero_center, "use_highly_variable": isinstance(mask_var, str) and mask_var == "highly_variable",
                   "mask_var": mask_var if isinstance(mask_var, str) else None, "scale": scale,
                   "layer": layer, "solver": "covariance_eigh", "backend": backend},
        "variance": values, "variance_ratio": values / total_variance,
        "singlecell_npu": {"wall_seconds": time.perf_counter() - start, "block_rows": block_rows,
                           "gram_dtype": "float32" if backend == "npu" else "float64",
                           "eigensolver": "scipy.linalg.eigh", "allow_hf32": False},
    }
    return result if copy else None
