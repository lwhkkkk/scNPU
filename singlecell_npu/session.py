"""Explicit device snapshots; AnnData itself always retains host arrays."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def _matrix(adata, layer=None):
    if adata.isbacked:
        raise ValueError("Backed AnnData is not supported here; use the streaming pipeline.")
    x = adata.X if layer is None else adata.layers[layer]
    if x is None or not (isinstance(x, np.ndarray) or sparse.issparse(x)):
        raise TypeError("Expected a NumPy array or SciPy sparse matrix.")
    return x


class DeviceSession:
    """Own an FP32 snapshot, optionally resident on an Ascend device.

    Mutations of the source AnnData are deliberately not tracked. ``writeback``
    checks axes, but replaces the selected matrix even if its values changed.
    Sparse input is canonicalized once and never expanded to a dense matrix.
    """

    def __init__(self, adata, *, layer=None, device="npu:0"):
        import torch
        if str(device).startswith("npu"):
            import torch_npu  # noqa: F401
        self._torch = torch
        self.device = torch.device(device)
        self.layer = layer
        self._obs_names = adata.obs_names.copy()
        self._var_names = adata.var_names.copy()
        self.shape = adata.shape
        x = _matrix(adata, layer)
        self.sparse = sparse.issparse(x)
        if self.sparse:
            x = sparse.csr_matrix(x, dtype=np.float32, copy=True)
            x.sum_duplicates()
            x.eliminate_zeros()
            x.sort_indices()
            self._indices, self._indptr = x.indices.copy(), x.indptr.copy()
            rows = np.repeat(np.arange(x.shape[0], dtype=np.int64), np.diff(x.indptr))
            self._rows = torch.from_numpy(rows).to(self.device)
            values = x.data
        else:
            values = np.asarray(x, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Input contains non-finite values or overflows FP32.")
        self._values = torch.from_numpy(np.array(values, dtype=np.float32, order="C", copy=True)).to(self.device)

    def _check_open(self):
        if self._values is None:
            raise RuntimeError("DeviceSession is closed.")

    def normalize_total(self, target_sum=1e4):
        """Normalize rows using all columns in this snapshot; zero rows stay zero."""
        self._check_open()
        if target_sum is None or not np.isfinite(target_sum) or not 0 < target_sum <= np.finfo(np.float32).max:
            raise ValueError("DeviceSession requires a finite positive target_sum.")
        torch = self._torch
        if bool((self._values < 0).any().item()):
            raise ValueError("normalize_total requires nonnegative values.")
        if self.sparse:
            totals = torch.zeros(self.shape[0], dtype=torch.float32, device=self.device)
            totals.scatter_add_(0, self._rows, self._values)
        else:
            totals = self._values.sum(dim=1)
        if not bool(torch.isfinite(totals).all().item()):
            raise ValueError("Row totals overflow FP32; rescale the input on CPU.")
        divisors = torch.where(totals > 0, totals, torch.ones_like(totals))
        # Divide first: target / tiny_total could overflow even when normalized
        # values themselves are bounded by target_sum.
        normalized = self._values / (divisors[self._rows] if self.sparse else divisors[:, None]) * target_sum
        if not bool(torch.isfinite(normalized).all().item()):
            raise ValueError("Normalized values overflow FP32.")
        self._values = normalized
        return self

    def log1p(self, base=None):
        self._check_open()
        if base is not None and (not np.isfinite(base) or base <= 0 or base == 1):
            raise ValueError("Logarithm base must be positive and different from one.")
        if bool((self._values <= -1).any().item()):
            raise ValueError("log1p requires values greater than -1.")
        self._values = self._torch.log1p(self._values)
        if base is not None:
            self._values = self._values / np.log(base)
        return self

    def to_host(self):
        self._check_open()
        values = self._values.detach().cpu().numpy().copy()
        if self.sparse:
            return sparse.csr_matrix((values, self._indices.copy(), self._indptr.copy()), shape=self.shape)
        return values

    def writeback(self, adata):
        if adata.isbacked:
            raise ValueError("Writeback to backed AnnData is unsupported.")
        if adata.is_view:
            raise ValueError("Writeback to an AnnData view is ambiguous; copy it first.")
        if adata.shape != self.shape or not adata.obs_names.equals(self._obs_names) or not adata.var_names.equals(self._var_names):
            raise ValueError("AnnData axes changed since the device snapshot was created.")
        result = self.to_host()
        if self.layer is None:
            adata.X = result
        else:
            adata.layers[self.layer] = result

    def close(self):
        self._values = None
        self._rows = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
