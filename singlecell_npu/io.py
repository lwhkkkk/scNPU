"""10x HDF5 and immutable, canonical, block-local CSR caches.

Global offsets remain int64. Only independently bounded blocks use int32 indices.
The cache is optional and contains no pickle or executable metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

FORMAT_VERSION = 1
INT32_MAX = np.iinfo(np.int32).max


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


class TenXReader:
    """Read one legacy genome or a v3 Gene Expression matrix by cell blocks."""

    def __init__(self, path, *, rows_per_block=8192, max_block_nnz=32_000_000):
        if rows_per_block < 1 or not 1 <= max_block_nnz <= INT32_MAX:
            raise ValueError("Positive row limit and int32-safe nnz limit required")
        self.path = Path(path)
        self.rows_per_block = int(rows_per_block)
        self.max_block_nnz = int(max_block_nnz)
        with h5py.File(self.path, "r") as handle:
            groups = [key for key in handle if isinstance(handle[key], h5py.Group)
                      and {"data", "indices", "indptr", "shape"} <= set(handle[key])]
            if len(groups) != 1:
                raise ValueError("Expected exactly one 10x matrix/genome group")
            self.group = groups[0]
            group = handle[self.group]
            if any(group[key].dtype.kind not in "iu" for key in ("shape", "indices", "indptr")):
                raise ValueError("10x shape, indices and indptr must use integer dtypes")
            if group["data"].dtype.kind not in "iuf":
                raise ValueError("10x counts must use a real numeric dtype")
            self.n_genes, self.n_cells = map(int, group["shape"][:])
            self.ptr = group["indptr"][:].astype(np.int64)
            if (self.n_genes < 1 or self.n_genes > INT32_MAX or self.n_cells < 1
                    or len(self.ptr) != self.n_cells + 1 or self.ptr[0] != 0
                    or np.any(self.ptr[1:] < self.ptr[:-1])
                    or self.ptr[-1] != len(group["data"])
                    or len(group["indices"]) != len(group["data"])):
                raise ValueError("Invalid 10x shape, offsets, or index lengths")
            if "features" in group:
                features = group["features"]
                if "feature_type" in features and np.any(features["feature_type"].asstr()[:] != "Gene Expression"):
                    raise ValueError("Mixed feature types require explicit upstream Gene Expression selection")
                self.ids = features["id"].asstr()[:]
                self.names = features["name"].asstr()[:]
            else:
                self.ids = group["genes"].asstr()[:]
                self.names = group["gene_names"].asstr()[:]
            self.barcodes = group["barcodes"].asstr()[:]
            if len(self.ids) != self.n_genes or len(self.names) != self.n_genes or len(self.barcodes) != self.n_cells:
                raise ValueError("Feature/barcode metadata has inconsistent length")

    def ranges(self):
        start = 0
        while start < self.n_cells:
            stop = min(start + self.rows_per_block, self.n_cells)
            stop = min(stop, int(np.searchsorted(self.ptr, self.ptr[start] + self.max_block_nnz, side="right")) - 1)
            if stop <= start:
                raise ValueError(f"Row {start} exceeds max_block_nnz; increase the explicit bound")
            yield start, stop
            start = stop

    def blocks(self):
        with h5py.File(self.path, "r") as handle:
            group = handle[self.group]
            for start, stop in self.ranges():
                lo, hi = int(self.ptr[start]), int(self.ptr[stop])
                # Canonicalize in float64, preventing integer duplicate-sum overflow.
                source_values = group["data"][lo:hi]
                if source_values.dtype.kind in "iu" and np.any(source_values > 2**53):
                    raise ValueError("Integer counts exceed exact FP64 representation")
                values = source_values.astype(np.float64)
                indices = group["indices"][lo:hi]
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError("Counts must be finite and nonnegative")
                if np.any(indices < 0) or np.any(indices >= self.n_genes):
                    raise ValueError("CSR column index outside matrix shape")
                matrix = sparse.csr_matrix((values, indices.astype(np.int32),
                                           (self.ptr[start:stop + 1] - lo).astype(np.int32)),
                                          shape=(stop - start, self.n_genes))
                matrix.sum_duplicates()
                matrix.eliminate_zeros()
                if not np.isfinite(matrix.data).all():
                    raise ValueError("Count accumulation overflow")
                yield start, stop, matrix


class CSRCache:
    """A source-hash-bound cache. Incomplete caches are never accepted for reads.

    Files are treated as trusted local storage, not as an adversarial interchange
    format. Source identity and all matrix dimensions are checked on every run.
    """

    def __init__(self, path, reader, source_sha256):
        self.path = Path(path)
        self.reader = reader
        self.identity = dict(format=FORMAT_VERSION, source_sha256=source_sha256,
                             shape=[reader.n_cells, reader.n_genes], group=reader.group,
                             rows_per_block=reader.rows_per_block, max_block_nnz=reader.max_block_nnz)
        self.entries = []
        self.ready = False
        if self.path.exists():
            manifest = self.path / "manifest.json"
            if not manifest.is_file():
                raise ValueError(f"Incomplete cache at {self.path}; use a new cache path")
            metadata = json.loads(manifest.read_text())
            if metadata.get("identity") != self.identity:
                raise ValueError("Cache identity mismatch (source hash, shape, or block policy)")
            self.entries = metadata["blocks"]
            ranges = list(reader.ranges())
            if [(item["start"], item["stop"]) for item in self.entries] != ranges:
                raise ValueError("Cache block coverage is incomplete or inconsistent")
            self.ready = True
        else:
            self.path.mkdir(parents=True, exist_ok=False)

    def append(self, start, stop, matrix):
        if self.ready:
            raise RuntimeError("A complete cache is immutable")
        number = len(self.entries)
        prefix = f"{number:06d}"
        data = matrix.data
        # Compact integer UMI counts without changing fractional/large counts.
        if np.all(data <= INT32_MAX) and np.all(data == np.floor(data)):
            data = data.astype(np.int32)
        else:
            data = data.astype(np.float64, copy=False)
        for name, array in (("data", data), ("indices", matrix.indices.astype(np.int32)),
                            ("indptr", matrix.indptr.astype(np.int32))):
            np.save(self.path / f"{prefix}-{name}.npy", array, allow_pickle=False)
        self.entries.append(dict(start=int(start), stop=int(stop), nnz=matrix.nnz))

    def finish(self):
        if [(e["start"], e["stop"]) for e in self.entries] != list(self.reader.ranges()):
            raise RuntimeError("Cannot publish an incomplete cache")
        pending = self.path / "manifest.pending"
        pending.write_text(json.dumps(dict(identity=self.identity, blocks=self.entries), indent=2) + "\n")
        pending.replace(self.path / "manifest.json")
        self.ready = True

    def blocks(self):
        if not self.ready:
            raise RuntimeError("Cache has no completed manifest")
        for number, entry in enumerate(self.entries):
            prefix = f"{number:06d}"
            data, indices, ptr = [np.load(self.path / f"{prefix}-{name}.npy", mmap_mode="r", allow_pickle=False)
                                  for name in ("data", "indices", "indptr")]
            rows = entry["stop"] - entry["start"]
            if (data.ndim != 1 or indices.ndim != 1 or ptr.ndim != 1
                    or len(data) != entry["nnz"] or len(indices) != len(data)
                    or len(ptr) != rows + 1 or ptr[0] != 0 or ptr[-1] != len(data)
                    or data.dtype not in (np.dtype("int32"), np.dtype("float64"))
                    or indices.dtype != np.dtype("int32") or ptr.dtype != np.dtype("int32")
                    or np.any(ptr[1:] < ptr[:-1]) or np.any(indices < 0)
                    or np.any(indices >= self.reader.n_genes)
                    or not np.isfinite(data).all() or np.any(data <= 0)):
                raise ValueError("Invalid cached CSR array layout")
            matrix = sparse.csr_matrix(
                (data, indices, ptr), shape=(rows, self.reader.n_genes), copy=False)
            if not matrix.has_canonical_format:
                raise ValueError("Cache must contain sorted, unique CSR coordinates")
            yield entry["start"], entry["stop"], matrix
