import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

from singlecell_npu.io import CSRCache, TenXReader, sha256_file


def write_10x(path, matrix=None, *, v3=False):
    if matrix is None:
        # Empty row, duplicate indices, stored zero, unsorted columns.
        matrix = sparse.csr_matrix((np.array([1, 3, 0, 4, 2, 1]),
                                    np.array([2, 0, 1, 2, 1, 1]),
                                    np.array([0, 4, 4, 6])), shape=(3, 4))
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix" if v3 else "genome")
        for key, value in dict(data=matrix.data, indices=matrix.indices.astype(np.int64),
                               indptr=matrix.indptr.astype(np.int64), shape=matrix.shape[::-1]).items():
            group.create_dataset(key, data=value)
        group.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(matrix.shape[0])], dtype="S"))
        features = group.create_group("features") if v3 else group
        features.create_dataset("id" if v3 else "genes", data=np.array([f"id{i}" for i in range(matrix.shape[1])], dtype="S"))
        features.create_dataset("name" if v3 else "gene_names", data=np.array([f"g{i}" for i in range(matrix.shape[1])], dtype="S"))
        if v3:
            features.create_dataset("feature_type", data=np.array(["Gene Expression"] * matrix.shape[1], dtype="S"))


class IOTests(unittest.TestCase):
    def test_canonicalization_blocks_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            write_10x(path)
            reader = TenXReader(path, rows_per_block=2, max_block_nnz=4)
            blocks = list(reader.blocks())
            self.assertEqual([(s, e) for s, e, _ in blocks], [(0, 2), (2, 3)])
            expected = np.array([[3, 0, 5, 0], [0, 0, 0, 0], [0, 3, 0, 0]])
            np.testing.assert_array_equal(sparse.vstack([m for _, _, m in blocks]).toarray(), expected)
            cache = CSRCache(Path(tmp) / "cache", reader, sha256_file(path))
            for s, e, m in blocks:
                self.assertTrue(m.has_canonical_format)
                self.assertEqual(m.indptr.dtype, np.int32)
                cache.append(s, e, m)
            with self.assertRaises(RuntimeError):
                list(cache.blocks())
            cache.finish()
            reused = CSRCache(cache.path, reader, sha256_file(path))
            self.assertTrue(reused.ready)
            np.testing.assert_array_equal(sparse.vstack([m for _, _, m in reused.blocks()]).toarray(), expected)
            with self.assertRaises(ValueError):
                CSRCache(cache.path, reader, "different-input-hash")

    def test_fractional_and_large_duplicate_counts_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            matrix = sparse.csr_matrix((np.array([2**31-1, 2**31-1, .25]),
                                        [0, 0, 1], [0, 3]), shape=(1, 2))
            write_10x(path, matrix, v3=True)
            reader = TenXReader(path)
            cache = CSRCache(Path(tmp) / "cache", reader, sha256_file(path))
            for s, e, m in reader.blocks():
                cache.append(s, e, m)
            cache.finish()
            result = next(cache.blocks())[2]
            np.testing.assert_array_equal(result.toarray(), [[2*(2**31-1), .25]])
            self.assertEqual(result.data.dtype, np.float64)

    def test_guards_incomplete_cache_negative_and_long_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            write_10x(path)
            reader = TenXReader(path, max_block_nnz=2)
            with self.assertRaises(ValueError):
                list(reader.blocks())
            reader = TenXReader(path)
            cache = CSRCache(Path(tmp) / "cache", reader, sha256_file(path))
            with self.assertRaises(ValueError):
                CSRCache(cache.path, reader, sha256_file(path))
            with h5py.File(path, "a") as handle:
                handle["genome/data"][0] = -1
            with self.assertRaises(ValueError):
                list(reader.blocks())

    def test_corrupt_cache_and_float_indices_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.h5"
            write_10x(path)
            reader = TenXReader(path)
            cache = CSRCache(Path(tmp) / "cache", reader, sha256_file(path))
            for start, stop, matrix in reader.blocks():
                cache.append(start, stop, matrix)
            cache.finish()
            indices = np.load(cache.path / "000000-indices.npy")
            indices[0] = reader.n_genes
            np.save(cache.path / "000000-indices.npy", indices)
            with self.assertRaises(ValueError):
                list(cache.blocks())
            with h5py.File(path, "a") as handle:
                values = handle["genome/indices"][:].astype(np.float64)
                del handle["genome/indices"]
                handle["genome"].create_dataset("indices", data=values)
            with self.assertRaises(ValueError):
                TenXReader(path)


if __name__ == "__main__":
    unittest.main()
