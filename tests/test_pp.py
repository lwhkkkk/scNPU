"""Numerical contracts against Scanpy/sklearn, declared before NPU execution."""
import os
import unittest
import warnings

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

from singlecell_npu import pp
from singlecell_npu.session import DeviceSession

# FP32 operation contracts; do not widen after a failed run.
ELEMENT_ATOL = 2e-4
ELEMENT_RTOL = 2e-5
PCA_REL = 2e-4
SUBSPACE_MIN = 0.9999


def counts():
    rng = np.random.default_rng(73)
    x = rng.poisson(rng.uniform(0.1, 4, 23), (137, 23)).astype(np.float32)
    x[0] = 0
    x[:, -1] = 0
    return x


class APIContract(unittest.TestCase):
    def test_cpu_normalize_log1p_matches_scanpy(self):
        for sparse_input in (False, True):
            x = counts()
            source = ad.AnnData(sparse.csr_matrix(x) if sparse_input else x)
            source.layers["counts"] = source.X.copy()
            original = source.copy()
            expected = source.copy()
            sc.pp.normalize_total(expected, target_sum=1e4, layer="counts")
            sc.pp.log1p(expected, layer="counts", base=2)
            actual = pp.normalize_total(source, target_sum=1e4, layer="counts", copy=True)
            pp.log1p(actual, layer="counts", base=2)
            a, b = actual.layers["counts"], expected.layers["counts"]
            np.testing.assert_allclose(a.toarray() if sparse_input else a, b.toarray() if sparse_input else b)
            np.testing.assert_array_equal(source.X.toarray() if sparse_input else source.X, original.X.toarray() if sparse_input else original.X)
            self.assertEqual(actual.uns["log1p"], expected.uns["log1p"])

    def test_qc_hvg_are_scanpy_exact(self):
        a = ad.AnnData(sparse.csr_matrix(counts()))
        b = a.copy()
        pp.calculate_qc_metrics(a, percent_top=None, inplace=True)
        sc.pp.calculate_qc_metrics(b, percent_top=None, inplace=True)
        np.testing.assert_array_equal(a.obs.total_counts, b.obs.total_counts)
        pp.normalize_total(a)
        pp.log1p(a)
        b = a.copy()
        pp.highly_variable_genes(a, n_top_genes=10, flavor="seurat")
        sc.pp.highly_variable_genes(b, n_top_genes=10, flavor="seurat")
        np.testing.assert_array_equal(a.var.highly_variable, b.var.highly_variable)
        np.testing.assert_allclose(a.var.dispersions_norm, b.var.dispersions_norm, equal_nan=True)

    def test_view_and_mask_guards(self):
        a = ad.AnnData(counts())
        with self.assertRaises(ValueError):
            pp.normalize_total(a[:10])
        self.assertEqual(pp.normalize_total(a[:10], copy=True).n_obs, 10)
        with self.assertRaises(ValueError):
            pp.pca(a, 3, mask_var=np.ones(a.n_vars, dtype=int))
        with self.assertRaises(MemoryError):
            pp.pca(a, 3, max_covariance_bytes=10)
        with self.assertRaises(ValueError):
            pp.pca(a, 3, backend="cuda")

    def test_pca_against_sklearn(self):
        for sparse_input in (False, True):
            for scale in (False, True):
                x = np.log1p(counts()).astype(np.float64)
                a = ad.AnnData(sparse.csr_matrix(x) if sparse_input else x)
                a.var["hvg"] = np.arange(a.n_vars) < 17
                result = pp.pca(a, 7, copy=True, mask_var="hvg", scale=scale, block_rows=19)
                z = x[:, :17]
                if scale:
                    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=1)
                ref = PCA(n_components=7, svd_solver="full").fit(z)
                vectors = result.varm["PCs"][:17]
                self.assertGreater(np.linalg.svd(vectors.T @ ref.components_.T, compute_uv=False).min(), 1 - 1e-10)
                np.testing.assert_allclose(result.uns["pca"]["variance"], ref.explained_variance_, rtol=1e-10)
                np.testing.assert_allclose(result.uns["pca"]["variance_ratio"], ref.explained_variance_ratio_, rtol=1e-10)
                np.testing.assert_allclose(result.obsm["X_pca"].var(axis=0, ddof=1), ref.explained_variance_, rtol=1e-6)
                self.assertTrue(np.all(result.varm["PCs"][17:] == 0))
                self.assertNotIn("X_pca", a.obsm)

    def test_uncentered(self):
        x = np.log1p(counts()).astype(np.float64)
        a = ad.AnnData(x)
        pp.pca(a, 4, zero_center=False, block_rows=31)
        ref = TruncatedSVD(n_components=4, algorithm="arpack").fit(x)
        self.assertGreater(np.linalg.svd(a.varm["PCs"].T @ ref.components_.T, compute_uv=False).min(), 1 - 1e-9)
        np.testing.assert_allclose(a.uns["pca"]["variance"], ref.explained_variance_, rtol=1e-10)
        np.testing.assert_allclose(a.uns["pca"]["variance_ratio"], ref.explained_variance_ratio_, rtol=1e-10)

    def test_session_canonicalization_and_snapshot(self):
        # Duplicate indices must be summed BEFORE the nonlinear transform.
        x = sparse.csr_matrix((np.array([1, 2, 4, 0], dtype=np.float32), [2, 2, 0, 1], [0, 3, 4, 4]), shape=(3, 4))
        a = ad.AnnData(x)
        ref = np.log1p(np.array([[4, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32) * np.array([[1e4 / 7], [1], [1]]))
        with DeviceSession(a, device="cpu") as session:
            session.normalize_total().log1p()
            np.testing.assert_allclose(session.to_host().toarray(), ref, atol=ELEMENT_ATOL, rtol=ELEMENT_RTOL)
            with self.assertRaises(ValueError):
                session.writeback(a[:2].copy())
            session.writeback(a)
        with self.assertRaises(RuntimeError):
            session.to_host()
        self.assertTrue(sparse.isspmatrix_csr(a.X))

    def test_session_validation(self):
        a = ad.AnnData(counts())
        with DeviceSession(a, device="cpu") as session:
            for target in (0, -1, None, np.inf):
                with self.assertRaises(ValueError):
                    session.normalize_total(target)
            with self.assertRaises(ValueError):
                session.log1p(1)
        tiny = ad.AnnData(np.array([[1e-38, 2e-38], [0, 0]], dtype=np.float32))
        with DeviceSession(tiny, device="cpu") as session:
            actual = session.normalize_total().to_host()
            np.testing.assert_allclose(actual[0], [1e4 / 3, 2e4 / 3], rtol=ELEMENT_RTOL)
            self.assertTrue(np.isfinite(actual).all())


@unittest.skipUnless(os.environ.get("SINGLECELL_NPU_TEST_DEVICE") == "1", "Set SINGLECELL_NPU_TEST_DEVICE=1 on an Ascend host")
class NPUContract(unittest.TestCase):
    def test_sparse_dense_normalize_log1p(self):
        for sparse_input in (False, True):
            x = counts()
            a = ad.AnnData(sparse.csr_matrix(x) if sparse_input else x)
            ref = a.copy()
            sc.pp.normalize_total(ref, target_sum=1e4)
            sc.pp.log1p(ref)
            with DeviceSession(a) as session:
                session.normalize_total().log1p().writeback(a)
            actual = a.X.toarray() if sparse_input else a.X
            expected = ref.X.toarray() if sparse_input else ref.X
            np.testing.assert_allclose(actual, expected, atol=ELEMENT_ATOL, rtol=ELEMENT_RTOL)
            # Exercise one-call wrappers as well as resident chained operations.
            b = ad.AnnData(sparse.csr_matrix(x) if sparse_input else x)
            pp.normalize_total(b, backend="npu")
            pp.log1p(b, backend="npu")
            np.testing.assert_allclose(b.X.toarray() if sparse_input else b.X, expected, atol=ELEMENT_ATOL, rtol=ELEMENT_RTOL)

    def test_pca_subspace_and_scores(self):
        x = np.log1p(counts())
        for scale in (False, True):
            a = ad.AnnData(sparse.csr_matrix(x))
            ref = pp.pca(a, 7, copy=True, scale=scale, block_rows=19)
            pp.pca(a, 7, scale=scale, backend="npu", block_rows=19)
            overlap = np.linalg.svd(a.varm["PCs"].T @ ref.varm["PCs"], compute_uv=False).min()
            self.assertGreater(overlap, SUBSPACE_MIN)
            np.testing.assert_allclose(a.uns["pca"]["variance"], ref.uns["pca"]["variance"], rtol=PCA_REL)
            np.testing.assert_allclose(a.obsm["X_pca"], ref.obsm["X_pca"], atol=ELEMENT_ATOL, rtol=PCA_REL)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some cells have zero counts")
        unittest.main(verbosity=2)
