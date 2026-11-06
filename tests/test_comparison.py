import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from benchmarks.compare_artifacts import compare


def fixture(path, *, nan_expression=False):
    path.mkdir()
    obs = pd.DataFrame({"total_counts": [1., 1., 1.], "n_genes_by_counts": [1, 1, 1],
                        "pct_counts_mt": [0., 0., 0.]}, index=["a", "b", "c"])
    var = pd.DataFrame(index=["g1", "g2", "g3", "g4"])
    data = ad.AnnData(X=None, obs=obs, var=var, shape=(3, 4))
    scores = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    data.obsm["X_pca"] = scores
    data.varm["PCs"] = np.eye(4, 2, dtype=np.float32)
    data.uns["pca"] = {"variance": np.array([1., .5])}
    data.write_h5ad(path / "brain_1m_pca.h5ad")
    expression = np.ones((3, 4), dtype=np.float32)
    if nan_expression:
        expression[0, 0] = np.nan
    np.save(path / "log1p_hvg.npy", expression)
    np.save(path / "pca_scores.npy", scores)


class ComparisonTests(unittest.TestCase):
    def test_identical_duplicate_scores_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            fixture(a)
            fixture(b)
            self.assertTrue(compare(a, b)["all_passed"])

    def test_nan_cannot_silently_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            fixture(a)
            fixture(b, nan_expression=True)
            with self.assertRaisesRegex(AssertionError, "Non-finite"):
                compare(a, b)


if __name__ == "__main__":
    unittest.main()
