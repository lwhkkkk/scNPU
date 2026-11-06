"""Complete synthetic 10x -> cached preprocessing -> PCA -> artifact readback."""
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from singlecell_npu.streaming import main
from test_io import write_10x


class StreamingTest(unittest.TestCase):
    def test_uncached_build_reuse_same_results(self):
        rng = np.random.default_rng(911)
        x = rng.poisson(rng.uniform(.2, 3, 256), size=(513, 256)).astype(np.int32)
        x[0] = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "counts.h5"
            write_10x(source, sparse.csr_matrix(x), v3=True)
            common = ["--input", str(source), "--device", "cpu", "--n-hvg", "64",
                      "--n-pcs", "10", "--chunk-size", "256"]
            summaries = []
            for name, extra in (("direct", []), ("build", ["--cache", str(root / "cache")]),
                                ("reuse", ["--cache", str(root / "cache")])):
                summaries.append(main(common + ["--output", str(root / name)] + extra))
            self.assertFalse(summaries[1]["cache_reused"])
            self.assertTrue(summaries[2]["cache_reused"])
            for name in ("build", "reuse"):
                for file in ("log1p_hvg.npy", "pca_scores.npy"):
                    np.testing.assert_allclose(np.load(root / name / file), np.load(root / "direct" / file), atol=1e-6)
                result = ad.read_h5ad(root / name / "brain_1m_pca.h5ad")
                self.assertEqual(result.shape, (512, 64))
                self.assertIsNone(result.X)
                self.assertTrue((root / name / "SUCCESS").is_file())


if __name__ == "__main__":
    unittest.main()
