import os
import unittest

import numpy as np

from singlecell_npu.backend import DenseBackend


class BackendTests(unittest.TestCase):
    def exercise(self, device):
        rng = np.random.default_rng(741)
        x = rng.normal(size=(129, 19)).astype(np.float32)
        with DenseBackend(device) as backend:
            np.testing.assert_allclose(backend.gram(x), x.astype(np.float64).T @ x,
                                       atol=2e-4, rtol=2e-5)
            for _ in range(2):
                projection = rng.normal(size=(19, 7))
                offset = rng.normal(size=7)
                np.testing.assert_allclose(backend.project(x, projection, offset),
                                           x.astype(np.float64) @ projection - offset,
                                           atol=2e-4, rtol=2e-5)
            covariance = x.astype(np.float64).T @ x
            values, vectors = backend.eigh_top(covariance, 7)
            np.testing.assert_allclose(covariance @ vectors, vectors * values, atol=1e-10)

    def test_cpu(self):
        self.exercise("cpu")

    @unittest.skipUnless(os.environ.get("SINGLECELL_NPU_TEST_DEVICE") == "1", "NPU opt-in")
    def test_npu_and_state_restoration(self):
        import torch
        import torch_npu  # noqa: F401
        before = (torch.npu.current_device(), torch.npu.matmul.allow_hf32,
                  torch.npu.is_jit_compile_false(), torch.get_num_threads())
        self.exercise("npu:0")
        after = (torch.npu.current_device(), torch.npu.matmul.allow_hf32,
                 torch.npu.is_jit_compile_false(), torch.get_num_threads())
        self.assertEqual(before, after)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with DenseBackend("npu:0"):
                raise RuntimeError("intentional")
        self.assertEqual(before, (torch.npu.current_device(), torch.npu.matmul.allow_hf32,
                                 torch.npu.is_jit_compile_false(), torch.get_num_threads()))


if __name__ == "__main__":
    unittest.main()
