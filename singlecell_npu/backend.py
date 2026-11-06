"""Small, explicit dense compute boundary; covariance eigensolve stays on CPU."""
import time

import numpy as np
from scipy import linalg

# Frozen before measurement. These are correctness gates, not tuning targets.
TOLERANCES = dict(gram_relative_frobenius=2e-5, eigenvalue_relative=2e-4,
                  eigenspace_min_cosine=0.9999, eigen_residual=2e-4,
                  orthogonality_abs=2e-4, projection_abs=2e-4,
                  score_mean_abs=2e-4, score_covariance_abs=2e-4,
                  score_covariance_relative=2e-4)


class DenseBackend:
    def __init__(self, device="cpu"):
        self.device = device
        self.torch = None
        self._projection_key = None
        self.info = dict(device=device, eigensolver="scipy.linalg.eigh/evr/cpu/float64",
                         gram_calls=0, projection_calls=0, gram_wall_seconds=0.,
                         projection_wall_seconds=0., eigh_wall_seconds=0.,
                         tolerances=TOLERANCES)
        if device != "cpu":
            import torch
            import torch_npu
            if not device.startswith("npu:"):
                raise ValueError("device must be cpu or npu:<index>")
            self.torch = torch
            self._torch_npu = torch_npu
            self._previous_hf32 = torch.npu.matmul.allow_hf32
            self._previous_threads = torch.get_num_threads()
            self._previous_device = torch.npu.current_device()
            self._previous_jit_compile = not torch.npu.is_jit_compile_false()
            torch.npu.set_device(device)
            torch.npu.set_compile_mode(jit_compile=False)
            torch.npu.matmul.allow_hf32 = False
            torch.set_num_threads(8)
            self.info.update(name=torch.npu.get_device_name(), torch=torch.__version__,
                             torch_npu=torch_npu.__version__, dtype="float32", allow_hf32=False)

    def close(self):
        if self.torch is not None:
            self.torch.npu.matmul.allow_hf32 = self._previous_hf32
            self.torch.set_num_threads(self._previous_threads)
            self.torch.npu.set_compile_mode(jit_compile=self._previous_jit_compile)
            self.torch.npu.set_device(self._previous_device)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def upload(self, array, dtype=np.float32):
        if self.torch is None:
            raise RuntimeError("upload requires an NPU backend")
        value = np.ascontiguousarray(array, dtype=dtype)
        if not value.flags.writeable:
            value = value.copy()
        return self.torch.from_numpy(value).to(self.device)

    def host(self, value):
        return value.detach().cpu().numpy() if self.torch is not None and isinstance(value, self.torch.Tensor) else value

    def gram(self, array):
        start = time.perf_counter()
        if self.torch is None:
            x = np.asarray(array, dtype=np.float64)
            result = x.T @ x
        else:
            x = array if isinstance(array, self.torch.Tensor) else self.upload(array)
            with self.torch.no_grad():
                result = (x.T @ x).cpu().numpy().astype(np.float64)
        self.info["gram_calls"] += 1
        self.info["gram_wall_seconds"] += time.perf_counter() - start
        if self.info["gram_calls"] == 1:
            x64 = np.asarray(self.host(array), dtype=np.float64)
            reference = x64.T @ x64
            error = np.linalg.norm(result - reference) / max(np.linalg.norm(reference), 1e-30)
            if error > TOLERANCES["gram_relative_frobenius"]:
                raise ArithmeticError(f"Gram precision gate failed: {error}")
            self.info["first_gram_relative_error"] = float(error)
        return result

    def eigh_top(self, covariance, k):
        start = time.perf_counter()
        h = len(covariance)
        values, vectors = linalg.eigh(covariance, subset_by_index=[h-k, h-1], driver="evr")
        self.info["eigh_wall_seconds"] += time.perf_counter() - start
        return values[::-1].copy(), vectors[:, ::-1].copy()

    def project(self, array, projection, offset):
        start = time.perf_counter()
        if self.torch is None:
            result = (np.asarray(array, dtype=np.float64) @ projection - offset).astype(np.float32)
        else:
            # Small host arrays are compared to avoid stale projections across datasets.
            if (self._projection_key is None or not np.array_equal(projection, self._projection_key[0])
                    or not np.array_equal(offset, self._projection_key[1])):
                self._projection_key = (projection.copy(), offset.copy())
                self._projection = self.upload(projection)
                self._offset = self.upload(offset)
            with self.torch.no_grad():
                result = (self.upload(array) @ self._projection - self._offset).cpu().numpy()
        self.info["projection_calls"] += 1
        self.info["projection_wall_seconds"] += time.perf_counter() - start
        if self.info["projection_calls"] == 1:
            reference = np.asarray(array, dtype=np.float64) @ projection - offset
            error = np.abs(result - reference).max(initial=0)
            if error > TOLERANCES["projection_abs"]:
                raise ArithmeticError(f"Projection precision gate failed: {error}")
            self.info["first_projection_max_abs_error"] = float(error)
        return result
