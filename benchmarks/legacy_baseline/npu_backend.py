"""Native NPU PCA kernels, with declared FP32 numerical validation tolerances."""
import time
import numpy as np
import torch
import torch_npu
from scipy import linalg

# Declared before validation; compare to CPU float64 references.
TOLERANCES = dict(gram_relative_frobenius=2e-5, eigenvalue_relative=2e-4,
                  eigenspace_min_cosine=0.9999, eigen_residual=2e-4,
                  orthogonality_abs=2e-4, projection_abs=2e-4,
                  score_mean_abs=2e-4, score_covariance_abs=2e-4,
                  score_covariance_relative=2e-4)


class NPUBackend:
    def __init__(self, index=0):
        torch.npu.set_device(index)
        torch.npu.set_compile_mode(jit_compile=False)
        torch_npu.npu.config.allow_internal_format = False
        torch.npu.matmul.allow_hf32 = False
        torch.set_num_threads(8)
        self.device = torch.device(f'npu:{index}')
        self.info = dict(device=str(self.device), name=torch.npu.get_device_name(index),
                         torch=torch.__version__, torch_npu=torch_npu.__version__,
                         dtype='float32', allow_hf32=False, tolerances=TOLERANCES,
                         gram_calls=0, svd_calls=0, projection_calls=0,
                         gram_wall_seconds=0., svd_wall_seconds=0., projection_wall_seconds=0.)
        self.projection = None

    def tensor(self, array):
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)).to(self.device)

    @torch.no_grad()
    def gram(self, array):
        start = time.perf_counter()
        x = self.tensor(array)
        result = (x.T @ x).cpu().numpy().astype(np.float64)
        self.info['gram_calls'] += 1
        self.info['gram_wall_seconds'] += time.perf_counter() - start
        if self.info['gram_calls'] == 1:
            reference = np.asarray(array, dtype=np.float64).T @ np.asarray(array, dtype=np.float64)
            error = np.linalg.norm(result - reference) / np.linalg.norm(reference)
            assert error < TOLERANCES['gram_relative_frobenius'], error
            self.info['first_gram_relative_error'] = float(error)
        return result

    @torch.no_grad()
    def eigh_top(self, covariance, k):
        # torch.linalg.eigh falls back to CPU in this torch_npu version.
        # Native NPU SVD of a positive semidefinite covariance gives its eigenpairs.
        start = time.perf_counter()
        matrix = self.tensor(covariance)
        u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
        values = singular[:k].cpu().numpy().astype(np.float64)
        vectors = u[:, :k].cpu().numpy().astype(np.float64)
        self.info['svd_calls'] += 1
        self.info['svd_wall_seconds'] += time.perf_counter() - start
        ref_val, ref_vec = linalg.eigh(covariance, subset_by_index=[len(covariance)-k, len(covariance)-1])
        ref_val, ref_vec = ref_val[::-1], ref_vec[:, ::-1]
        error = np.max(np.abs(values - ref_val) / np.maximum(ref_val, 1e-12))
        overlap = np.linalg.svd(vectors.T @ ref_vec, compute_uv=False).min()
        assert error < TOLERANCES['eigenvalue_relative'], error
        assert overlap > TOLERANCES['eigenspace_min_cosine'], overlap
        self.info.update(eigenvalue_max_relative_error=float(error), eigenspace_min_cosine=float(overlap))
        return values, vectors

    @torch.no_grad()
    def project(self, array, projection, offset):
        start = time.perf_counter()
        if self.projection is None:
            self.projection = self.tensor(projection)
            self.offset = self.tensor(offset)
        x = self.tensor(array)
        result = (x @ self.projection - self.offset).cpu().numpy()
        self.info['projection_calls'] += 1
        self.info['projection_wall_seconds'] += time.perf_counter() - start
        if self.info['projection_calls'] == 1:
            reference = np.asarray(array, dtype=np.float64) @ projection - offset
            error = np.abs(result - reference).max()
            assert error < TOLERANCES['projection_abs'], error
            self.info['first_projection_max_abs_error'] = float(error)
        return result
