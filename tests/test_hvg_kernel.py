"""CPU contract tests always run; device accuracy tests skip without Ascend."""

import importlib.util

import numpy as np
import pytest

from benchmarks.bench_hvg_kernel import cpu_path, make_case, validate_host_case


def test_all_gene_totals_not_hvg_row_sums():
    case = (
        np.array([2, 3], dtype=np.float32), np.array([0, 2], dtype=np.int32),
        np.array([0, 2], dtype=np.int32), np.array([100], dtype=np.float32), 3,
    )
    validate_host_case(case)
    np.testing.assert_allclose(cpu_path(case), [[np.log1p(200), 0, np.log1p(300)]], rtol=1e-6)


def test_host_rejects_duplicates_and_wrong_total():
    values = np.array([2, 3], dtype=np.float32)
    ptr = np.array([0, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="duplicate"):
        validate_host_case((values, np.array([1, 1], dtype=np.int32), ptr, np.array([5], dtype=np.float32), 3))
    with pytest.raises(ValueError, match="smaller"):
        validate_host_case((values, np.array([0, 1], dtype=np.int32), ptr, np.array([4], dtype=np.float32), 3))


def _npu_modules():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available() or importlib.util.find_spec("tilelang") is None:
        pytest.skip("Ascend NPU with TileLang required")
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    return torch


@pytest.mark.parametrize("rows,genes,density", [(1, 1, .0), (11, 17, .2), (35, 257, .3), (67, 2000, .08), (13, 6001, .9)])
def test_npu_kernel_against_fp64(rows, genes, density):
    _npu_modules()
    from benchmarks.bench_hvg_kernel import device_inputs
    from singlecell_npu.kernels import hvg_to_dense

    case = make_case(rows, genes, density)
    validate_host_case(case)
    inputs = device_inputs(case)
    expected = cpu_path(case, dtype=np.float64)
    for _ in range(2):
        result = hvg_to_dense(*inputs, n_genes=genes, max_row_nnz=int(np.diff(case[2]).max(initial=0))).cpu().numpy()
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-6)
        assert np.array_equal(result == 0, expected == 0)


def test_npu_rejects_metadata_without_device_value_scan():
    torch = _npu_modules()
    from benchmarks.bench_hvg_kernel import device_inputs
    from singlecell_npu.kernels import hvg_to_dense

    values, indices, ptr, totals = device_inputs(make_case(3, 17, .1))
    with pytest.raises(TypeError, match="int32"):
        hvg_to_dense(values, indices.to(torch.int64), ptr, totals, n_genes=17, max_row_nnz=17)
    with pytest.raises(ValueError, match="8192"):
        hvg_to_dense(values, indices, ptr, totals, n_genes=9000, max_row_nnz=17)


def test_npu_variable_nnz_reuses_specialization():
    _npu_modules()
    from benchmarks.bench_hvg_kernel import device_inputs
    from singlecell_npu.kernels import hvg_to_dense
    from singlecell_npu.kernels._hvg_tilelang import hvg_kernel

    kernel = hvg_kernel(35, 257, 320, 10000.0)
    for density in (.01, .7):
        case = make_case(35, 257, density)
        result = hvg_to_dense(*device_inputs(case), n_genes=257, max_row_nnz=257).cpu().numpy()
        np.testing.assert_allclose(result, cpu_path(case, dtype=np.float64), rtol=2e-5, atol=2e-6)
        assert hvg_kernel(35, 257, 320, 10000.0) is kernel


def test_npu_fractional_total_and_empty_batch():
    torch = _npu_modules()
    from benchmarks.bench_hvg_kernel import device_inputs
    from singlecell_npu.kernels import hvg_to_dense

    case = (np.array([.125], dtype=np.float32), np.array([0], dtype=np.int32),
            np.array([0, 1], dtype=np.int32), np.array([.25], dtype=np.float32), 3)
    result = hvg_to_dense(*device_inputs(case), n_genes=3, max_row_nnz=1).cpu().numpy()
    np.testing.assert_allclose(result, cpu_path(case, dtype=np.float64), rtol=2e-5, atol=2e-6)
    empty = torch.empty(0, device="npu", dtype=torch.float32)
    indices = torch.empty(0, device="npu", dtype=torch.int32)
    ptr = torch.zeros(1, device="npu", dtype=torch.int32)
    result = hvg_to_dense(empty, indices, ptr, empty, n_genes=3, max_row_nnz=0)
    assert result.shape == (0, 3)
