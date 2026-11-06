"""Checked metadata boundary for the optional FP32 HVG packing prototype."""

import math


def hvg_to_dense(
    values, indices, indptr, totals, *, n_genes, max_row_nnz, target_sum=10000.0
):
    """Pack canonical HVG CSR into dense log1p normalized expression on NPU.

    ``totals`` MUST contain all-gene totals before HVG selection. The caller
    must validate CSR values/indices/offsets on the host before transfer: finite
    nonnegative float32 counts/totals, sorted unique indices in [0, n_genes),
    nondecreasing zero-based int32 offsets ending at len(values), and the true
    maximum row length <= max_row_nnz. A zero total is valid only for a row
    whose counts are zero. No device data is copied back by this function.

    This experimental implementation supports 1..8192 selected genes. Empty
    inputs use torch.zeros. All nonempty rows execute one TileLang kernel;
    row length is chunked and is not limited to 4096. Output stays on device.
    """
    import torch

    if not isinstance(n_genes, int) or not 1 <= n_genes <= 8192:
        raise ValueError("prototype n_genes must be an integer in [1, 8192]")
    if not isinstance(max_row_nnz, int) or not 0 <= max_row_nnz <= n_genes:
        raise ValueError("max_row_nnz must be an integer in [0, n_genes]")
    if not math.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("target_sum must be positive and finite")
    tensors = (values, indices, indptr, totals)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("all inputs must be torch tensors")
    if any(x.ndim != 1 or not x.is_contiguous() for x in tensors):
        raise ValueError("inputs must be contiguous one-dimensional tensors")
    if values.dtype != torch.float32 or totals.dtype != torch.float32:
        raise TypeError("values and all-gene totals must be float32")
    if indices.dtype != torch.int32 or indptr.dtype != torch.int32:
        raise TypeError("indices and local rebased indptr must be int32")
    if values.device.type != "npu" or any(x.device != values.device for x in tensors):
        raise ValueError("all inputs must reside on the same NPU")
    rows = totals.numel()
    if indptr.numel() != rows + 1 or indices.numel() != values.numel():
        raise ValueError("incompatible CSR/totals shapes")
    if values.numel() > 2**31 - 1 or rows * n_genes > 2**31 - 1:
        raise ValueError("prototype requires int32-addressable local blocks")
    if values.numel() and (rows == 0 or max_row_nnz == 0):
        raise ValueError("nonempty values require nonempty rows and max_row_nnz > 0")
    if not values.numel():
        return torch.zeros((rows, n_genes), device=values.device, dtype=torch.float32)
    from ._hvg_tilelang import hvg_kernel

    padded_genes = (n_genes + 63) // 64 * 64
    # nnz is a symbolic dimension: streaming blocks must not each trigger JIT.
    kernel = hvg_kernel(rows, n_genes, padded_genes, float(target_sum))
    return kernel(values, indices, indptr, totals).reshape(rows, n_genes)
