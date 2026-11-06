"""AIV-only fused CSR packing prototype; no Cube/low-precision arithmetic."""

import tilelang
import tilelang.language as T

from ._copy_1d import copy_1d

CONFIG = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
}


@tilelang.jit(out_idx=[4], pass_configs=CONFIG)
def hvg_kernel(Rows, D, P, target, K=256):
    N = T.symbolic("nnz")
    @T.prim_func
    def main(
        Values: T.Tensor((N,), "float32"),
        Indices: T.Tensor((N,), "int32"),
        Ptr: T.Tensor((Rows + 1,), "int32"),
        Totals: T.Tensor((Rows,), "float32"),
        Y: T.Tensor((Rows * D,), "float32"),
    ):
        with T.Kernel(T.ceildiv(Rows, 2), is_npu=True) as (cid, vid):
            dense = T.alloc_ub((P,), "float32")
            tmp = T.alloc_ub((P,), "float32")
            out = T.alloc_ub((P,), "float32")
            mask = T.alloc_ub((max(32, P // 8),), "uint8")
            counts = T.alloc_ub((K,), "float32")
            columns = T.alloc_ub((K,), "int32")
            meta = T.alloc_ub((2,), "int32")
            total = T.alloc_ub((1,), "float32")
            with T.Scope("V"):
                row = cid * 2 + vid
                if row < Rows:
                    copy_1d(Ptr[row : row + 2], meta)
                    copy_1d(Totals[row : row + 1], total)
                    T.tile.fill(dense, 0.0)
                    T.barrier_all()
                    length = meta[1] - meta[0]
                    # Only UB scalar scatter: all global reads/writes use DMA.
                    # The wheel has gather but no public vector scatter primitive.
                    for k in T.serial(T.ceildiv(length, K)):
                        if k * K < length:
                            valid = T.min(T.cast(K, "int32"), T.cast(length - k * K, "int32"))
                            offset = meta[0] + k * K
                            copy_1d(Values[offset : offset + valid], counts[0:valid])
                            copy_1d(Indices[offset : offset + valid], columns[0:valid])
                            for j in T.serial(valid):
                                dense[columns[j]] = counts[j]
                            T.barrier_all()
                    if total[0] > 0.0:
                        T.tile.div(dense, dense, total[0])
                        T.tile.mul(dense, dense, target)
                        # Preserve tiny positive log1p inputs: ln(1+x) alone
                        # loses them in FP32. Second-order error <3.34e-13
                        # for 0 <= x < 1e-4; adapted from scops math_compat.py.
                        T.tile.add(tmp, dense, 1.0)
                        T.tile.ln(out, tmp)
                        T.tile.mul(tmp, dense, dense)
                        T.tile.mul(tmp, tmp, -0.5)
                        T.tile.add(tmp, tmp, dense)
                        T.tile.compare(mask, dense, 1e-4, "LT")
                        T.tile.select(out, mask, tmp, out, "VSEL_TENSOR_TENSOR_MODE")
                    else:
                        T.tile.fill(out, 0.0)
                    copy_1d(out[0:D], Y[row * D : (row + 1) * D])

    return main
