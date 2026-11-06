"""1D-only DMA adapter for TileLang-Ascend 0.1.1.10/cann900.

Adapted from the user's scops-tilelang/scops/kernels/copy_compat.py.
Calls existing wheel C++ helpers with runtime valid lengths; avoids that
wheel's dynamic-region T.copy lowering issue. Deliberately rejects 2D copies.
No modification to the installed compiler is required.
"""

import tilelang.language as T
from tilelang import tvm


def _region(value):
    if isinstance(value, tvm.tir.Buffer):
        return value, 0, value.shape[0]
    if isinstance(value, tvm.tir.BufferRegion):
        return value.buffer, value.region[0].min, value.region[0].extent
    raise TypeError(f"unsupported copy region {type(value)}")


def _call(src, dst):
    sb, si, se = _region(src)
    db, di, de = _region(dst)
    if len(sb.shape) != 1 or len(db.shape) != 1:
        raise ValueError("only contiguous one-dimensional copies are supported")
    load = sb.scope() == "global"
    gm, ub = (sb, db) if load else (db, sb)
    valid = se if load else de
    dtype = {"float32": "float", "int32": "int32_t"}[str(ub.dtype)]
    name = "copy_gm_to_ub" if load else "copy_ub_to_gm"
    args = [
        sb.access_ptr("r", offset=si),
        db.access_ptr("w", offset=di),
        T.cast(gm.shape[0], "uint32"),
        T.cast(1, "uint32"),
        T.cast(valid, "uint32"),
    ]
    if load:
        args.append(T.cast(0, str(ub.dtype)))
    return T.call_extern("handle", f"tl::ascend::{name}<{dtype}, {ub.shape[0]}, 1>", *args)


@T.macro
def copy_1d(src, dst):
    T.barrier_all()
    T.evaluate(_call(src, dst))
    T.barrier_all()
