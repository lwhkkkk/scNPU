"""Isolated HVG packing benchmark; run from the project root in the NPU env.

python -m benchmarks.bench_hvg_kernel --rows 8192 --genes 2000 --density .08
The optional --input-npz accepts values/indices/indptr/totals/n_genes arrays.
Do not provide HVG-only totals: totals must precede HVG selection.
"""

import argparse
import csv
import gzip
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def make_case(rows=257, genes=2000, density=0.08, seed=137):
    """Canonical sparse counts plus omitted-gene counts in all-gene totals."""
    rng = np.random.default_rng(seed)
    lengths = rng.binomial(genes, density, size=rows)
    edge_lengths = [0, 1, 2, 63, 64, 65, 255, 256, 257, genes]
    lengths[: min(rows, len(edge_lengths))] = np.minimum(edge_lengths[:rows], genes)
    ptr64 = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    if ptr64[-1] > np.iinfo(np.int32).max:
        raise ValueError("test case exceeds local int32 CSR range")
    indices = np.concatenate([np.sort(rng.choice(genes, n, replace=False)) for n in lengths])
    values = (rng.poisson(3, size=len(indices)) + 1).astype(np.float32)
    ptr = ptr64.astype(np.int32)
    indices = indices.astype(np.int32)
    sums = np.asarray(sp.csr_matrix((values, indices, ptr), shape=(rows, genes)).sum(1)).ravel()
    totals = (sums * np.float32(2) + np.float32(17)).astype(np.float32)
    totals[lengths == 0] = 0
    # Exercise log1p cancellation, not just usual integer-count values.
    if rows > 1 and lengths[1]:
        values[ptr[1]] = np.float32(1e-5)
        totals[1] = np.float32(1e8)
    return values, indices, ptr, totals, genes


def validate_host_case(case):
    values, indices, ptr, totals, genes = case
    if values.dtype != np.float32 or totals.dtype != np.float32:
        raise TypeError("values/totals must be float32")
    if indices.dtype != np.int32 or ptr.dtype != np.int32:
        raise TypeError("local indices/indptr must be int32")
    if not all(a.ndim == 1 and a.flags.c_contiguous for a in case[:4]):
        raise ValueError("arrays must be contiguous and one-dimensional")
    if len(ptr) != len(totals) + 1 or len(indices) != len(values):
        raise ValueError("incompatible array lengths")
    if ptr[0] != 0 or ptr[-1] != len(values) or np.any(np.diff(ptr.astype(np.int64)) < 0):
        raise ValueError("invalid rebased indptr")
    if not np.isfinite(values).all() or not np.isfinite(totals).all():
        raise ValueError("nonfinite counts/totals")
    if np.any(values < 0) or np.any(totals < 0):
        raise ValueError("negative counts/totals")
    if np.any(indices < 0) or np.any(indices >= genes):
        raise ValueError("index out of bounds")
    for row in range(len(totals)):
        a, b = int(ptr[row]), int(ptr[row + 1])
        if np.any(np.diff(indices[a:b]) <= 0):
            raise ValueError("CSR must be sorted and duplicate free")
        if totals[row] == 0 and np.any(values[a:b] != 0):
            raise ValueError("zero total with positive HVG counts")
        # Small allowance for all-gene FP32 accumulation rounding.
        if values[a:b].sum(dtype=np.float64) > float(totals[row]) * (1 + 1e-5):
            raise ValueError("all-gene total is smaller than selected count sum")


def real_tenx_case(path, hvg_table, rows):
    """First real cell block; verify table feature order before selecting HVGs."""
    from singlecell_npu.io import TenXReader

    reader = TenXReader(path, rows_per_block=rows)
    opener = gzip.open if str(hvg_table).endswith(".gz") else open
    with opener(hvg_table, "rt") as handle:
        records = list(csv.DictReader(handle))
    if [item["gene_id"] for item in records] != list(reader.ids):
        raise ValueError("HVG table feature IDs/order do not match the 10x source")
    selected = np.asarray([item["highly_variable"].lower() == "true" for item in records])
    _, _, raw = next(reader.blocks())
    totals = np.asarray(raw.sum(axis=1)).ravel().astype(np.float32)
    subset = raw[:, selected].tocsr()
    subset.sort_indices()
    return (subset.data.astype(np.float32), subset.indices.astype(np.int32),
            subset.indptr.astype(np.int32), totals, subset.shape[1])


def cpu_path(case, target=10000.0, dtype=np.float32):
    values, indices, ptr, totals, genes = case
    lengths = np.diff(ptr)
    row_ids = np.repeat(np.arange(len(totals)), lengths)
    denominator = np.where(totals > 0, totals, 1).astype(dtype)
    # dtype is explicit: FP32 CPU timing, independent FP64 accuracy reference.
    data = np.log1p(values.astype(dtype) / denominator[row_ids] * dtype(target))
    return sp.csr_matrix((data, indices, ptr), shape=(len(totals), genes)).toarray()


def device_inputs(case, device="npu:0"):
    import torch

    return tuple(torch.from_numpy(a).to(device) for a in case[:4])


def native_metadata(case, device="npu:0"):
    import torch

    _, indices, ptr, totals, genes = case
    rows = np.repeat(np.arange(len(totals), dtype=np.int64), np.diff(ptr))
    flat = rows * genes + indices
    return torch.from_numpy(rows).to(device), torch.from_numpy(flat).to(device)


def torch_native(inputs, metadata, genes, target=10000.0):
    """Mature torch ops, cached host row/flat indices; no sparse NPU dependency."""
    import torch

    values, _, _, totals = inputs
    rows, flat = metadata
    denominator = torch.where(totals > 0, totals, torch.ones_like(totals))
    data = torch.log1p(values / denominator[rows] * target)
    out = torch.zeros(totals.numel() * genes, dtype=torch.float32, device=values.device)
    out.scatter_(0, flat, data)
    return out.reshape(totals.numel(), genes)


def summarize_ms(samples):
    return {"median": statistics.median(samples), "min": min(samples), "max": max(samples)}


def run_benchmark(case, repeats=10):
    import torch
    import torch_npu  # noqa: F401
    import tilelang

    from singlecell_npu.kernels import hvg_to_dense

    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    validate_host_case(case)
    values, indices, ptr, totals, genes = case
    max_row = int(np.diff(ptr).max(initial=0))
    inputs = device_inputs(case)
    started = time.perf_counter()
    native_meta = native_metadata(case)
    torch.npu.synchronize()
    native_metadata_ms = (time.perf_counter() - started) * 1000
    kwargs = {"n_genes": genes, "max_row_nnz": max_row}
    started = time.perf_counter()
    result = hvg_to_dense(*inputs, **kwargs)
    torch.npu.synchronize()
    first_call_ms = (time.perf_counter() - started) * 1000
    expected = cpu_path(case, dtype=np.float64)
    actual = result.cpu().numpy()
    native = torch_native(inputs, native_meta, genes).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(native, expected, rtol=2e-5, atol=2e-6)
    assert np.array_equal(actual == 0, expected == 0), "zeros/tiny-positive support changed"
    functions = {
        "tilelang": lambda: hvg_to_dense(*inputs, **kwargs),
        "torch_native_cached_indices": lambda: torch_native(inputs, native_meta, genes),
    }
    report = {
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__,
            "tilelang_module": tilelang.__file__, "device": torch.npu.get_device_name(0),
            "hf32": False,
        },
        "shape": [len(totals), genes], "nnz": len(values), "max_row_nnz": max_row,
        "density": len(values) / max(1, len(totals) * genes),
        "accuracy": {"max_abs_error_vs_fp64": float(np.max(np.abs(actual - expected), initial=0)),
                     "rtol": 2e-5, "atol": 2e-6, "zero_support_exact": True},
        "first_call_including_jit_ms": first_call_ms,
        "native_metadata_host_and_h2d_ms": native_metadata_ms,
        "cpu_threads": {key: __import__("os").environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "repeats": repeats, "device_event_ms": {}, "host_wall_ms": {},
        "h2d_inclusive_host_wall_ms": {},
        "timing_scope": "Output remains on NPU; no D2H. JIT excluded after first call. H2D uses pageable host arrays; native H2D includes rebuilding/uploading extra row and flat indices.",
    }
    for name, fn in functions.items():
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        events, walls = [], []
        for _ in range(repeats):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            wall = time.perf_counter()
            start.record()
            fn()
            end.record()
            end.synchronize()
            walls.append((time.perf_counter() - wall) * 1000)
            events.append(start.elapsed_time(end))
        report["device_event_ms"][name] = summarize_ms(events)
        report["host_wall_ms"][name] = summarize_ms(walls)
    cpu_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        cpu_path(case)
        cpu_times.append((time.perf_counter() - start) * 1000)
    report["host_wall_ms"]["cpu_fp32"] = summarize_ms(cpu_times)
    for mode in ("tilelang", "torch_native"):
        samples = []
        for _ in range(repeats):
            torch.npu.synchronize()
            start = time.perf_counter()
            fresh = device_inputs(case)
            if mode == "tilelang":
                hvg_to_dense(*fresh, **kwargs)
            else:
                torch_native(fresh, native_metadata(case), genes)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        report["h2d_inclusive_host_wall_ms"][mode] = summarize_ms(samples)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--genes", type=int, default=2000)
    parser.add_argument("--density", type=float, default=0.08)
    parser.add_argument("--repeats", type=int, default=10)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-npz", type=Path)
    source.add_argument("--tenx", type=Path)
    parser.add_argument("--hvg-table", type=Path)
    parser.add_argument("--save-input-npz", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="prepare NPZ using a CPU environment with h5py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.input_npz:
        with np.load(args.input_npz, allow_pickle=False) as source:
            case = tuple(source[k] for k in ("values", "indices", "indptr", "totals")) + (int(source["n_genes"]),)
    elif args.tenx:
        if not args.hvg_table:
            parser.error("--tenx requires --hvg-table")
        case = real_tenx_case(args.tenx, args.hvg_table, args.rows)
    else:
        case = make_case(args.rows, args.genes, args.density)
    if args.save_input_npz:
        args.save_input_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_input_npz, **dict(zip(("values", "indices", "indptr", "totals", "n_genes"), case)))
    if args.prepare_only:
        if not args.save_input_npz:
            parser.error("--prepare-only requires --save-input-npz")
        validate_host_case(case)
        print(json.dumps({"saved": str(args.save_input_npz), "shape": [len(case[3]), case[4]], "nnz": len(case[0])}))
        return
    report = run_benchmark(case, args.repeats)
    report["input_source"] = str(args.input_npz or args.tenx or "synthetic seed137 + edge cases")
    report["hvg_table"] = str(args.hvg_table) if args.hvg_table else None
    result = json.dumps(report, indent=2)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
