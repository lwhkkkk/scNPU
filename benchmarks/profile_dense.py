"""Representative real-data Gram + CPU eigh + projection, wall time and NPU L0.

This intentionally does not label a device-stage trace as a full pipeline trace.
"""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from singlecell_npu.backend import DenseBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, out = Path(args.artifacts), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    expression = np.load(source / "log1p_hvg.npy", mmap_mode="r")
    x = np.array(expression[min(80*8192, max(0, len(expression)-8192)):][:8192], dtype=np.float32)
    with h5py.File(source / "brain_1m_pca.h5ad") as handle:
        std = handle["var/pca_std_log1p"][:]
        mean = handle["var/pca_mean_log1p"][:]
        loading = handle["varm/PCs"][:].astype(np.float64)
    projection = loading / std[:, None]
    offset = mean @ projection
    z = (x.astype(np.float64) - x.mean(0, dtype=np.float64)) / std
    covariance = z.T @ z / (len(x)-1)
    with DenseBackend("npu:0") as backend:
        import torch_npu

        def workload():
            backend.gram(x)
            backend.eigh_top(covariance, loading.shape[1])
            backend.project(x, projection, offset)

        for _ in range(3):
            workload()
        times = []
        for _ in range(7):
            start = time.perf_counter()
            workload()
            times.append(time.perf_counter()-start)
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out / "L0"))) as profiler:
            workload()
            backend.torch.npu.synchronize()
            profiler.step()
        result = dict(scope="one real 8192x2000 Gram + sample covariance CPU eigh(top50) + projection; not full pipeline",
                      seconds=times, median_seconds=float(np.median(times)), backend=backend.info,
                      trace=str(out / "L0"))
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
