"""A/B a redundant writable-host FP32 copy; no dtype/precision change."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from singlecell_npu.backend import DenseBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    x = np.array(np.load(args.expression, mmap_mode="r")[:8192], dtype=np.float32)
    weights = np.random.default_rng(5).normal(0, .01, (x.shape[1], 50))
    offset = np.zeros(50)
    with DenseBackend("npu:0") as backend:
        def copy_upload(array, dtype=np.float32):
            return backend.torch.from_numpy(np.array(array, dtype=dtype, order="C", copy=True)).to(backend.device)

        def view_upload(array, dtype=np.float32):
            value = np.ascontiguousarray(array, dtype=dtype)
            if not value.flags.writeable:
                value = value.copy()
            return backend.torch.from_numpy(value).to(backend.device)

        outputs, measurements = {}, {}
        for name, upload in (("copy", copy_upload), ("view", view_upload)):
            backend.upload = upload
            for _ in range(3):
                outputs[name] = backend.project(x, weights, offset)
            timings = []
            for _ in range(10):
                start = time.perf_counter()
                backend.project(x, weights, offset)
                timings.append(time.perf_counter()-start)
            measurements[name] = dict(seconds=timings, median_seconds=statistics.median(timings))
        np.testing.assert_array_equal(outputs["copy"], outputs["view"])
        result = dict(scope="writable real 8192x2000 FP32 block H2D + projection + D2H; 3 warmups, 10 repetitions",
                      exact_output_match=True, measurements=measurements, device=backend.info.get("name"))
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
