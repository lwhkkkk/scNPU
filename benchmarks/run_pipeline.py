"""Measure complete child-process wall time, preserving the exact command/log."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--tilelang-site", help="Explicit trusted read-only package overlay for experimental TileLang runs")
    parser.add_argument("--warm-cache", help="Read this existing CSR cache before timing; report prewarm separately")
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    record = Path(args.record)
    record.parent.mkdir(parents=True, exist_ok=True)
    if record.exists() or record.with_suffix(".log").exists():
        parser.error("Record/log already exists; use a new destination")
    arguments = args.pipeline_args[1:] if args.pipeline_args[:1] == ["--"] else args.pipeline_args
    command = [sys.executable, "-m", "singlecell_npu.streaming", *arguments]
    if args.tilelang_site:
        command = [sys.executable, "-m", "benchmarks.tilelang_launcher", "--tilelang-site", args.tilelang_site,
                   "singlecell_npu.streaming", *arguments]
    source_root = Path(__file__).resolve().parents[1]
    code_hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((source_root / "singlecell_npu").rglob("*.py"))}
    snapshot = record.with_suffix(".source.zip")
    with zipfile.ZipFile(snapshot, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ("singlecell_npu", "benchmarks"):
            for path in sorted((source_root / folder).rglob("*.py")):
                archive.write(path, str(path.relative_to(source_root)))
        archive.write(source_root / "pyproject.toml", "pyproject.toml")
    warmup = None
    if args.warm_cache:
        cache_root = Path(args.warm_cache)
        if not (cache_root / "manifest.json").is_file():
            parser.error("--warm-cache requires a complete canonical CSR cache")
        warm_start, warm_bytes = time.perf_counter(), 0
        for path in sorted(cache_root.glob("*.npy")):
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024**2), b""):
                    warm_bytes += len(block)
        warmup = dict(cache=str(cache_root), bytes=warm_bytes, seconds=time.perf_counter()-warm_start,
                      included_in_process_wall=False)
    started = time.perf_counter()
    with record.with_suffix(".log").open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    report = dict(command=command, exit_code=process.returncode,
                  process_wall_seconds=time.perf_counter()-started,
                  child_peak_rss_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
                  code_sha256=code_hashes, source_snapshot=str(snapshot),
                  warmup=warmup,
                  cache_policy=("CSR files explicitly read before timing; no OS cache drop or isolation"
                                if warmup else "OS page cache uncontrolled; report cache construction/reuse separately"))
    record.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
