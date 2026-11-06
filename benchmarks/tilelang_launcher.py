"""Explicit experimental launcher reusing two existing read-only environments.

Run with the PCA interpreter; its NumPy/SciPy/h5py stay first on sys.path.
The supplied TileLang site is appended using site.addsitedir. Nothing is
installed, copied into an environment, or persisted in shell configuration.

python -m benchmarks.tilelang_launcher --inspect
python -m benchmarks.tilelang_launcher singlecell_npu.streaming [args...]
"""

import argparse
import importlib
import json
from pathlib import Path
import runpy
import site
import sys

DEFAULT_TILELANG_SITE = "/home/naie/work/slai/user-liuweiheng/envs/tilelang-ascend/lib/python3.11/site-packages"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tilelang-site", type=Path, default=Path(DEFAULT_TILELANG_SITE))
    parser.add_argument("--inspect", action="store_true", help="imports and versions only; no NPU availability probe or kernel")
    parser.add_argument("module", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.tilelang_site.is_dir():
        parser.error(f"TileLang site does not exist: {args.tilelang_site}")
    # This is an explicitly trusted, user-owned environment. Existing
    # interpreter package precedence is preserved; .pth entries are handled.
    site.addsitedir(str(args.tilelang_site))
    if args.inspect:
        modules = {}
        for name in ("numpy", "scipy", "h5py", "torch", "torch_npu", "tilelang"):
            module = importlib.import_module(name)
            modules[name] = {"version": getattr(module, "__version__", None), "path": getattr(module, "__file__", None)}
        print(json.dumps({"python": sys.executable, "added_site": str(args.tilelang_site), "modules": modules}, indent=2))
        return
    if not args.module:
        parser.error("provide a module to run, or --inspect")
    sys.argv = [args.module] + args.arguments
    runpy.run_module(args.module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
