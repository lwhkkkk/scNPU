#!/usr/bin/env bash
# Read-only reuse of the validated PCA environment; all output stays in this project.
set -eo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PCA_ROOT="${PCA_ROOT:-/home/naie/work/slai/user-liuweiheng/PCA}"
INPUT_PATH="${INPUT_PATH:-$PCA_ROOT/brain_1m.h5}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/artifacts/legacy-baseline-$RUN_TAG}"
EVIDENCE_PREFIX="$PROJECT_ROOT/benchmarks/evidence/legacy-$RUN_TAG"
cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
source "$PCA_ROOT/env_npu.sh"
mkdir -p "$PROJECT_ROOT/artifacts" "$PROJECT_ROOT/benchmarks/evidence"
test -f "$INPUT_PATH"
test ! -e "$OUTPUT_PATH"
export PROJECT_ROOT PCA_ROOT INPUT_PATH OUTPUT_PATH EVIDENCE_PREFIX
"$PCA_ROOT/.venv_npu/bin/python" - <<'PY'
import hashlib
import json
import os
import pathlib
import resource
import subprocess
import time
from datetime import datetime, timezone

root = pathlib.Path(os.environ['PROJECT_ROOT'])
prefix = pathlib.Path(os.environ['EVIDENCE_PREFIX'])
env_root = pathlib.Path(os.environ['PCA_ROOT'])
output = pathlib.Path(os.environ['OUTPUT_PATH'])
sources = [root / 'benchmarks/legacy_baseline' / name for name in ['run_pca.py', 'npu_backend.py']]
source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
command = [str(env_root / '.venv_npu/bin/python'), str(sources[0]), '--input', os.environ['INPUT_PATH'], '--output', str(output), '--device', 'npu']
started_utc = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
with prefix.with_suffix('.log.txt').open('w') as log:
    completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
elapsed = time.perf_counter() - started
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
result = dict(started_utc=started_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
              command=command, cwd=str(root), output=str(output), exit_code=completed.returncode,
              process_wall_seconds=elapsed, child_peak_rss_gib=usage.ru_maxrss / 1024**2,
              source_sha256=source_hashes, cache_state='uncontrolled; no page-cache eviction',
              benchmark_repetitions=1,
              hardware_scope='cellfm actual Ascend910_9382 (A3); not an Ascend 910B result')
if (output / 'summary.json').is_file():
    result['summary'] = json.loads((output / 'summary.json').read_text())
prefix.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
raise SystemExit(completed.returncode)
PY
