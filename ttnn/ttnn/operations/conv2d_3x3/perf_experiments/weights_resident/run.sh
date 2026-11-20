#!/usr/bin/env bash
# One profiled session of test_wr.py.   WR_CASES=idx41,idx42 WR_VARIANTS=base,solve [WR_ITERS=1] [WR_DEV=1] run.sh <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
TAG="$1"
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
export WR_LOG="${TAG}.jsonl"
mkdir -p "$HERE/runs"
rm -f "$HERE/runs/${TAG}.jsonl"
FLAGS="--run-all"
[[ "${WR_DEV:-0}" == "1" ]] && FLAGS="--dev $FLAGS" || FLAGS="--profile $FLAGS"
LOG="$HERE/runs/${TAG}.log"
timeout 3000 scripts/run_safe_pytest.sh $FLAGS "$HERE/test_wr.py" --import-mode=prepend --timeout 2400 > "$LOG" 2>&1
echo "exit=$?"
[[ -f generated/profiler/.logs/profile_log_device.csv ]] && gzip -c generated/profiler/.logs/profile_log_device.csv > "$HERE/runs/${TAG}_device.csv.gz"
grep -E " passed| failed|SAFE_PYTEST|HANG|hang detected|triage|Error|error:" "$LOG" | grep -v "^PRECOMPILE" | tail -12
grep "^WR:" "$LOG" | sed 's/^WR: //' | python3 -c '
import sys, json
for l in sys.stdin:
    r = json.loads(l)
    print(r["case"], r["variant"], "bm", r["bm"], "bk", r["bk"], "mb", r["m_blocks"], "wdepth", r["weight_depth"], "res", r["resident"], "im2col", r.get("im2col"), "fp", r["footprint"], "sha", r["sha"], "ident", r["identical_to_base"], "pcc", r.get("pcc"))'
CSV=$(grep "PROFILER CSV:" "$LOG" | tail -1 | sed 's/.*CSV: //')
if [[ -n "$CSV" && -f "$CSV" ]]; then
    cp "$CSV" "$HERE/runs/${TAG}_ops.csv"
    python3 "$HERE/../asym_padding/cmp_windows.py" "$CSV" | tee "$HERE/runs/${TAG}_ns.txt"
fi
