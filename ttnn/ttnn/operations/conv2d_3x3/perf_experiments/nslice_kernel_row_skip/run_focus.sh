#!/usr/bin/env bash
# Profile the 22 [perf] focus cases on the graduation candidate (grad) or the real op (real), one process each, into
# this dir's own profiler folder. usage: run_focus.sh grad|real <tag>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
WHICH="$1"; TAG="$2"
if [[ "$WHICH" == grad ]]; then
    T="$HERE/test_nks_grad_perf.py::test_conv2d_3x3_perf_focus"
else
    T="$REPO/tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus"
fi
export TT_METAL_PROFILER_DIR="$HERE/prof/$TAG"
rm -rf "$HERE/prof/$TAG"
cd "$REPO"
export TT_METAL_HOME=$REPO PYTHONPATH=$REPO
source python_env/bin/activate
scripts/run_safe_pytest.sh --profile --run-all "$T" --import-mode=prepend -q 2>&1 | \
    grep -E " passed| failed|FAILED|SAFE_PYTEST_RESULT|HANG" | tail -3
CSV=$(ls -t "$HERE/prof/$TAG"/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[[ -n "$CSV" ]] && cp "$CSV" "$HERE/runs/${TAG}_ops_perf.csv" && python3 - "$CSV" "$WHICH $TAG" <<'PY' | tee -a "$HERE/runs/focus_sets.txt"
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["OP CODE"].startswith("GenericOp")]
print(sys.argv[2], [int(float(r["DEVICE KERNEL DURATION [ns]"])) for r in rows])
PY
rm -rf "$HERE/prof/$TAG"
