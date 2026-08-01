#!/usr/bin/env bash
# Measure one configuration of the op and print `DEVICE KERNEL DURATION [ns]` per case.
#
#   perf_experiments/measure.sh "<label>" "<MOE_R2_CASES>"
#
# Every MOE_SWIGLU_* knob is inherited from the caller's environment, so an A/B is
#
#   MOE_SWIGLU_GRID=11x8 perf_experiments/measure.sh 88cores "7168,5120,256,bf16_rm"
#
# The profiler report root is wherever run_safe_pytest.sh puts it; the newest report dir wins.
set -u
LABEL="${1:-run}"
CASES="${2:-7168,5120,256,bf16_rm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT" || exit 1

# run_safe_pytest.sh may write the report under this clone OR under the shared tt-metal root
# (TT_METAL_HOME), so both roots are searched and the newest dir overall wins.
newest_report() {
    ls -1dt /localdev/mstaletovic/tt-metal/generated/profiler/reports/*/ \
        "$ROOT"/generated/profiler/reports/*/ 2>/dev/null | head -1
}
BEFORE=$(newest_report)

MOE_R2_CASES="$CASES" timeout 1200 scripts/run_safe_pytest.sh --profile \
    tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_r2_perf.py \
    >"/tmp/moe_measure_$$.log" 2>&1
RC=$?
# Prefer the path run_safe_pytest.sh reports for THIS run. Picking "newest report dir" by mtime is
# wrong whenever anything else is using the device concurrently (e.g. a perf subagent), and it fails
# silently: you get a valid CSV belonging to someone else's cases and zero matching rows.
# tracy prints the authoritative path; run_safe_pytest's own PROFILER CSV line looks in the clone
# while tracy writes under TT_METAL_HOME, so it is often absent. Take the last one in case of retries.
CSV=$(grep "OPs csv generated at:" "/tmp/moe_measure_$$.log" 2>/dev/null | tail -1 | sed "s/.*generated at: //")
[ -f "$CSV" ] || CSV=$(grep -m1 "SAFE_PYTEST: PROFILER CSV:" "/tmp/moe_measure_$$.log" 2>/dev/null | sed "s/.*PROFILER CSV: //")
if [ -n "$CSV" ] && [ -f "$CSV" ]; then
    AFTER="$(dirname "$CSV")/"
else
    AFTER=$(newest_report)
fi
if [ "$RC" != "0" ] || [ "$AFTER" = "$BEFORE" ]; then
    echo "$LABEL: FAILED (rc=$RC) — see /tmp/moe_measure_$$.log"
    tail -25 "/tmp/moe_measure_$$.log"
    exit 1
fi

python3 - "$AFTER" "$LABEL" "$CASES" <<'PYEOF'
import csv, glob, sys
report, label, cases = sys.argv[1], sys.argv[2], sys.argv[3]
if cases == "guard":  # the harness's own alias; mirror it so the labels line up
    sys.path.insert(0, "tests/ttnn/unit_tests/operations/moe_fused_swiglu")
    from test_moe_fused_swiglu_r2_perf import GUARD_SET

    cases = GUARD_SET
names = [c.strip() for c in cases.split(";") if c.strip()]
ns = []
for p in glob.glob(f"{report}/ops_perf_results*.csv"):
    for r in csv.DictReader(open(p)):
        if r.get("OP CODE") == "GenericOpDeviceOperation":
            ns.append((int(r["GLOBAL CALL COUNT"]), int(r["DEVICE KERNEL DURATION [ns]"]), int(r["CORE COUNT"])))
ns.sort()
for (name, (_, dur, cores)) in zip(names, ns):
    print(f"{label:28s} {name:28s} {dur:9,d} ns   cores={cores}")
print(f"# report: {report}")
PYEOF
