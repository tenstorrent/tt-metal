#!/bin/bash
# usage: bench.sh <variant: kernels_baseline|kernels_c1|...> <case> [<case> ...]
# One fresh profiled dispatch per case; prints "P1M_NS <variant> <case> <DEVICE KERNEL DURATION ns>".
# Set GN_BENCH_CHUNK_ZONES=1 to also print the per-stage zone breakdown (zone_breakdown.py).
set -u
D="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$D/../../../../../.." && pwd)"
V="$1"; shift
cd "$ROOT"
source python_env/bin/activate
for c in "$@"; do
    LOG="$D/logs/${V}_${c}_$(date +%H%M%S).log"
    mkdir -p "$D/logs"
    GN_KERNEL_DIR="$D/$V" scripts/run_safe_pytest.sh --profile --no-py-zones \
        "$D/test_p1m.py::test_bench[$c]" --import-mode=prepend > "$LOG" 2>&1
    CSV=$(grep -oP "PROFILER CSV: \K.*" "$LOG")
    if [[ -z "$CSV" ]]; then echo "P1M_NS $V $c NO_CSV ($LOG)"; continue; fi
    NS=$(python3 - "$CSV" <<'EOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if "groupnorm" in (r.get("OP CODE", "") + r.get("OP NAME", "")).lower() or r.get("OP TYPE", "") == "python_fallback" or "Generic" in r.get("OP CODE", "")]
print(" ".join(r["DEVICE KERNEL DURATION [ns]"] for r in rows if r.get("DEVICE KERNEL DURATION [ns]")))
EOF
)
    echo "P1M_NS $V $c $NS"
    if [[ -n "${GN_BENCH_CHUNK_ZONES:-}" ]]; then
        python3 ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/_tools/zone_breakdown.py \
            "$(dirname "$CSV")/profile_log_device.csv" 2>/dev/null | sed "s/^/  [$V $c] /"
    fi
done
