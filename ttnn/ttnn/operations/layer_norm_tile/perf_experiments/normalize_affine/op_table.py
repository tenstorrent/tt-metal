"""Print DEVICE KERNEL DURATION per case for the newest profiled whole-op run (CSV rows follow CASES order)."""
import glob, os, sys

NAMES = [
    "focus1_inL1_resDRAM_outDRAM",
    "focus2_allDRAM_res",
    "focus3_allDRAM_nores",
    "allL1_res",
    "L1_nores",
    "8x384_allDRAM_res",
    "32x32_allDRAM_res",
    "affineTILE_DRAM",
    "focus1_fp32dest_off",
    "h_non_aligned_3x500",
]
csv = max(
    glob.glob("/localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime
)
durs = [int(l.split(",")[19]) for l in open(csv).read().splitlines()[1:]]
tag = sys.argv[1] if len(sys.argv) > 1 else ""
for name, d in zip(NAMES, durs):
    print(f"OPROW {tag} {name} {d}")
