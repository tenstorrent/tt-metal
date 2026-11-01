import csv, glob, os, sys

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else max(
        glob.glob("/localdev/kgrujcic/tt-metal/generated/profiler/reports/*/ops_perf_results_*.csv"),
        key=os.path.getmtime,
    )
)
ids = [
    "focus1",
    "focus2",
    "focus3",
    "allL1_res",
    "L1_nores",
    "8x384",
    "32x32",
    "affTILE",
    "g_focus1_bf16",
    "g_3x500",
    "g_64x17",
    "g_2x512",
]
rows = list(csv.DictReader(open(path)))
print(path)
for i, r in enumerate(rows):
    name = ids[i] if i < len(ids) else str(i)
    print(
        f"{name:14s} {int(r['DEVICE KERNEL DURATION [ns]'])/1000:7.1f} us  cores {r['CORE COUNT']:>4s}  in0 {r['INPUT_0_W_PAD[LOGICAL]']}x{r['INPUT_0_Y_PAD[LOGICAL]']} {r['INPUT_0_MEMORY']}"
    )
