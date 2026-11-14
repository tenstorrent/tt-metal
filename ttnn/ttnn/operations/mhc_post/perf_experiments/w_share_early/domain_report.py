"""python domain_report.py <ops_perf_results.csv> [...]: min / max DEVICE KERNEL DURATION (us) per test_domain_perf case."""
import csv, sys
import ast, pathlib

_src = (pathlib.Path(__file__).parent / "test_domain_perf.py").read_text()  # no ttnn import
CASES = ast.literal_eval(_src[_src.index("CASES = [") + 8 : _src.index("]\n\n", _src.index("CASES = [")) + 1])
REPS = int(_src.split("REPS = ")[1].split()[0])
cols = []
for path in sys.argv[1:]:
    d = [
        int(x["DEVICE KERNEL DURATION [ns]"]) / 1000
        for x in csv.DictReader(open(path))
        if x["OP CODE"] == "GenericOpDeviceOperation"
    ]
    cols.append([d[i * REPS : (i + 1) * REPS] for i in range(len(CASES))])
print(f"{'case':36s}" + "".join(f"{p.split('/')[-2][:20]:>22s}" for p in sys.argv[1:]))
for i, c in enumerate(CASES):
    print(f"{' '.join(map(str, c)):36s}" + "".join(f"{min(col[i]):10.1f} ({max(col[i]):7.1f})" for col in cols))
