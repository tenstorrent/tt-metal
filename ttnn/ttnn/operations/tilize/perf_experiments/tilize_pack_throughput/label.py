"""Pair a --profile report's generic-op rows with the harness' `P2 case=.. variant=..` lines.

usage: python3 label.py <pytest -s log> [report_dir]
(default: the report the log's `SAFE_PYTEST: PROFILER CSV:` line names -- NOT the newest report,
which may belong to a sibling agent's session on the shared device)
Like p2_breakdown/label_ns.py (last-n pairing), but sorts the op rows by GLOBAL CALL COUNT and
reads the report path from the log itself.
"""
import csv, glob, os, sys

if len(sys.argv) > 2:
    csv_path = glob.glob(os.path.join(sys.argv[2], "ops_perf_results*.csv"))[0]
else:
    csv_path = [l.split("PROFILER CSV:")[1].strip() for l in open(sys.argv[1]) if "PROFILER CSV:" in l][-1]
rows = list(csv.DictReader(open(csv_path)))
rows = [r for r in rows if "generic" in r["OP CODE"].lower() and r["DEVICE KERNEL DURATION [ns]"]]
rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
lines = [l.split()[1:3] for l in open(sys.argv[1]) if l.startswith("P2 ")]
# The precompile pass prints (fake-device) P2 lines too, interleaved unpredictably: the real pass
# is the LAST len(rows) lines. Valid only when every test runs the op exactly once (a low_l1 A/B
# case runs it twice and shifts every later label -- keep such cases out of profiled sessions).
if len(lines) < len(rows):
    sys.exit(f"row/label mismatch: {len(rows)} op rows vs {len(lines)} P2 lines")
real = lines[-len(rows) :]
out = {}
for (c, v), r in zip(real, rows):
    out.setdefault(c.split("=")[1], {})[v.split("=")[1].split("kernels_")[-1]] = int(
        float(r["DEVICE KERNEL DURATION [ns]"])
    )
vs = list(dict.fromkeys(v for d in out.values() for v in d))
print("case     " + " ".join(f"{v:>8s}" for v in vs))
for c, d in out.items():
    print(f"{c:8s} " + " ".join(f"{d.get(v, 0):>8d}" for v in vs))
