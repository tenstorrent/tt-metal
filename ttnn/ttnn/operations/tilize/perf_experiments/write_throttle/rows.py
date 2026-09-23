"""Map the newest --profile report's ops rows to the test ids (collection order).
usage: TILIZE_WT=... TILIZE_WT_CASES=... python rows.py  (python_env active)"""
import csv, glob, os, subprocess, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
rep = sorted(glob.glob(os.path.join(root, "generated/profiler/reports/*/")))[-1]
rows = list(csv.DictReader(open(glob.glob(os.path.join(rep, "ops_perf_results*.csv"))[0])))
ids = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        os.path.join(root, "tests/ttnn/unit_tests/operations/tilize/test_tilize_perf1_write_throttle.py"),
    ],
    capture_output=True,
    text=True,
    cwd=root,
).stdout.split("\n")
import re

ids = [m.group(1) for i in ids for m in [re.search(r"test_wt\[(.*?)\]", i)] if m]
print(rep, len(rows), "rows", len(ids), "ids", "" if len(rows) == len(ids) else "MISMATCH - DO NOT TRUST")
for i, r in enumerate(rows):
    print(
        f"{ids[i] if i < len(ids) else '?':70s} {r['OP CODE'][:20]:20s} cores={r['CORE COUNT']:>3s} {r['DEVICE KERNEL DURATION [ns]']:>8s}"
    )
