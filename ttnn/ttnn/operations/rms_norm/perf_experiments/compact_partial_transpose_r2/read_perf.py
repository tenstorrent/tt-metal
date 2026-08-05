# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join perf_manifest.csv (launch order) against the profiler's DEVICE KERNEL DURATION [ns].

    python3 <this dir>/read_perf.py [ops_perf_results_*.csv]

Per config prints
    t1     the ITERS=1 launch (one combine round + the fixed kernel-launch floor)
    t9     the ITERS=9 launch
    per    (t9 - t1) / 8 -- ONE combine round with the launch floor cancelled exactly
plus the precision recorded for the same spelling.
"""
import csv
import glob
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

KEYS = ("variant", "group_size", "rows", "fin", "bank", "bankdt", "seed", "batch")


def main():
    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))[-1]
    )

    man = list(csv.DictReader(open(HERE / "perf_manifest.csv")))
    with open(csv_path) as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        col = hdr.index("DEVICE KERNEL DURATION [ns]")
        ns = [int(r[col]) for r in rdr if len(r) > col and r[col].strip().isdigit()]

    if len(ns) != len(man):
        print(f"WARNING: {len(ns)} profiled launches vs {len(man)} manifest rows; joining the common prefix")
    n = min(len(ns), len(man))

    rec = {}
    for m, t in zip(man[:n], ns[:n]):
        key = tuple(m[k] for k in KEYS)
        e = rec.setdefault(key, {})
        if m["iters"] == "pcc_probe":
            e["pcc"], e["rel_rms"], e["zero"] = m["pcc"], m["rel_rms"], m["zero_leak"]
        else:
            e[f"t{m['iters']}"] = t

    print(
        f"{'variant':12s} {'G':>3s} {'rows':>5s} {'fin':>3s} {'bk':>3s} {'dt':>5s} {'sd':>3s} {'bat':>4s} "
        f"{'t1':>8s} {'t9':>9s} {'per_round':>10s}   pcc        rel_rms   zero"
    )
    order = sorted(rec, key=lambda k: (int(k[2]), int(k[1]), k[0], int(k[3]), int(k[4]), k[5], int(k[6]), int(k[7])))
    for k in order:
        e = rec[k]
        t1, t9 = e.get("t1"), e.get("t9")
        per = (t9 - t1) / 8.0 if (t1 and t9) else float("nan")
        print(
            f"{k[0]:12s} {int(k[1]):3d} {int(k[2]):5d} {int(k[3]):3d} {int(k[4]):3d} {k[5]:>5s} "
            f"{int(k[6]):3d} {int(k[7]):4d} {t1 or -1:8d} {t9 or -1:9d} {per:10.0f}   "
            f"{e.get('pcc','-')}  {e.get('rel_rms','-')}  {e.get('zero','-')}"
        )


if __name__ == "__main__":  # NEVER at import time: ttnn.operations auto-imports this tree.
    main()
