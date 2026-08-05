# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join perf_manifest.csv (launch order) against the profiler's DEVICE KERNEL DURATION [ns].

    python3 <this dir>/read_perf.py [ops_perf_results_*.csv]

Per (variant, GROUP_SIZE, rows, seed) prints
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

    launches = [m for m in man]  # every manifest row is exactly one launch, in order
    if len(ns) != len(launches):
        print(f"WARNING: {len(ns)} profiled launches vs {len(launches)} manifest rows; joining the common prefix")
    n = min(len(ns), len(launches))

    rec = {}
    for m, t in zip(launches[:n], ns[:n]):
        key = (m["variant"], int(m["group_size"]), int(m["rows"]), int(m["seed"]), int(m.get("batch") or 4))
        e = rec.setdefault(key, {})
        if m["iters"] == "pcc_probe":
            e["pcc"], e["rel_rms"], e["zero"] = m["pcc"], m["rel_rms"], m["zero_leak"]
        else:
            e[f"t{m['iters']}"] = t

    print(
        f"{'variant':13s} {'G':>3s} {'rows':>5s} {'seed':>4s} {'bat':>4s} {'t1':>8s} {'t9':>9s} "
        f"{'per_round':>10s}   pcc        rel_rms"
    )
    order = sorted(rec, key=lambda k: (k[1], k[2], k[3], k[4], k[0]))
    for k in order:
        e = rec[k]
        t1, t9 = e.get("t1"), e.get("t9")
        per = (t9 - t1) / 8.0 if (t1 and t9) else float("nan")
        print(
            f"{k[0]:13s} {k[1]:3d} {k[2]:5d} {k[3]:4d} {k[4]:4d} {t1:8d} {t9:9d} {per:10.0f}   "
            f"{e.get('pcc','-')}  {e.get('rel_rms','-')}"
        )


if __name__ == "__main__":  # NEVER at import time: ttnn.operations auto-imports this tree.
    main()
