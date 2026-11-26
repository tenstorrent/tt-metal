#!/usr/bin/env python3
"""Join the bench log's real-pass PTC_RESULT lines with the profiler CSV's GenericOp rows (same order)."""
import csv
import re
import sys


def main(log, csv_path):
    text = open(log, errors="replace").read()
    real = text.split("starting the real pass", 1)[-1]
    res = re.findall(r"PTC_RESULT case=(\S+) variant=(\S+) pcc=(\S+) rms=(\S+)", real)
    durs = []
    if csv_path:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if "Generic" in r.get("OP CODE", "") or "generic" in r.get("OP CODE", "").lower():
                    durs.append(int(float(r["DEVICE KERNEL DURATION [ns]"])))
    print(f"{'case':28s} {'variant':14s} {'pcc':>9s} {'rms':>8s} {'ns':>9s}")
    for i, (c, v, p, m) in enumerate(res):
        d = durs[i] if i < len(durs) else -1
        print(f"{c:28s} {v:14s} {p:>9s} {m:>8s} {d:>9d}")
    if len(durs) != len(res):
        print(f"WARNING: {len(res)} results vs {len(durs)} GenericOp rows; csv={csv_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
