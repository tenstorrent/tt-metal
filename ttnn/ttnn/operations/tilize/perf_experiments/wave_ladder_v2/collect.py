#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Align `WLV2_MARK` lines from a pytest log against the newest ops_perf CSV.

Every ladder rung dispatches exactly REPS GenericOpDeviceOperation rows, in the
order the rungs printed their marker, so the alignment is positional and then
CROSS-CHECKED against the row's own INPUT_0_Y/X and CORE COUNT before anything
is reported. A mismatch aborts rather than mislabels a number.

    python3 .../wave_ladder_v2/collect.py <pytest_log> [--report DIR]
"""
import csv
import glob
import os
import re
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))


def newest_report():
    cands = sorted(
        glob.glob(os.path.join(ROOT, "generated/profiler/reports/*/ops_perf_results*.csv")), key=os.path.getmtime
    )
    return cands[-1]


def main():
    log = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else newest_report()
    marks = []
    for line in open(log, errors="replace"):
        if "WLV2_MARK" not in line:
            continue
        body = line[line.index("WLV2_MARK") :].strip()
        kv = dict(re.findall(r"(\w+)=(\S+)", body.replace("WLV2_MARK ", "")))
        shape = tuple(int(x) for x in re.search(r"shape=\((.*?)\)", body).group(1).replace(" ", "").split(","))
        kv["shape"] = shape
        marks.append(kv)

    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if r["OP CODE"] == "GenericOpDeviceOperation"]
    reps = int(marks[0]["reps"])
    print(f"# csv={csv_path}\n# {len(rows)} device rows / {len(marks)} rungs x {reps} reps")
    assert len(rows) == len(marks) * reps, f"row/marker mismatch: {len(rows)} vs {len(marks)*reps}"

    print(
        f"{'label':18s} {'w':>2s} {'bw':>3s} {'read':>6s} {'cores':>6s} {'spl':>3s} "
        f"{'ns rep1':>8s} {'rep2':>8s} {'rep3':>8s} {'median':>8s}"
    )
    out = {}
    for i, m in enumerate(marks):
        chunk = rows[i * reps : (i + 1) * reps]
        for r in chunk:
            y = int(r["INPUT_0_Y_PAD[LOGICAL]"].split("[")[0])
            x = int(r["INPUT_0_X_PAD[LOGICAL]"].split("[")[0])
            assert (y, x) == (m["shape"][2], m["shape"][3]), f"row {i} shape mismatch {y}x{x} vs {m['shape']}"
            assert int(r["CORE COUNT"]) == int(
                m["cores"].split("/")[0]
            ), f"row {i} core mismatch {r['CORE COUNT']} vs {m['cores']}"
        ns = [int(r["DEVICE KERNEL DURATION [ns]"]) for r in chunk]
        med = statistics.median(ns)
        out[(m["label"], int(m["waves"]))] = med
        print(
            f"{m['label']:18s} {m['waves']:>2s} {m['bw']:>3s} {m['read']:>6s} {m['cores']:>6s} "
            f"{m['split_reader']:>3s} " + " ".join(f"{v:8d}" for v in ns) + f" {med:8.0f}"
        )

    print("\n# ratio vs w1 (>1 = w1 faster => rung is a regression)")
    labels = sorted({k[0] for k in out}, key=lambda l: [m["label"] for m in marks].index(l))
    for lab in labels:
        base = out.get((lab, 1))
        if base is None:
            continue
        cells = " ".join(f"w{w}={out[(lab,w)]/base:.3f}" for w in (1, 2, 4, 8) if (lab, w) in out)
        best = min(((w, out[(lab, w)]) for w in (1, 2, 4, 8) if (lab, w) in out), key=lambda t: t[1])
        print(f"{lab:18s} {cells}   best=w{best[0]} ({best[1]:.0f} ns, {base/best[1]:.3f}x)")


if __name__ == "__main__":
    main()
