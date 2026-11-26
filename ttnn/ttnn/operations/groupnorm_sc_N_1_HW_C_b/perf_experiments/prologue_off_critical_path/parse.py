#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Align the GenericOp rows of an ops_perf_results CSV with runs.jsonl (one line per bench test, same order).

usage: parse.py <ops_perf_results.csv> [runs.jsonl] [--save <tag>]
Prints variant, case, DEVICE KERNEL DURATION [ns], pcc, rms; --save appends the rows to results.jsonl with the tag.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main(argv):
    ops = argv[0]
    runs_path = HERE / "runs.jsonl"
    tag = None
    rest = argv[1:]
    if "--save" in rest:
        tag = rest[rest.index("--save") + 1]
        rest = [a for a in rest if a not in ("--save", tag)]
    if rest:
        runs_path = Path(rest[0])
    runs = [json.loads(l) for l in runs_path.read_text().splitlines() if l.strip()]
    with open(ops) as f:
        rows = [r for r in csv.DictReader(f) if "Generic" in r["OP CODE"]]
    if len(rows) != len(runs):
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(runs)} runs")
    out = []
    for r, run in zip(rows, runs):
        ns = int(float(r["DEVICE KERNEL DURATION [ns]"]))
        out.append(dict(run, ns=ns, tag=tag))
        print(f"{run['variant']:>10s} {run['case']:<28s} {ns:>10d} ns  pcc {run['pcc']:.6f} rms {run['rms']:.5f}")
    if tag:
        with (HERE / "results.jsonl").open("a") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
