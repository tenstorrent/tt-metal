#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Align the GenericOp rows of an ops_perf_results CSV with runs.jsonl (one line per bench run, same order), and
optionally summarize the combine-path zones of the profile_log_device.csv copy taken right after the run.

usage: parse.py <ops_perf_results.csv> [--zones <profile_log_device.csv>] [--save <tag>]
Prints variant, case, DEVICE KERNEL DURATION [ns], pcc, rms, ndiff (elements differing from the case's first
variant). --zones adds, per run, the p50-over-cores and max-over-cores per-core sum of the combine-path zones.
--save appends the rows to results.jsonl with the tag.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).parent
ZONES = [
    ("TRISC_1", "c_collapse"),
    ("TRISC_1", "c_publish"),
    ("TRISC_1", "c_collapse_mm"),
    ("TRISC_0", "c_wait_gather"),
    ("TRISC_1", "c_finalize"),
    ("TRISC_0", "c_wait_stats"),
    ("TRISC_1", "c_expand"),
    ("TRISC_1", "c_pass1"),
    ("TRISC_1", "c_pass2"),
]


def zone_table(path):
    with open(path) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(f, skipinitialspace=True))
    k = 1000.0 / mhz
    stacks = defaultdict(list)
    dur = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if r["type"] not in ("ZONE_START", "ZONE_END"):
            continue
        run, core, risc, name = r["run host ID"], (r["core_x"], r["core_y"]), r["RISC processor type"], r["zone name"]
        t = int(r["time[cycles since reset]"])
        key = (run, risc, name, core)
        if r["type"] == "ZONE_START":
            stacks[key].append(t)
        elif stacks[key]:
            dur[(run, risc, name)][core] += (t - stacks[key].pop()) * k
    runs = sorted({r[0] for r in dur}, key=int)
    table = {}
    for run in runs:
        cells = {}
        for risc, name in ZONES:
            v = list(dur[(run, risc, name)].values())
            cells[name] = (median(v), max(v), len(v)) if v else None
        table[run] = cells
    return runs, table


def main(argv):
    ops = argv[0]
    rest = argv[1:]
    tag = zones = None
    if "--save" in rest:
        tag = rest[rest.index("--save") + 1]
    if "--zones" in rest:
        zones = rest[rest.index("--zones") + 1]
    runs = [json.loads(l) for l in (HERE / "runs.jsonl").read_text().splitlines() if l.strip()]
    with open(ops) as f:
        rows = [r for r in csv.DictReader(f) if "Generic" in r["OP CODE"]]
    if len(rows) != len(runs):
        print(f"WARNING: {len(rows)} GenericOp rows vs {len(runs)} runs")
    ztab = zone_table(zones) if zones else None
    out = []
    for i, (r, run) in enumerate(zip(rows, runs)):
        ns = int(float(r["DEVICE KERNEL DURATION [ns]"]))
        o = dict(run, ns=ns, tag=tag)
        line = (
            f"{run['variant']:>6s} {run['case']:<26s} {ns:>9d} ns  pcc {run['pcc']:.6f} rms {run['rms']:.5f}"
            f"  ndiff_vs_{run['vs']} {run['ndiff']}"
        )
        if ztab:
            zruns, table = ztab
            if i < len(zruns):
                cells = table[zruns[i]]
                o["zones"] = {n: c[:2] for n, c in cells.items() if c}
                line += "  |" + " ".join(f" {n.replace('c_', '')} {c[0]:.0f}/{c[1]:.0f}" for n, c in cells.items() if c)
        out.append(o)
        print(line)
    if tag:
        with (HERE / "results.jsonl").open("a") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
