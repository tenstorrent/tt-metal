#!/usr/bin/env python3
"""Per run: p50-over-cores per-core-sum (ns) of each compute zone on TRISC_0/1/2 (unpack/math/pack).
usage: zones3.py <profile_log_device.csv> [zone,zone,...]"""
import csv
import sys
from collections import defaultdict
from statistics import median

ZONES = [
    "c_tilize",
    "c_p1_wait_x",
    "c_p1_xk_chain",
    "c_p1_reduce",
    "c_pass1",
    "c_p2_wait_x",
    "c_p2_normalize",
    "c_untilize",
    "c_pass2",
]


def main(path, zones):
    with open(path) as f:
        head = f.readline()
        mhz = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = list(csv.DictReader(f, skipinitialspace=True))
    k = 1000.0 / mhz
    stacks = defaultdict(list)
    dur = defaultdict(lambda: defaultdict(float))
    span = defaultdict(dict)
    for r in rows:
        if r["type"] not in ("ZONE_START", "ZONE_END"):
            continue
        run, core, risc, name = r["run host ID"], (r["core_x"], r["core_y"]), r["RISC processor type"], r["zone name"]
        t = int(r["time[cycles since reset]"])
        if name.endswith("-KERNEL"):
            span[(run, risc)].setdefault(core, [0, 0])[0 if r["type"] == "ZONE_START" else 1] = t
            continue
        key = (run, risc, name, core)
        if r["type"] == "ZONE_START":
            stacks[key].append(t)
        elif stacks[key]:
            dur[(run, risc, name)][core] += (t - stacks[key].pop()) * k
    runs = sorted({r[0] for r in dur}, key=int)
    for run in runs:
        sp = [(e - s) * k for s, e in span[(run, "TRISC_1")].values() if e > s]
        print(f"run {run}: TRISC_1 span p50 {median(sp):.0f} ns" if sp else f"run {run}")
        print(f"  {'zone':16s} {'UNPACK':>8s} {'MATH':>8s} {'PACK':>8s}")
        for z in zones:
            vals = []
            for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
                v = list(dur[(run, risc, z)].values())
                vals.append(median(v) if v else float("nan"))
            print(f"  {z:16s} " + " ".join(f"{v:8.0f}" for v in vals))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else ZONES)
