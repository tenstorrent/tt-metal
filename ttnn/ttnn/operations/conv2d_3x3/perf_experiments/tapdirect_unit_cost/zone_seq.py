"""Per-execution zone timeline of one core (no device needed).

    python3 zone_seq.py <profile_log_device.csv> [--run-index -1] [--core x,y] [--zones a,b,c]

Prints, for the chosen run (default: the last one) and core, every execution of the listed zones in time order:
start / end (us since the core's first marker) and duration, with the RISC.
"""

import argparse
import collections
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--run-index", type=int, default=-1)
    ap.add_argument("--core", default="")
    ap.add_argument("--zones", default="")
    a = ap.parse_args()
    want = set(z for z in a.zones.split(",") if z)
    with open(a.log) as f:
        header = f.readline()
        mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = csv.reader(f)
        cols = [c.strip() for c in next(rows)]
        ix = {c: i for i, c in enumerate(cols)}
        runs = collections.defaultdict(list)
        for r in rows:
            if len(r) < len(cols):
                continue
            runs[int(r[ix["run host ID"]])].append(
                (
                    (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                    r[ix["RISC processor type"]].strip(),
                    int(r[ix["time[cycles since reset]"]]),
                    r[ix["zone name"]].strip(),
                    r[ix["type"]].strip(),
                )
            )
    rid = sorted(runs)[a.run_index]
    ev = runs[rid]
    core = (
        tuple(int(v) for v in a.core.split(","))
        if a.core
        else sorted(set(e[0] for e in ev))[len(set(e[0] for e in ev)) // 2]
    )
    ev = sorted((e for e in ev if e[0] == core), key=lambda e: e[2])
    t0 = ev[0][2]
    stacks = collections.defaultdict(list)
    out = []
    for _, risc, t, zone, kind in ev:
        if want and zone not in want:
            continue
        if kind == "ZONE_START":
            stacks[(risc, zone)].append(t)
        elif kind == "ZONE_END" and stacks[(risc, zone)]:
            s = stacks[(risc, zone)].pop()
            out.append((s, t, risc, zone))
    print(f"run {rid} core {core}")
    for s, t, risc, zone in sorted(out):
        print(f"{(s - t0) / mhz:9.2f} {(t - t0) / mhz:9.2f} {(t - s) / mhz:8.2f}  {risc:7s} {zone}")


if __name__ == "__main__":
    main()
