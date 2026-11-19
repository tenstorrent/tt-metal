# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core walk landmarks of one profiled run (zoned build): python3 walk_stats.py <profile_log_device.csv> <run idx>.

For every core: first weight round end (w.weight_fetch #1), walk start / end (TRISC_1 compute_matmul_walk), combine end,
store end and kernel end, in us from the run's earliest kernel start. Prints p50 / max of each landmark, grouped by
grid-row vs spare-column cores (x of the core >= the first spare column given by --spare-x)."""
import collections
import csv
import statistics
import sys


def main():
    path, idx = sys.argv[1], int(sys.argv[2])
    spare_x = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    with open(path) as f:
        hdr = f.readline()
        freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
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
    big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
    ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
    t0 = min(t for _, _, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
    us = lambda c: (c - t0) / freq
    zones = collections.defaultdict(list)  # (core, risc, zone) -> [(s, e)]
    stack = collections.defaultdict(list)
    for core, risc, t, z, kind in ev:
        if kind == "ZONE_START":
            stack[(core, risc, z)].append(t)
        elif stack[(core, risc, z)]:
            zones[(core, risc, z)].append((us(stack[(core, risc, z)].pop()), us(t)))
    cores = sorted({k[0] for k in zones})
    lm = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in cores:
        grp = "spare" if c[0] >= spare_x else "grid"
        wf = sorted(zones.get((c, "BRISC", "writer_weight_fetch"), []))
        walk = zones.get((c, "TRISC_1", "compute_matmul_walk"), [])
        comb = zones.get((c, "TRISC_2", "compute_combine"), [])
        store = zones.get((c, "BRISC", "writer_store"), [])
        kend = max(e for (cc, r, z), v in zones.items() if cc == c and z.endswith("-KERNEL") for _, e in v)
        if wf:
            lm[grp]["round0_end"].append(wf[0][1])
            if len(wf) > 1:
                lm[grp]["round_period"].append((wf[-1][1] - wf[0][1]) / (len(wf) - 1))
        if walk:
            lm[grp]["walk_start"].append(walk[0][0])
            lm[grp]["walk_end"].append(walk[0][1])
            lm[grp]["walk_len"].append(walk[0][1] - walk[0][0])
        if comb:
            lm[grp]["combine_len"].append(comb[0][1] - comb[0][0])
        if store:
            lm[grp]["store_end"].append(store[0][1])
        lm[grp]["kernel_end"].append(kend)
    for grp in ("grid", "spare"):
        print(f"{grp}: {len(lm[grp]['kernel_end'])} cores")
        for k, v in lm[grp].items():
            print(f"   {k:<13} p50 {statistics.median(v):6.2f}  max {max(v):6.2f}  min {min(v):6.2f}")


if __name__ == "__main__":
    main()
