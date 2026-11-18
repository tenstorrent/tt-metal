# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per grid row (virtual y) of each profiled dispatch: the row's slowest core (latest kernel end over all RISCs) and its
zone windows in us from the dispatch's first kernel start. No ttnn.

    python3 rows.py <profile_log_device.csv[.gz]> <runs/<tag>.jsonl> [zone,zone,...]
"""
import collections
import csv
import gzip
import json
import sys

ZONES = [
    ("TRISC_1", "compute_untilize", "unt"),
    ("TRISC_1", "compute_tilize", "til"),
    ("TRISC_1", "compute_matmul_walk", "walk"),
    ("TRISC_1", "compute_combine", "comb"),
    ("BRISC", "writer_store", "store"),
    ("NCRISC", "reader_fill", "fill"),
    ("NCRISC", "reader_gather", "gath"),
]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
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
    return freq, runs


def main():
    path, log = sys.argv[1], sys.argv[2]
    freq, runs = load(path)
    recs = [json.loads(l) for l in open(log)]
    big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
    recs = recs[len(recs) - len(big) :]
    for rec, k in zip(recs, big):
        ev = sorted(runs[k], key=lambda e: e[2])
        t0 = min(t for c, r, t, z, kd in ev if z.endswith("-KERNEL") and kd == "ZONE_START")
        us = lambda t: (t - t0) / freq
        open_ = {}
        win = collections.defaultdict(list)  # (core, risc, zone) -> [(start, end)]
        kend = {}
        for core, risc, t, z, kind in ev:
            if kind == "ZONE_START":
                open_[(core, risc, z)] = t
            elif (core, risc, z) in open_:
                s = open_.pop((core, risc, z))
                if z.endswith("-KERNEL"):
                    kend[core] = max(kend.get(core, 0), us(t))
                else:
                    win[(core, risc, z)].append((us(s), us(t)))
        ys = sorted({c[1] for c in kend})
        print(f"== {rec['case']} {rec['variant']} ns={rec.get('ns', '?')} ranges={rec.get('ranges')}")
        for y in ys:
            cores = [c for c in kend if c[1] == y]
            slow = max(cores, key=lambda c: kend[c])
            parts = []
            for risc, z, short in ZONES:
                w = win.get((slow, risc, z))
                if w:
                    parts.append(f"{short}[{w[0][0]:.1f}-{w[-1][1]:.1f}]")
            walk = [sum(b - a for a, b in win.get((c, "TRISC_1", "compute_matmul_walk"), [])) for c in cores]
            print(
                f"  y{y:<2} end {kend[slow]:5.1f} (min {min(kend[c] for c in cores):5.1f}) slow {slow} "
                + " ".join(parts)
                + f"  walk max {max(walk):.1f}"
            )


if __name__ == "__main__":
    main()
