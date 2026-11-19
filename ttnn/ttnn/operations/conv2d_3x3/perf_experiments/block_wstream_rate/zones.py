# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core zone timeline of one dispatch from a device profiler log (no device needed).

    python3 zones.py <profile_log_device.csv[.gz]> <run#> grid [ZONE]       kernel end (or ZONE's last end) per core
    python3 zones.py <log> <run#> core X,Y                                    every zone instance on core (x, y)
    python3 zones.py <log> <run#> summary                                     per zone: first start / last end p50, max
Times are us from the run's first kernel start. run# counts dispatches on >= 8 cores, from 1.
"""
import collections
import csv
import gzip
import statistics
import sys


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


def spans(ev, freq):
    t0 = min(t for c, rk, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
    stack = collections.defaultdict(list)
    out = []
    for c, rk, t, z, k in sorted(ev, key=lambda e: e[2]):
        if k == "ZONE_START":
            stack[(c, rk, z)].append(t)
        elif k == "ZONE_END" and stack[(c, rk, z)]:
            s = stack[(c, rk, z)].pop()
            out.append((c, rk, z, (s - t0) / freq, (t - t0) / freq))
    return out


def main():
    freq, runs = load(sys.argv[1])
    big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
    ev = runs[big[int(sys.argv[2]) - 1]]
    sp = spans(ev, freq)
    mode = sys.argv[3]
    if mode == "grid":
        zone = sys.argv[4] if len(sys.argv) > 4 else None
        end = collections.defaultdict(float)
        for c, rk, z, s, e in sp:
            if (zone is None and z.endswith("-KERNEL")) or z == zone:
                end[c] = max(end[c], e)
        xs = sorted({c[0] for c in end})
        ys = sorted({c[1] for c in end})
        print("y\\x " + " ".join(f"{x:5d}" for x in xs))
        for y in ys:
            print(f"{y:3d} " + " ".join(f"{end[(x, y)]:5.1f}" if (x, y) in end else "    -" for x in xs))
        worst = max(end, key=end.get)
        print("max", worst, round(end[worst], 2), "p50", round(statistics.median(end.values()), 2))
    elif mode == "core":
        core = tuple(int(v) for v in sys.argv[4].split(","))
        agg = collections.OrderedDict()
        for c, rk, z, s, e in sorted(sp, key=lambda r: r[3]):
            if c != core:
                continue
            key = (rk, z)
            if key not in agg:
                agg[key] = [s, e, 0.0, 0]
            a = agg[key]
            a[0], a[1], a[2], a[3] = min(a[0], s), max(a[1], e), a[2] + e - s, a[3] + 1
        for (rk, z), (s, e, tot, n) in agg.items():
            print(f"{rk:<8} {z:<32} first {s:7.2f} last {e:7.2f} sum {tot:6.2f} n {n}")
    elif mode == "summary":
        per = collections.defaultdict(lambda: collections.defaultdict(lambda: [1e9, 0.0, 0.0]))
        for c, rk, z, s, e in sp:
            a = per[(rk, z)][c]
            a[0], a[1], a[2] = min(a[0], s), max(a[1], e), a[2] + e - s
        for (rk, z), cores in sorted(per.items(), key=lambda kv: statistics.median(v[0] for v in kv[1].values())):
            fs = [v[0] for v in cores.values()]
            le = [v[1] for v in cores.values()]
            sm = [v[2] for v in cores.values()]
            print(
                f"{rk:<8} {z:<32} n{len(cores):4d} start p50 {statistics.median(fs):6.2f} end p50 "
                f"{statistics.median(le):6.2f} max {max(le):6.2f}  sum p50 {statistics.median(sm):6.2f} max {max(sm):6.2f}"
            )


if __name__ == "__main__":
    main()
