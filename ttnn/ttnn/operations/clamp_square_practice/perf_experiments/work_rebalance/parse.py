# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Parse a --profile report for the work_rebalance bench.

usage: python3 parse.py <report_dir> <pytest_log>
Pairs each generic op (in call order) with the BENCH_ROW lines of the log, prints DEVICE KERNEL
DURATION and per-core spread: writer_write span p50/max (us) and per-core BRISC kernel end
relative to the earliest kernel start (finish time) p50/max (us).
"""
import csv
import glob
import re
import statistics
import sys
from collections import defaultdict

MHZ = 1350.0


def main(rep, log):
    rows = [l for l in open(log) if l.startswith("BENCH_ROW")]
    seen, bench = set(), []
    for l in rows:  # up-front collect may print duplicates; keep order of the real run (last occurrence)
        bench.append(l.strip())
    ops_csv = glob.glob(f"{rep}/ops_perf_results*.csv")[0]
    ops = list(csv.DictReader(open(ops_csv)))
    ops = [o for o in ops if "Generic" in o["OP CODE"] or "generic" in o["OP CODE"].lower()]
    bench = bench[-len(ops) :]
    zones = defaultdict(dict)  # call -> (x,y,risc,zone,type) -> time
    with open(f"{rep}/profile_log_device.csv") as f:
        f.readline()
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 12:
                continue
            x, y, risc, t, call, zone, typ = row[1], row[2], row[3], int(row[5]), row[7], row[10], row[11]
            if zone in ("writer_write", "BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL"):
                key = (int(x), int(y), risc.strip(), zone, typ)
                d = zones[call.strip()]
                if typ == "ZONE_START":
                    d[key] = min(d.get(key, t), t)
                else:
                    d[key] = max(d.get(key, t), t)
    for o, b in zip(ops, bench):
        call = o["GLOBAL CALL COUNT"]
        d = zones.get(call, {})
        starts = [v for k, v in d.items() if k[4] == "ZONE_START" and k[3].endswith("KERNEL")]
        t0 = min(starts) if starts else 0
        ww = []
        fin = []
        for k, v in d.items():
            if k[3] == "writer_write" and k[4] == "ZONE_START":
                e = d.get((k[0], k[1], k[2], k[3], "ZONE_END"))
                if e:
                    ww.append((e - v) / MHZ)
            if k[3] == "BRISC-KERNEL" and k[4] == "ZONE_END":
                fin.append((v - t0) / MHZ)
        m = re.search(r"variant=(\S+) shape=(\([^)]*\))", b)
        if HEAT and HEAT == f"{m.group(1)}:{m.group(2)}":
            heat = {}
            for k, v in d.items():
                if k[3] == "writer_write" and k[4] == "ZONE_START":
                    e = d.get((k[0], k[1], k[2], k[3], "ZONE_END"))
                    heat[(k[0], k[1])] = (e - v) / MHZ
            xs = sorted({k[0] for k in heat})
            ys = sorted({k[1] for k in heat})
            print("writer span (us) y\\x " + " ".join(f"{x:4d}" for x in xs))
            for y in ys:
                print(f"{y:20d} " + " ".join(f"{heat[(x, y)]:4.0f}" if (x, y) in heat else "   ." for x in xs))
        s = lambda a: f"{statistics.median(a):6.1f}/{max(a):6.1f}" if a else "   n/a"
        print(
            f"{m.group(1):16s} {m.group(2):22s} kernel_ns={int(o['DEVICE KERNEL DURATION [ns]']):9d} "
            f"cores={o['CORE COUNT']:>4s} writer_span p50/max={s(ww)} finish p50/max={s(fin)}"
        )


HEAT = None
if __name__ == "__main__":
    HEAT = sys.argv[3] if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2])
