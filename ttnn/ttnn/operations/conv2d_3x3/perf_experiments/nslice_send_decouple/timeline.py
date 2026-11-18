# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-core send / walk / combine timeline of one profiled dispatch (no ttnn, no device).

    python3 timeline.py runs/<tag>_device.csv.gz [--run N] [--cores 3,3;0,0] [--rows]

Times are us from the dispatch's earliest marker over all cores (BH core timers share one reference). Columns per core:
walk start/end (compute_matmul_walk, TRISC_1), the reader's and the writer's slice sends (start-end each), the combine
wait and the combine end, and the core's last kernel end. The slowest core is printed first.
"""

import argparse
import collections
import csv
import gzip
import statistics


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        header = f.readline()
        freq = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
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


def zones(events, freq):
    t0 = min(e[2] for e in events)
    stacks = collections.defaultdict(list)
    out = collections.defaultdict(list)  # (core, risc, zone) -> [(s, e)]
    for core, risc, t, zone, kind in sorted(events, key=lambda e: e[2]):
        key = (core, risc, zone)
        if kind == "ZONE_START":
            stacks[key].append(t)
        elif kind == "ZONE_END" and stacks[key]:
            s = stacks[key].pop()
            out[key].append(((s - t0) / freq, (t - t0) / freq))
    return out


def core_summary(z, core):
    def get(risc, zone):
        return z.get((core, risc, zone), [])

    walk = get("TRISC_1", "compute_matmul_walk") or get("TRISC_0", "compute_matmul_walk")
    rs = get("NCRISC", "reader_combine_send")
    ws = get("BRISC", "writer_combine_send")
    cw = get("TRISC_1", "compute_combine_wait") or get("TRISC_0", "compute_combine_wait")
    cb = get("TRISC_2", "compute_combine") or get("TRISC_1", "compute_combine")
    ends = [e for (c, r, zn), v in z.items() if c == core and zn.endswith("-KERNEL") for _, e in v]
    return dict(
        walk=walk[0] if walk else None,
        rs=rs,
        ws=ws,
        cwait=cw[0] if cw else None,
        comb=cb[0] if cb else None,
        end=max(ends) if ends else None,
    )


def fmt(iv):
    return "   -   " if iv is None else f"{iv[0]:6.2f}-{iv[1]:6.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--run", type=int, default=None, help="run index (0-based, in run-id order)")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--min-cores", type=int, default=16)
    args = ap.parse_args()
    freq, runs = load(args.log)
    ids = [r for r in sorted(runs) if len({e[0] for e in runs[r]}) >= args.min_cores]
    sel = ids if args.run is None else [ids[args.run]]
    for rid in sel:
        z = zones(runs[rid], freq)
        cores = sorted({c for c, _, _ in z})
        summ = {c: core_summary(z, c) for c in cores}
        ends = {c: s["end"] for c, s in summ.items() if s["end"] is not None}
        order = sorted(ends, key=lambda c: -ends[c])
        walk_end = [s["walk"][1] for s in summ.values() if s["walk"]]
        last_rs = [s["rs"][-1][1] for s in summ.values() if s["rs"]]
        last_ws = [s["ws"][-1][1] for s in summ.values() if s["ws"]]
        comb_wait = [s["cwait"][1] - s["cwait"][0] for s in summ.values() if s["cwait"]]
        walk_dur = [s["walk"][1] - s["walk"][0] for s in summ.values() if s["walk"]]
        rs_dur = [e - s for v in summ.values() for s, e in v["rs"]]
        ws_dur = [e - s for v in summ.values() for s, e in v["ws"]]

        def st(xs):
            return f"p50 {statistics.median(xs):6.2f} max {max(xs):6.2f}" if xs else "-"

        print(f"== run {rid}: {len(cores)} cores, op end {max(ends.values()):.2f} us")
        print(f"   walk dur {st(walk_dur)} | walk end {st(walk_end)}")
        print(f"   reader send dur {st(rs_dur)} | last reader send end {st(last_rs)}")
        print(f"   writer send dur {st(ws_dur)} | last writer send end {st(last_ws)}")
        print(f"   combine wait {st(comb_wait)}")
        for c in order[: args.top]:
            s = summ[c]
            print(
                f"   core {c}: end {s['end']:6.2f} walk {fmt(s['walk'])} cwait {fmt(s['cwait'])} comb {fmt(s['comb'])}"
            )
            print("      reader sends: " + " ".join(fmt(x) for x in s["rs"]))
            print("      writer sends: " + " ".join(fmt(x) for x in s["ws"]))


if __name__ == "__main__":
    main()
