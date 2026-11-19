# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight-stream timeline of one profiled conv2d_3x3 dispatch (no device).

    python3 timeline.py runs/dev_<label>.csv.gz [--run N] [--cores x,y;x,y]

Per run (program dispatch, in order): t0 = the earliest *-KERNEL start; per core (us from t0) the compute prelude end
(last compute_tilize end), the walk span (compute_matmul_walk, TRISC_0 = the unpacker, which is what waits on the weight
CB), the writer's weight rounds (writer_weight_fetch end = the round pushed to cb_weights) and its slab_dead wait, and
the kernel end. Prints the critical core (latest end), the column senders, and per-round landing p50 / max over cores."""

import argparse
import collections
import csv
import gzip
import statistics

ROW = False


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
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
    return freq, [runs[k] for k in sorted(runs)]


def spans(events):
    """(core, risc, zone) -> [(start, end)] in time order."""
    out = collections.defaultdict(list)
    stack = collections.defaultdict(list)
    for core, risc, t, zone, kind in sorted(events, key=lambda e: e[2]):
        key = (core, risc, zone)
        if kind == "ZONE_START":
            stack[key].append(t)
        elif kind == "ZONE_END" and stack[key]:
            out[key].append((stack[key].pop(), t))
    return out


def analyze(events, freq, show_cores=()):
    sp = spans(events)
    cores = sorted({c for c, *_ in events})
    t0 = min(s for (c, r, z), v in sp.items() if z.endswith("-KERNEL") for s, _ in v)
    us = lambda t: (t - t0) / freq
    info = {}
    for c in cores:
        d = {}
        ends = [e for (cc, r, z), v in sp.items() if cc == c and z.endswith("-KERNEL") for _, e in v]
        d["end"] = us(max(ends)) if ends else 0
        walk = sp.get((c, "TRISC_0", "compute_matmul_walk"), [])
        d["walk"] = (us(walk[0][0]), us(walk[-1][1])) if walk else None
        til = sp.get((c, "TRISC_0", "compute_tilize"), [])
        d["tilize_end"] = us(til[-1][1]) if til else None
        wf = sp.get((c, "BRISC", "writer_weight_fetch"), [])
        d["rounds"] = [(us(s), us(e)) for s, e in wf]
        sd = sp.get((c, "BRISC", "writer_slab_dead_wait"), [])
        d["slab_dead"] = [(us(s), us(e)) for s, e in sd]
        rw = sp.get((c, "NCRISC", "reader_ring_wait_step"), [])
        d["ring_wait"] = sum(e - s for s, e in rw) / freq
        ww = sp.get((c, "TRISC_0", "compute_walk_weight_wait"), [])
        d["wwait"] = [(e - s) / freq for s, e in ww]
        d["issue"] = [(us(a), us(b)) for a, b in sp.get((c, "BRISC", "bwr_issue"), [])]
        d["send"] = [(us(a), us(b)) for a, b in sp.get((c, "BRISC", "bwr_send"), [])]
        d["fill_end"] = max(
            (us(e) for (cc, r, z), v in sp.items() if cc == c and z == "writer_fill" for _, e in v), default=None
        )
        info[c] = d
    crit = max(info, key=lambda c: info[c]["end"])
    span = max(d["end"] for d in info.values())
    print(f"  cores {len(cores)}  span {span:.2f} us  critical core {crit}")
    # senders: the cores whose round-0 fetch is the longest in their column
    cols = collections.defaultdict(list)
    for c in cores:
        cols[c[0]].append(c)
    senders = [c for c in cores if info[c]["send"]]
    for x, cs in [] if senders else sorted(cols.items()):
        with_r = [c for c in cs if info[c]["rounds"]]
        if not with_r:
            continue
        s = max(with_r, key=lambda c: sum(e - b for b, e in info[c]["rounds"]))
        senders.append(s)
    nr = max(len(d["rounds"]) for d in info.values())
    land = [[d["rounds"][i][1] for d in info.values() if len(d["rounds"]) > i] for i in range(nr)]
    print("  round land  p50: " + " ".join(f"{statistics.median(v):5.1f}" for v in land))
    print("  round land  max: " + " ".join(f"{max(v):5.1f}" for v in land))
    wsum = sorted((sum(d["wwait"]), c) for c, d in info.items() if d["wwait"])
    if wsum:
        print(
            "  walk weight wait (sum per core) p50 %.2f max %.2f at %s; crit %.2f"
            % (statistics.median(v for v, _ in wsum), wsum[-1][0], wsum[-1][1], sum(info[crit]["wwait"]))
        )
    walks = [d["walk"] for d in info.values() if d["walk"]]
    if walks:
        print(
            "  walk start p50/max %.1f/%.1f  walk end p50/max %.1f/%.1f  walk len p50/max %.1f/%.1f"
            % (
                statistics.median(w[0] for w in walks),
                max(w[0] for w in walks),
                statistics.median(w[1] for w in walks),
                max(w[1] for w in walks),
                statistics.median(w[1] - w[0] for w in walks),
                max(w[1] - w[0] for w in walks),
            )
        )

    def line(c, tag):
        d = info[c]
        r = " ".join(f"{e:5.1f}" for _, e in d["rounds"])
        rs = " ".join(f"{s:5.1f}" for s, _ in d["rounds"])
        sd = " ".join(f"{s:.1f}-{e:.1f}" for s, e in d["slab_dead"])
        w = d["walk"]
        ws = f"{w[0]:.1f}-{w[1]:.1f}" if w else "-"
        fe = f"{d['fill_end']:.1f}" if d["fill_end"] is not None else "-"
        print(
            f"  {tag}{c}: end {d['end']:.1f} fill_end {fe} tilize_end {d['tilize_end'] or 0:.1f} walk {ws} ring_wait {d['ring_wait']:.1f} slab_dead {sd}"
        )
        print(f"      rounds start: {rs}")
        print(f"      rounds land : {r}")
        if d["issue"]:
            print("      issue      : " + " ".join(f"{a:.1f}-{b:.1f}" for a, b in d["issue"]))
        if d["send"]:
            print("      send       : " + " ".join(f"{a:.1f}-{b:.1f}" for a, b in d["send"]))
        if d["wwait"]:
            print("      walk wwait : " + " ".join(f"{v:5.2f}" for v in d["wwait"]))

    line(crit, "CRIT ")
    if ROW:
        for c in sorted(info):
            if c[1] == crit[1] and c != crit:
                line(c, "row  ")
        return info
    for s in senders:
        line(s, "send ")
    for c in show_cores:
        if c in info:
            line(c, "core ")
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--cores", default="")
    ap.add_argument("--min-cores", type=int, default=60)
    ap.add_argument("--row", action="store_true", help="print the critical core's grid row instead of the senders")
    a = ap.parse_args()
    ROW = a.row
    freq, runs = load(a.path)
    show = [tuple(int(v) for v in s.split(",")) for s in a.cores.split(";") if s]
    big = [r for r in runs if len({c for c, *_ in r}) >= a.min_cores]
    for i, ev in enumerate(big):
        if a.run is not None and i != a.run:
            continue
        print(f"run {i}")
        analyze(ev, freq, show)
