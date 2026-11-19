# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: round-0 view of one BLOCK N-slice dispatch from a device profiler log (zoned run).

    python3 r0view.py <profile_log_device.csv[.gz]> <run# (1-based among dispatches on >= 8 cores)> [k_cols] [rows] [sender shift]

Per core (logical x, y; ROW_MAJOR 11x10 Blackhole p150b mapping), its role in the column weight stream (S = column
sender, F = forwarder, R = column receiver, P = spare member), then (us after the run's earliest kernel start):
  fetch0 / fetch1   writer_weight_fetch of rounds 0 / 1: start-end (end = round landed and pushed / multicast issued)
  fwd0              writer_spare_forward of round 0 (forwarders)
  fill              reader_fill end;  pre = compute_tilize end (TRISC_0..2 max);  walk = compute_matmul_walk start
  idle              walk start - prelude end (the round-0 bubble);  end = kernel end
"""
import collections
import csv
import gzip
import sys

XMAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 11: 6, 12: 7, 13: 8, 14: 9, 15: 10}


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
    out = collections.defaultdict(list)
    for c, rk, t, z, k in sorted(ev, key=lambda e: e[2]):
        if k == "ZONE_START":
            stack[(c, rk, z)].append(t)
        elif k == "ZONE_END" and stack[(c, rk, z)]:
            s = stack[(c, rk, z)].pop()
            out[(c, z)].append((rk, (s - t0) / freq, (t - t0) / freq))
    return out


def main():
    freq, runs = load(sys.argv[1])
    big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
    ev = runs[big[int(sys.argv[2]) - 1]]
    k_cols = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rows = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    shift = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    sp = spans(ev, freq)
    cores = sorted({c for c, _ in sp}, key=lambda c: (XMAP[c[0]], c[1]))

    def role(lx, ly):
        if lx >= k_cols:
            return "P"
        if ly == (lx + shift) % rows:
            return "S"
        if ly == (lx + shift + 1) % rows:
            return "F"
        return "R"

    def fmt(v):
        return f"{v[1]:5.2f}-{v[2]:5.2f}" if v else "     -     "

    print(
        f"{'core':>8} {'log':>7} r  {'fetch0':>11} {'fetch1':>11} {'fwd0':>11} {'fill':>5} {'pre':>5} "
        f"{'walk':>5} {'idle':>5} {'end':>5}"
    )
    worst = []
    for c in cores:
        lx, ly = XMAP[c[0]], c[1] - 2
        f = sorted(sp.get((c, "writer_weight_fetch"), []), key=lambda v: v[1])
        fw = sorted(sp.get((c, "writer_spare_forward"), []), key=lambda v: v[1])
        fill = max((v[2] for v in sp.get((c, "reader_fill"), [])), default=0)
        pre = max((v[2] for v in sp.get((c, "compute_tilize"), [])), default=0)
        walk = min((v[1] for v in sp.get((c, "compute_matmul_walk"), [])), default=0)
        end = max((v[2] for (cc, z), vs in sp.items() if cc == c and z.endswith("-KERNEL") for v in vs), default=0)
        r = role(lx, ly)
        print(
            f"{str(c):>8} {str((lx, ly)):>7} {r}  {fmt(f[0] if f else None)} {fmt(f[1] if len(f) > 1 else None)} "
            f"{fmt(fw[0] if fw else None)} {fill:5.2f} {pre:5.2f} {walk:5.2f} {walk - pre:5.2f} {end:5.2f}"
        )
        worst.append((end, c, (lx, ly), r, f[0][2] if f else 0, pre, walk))
    worst.sort(reverse=True)
    print(
        "latest 6:",
        " ".join(f"{c}{l}{r} end {e:.2f} r0 {a:.2f} pre {p:.2f} walk {w:.2f}" for e, c, l, r, a, p, w in worst[:6]),
    )
    by = collections.defaultdict(list)
    for e, c, l, r, a, p, w in worst:
        by[r].append((a, p, w, e))
    for r in "SFRP":
        if by[r]:
            v = by[r]
            med = lambda i: sorted(x[i] for x in v)[len(v) // 2]
            mx = lambda i: max(x[i] for x in v)
            print(
                f"role {r} n {len(v):3d}: r0 land p50 {med(0):.2f} max {mx(0):.2f}  pre p50 {med(1):.2f} max {mx(1):.2f}"
                f"  walk p50 {med(2):.2f} max {mx(2):.2f}  end p50 {med(3):.2f} max {mx(3):.2f}"
            )


if __name__ == "__main__":
    main()
