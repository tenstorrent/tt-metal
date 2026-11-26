"""Mask search with priority: keep the max link load (ri_linkmodel) under CAP MB while putting as few NoC 1
units as possible on remote-head cores (rows y4..y9; local heads sit on y2..y3 under HEAD_PLACEMENT=by_stream).
usage: ri_masksearch2.py CAP_MB OUT.json [seeds]"""
import random, json, sys
from collections import defaultdict
from ri_linkmodel import CORES, EP, PER_BANK, path

U = PER_BANK / 2
CAP = float(sys.argv[1]) * 1e6
P = {(noc, b, c): path(noc, EP[noc][b], c) for noc in (0, 1) for b in range(8) for c in CORES}
LOCAL = {c for c in CORES if c[1] in (2, 3)}


def run(seed, iters=150000):
    rnd = random.Random(seed)
    st = {c: (0xFFFF if c in LOCAL else 0x0000) for c in CORES}
    L = defaultdict(float)
    nrem = 0
    for c in CORES:
        for u in range(16):
            for l in P[((st[c] >> u) & 1, u % 8, c)]:
                L[l] += U

    def cost():
        over = max(0.0, max(L.values()) - CAP)
        remote1 = sum(bin(st[c]).count("1") for c in CORES if c not in LOCAL)
        return over / U * 50 + remote1

    cur = cost()
    T = 5.0
    for it in range(iters):
        c = rnd.choice(CORES)
        u = rnd.randrange(16)
        old = (st[c] >> u) & 1
        new = 1 - old
        for l in P[(old, u % 8, c)]:
            L[l] -= U
        for l in P[(new, u % 8, c)]:
            L[l] += U
        st[c] ^= 1 << u
        nc = cost()
        if nc <= cur or rnd.random() < pow(2.718, -(nc - cur) / max(T, 1e-6)):
            cur = nc
        else:
            st[c] ^= 1 << u
            for l in P[(new, u % 8, c)]:
                L[l] -= U
            for l in P[(old, u % 8, c)]:
                L[l] += U
        T *= 0.99995
    return max(L.values()), cur, st


best = None
for s in range(int(sys.argv[3]) if len(sys.argv) > 3 else 2):
    m, c, st = run(s)
    rem = sum(bin(st[x]).count("1") for x in CORES if x not in LOCAL) / (16 * 72)
    loc = sum(bin(st[x]).count("1") for x in CORES if x in LOCAL) / (16 * 24)
    print(
        f"seed {s}: max link {m/1e6:.2f} MB ({m/64.5e9*1e6:.1f} us) remote NoC1 frac {rem:.2f} local {loc:.2f} cost {c:.0f}"
    )
    if best is None or c < best[0]:
        best = (c, st)
json.dump({f"{x},{y}": v for (x, y), v in best[1].items()}, open(sys.argv[2], "w"), indent=0)
from collections import defaultdict as dd

rows = dd(list)
for (x, y), v in best[1].items():
    rows[y].append(bin(v).count("1") / 16)
print("NoC1 frac by row", {y: round(sum(v) / len(v), 2) for y, v in sorted(rows.items())})
