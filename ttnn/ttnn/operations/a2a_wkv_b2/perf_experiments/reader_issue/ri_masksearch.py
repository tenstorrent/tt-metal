"""Local search over per-core NoC choice per (bank, half) unit to minimise the max link load (ri_linkmodel)."""
import random, json, sys
from collections import defaultdict
from ri_linkmodel import CORES, EP, PER_BANK, path

U = PER_BANK / 2
P = {(noc, b, c): path(noc, EP[noc][b], c) for noc in (0, 1) for b in range(8) for c in CORES}


def run(seed, iters=60000, init=None):
    rnd = random.Random(seed)
    # state: per core 16-bit mask (bit b: bank b half 0 on NoC1; bit 8+b: half 1)
    st = dict(init) if init else {c: (0x00 if c[0] < 9 else 0xFFFF) for c in CORES}
    L = defaultdict(float)
    for c in CORES:
        for u in range(16):
            for l in P[((st[c] >> u) & 1, u % 8, c)]:
                L[l] += U

    def cost():
        v = sorted(L.values(), reverse=True)
        return v[0] + 0.05 * v[1] + 0.02 * v[2]

    cur = cost()
    T = cur * 0.002
    for it in range(iters):
        c = rnd.choice(CORES)
        u = rnd.randrange(16)
        old = (st[c] >> u) & 1
        new = 1 - old
        for l in P[(old, u % 8, c)]:
            L[l] -= U
        for l in P[(new, u % 8, c)]:
            L[l] += U
        nc = cost()
        if nc <= cur or rnd.random() < pow(2.718, -(nc - cur) / max(T, 1e-9)):
            st[c] ^= 1 << u
            cur = nc
        else:
            for l in P[(new, u % 8, c)]:
                L[l] -= U
            for l in P[(old, u % 8, c)]:
                L[l] += U
        T *= 0.9999
    return max(L.values()), st


best = None
for s in range(int(sys.argv[1]) if len(sys.argv) > 1 else 4):
    m, st = run(s)
    print("seed", s, "max link MB", round(m / 1e6, 3), "-> us @64.5GB/s", round(m / 64.5e9 * 1e6, 1))
    if best is None or m < best[0]:
        best = (m, st)
json.dump({f"{x},{y}": v for (x, y), v in best[1].items()}, open("mask_search.json", "w"), indent=0)
# summary: fraction on NoC1 per column
for x in sorted({c[0] for c in CORES}):
    print(x, [f"{best[1][(x, y)]:04x}" for y in range(2, 10)])
