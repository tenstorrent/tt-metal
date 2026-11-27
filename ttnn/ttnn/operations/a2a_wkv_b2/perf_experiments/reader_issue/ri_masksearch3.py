"""Mask search with a fixed NoC 1 unit count per core (N1 of 16): minimise the max link load (ri_linkmodel).
usage: ri_masksearch3.py N1 OUT.json [seeds]"""
import random, json, sys
from collections import defaultdict
from ri_linkmodel import CORES, EP, PER_BANK, path

U = PER_BANK / 2
N1 = int(sys.argv[1])
P = {(noc, b, c): path(noc, EP[noc][b], c) for noc in (0, 1) for b in range(8) for c in CORES}


def run(seed, iters=150000):
    rnd = random.Random(seed)
    st = {}
    for c in CORES:
        bits = rnd.sample(range(16), N1)
        st[c] = sum(1 << b for b in bits)
    L = defaultdict(float)
    for c in CORES:
        for u in range(16):
            for l in P[((st[c] >> u) & 1, u % 8, c)]:
                L[l] += U

    def cost():
        v = sorted(L.values(), reverse=True)
        return v[0] + 0.3 * v[1] + 0.1 * v[2]

    cur = cost()
    T = cur * 0.003
    for it in range(iters):
        c = rnd.choice(CORES)
        ones = [u for u in range(16) if (st[c] >> u) & 1]
        zeros = [u for u in range(16) if not (st[c] >> u) & 1]
        if not ones or not zeros:
            continue
        a = rnd.choice(ones)
        b = rnd.choice(zeros)

        def mv(sign):
            for l in P[(1, a % 8, c)]:
                L[l] -= sign * U
            for l in P[(0, a % 8, c)]:
                L[l] += sign * U
            for l in P[(0, b % 8, c)]:
                L[l] -= sign * U
            for l in P[(1, b % 8, c)]:
                L[l] += sign * U

        mv(1)
        nc = cost()
        if nc <= cur or rnd.random() < pow(2.718, -(nc - cur) / max(T, 1e-9)):
            st[c] ^= (1 << a) | (1 << b)
            cur = nc
        else:
            mv(-1)
        T *= 0.99995
    return max(L.values()), st


best = None
for s in range(int(sys.argv[3]) if len(sys.argv) > 3 else 2):
    m, st = run(s)
    print(f"N1={N1} seed {s}: max link {m/1e6:.2f} MB ({m/64.5e9*1e6:.1f} us)")
    if best is None or m < best[0]:
        best = (m, st)
json.dump({f"{x},{y}": v for (x, y), v in best[1].items()}, open(sys.argv[2], "w"), indent=0)
