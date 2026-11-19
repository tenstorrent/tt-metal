# Host-only NoC link-load model of the kw-fold slab fill (no device): every core reads its slab's live tile-rows from
# the input shards, shards <= split over NoC0 (+x, then +y, torus) and > split over NoC1 (-y, then -x, torus), on the
# p150b virtual grid (worker x 1-6, 11-15 of a 17-ring; y 2-11 of a 12-ring). Prints the busiest link's bytes and the
# per-core hop-bytes for a layout, to rank layouts before running them.
#   python3 fill_model.py
import collections

VX = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]
VY = list(range(2, 12))
RX, RY = 17, 12
GX, GY = 11, 10
M, SHARD_ROWS, TPR, CT, TILE_B = 256, 3, 4, 10, 1088
ROW_B = CT * TILE_B


def pos(p):
    return p % GX, p // GX


def slab_rows(t0, n):
    # tile-rows [t0 - 1 - R, t0 + n + R], clamped (live-row skip ignored: an upper bound)
    lo, hi = max(0, t0 - 1 - TPR), min(M - 1, t0 + n + TPR)
    return range(lo, hi + 1)


def route(src, dst, noc):
    (sx, sy), (dx, dy) = src, dst
    x, y = VX[sx], VY[sy]
    tx, ty = VX[dx], VY[dy]
    links = []
    if noc == 0:
        while x != tx:
            links.append(("x+", x, y))
            x = (x + 1) % RX
        while y != ty:
            links.append(("y+", x, y))
            y = (y + 1) % RY
    else:
        while y != ty:
            links.append(("y-", x, y))
            y = (y - 1) % RY
        while x != tx:
            links.append(("x-", x, y))
            x = (x - 1) % RX
    return links


def model(items, name):
    """items: (core position, t0, n, split shard)"""
    load = collections.Counter()
    nic = collections.Counter()
    hopb = 0
    for c, t0, n, split in items:
        for r in slab_rows(t0, n):
            s = r // SHARD_ROWS
            if s == c:
                continue
            noc = 0 if s <= split else 1
            ls = route(pos(s), pos(c), noc)
            for l in ls:
                load[l] += ROW_B
            nic[(s, noc)] += ROW_B
            hopb += ROW_B * len(ls)
    mx = max(load.values())
    print(
        f"{name:12s} max link {mx/1e3:7.0f} KB -> {mx/64/1350:5.2f} us   max shard-port {max(nic.values())/1e3:5.0f} KB   "
        f"hop-MB {hopb/1e6:6.1f}"
    )


def balanced(k):
    r = -(-M // k)
    big = M - k * (r - 1)
    out = []
    t = 0
    for i in range(k):
        n = r if i < big else r - 1
        out.append((t, n))
        t += n
    return out


base = [(i, t0, n, i) for i, (t0, n) in enumerate([(3 * i, min(3, M - 3 * i)) for i in range(86)])]
model(base, "baseline")
rng = balanced(52)
pairs = [(2 * j + p, t0, n, 2 * j + p) for p in range(2) for j, (t0, n) in enumerate(rng)]
model(pairs, "pairs")
rows = [(52 * p + j, t0, n, 52 * p + j) for p in range(2) for j, (t0, n) in enumerate(rng)]
model(rows, "rows")


def best_split(c, t0, n):
    """(hop-bytes, split) of the cheapest shard threshold for a core at position c reading range (t0, n)."""
    shards = sorted({r // SHARD_ROWS for r in slab_rows(t0, n)})
    best = None
    for q in [shards[0] - 1] + shards:
        hb = 0
        for r in slab_rows(t0, n):
            s = r // SHARD_ROWS
            if s == c:
                continue
            hb += ROW_B * len(route(pos(s), pos(c), 0 if s <= q else 1))
        if best is None or hb < best[0]:
            best = (hb, q)
    return best


def assign(slots, rng, planes=2):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    items = [(p, j) for p in range(planes) for j in range(len(rng))]
    cost = np.zeros((len(items), len(slots)))
    split = {}
    for a, (p, j) in enumerate(items):
        t0, n = rng[j]
        for b, c in enumerate(slots):
            hb, q = best_split(c, t0, n)
            cost[a, b] = hb
            split[(a, b)] = q
    ri, ci = linear_sum_assignment(cost)
    out = {}
    for a, b in zip(ri, ci):
        p, j = items[a]
        out[(p, j)] = (slots[b], split[(a, b)])
    return out


if __name__ == "__main__":
    import sys

    slots = list(range(104))
    a = assign(slots, rng)
    opt = [(c, rng[j][0], rng[j][1], q) for (p, j), (c, q) in a.items()]
    model(opt, "hungarian")
    # Baseline with the best split per core (HEAD uses its own shard).
    model([(i, t0, n, best_split(i, t0, n)[1]) for i, t0, n, _ in base], "base-bestq")
    # pairs with the best split per core
    model([(c, t0, n, best_split(c, t0, n)[1]) for c, t0, n, _ in pairs], "pairs-bestq")
    if "dump" in sys.argv:
        for (p, j), (c, q) in sorted(a.items()):
            print(p, j, c, q)


def link_loads(c, t0, n, q):
    load = collections.Counter()
    for r in slab_rows(t0, n):
        s = r // SHARD_ROWS
        if s == c:
            continue
        for l in route(pos(s), pos(c), 0 if s <= q else 1):
            load[l] += ROW_B
    return load


def anneal(a, iters=20000, seed=0):
    """Swap two items' positions (each re-takes its best split) while the busiest link does not get busier."""
    import random

    rnd = random.Random(seed)
    keys = list(a)
    cache = {}

    def contrib(k, c):
        p, j = k
        t0, n = rng[j]
        if (j, c) not in cache:
            q = best_split(c, t0, n)[1]
            cache[(j, c)] = (q, link_loads(c, t0, n, q))
        return cache[(j, c)]

    cur = {k: a[k][0] for k in keys}
    total = collections.Counter()
    for k in keys:
        total.update(contrib(k, cur[k])[1])

    def score(t):
        v = sorted(t.values(), reverse=True)
        return v[0] + 0.05 * sum(v[:20])

    best = score(total)
    for it in range(iters):
        k1, k2 = rnd.sample(keys, 2)
        c1, c2 = cur[k1], cur[k2]
        t = total.copy()
        t.subtract(contrib(k1, c1)[1])
        t.subtract(contrib(k2, c2)[1])
        t.update(contrib(k1, c2)[1])
        t.update(contrib(k2, c1)[1])
        s = score(t)
        if s <= best:
            best, total = s, t
            cur[k1], cur[k2] = c2, c1
    return {k: (cur[k], contrib(k, cur[k])[0]) for k in keys}


if __name__ == "__main__" and "anneal" in sys.argv:
    b2 = anneal(a, iters=int(sys.argv[-1]) if sys.argv[-1].isdigit() else 20000)
    opt2 = [(c, rng[j][0], rng[j][1], q) for (p, j), (c, q) in b2.items()]
    model(opt2, "annealed")
    import json

    json.dump({f"{p},{j}": [c, q] for (p, j), (c, q) in b2.items()}, open("layout_annealed.json", "w"))


def anneal_planes(plane_slots, iters=50000, seed=0):
    """Per-plane block layout: plane p's ranges (balanced over its slots) on its own slots, each range's core and split
    chosen to keep the busiest link low (swaps within a plane only). Returns {(p, j): (position, split)}."""
    import random

    rnd = random.Random(seed)
    rngs = [balanced(len(s)) for s in plane_slots]
    cache = {}

    def contrib(p, j, c):
        t0, n = rngs[p][j]
        if (t0, n, c) not in cache:
            q = best_split(c, t0, n)[1]
            cache[(t0, n, c)] = (q, link_loads(c, t0, n, q))
        return cache[(t0, n, c)]

    # start: each plane row-major over its slots
    cur = {(p, j): s for p, slots in enumerate(plane_slots) for j, s in enumerate(slots)}
    total = collections.Counter()
    for (p, j), c in cur.items():
        total.update(contrib(p, j, c)[1])

    def score(t):
        v = sorted(t.values(), reverse=True)
        return v[0] + 0.05 * sum(v[:20])

    best = score(total)
    for it in range(iters):
        p = rnd.randrange(len(plane_slots))
        j1, j2 = rnd.sample(range(len(plane_slots[p])), 2)
        c1, c2 = cur[(p, j1)], cur[(p, j2)]
        t = total.copy()
        t.subtract(contrib(p, j1, c1)[1])
        t.subtract(contrib(p, j2, c2)[1])
        t.update(contrib(p, j1, c2)[1])
        t.update(contrib(p, j2, c1)[1])
        s = score(t)
        if s <= best:
            best, total = s, t
            cur[(p, j1)], cur[(p, j2)] = c2, c1
    out = {k: (c, contrib(k[0], k[1], c)[0]) for k, c in cur.items()}
    items = [(c, rngs[p][j][0], rngs[p][j][1], q) for (p, j), (c, q) in out.items()]
    return out, items


if __name__ == "__main__" and "blocks" in sys.argv:
    P = lambda x, y: y * GX + x
    right = [P(x, y) for y in range(9) for x in range(5, 11)]
    left = [P(x, y) for y in range(10) for x in range(0, 5)] + [P(5, 9), P(6, 9)]
    for name, slots in (("blocks", [right, left]),):
        rm = [(c, rr[0], rr[1], c) for p, s in enumerate(slots) for c, rr in zip(s, balanced(len(s)))]
        model(rm, name + "-rowmaj")
        out, items = anneal_planes(slots, iters=int(sys.argv[-1]) if sys.argv[-1].isdigit() else 50000)
        model(items, name + "-annealed")
        import json

        json.dump({f"{p},{j}": [c, q] for (p, j), (c, q) in out.items()}, open(f"layout_{name}.json", "w"))
