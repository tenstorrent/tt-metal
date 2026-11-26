"""Link-load model of the worker DRAM read stream on one Blackhole p150 (physical coords).
NoC0 response: east along the DRAM endpoint row, then south along the core column. NoC1 response: north along
the DRAM column, then west along the core row (examples/noc_placement). Bank b = channel b (dram_views)."""
from collections import defaultdict

X, Y = 17, 12
EP = {
    0: {0: (0, 11), 1: (0, 2), 2: (0, 9), 3: (0, 5), 4: (9, 11), 5: (9, 3), 6: (9, 8), 7: (9, 6)},
    1: {0: (0, 1), 1: (0, 10), 2: (0, 4), 3: (0, 7), 4: (9, 1), 5: (9, 10), 6: (9, 4), 7: (9, 7)},
}
COLS = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15]
ROWS = list(range(2, 10))
CORES = [(x, y) for y in ROWS for x in COLS]
PER_BANK = (160 * 1024 + 64 * 1088) / 8  # bytes per core per bank


def path(noc, src, dst):
    (x, y), (cx, cy) = src, dst
    links = []
    if noc == 0:
        while x != cx:
            links.append((0, x, y, "E"))
            x = (x + 1) % X
        while y != cy:
            links.append((0, x, y, "S"))
            y = (y + 1) % Y
    else:
        while y != cy:
            links.append((1, x, y, "N"))
            y = (y - 1) % Y
        while x != cx:
            links.append((1, x, y, "W"))
            x = (x - 1) % X
    return links


def loads(maskfn):
    L = defaultdict(float)
    for c in CORES:
        m = maskfn(c)
        for b in range(8):
            noc = (m >> b) & 1
            for l in path(noc, EP[noc][b], c):
                L[l] += PER_BANK
    return L


def score(maskfn, bw=86e9 * 0.75):
    L = loads(maskfn)
    worst = max(L.values())
    # per core: worst link on any of its paths
    return worst / bw * 1e6, L


CONFIGS = {
    "halves": lambda c: 0x00 if c[0] < 9 else 0xFF,
    "00": lambda c: 0x00,
    "ff": lambda c: 0xFF,
    "0f": lambda c: 0x0F,
    "f0": lambda c: 0xF0,
    "geo": lambda c: 0xF0 if c[0] < 9 else 0x0F,
    "geo_inv": lambda c: 0x0F if c[0] < 9 else 0xF0,
}
if __name__ == "__main__":
    meas = {"halves": 52.4, "00": 60.3, "ff": 103.4, "0f": 100.8, "f0": 101.0, "geo": 72.3, "geo_inv": 60.4}
    for k, f in CONFIGS.items():
        s, L = score(f)
        top = sorted(L.items(), key=lambda kv: -kv[1])[:3]
        print(
            f"{k:8s} model max-link {s:6.1f} us  measured {meas[k]:6.1f}   top {[(l, round(v/1e6,2)) for l, v in top]}"
        )
