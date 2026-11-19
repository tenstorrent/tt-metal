# Host-only link-load model of one column's halo fill (runs/geom.jsonl from geom_dump.py).
#   pull: every core reads its slab rows from the owner shards (NoC0 response flows down; rows of higher shards come up
#         over NoC1 where `split` says so, else wrap the torus on NoC0).
#   push: each row is multicast once by its owner: down over NoC0 to the lowest core needing it, up over NoC1 to the
#         highest. Link loads in KB; the column's busiest link bounds the fill at ~86 GB/s (64 B / cycle at 1.35 GHz).
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
case = sys.argv[1] if len(sys.argv) > 1 else "aspp_convs_1"
rec = [json.loads(l) for l in open(HERE / "runs" / "geom.jsonl") if json.loads(l)["case"] == case][-1]
ct, tb = rec["c_in_tiles"], rec["in0_tile_bytes"]
row_kb = ct * tb / 1024
shard_rows = 8 if case.startswith("aspp") else None
H_rows = 12  # NoC torus rows (BH): grid rows 0..9 at y 2..11, wrap through y 0, 1
grid_rows = len(rec["m_ranges"])
n_tile_rows = rec["H"] * rec["W"] // 32
shard_rows = shard_rows or (n_tile_rows // 8)
owner = lambda t: t // shard_rows
need = []
for b in rec["bands"]:
    s = set()
    for x in b:
        if x:
            s.update(range(x[0], x[1]))
    need.append(s)
Y = lambda r: r + 2


def add(load, a, b, kb, down):
    # links on the path from grid row a to b (NoC0 down: y+1; NoC1 up: y-1), physical y on a 12-row torus
    y, yb = Y(a), Y(b)
    while y != yb:
        ny = (y + 1) % H_rows if down else (y - 1) % H_rows
        key = (y, ny)
        load[key] = load.get(key, 0) + kb
        y = ny


def report(name, load):
    worst = max(load.items(), key=lambda kv: kv[1])
    tot = sum(load.values())
    print(
        f"{name:14s} busiest link {worst[0]} {worst[1]:7.1f} KB = {worst[1] * 1024 / 86e3:5.2f} us; link-KB total {tot:8.1f}"
    )
    return worst[1]


for split in ("all", "none"):
    load = {}
    for r in range(grid_rows):
        for t in need[r]:
            o = owner(t)
            if o == r:
                continue
            if o < r:
                add(load, o, r, row_kb, True)
            elif split == "all":
                add(load, o, r, row_kb, False)
            else:
                add(load, o, r, row_kb, True)
    report(f"pull split={split}", load)
load = {}
for t in range(n_tile_rows):
    rs = [r for r in range(grid_rows) if t in need[r]]
    o = owner(t)
    lo, hi = min(rs), max(rs)
    if hi > o:
        add(load, o, hi, row_kb, True)
    if lo < o:
        add(load, o, lo, row_kb, False)
report("push exact", load)
# push per shard: whole shard to the span of cores needing any of its rows
load = {}
for s in range(n_tile_rows // shard_rows):
    rs = [r for r in range(grid_rows) if any(t in need[r] for t in range(s * shard_rows, (s + 1) * shard_rows))]
    lo, hi = min(rs), max(rs)
    if hi > s:
        add(load, s, hi, row_kb * shard_rows, True)
    if lo < s:
        add(load, s, lo, row_kb * shard_rows, False)
report("push shard", load)
fan = [len([r for r in range(grid_rows) if t in need[r]]) for t in range(n_tile_rows)]
print("rows/core", [len(n) for n in need], "consumers per row: mean %.2f max %d" % (sum(fan) / len(fan), max(fan)))
print("bytes per core (KB)", [round(len(n) * row_kb) for n in need])
# Actual HEAD split: every row splits except the column's staggered weight sender (row col % rows) and rows reading
# under 8 KB from higher shards.
print("HEAD config per column (sender row unsplit):")
for col in range(rec["k_cols"]):
    snd = col % grid_rows
    load = {}
    for r in range(grid_rows):
        hi_kb = sum(row_kb for t in need[r] if owner(t) > r)
        split_r = r != snd and hi_kb * 1024 >= 8192
        for t in need[r]:
            o = owner(t)
            if o == r:
                continue
            if o < r:
                add(load, o, r, row_kb, True)
            elif split_r:
                add(load, o, r, row_kb, False)
            else:
                add(load, o, r, row_kb, True)
    w = max(load.items(), key=lambda kv: kv[1])
    print(f"  col {col} sender row {snd}: busiest {w[0]} {w[1]:6.1f} KB = {w[1] * 1024 / 86e3:4.2f} us")


# Per-link loads (KB), HEAD (col 2: sender row 2 unsplit) vs exact push.
def head_loads(col):
    snd = col % grid_rows
    load = {}
    for r in range(grid_rows):
        hi_kb = sum(row_kb for t in need[r] if owner(t) > r)
        split_r = r != snd and hi_kb * 1024 >= 8192
        for t in need[r]:
            o = owner(t)
            if o == r:
                continue
            add(load, o, r, row_kb, o < r or not split_r)
    return load


def push_loads():
    load = {}
    for t in range(n_tile_rows):
        rs = [r for r in range(grid_rows) if t in need[r]]
        o = owner(t)
        if max(rs) > o:
            add(load, o, max(rs), row_kb, True)
        if min(rs) < o:
            add(load, o, min(rs), row_kb, False)
    return load


h, p = head_loads(2), push_loads()
print("link (y0,y1)   HEAD col2   push-exact")
for k in sorted(set(h) | set(p), key=lambda k: (k[1] - k[0] != 1 and k[1] - k[0] != -11, k)):
    print(f"  {k}  {h.get(k, 0):8.1f}  {p.get(k, 0):8.1f}")
