# Host-only: per-shard multicast plan (consumer span, down / up rectangles), ring size S and flag count K from
# runs/geom.jsonl, and the per-link count of overlapping multicast rectangles.
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
case = sys.argv[1] if len(sys.argv) > 1 else "aspp_convs_1"
shard_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rec = [json.loads(l) for l in open(HERE / "runs" / "geom.jsonl") if json.loads(l)["case"] == case][-1]
R = len(rec["m_ranges"])
n_rows = rec["H"] * rec["W"] // 32
need = []
for b in rec["bands"]:
    s = set()
    for x in b:
        if x:
            s.update(range(x[0], x[1]))
    need.append(s)
n_sh = (n_rows + shard_rows - 1) // shard_rows
span = {}
for o in range(n_sh):
    rows = set(range(o * shard_rows, min((o + 1) * shard_rows, n_rows)))
    cons = [r for r in range(R) if need[r] & rows]
    lo, hi = min(cons + [o]), max(cons + [o])
    span[o] = (lo, hi)
    print(f"shard {o}: consumers {cons} down [{o + 1},{hi}] up [{lo},{o - 1}]")
S, K = 0, 0
for r in range(R):
    cov = [o for o in range(n_sh) if span[o][0] <= r <= span[o][1]]
    recv = set()
    for o in cov:
        recv |= (
            set(range(o * shard_rows, min((o + 1) * shard_rows, n_rows)))
            if o != r
            else need[r] & set(range(o * shard_rows, (o + 1) * shard_rows))
        )
    ext = max(recv) - min(recv) + 1
    S = max(S, ext)
    K = max(K, max(cov) - min(cov) + 1)
    assert need[r] <= recv
    print(f"core {r}: covering shards {cov} extent {ext} rows")
print("S", S, "K", K, "ring KB", S * rec["c_in_tiles"] * rec["in0_tile_bytes"] / 1024)
for d, name in ((1, "down"), (-1, "up")):
    cnt = {}
    for o in range(n_sh):
        lo, hi = span[o]
        end = hi if d == 1 else lo
        y = o
        while y != end:
            cnt[(y, y + d)] = cnt.get((y, y + d), 0) + 1
            y += d
    print(name, "rectangles per link:", dict(sorted(cnt.items())))
