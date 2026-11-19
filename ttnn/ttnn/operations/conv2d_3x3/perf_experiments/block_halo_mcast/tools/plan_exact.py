# Host-only mirror of the descriptor's halo_mcast_plan (exact and shard runs): S, K per case from runs/geom.jsonl.
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
for case in sys.argv[1:]:
    rec = [json.loads(l) for l in open(HERE / "runs" / "geom.jsonl") if json.loads(l)["case"] == case][-1]
    R = len(rec["m_ranges"])
    n_rows = rec["H"] * rec["W"] // 32
    shard_rows = 8
    need = []
    for b in rec["bands"]:
        s = set()
        for x in b:
            if x:
                s.update(range(x[0], x[1]))
        need.append(sorted(s))
    readers_of = {}
    for r, rows in enumerate(need):
        for t in rows:
            readers_of.setdefault(t, []).append(r)
    for mode in ("shard", "exact"):
        runs, down, up = {}, {}, {}
        for o in range(8):
            rows = range(o * shard_rows, min((o + 1) * shard_rows, n_rows))
            readers = sorted({r for t in rows for r in readers_of.get(t, [])})
            if mode == "shard":
                lo, hi = min(readers + [o]), max(readers + [o])
                runs[o] = [(rows[0], rows[-1] + 1, hi, lo)]
            else:
                rr = []
                for t in rows:
                    rd = readers_of.get(t, [])
                    d_last, u_first = max(rd + [o]), min(rd + [o])
                    if rr and rr[-1][2] == d_last and rr[-1][3] == u_first:
                        rr[-1] = (rr[-1][0], t + 1, d_last, u_first)
                    else:
                        rr.append((t, t + 1, d_last, u_first))
                runs[o] = rr
            down[o] = max(x[2] for x in runs[o])
            up[o] = min(x[3] for x in runs[o])
        S, K = 0, 1
        for r in range(R):
            cov = [o for o in runs if up[o] <= r <= down[o]]
            got = set(need[r]) if r in cov else set()
            for o in cov:
                if o == r:
                    continue
                for lo_, hi_, d_last, u_first in runs[o]:
                    if (o < r <= d_last) or (u_first <= r < o):
                        got |= set(range(lo_, hi_))
            S = max(S, max(got) - min(got) + 1)
            K = max(K, max(cov) - min(cov) + 1)
        print(
            case,
            mode,
            "S",
            S,
            "K",
            K,
            "ring KB",
            S * rec["c_in_tiles"] * rec["in0_tile_bytes"] / 1024,
            "region KB",
            rec["shared_slab_bytes"] / 1024,
        )
