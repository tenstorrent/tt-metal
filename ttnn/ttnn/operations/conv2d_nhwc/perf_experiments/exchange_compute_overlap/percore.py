"""Per-core compute vs landing per block (same-core clock): python3 -I percore.py <csv>"""
import csv, sys, collections, statistics as st

rows = list(csv.reader(open(sys.argv[1])))[1:]
h = [c.strip() for c in rows[0]]
ix = {c: i for i, c in enumerate(h)}
last = max(int(r[ix["run host ID"]]) for r in rows[1:] if len(r) >= len(h) and r[ix["run host ID"]].strip().isdigit())
ev = collections.defaultdict(dict)
til = collections.defaultdict(list)
for r in rows[1:]:
    if len(r) < len(h) or r[ix["run host ID"]].strip() != str(last):
        continue
    n = r[ix["zone name"]].strip()
    c = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
    t = int(r[ix["time[cycles since reset]"]])
    if n.startswith("eco_"):
        ev[c][(n, int(r[ix["data"]]))] = t
    if n == "compute_tilize_own" and r[ix["RISC processor type"]].strip() == "TRISC_0":
        til[c].append((r[ix["type"]].strip(), t))
W = collections.defaultdict(list)
for c in sorted(ev):
    e = ev[c]
    ts = sorted(til[c], key=lambda x: x[1])
    tz = [(ts[i][1], ts[i + 1][1]) for i in range(0, len(ts) - 1, 2)]
    for b in range(21, 49):
        if ("eco_p", b) not in e or ("eco_p", b + 1) not in e or ("eco_l", b) not in e:
            continue
        hook, nxt, land = e[("eco_p", b)], e[("eco_p", b + 1)], e[("eco_l", b)]
        tl = sum(t1 - t0 for t0, t1 in tz if hook <= t0 < nxt)
        start = max(hook + tl, land)
        W["busy_notilize" if tl == 0 else "busy_tilize"].append(nxt - start)
        W["tilize"].append(tl) if tl else None
        W["data_wait"].append(max(0, land - (hook + tl)))
        W["pop_minus_land_next"].append(nxt - e.get(("eco_l", b + 1), nxt))
for k, v in W.items():
    v = sorted(v)
    print(
        f"{k:22s} n={len(v):4d} median {st.median(v):7.0f} mean {st.mean(v):7.0f} p10 {v[len(v)//10]:7.0f} p90 {v[9*len(v)//10]:7.0f}"
    )
