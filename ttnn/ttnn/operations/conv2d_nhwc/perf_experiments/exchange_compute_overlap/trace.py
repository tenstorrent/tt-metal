"""Per-round act-exchange timeline from an ECO_TRACE run: python3 -I trace.py <profile_log_device.csv> [row_y]
Events (data = round): writer eco_l land/sent, eco_a ack issued, eco_s own send start (before src wait),
eco_t own src ready; TRISC_0 eco_p at pre-hook of block b (= block b-1 popped).
Per-core clock skew is removed by aligning each row core's eco_l of receiver rounds to the row median."""
import csv, sys, collections, statistics as st

rows = list(csv.reader(open(sys.argv[1])))[1:]
h = [c.strip() for c in rows[0]]
ix = {c: i for i, c in enumerate(h)}
last = max(int(r[ix["run host ID"]]) for r in rows[1:] if len(r) >= len(h) and r[ix["run host ID"]].strip().isdigit())
ev = collections.defaultdict(dict)  # core -> (name, round) -> t
for r in rows[1:]:
    if len(r) < len(h) or r[ix["run host ID"]].strip() != str(last):
        continue
    n = r[ix["zone name"]].strip()
    if n.startswith("eco_"):
        c = (int(r[ix["core_x"]]), int(r[ix["core_y"]]))
        ev[c][(n, int(r[ix["data"]]))] = int(r[ix["time[cycles since reset]"]])
ys = sorted({c[1] for c in ev})
want = [int(a) for a in sys.argv[2:]] or ys
agg = collections.defaultdict(list)
for y in ys:
    cores = sorted(c for c in ev if c[1] == y)
    # skew: align eco_l (land) per round to the first core
    ref = cores[0]
    off = {}
    for c in cores:
        d = [
            ev[c][("eco_l", r)] - ev[ref][("eco_l", r)]
            for r in range(20, 50)
            if ("eco_l", r) in ev[c] and ("eco_l", r) in ev[ref]
        ]
        off[c] = st.median(d) if d else 0
    T = lambda c, k: ev[c].get(k, None) - off[c] if ev[c].get(k) is not None else None
    tbase = min(T(c, ("eco_l", 21)) for c in cores)
    lines = []
    for r in range(22, 49):
        owner = [c for c in cores if ("eco_s", r) in ev[c]]
        o = owner[0] if owner else None
        land = [T(c, ("eco_l", r)) for c in cores]
        acks = [(T(c, ("eco_a", r)), c) for c in cores if ("eco_a", r) in ev[c]]
        pops = [(T(c, ("eco_p", r - 1)), c) for c in cores if ("eco_p", r - 1) in ev[c]]  # pop of r-2
        lastack, ackc = max(acks)
        lastpop, popc = max(pops)
        s = T(o, ("eco_s", r)) if o else None
        t = T(o, ("eco_t", r)) if o else None
        own_prev_land = T(o, ("eco_l", r - 1)) if o else None
        landmin = min(land)
        land_prev = min(T(c, ("eco_l", r - 1)) for c in cores)
        rec = dict(
            gap=landmin - land_prev,  # round period
            ack_after_prevland=lastack - land_prev,  # last ack of r (rel. land r-1)
            pop2_after_prevland=lastpop - land_prev,  # last core pop of r-2
            send_start=s - land_prev,
            src_ready=t - land_prev,
            land_after_lastack=landmin - lastack,
            land_after_srcready=landmin - t,
        )
        for k, v in rec.items():
            agg[k].append(v)
        if y in want and y == want[0]:
            lines.append(
                f"r{r:2d} own={o[0] if o else '-'} "
                + " ".join(f"{k}={v}" for k, v in rec.items())
                + f" slowack={ackc[0]} slowpop={popc[0]}"
            )
    print("\n".join(lines))
print("== medians over rows/rounds (cycles, rel. to land of r-1):")
for k, v in agg.items():
    print(f"  {k:22s} median {st.median(v):8.0f}  mean {st.mean(v):8.0f}  p90 {sorted(v)[int(.9*len(v))]:8.0f}")
