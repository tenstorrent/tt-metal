# usage: relay_lat.py <device csv> <run_index> [k_cols]: per slice, the relay issue (column sender's writer_spread_relay
# start) to landing (spare member's writer_weight_fetch end) latency per round, and the column members' landing spread.
import re, subprocess, sys, statistics

out = subprocess.run(
    [
        sys.executable,
        __file__.replace("relay_lat.py", "rounds_tl.py"),
        sys.argv[1],
        sys.argv[2],
        "writer_weight_fetch,writer_spread_relay,writer_spread_mcast,BRISC-KERNEL",
    ],
    capture_output=True,
    text=True,
).stdout
k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
cores = {}
for l in out.splitlines():
    m = re.match(r"\(\s*(\d+),(\d+)\) phys\((\d+), (\d+)\) end ([0-9.]+)\s+(.*)", l)
    if not m:
        continue
    x, y = int(m.group(1)), int(m.group(2))
    d = {"end": float(m.group(5)), "phys": (int(m.group(3)), int(m.group(4)))}
    for part in m.group(6).split("|"):
        part = part.strip()
        if not part:
            continue
        name, *iv = part.split()
        d[name] = [tuple(map(float, s.split("-"))) for s in iv]
    cores[(x, y)] = d
for c in range(k):
    rel = [d for (x, y), d in cores.items() if x == c and "w.spread_relay" in d]
    if not rel:
        continue
    rs = [a for a, b in rel[0]["w.spread_relay"]]
    spares = [(x, d) for (x, y), d in cores.items() if x >= k and y == c]
    lats = []
    # Clock-offset corrected: each core's times relative to its own kernel start (the start spread by x is a per-core
    # clock offset: corrected, every relay lands at the same latency whatever its hop count).
    s0 = rel[0]["BRISC-KERNEL"][0][0]
    for x, d in spares:
        lands = [b for a, b in d["w.weight_fetch"]]
        s1 = d["BRISC-KERNEL"][0][0]
        lats.append([(l - s1) - (r - s0) for l, r in zip(lands, rs)])
    col_land = [
        statistics.median([d["w.weight_fetch"][r][1] for (x, y), d in cores.items() if x == c]) for r in range(len(rs))
    ]
    print(f"slice {c} relay phys{rel[0]['phys']} issue " + " ".join(f"{r:.2f}" for r in rs))
    print("   col median land " + " ".join(f"{v:.2f}" for v in col_land))
    for (x, d), l in zip(spares, lats):
        print(f"   spare ({x},{c}) lat " + " ".join(f"{v:.2f}" for v in l) + f"  end {d['end']:.2f}")
