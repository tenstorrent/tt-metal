# usage: read_lat.py <device csv> <run_index> [k_cols]: per column, each spread round's reader, the read's issue
# (round r - 2 landed on the reader: its writer_weight_fetch end; rounds 0 / 1: the writer_spread_issue start) and
# landing (the reader's writer_spread_mcast start for that round), same-core times (no clock offset). Host-only.
import re, subprocess, sys, statistics

out = subprocess.run(
    [
        sys.executable,
        __file__.replace("read_lat.py", "rounds_tl.py"),
        sys.argv[1],
        sys.argv[2],
        "writer_weight_fetch,writer_spread_mcast,writer_spread_issue",
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
    d = {}
    for part in m.group(6).split("|"):
        part = part.strip()
        if part:
            name, *iv = part.split()
            d[name] = [tuple(map(float, s.split("-"))) for s in iv]
    cores[(int(m.group(1)), int(m.group(2)))] = d
allat = []
for c in range(k):
    col = {y: d for (x, y), d in cores.items() if x == c}
    # Which rounds each member multicast: order the column's mcast zones by time; round r is the r-th overall.
    ev = sorted((a, y) for y, d in col.items() for a, b in d.get("w.spread_mcast", []))
    line = []
    for r, (t, y) in enumerate(ev):
        d = col[y]
        if r >= 2:
            iss = d["w.weight_fetch"][r - 2][1]
        else:
            iss = d.get("w.spread_issue", [(float("nan"), 0)])[0][0]
        line.append(f"r{r}@{y}:{t - iss:.2f}")
        if r >= 2:
            allat.append(t - iss)
    print(f"col {c}: " + " ".join(line))
print(f"rounds >= 2 read latency: median {statistics.median(allat):.2f} min {min(allat):.2f} max {max(allat):.2f}")
