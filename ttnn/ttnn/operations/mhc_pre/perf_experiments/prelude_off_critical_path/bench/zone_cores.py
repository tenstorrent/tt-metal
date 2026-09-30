"""Per-core start/end of given zones in the latest mhc_pre program of a profile_log_device.csv.
usage: python3 zone_cores.py <csv> zone1 [zone2 ...]  -> grid of zone END time (us) per core (x cols, y rows)"""
import collections, sys

CLK = 1350.0
f, zones = sys.argv[1], sys.argv[2:]
lines = open(f).read().splitlines()
hdr = [h.strip() for h in lines[1].split(",")]
ix = {h: i for i, h in enumerate(hdr)}
rows = [[x.strip() for x in l.split(",")] for l in lines[2:]]
rows = [r for r in rows if len(r) >= len(hdr) - 1 and r[ix["run host ID"]]]
rid = max(int(r[ix["run host ID"]]) for r in rows if r[ix["zone name"]] in zones)
rows = [r for r in rows if int(r[ix["run host ID"]]) == rid]
t0 = min(int(r[ix["time[cycles since reset]"]]) for r in rows if r[ix["zone name"]].endswith("-KERNEL"))
ev = collections.defaultdict(list)
for r in rows:
    ev[(r[ix["zone name"]], int(r[ix["core_x"]]), int(r[ix["core_y"]]))].append(
        ((int(r[ix["time[cycles since reset]"]]) - t0) / CLK, r[ix["type"]])
    )
for z in zones:
    xs = sorted({k[1] for k in ev if k[0] == z})
    ys = sorted({k[2] for k in ev if k[0] == z})
    print(f"== {z}: END us (start in parens) ==")
    print("     " + "".join(f"{x:>12}" for x in xs))
    for y in ys:
        cells = []
        for x in xs:
            v = sorted(ev.get((z, x, y), []))
            s = [t for t, ty in v if ty == "ZONE_START"]
            e = [t for t, ty in v if ty == "ZONE_END"]
            cells.append(f"{e[-1]:6.1f}({s[0]:4.1f})" if e else " " * 12)
        print(f"y{y:<3} " + "".join(f"{c:>12}" for c in cells))
