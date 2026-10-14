"""Per-core grid maps (profiler x,y) for the last run of a profile_log_device.csv.

Prints, in ns: kernel start skew, BRISC end (op end per core), sum of reader_read_barrier,
and per-zone sums named on the command line (default: reader_read_barrier writer_cb_wait).
Usage: python3 percore.py [csv] [--run N] [--zones a,b]
"""
import collections, csv, sys

args = sys.argv[1:]
run_sel = None
zones = ["reader_read_barrier", "writer_cb_wait"]
if "--run" in args:
    i = args.index("--run")
    run_sel = int(args[i + 1])
    del args[i : i + 2]
if "--zones" in args:
    i = args.index("--zones")
    zones = args[i + 1].split(",")
    del args[i : i + 2]
path = args[0] if args else "generated/profiler/.logs/profile_log_device.csv"
with open(path) as f:
    first = f.readline()
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
runs = sorted({int(r[7]) for r in rows})
run = runs[-1] if run_sel is None else run_sel
rows = [r for r in rows if int(r[7]) == run]
open_z = {}
zsum = collections.defaultdict(lambda: collections.defaultdict(int))
zends = collections.defaultdict(lambda: collections.defaultdict(list))
kern = {}
for r in rows:
    core = (int(r[1]), int(r[2]))
    risc = r[3]
    t = int(r[5])
    zone = r[10]
    typ = r[11]
    key = (core, risc, zone)
    if typ == "ZONE_START":
        open_z[key] = t
    elif typ == "ZONE_END" and key in open_z:
        s = open_z.pop(key)
        if zone.endswith("-KERNEL"):
            kern[(core, risc)] = (s, t)
        zsum[zone][core] += t - s
        zends[zone][core].append((s, t))
ns = lambda c: c * 1000.0 / freq
# largest timer cluster only (cores share a zero point within it)
starts = sorted(v[0] for v in kern.values())
cl = [[starts[0]]]
for s in starts[1:]:
    (cl[-1].append(s) if s - cl[-1][-1] < 1_000_000 else cl.append([s]))
xs = sorted({c[0] for (c, _) in kern})
ys = sorted({c[1] for (c, _) in kern})


def grid(title, f):
    print(title)
    print("     " + "".join(f"{x:>6d}" for x in xs))
    for y in ys:
        line = f"y{y:>3d} "
        for x in xs:
            v = f((x, y))
            line += f"{v:>6.0f}" if v is not None else "     ."
        print(line)


corecl = {}
for (c, risc), (s, e) in kern.items():
    for i, cc in enumerate(cl):
        if cc[0] <= s <= cc[-1]:
            corecl[c] = i
t0 = {i: min(s for (c, r), (s, e) in kern.items() if corecl.get(c) == i) for i in range(len(cl))}


def endf(risc):
    return lambda c: ns(kern[(c, risc)][1] - t0[corecl[c]]) if (c, risc) in kern else None


grid(f"run {run}: clusters {len(cl)}; BRISC end (ns from cluster start)", endf("BRISC"))
grid("NCRISC start skew", lambda c: ns(kern[(c, "NCRISC")][0] - t0[corecl[c]]) if (c, "NCRISC") in kern else None)
for z in zones:
    if z in zsum:
        grid(f"sum {z}", lambda c, z=z: ns(zsum[z][c]) if c in zsum[z] else None)
for z in zones:
    if z in zends:
        n = max(len(v) for v in zends[z].values())
        for i in range(min(n, 3)):
            grid(
                f"{z} occurrence {i}: END (ns from cluster start)",
                lambda c, z=z, i=i: ns(zends[z][c][i][1] - t0[corecl[c]])
                if c in zends[z] and len(zends[z][c]) > i
                else None,
            )
