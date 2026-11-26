"""reader_issue: from a trace-replay device profile, per replay instance (default last): span per chip,
and the worker reader (NCRISC) end per core = last reader_* zone end, by grid row / x-half.
Usage: python3 ri_reader_ends.py <csv> [--replay i] [--all]"""
import csv, sys, bisect
from collections import defaultdict

FREQ = 1350.0
args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = args[0]
rows = list(csv.reader(open(path)))[2:]
kst = defaultdict(list)
for r in rows:
    if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START":
        kst[r[0]].append(int(r[5]))
inst = {}
for chip, ts in kst.items():
    ts.sort()
    starts = [ts[0]]
    for a, b in zip(ts, ts[1:]):
        if b - a > 5 * FREQ:
            starts.append(b)
    inst[chip] = starts
ninst = min(len(v) for v in inst.values())


def report(which):
    kend = defaultdict(int)
    nend, readers = {}, set()
    for r in rows:
        chip, x, y, risc, _, t = r[:6]
        t = int(t)
        i = bisect.bisect_right(inst[chip], t) - 1
        if i != which:
            continue
        zone, typ = r[10].strip(), r[11].strip()
        if zone.endswith("-KERNEL") and typ == "ZONE_END":
            kend[chip] = max(kend[chip], t)
        if risc.strip() == "NCRISC" and zone.startswith("reader_"):
            readers.add((chip, int(x), int(y)))
        if risc.strip() == "NCRISC" and zone == "NCRISC-KERNEL" and typ == "ZONE_END":
            k = (chip, int(x), int(y))
            nend[k] = max(nend.get(k, 0), t - inst[chip][which])
    rend = {k: v for k, v in nend.items() if k in readers}
    chips = sorted(inst, key=int)
    spans = [(kend[c] - inst[c][which]) / FREQ for c in chips]
    ends = [v / FREQ for v in rend.values()]
    print(
        f"replay {which}: span max {max(spans):.1f} mean {sum(spans)/len(spans):.1f} | reader end max {max(ends):.1f} "
        f"mean {sum(ends)/len(ends):.1f} min {min(ends):.1f} (cores {len(ends)})"
    )
    return rend


which = int(sys.argv[sys.argv.index("--replay") + 1]) if "--replay" in sys.argv else ninst - 1
if "--all" in sys.argv:
    for i in range(1, ninst):
        report(i)
rend = report(which)
byrow = defaultdict(list)
for (chip, x, y), v in rend.items():
    byrow[y].append(v / FREQ)
print(
    "reader end by physical y (mean/max):",
    " ".join(f"y{y}:{sum(v)/len(v):.0f}/{max(v):.0f}" for y, v in sorted(byrow.items())),
)
byx = defaultdict(list)
for (chip, x, y), v in rend.items():
    byx[x].append(v / FREQ)
print(
    "reader end by physical x (mean/max):",
    " ".join(f"x{x}:{sum(v)/len(v):.0f}/{max(v):.0f}" for x, v in sorted(byx.items())),
)
loc = [v / FREQ for (c, x, y), v in rend.items() if y in (2, 3)]
rem = [v / FREQ for (c, x, y), v in rend.items() if y not in (2, 3)]
print(
    f"reader end local-head rows (y2-3) max {max(loc):.1f} mean {sum(loc)/len(loc):.1f} | remote rows max {max(rem):.1f} mean {sum(rem)/len(rem):.1f}"
)
