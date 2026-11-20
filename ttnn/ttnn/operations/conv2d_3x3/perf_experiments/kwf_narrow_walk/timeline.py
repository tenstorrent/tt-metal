# usage: timeline.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)> [core_x,core_y]
# Prints, for one core (default: the core whose kernel ends last), each zone's start/end in us relative to the
# earliest kernel start of the run (all cores), per RISC, plus every core's kernel start/end spread.
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
want = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else None
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = csv.reader(f)
    cols = [c.strip() for c in next(rows)]
    ix = {c: i for i, c in enumerate(cols)}
    runs = collections.defaultdict(list)
    for r in rows:
        if len(r) < len(cols):
            continue
        runs[int(r[ix["run host ID"]])].append(
            (
                (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                r[ix["RISC processor type"]].strip(),
                int(r[ix["time[cycles since reset]"]]),
                r[ix["zone name"]].strip(),
                r[ix["type"]].strip(),
            )
        )
big = [k for k in sorted(runs) if len({e[0] for e in runs[k]}) >= 8]
ev = sorted(runs[big[idx - 1]], key=lambda e: e[2])
kst, ken = {}, {}
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL"):
        if kind == "ZONE_START":
            kst[(core, risc)] = min(kst.get((core, risc), t), t)
        else:
            ken[(core, risc)] = max(ken.get((core, risc), t), t)
t0 = min(kst.values())
us = lambda c: (c - t0) / freq
starts = sorted(us(v) for v in kst.values())
ends = sorted(us(v) for v in ken.values())
print(
    f"run {big[idx-1]}: kernel start spread {starts[0]:.2f}..{starts[-1]:.2f} us, end spread {ends[0]:.2f}..{ends[-1]:.2f} us"
)
if want is None:
    want = max(ken, key=lambda k: ken[k])[0]
print(f"core {want}")
stack = collections.defaultdict(list)
out = []
for core, risc, t, z, kind in ev:
    if core != want:
        continue
    if kind == "ZONE_START":
        stack[(risc, z)].append(t)
    elif stack[(risc, z)]:
        s = stack[(risc, z)].pop()
        out.append((risc, us(s), us(t), z))
for risc in ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"):
    items = sorted([o for o in out if o[0] == risc], key=lambda o: o[1])
    print(
        f"  {risc}: "
        + "  ".join(
            f"{z.replace('compute_','c.').replace('reader_','r.').replace('writer_','w.')}[{a:.1f}-{b:.1f}]"
            for _, a, b, z in items
            if b - a >= 0.0
        )
    )
