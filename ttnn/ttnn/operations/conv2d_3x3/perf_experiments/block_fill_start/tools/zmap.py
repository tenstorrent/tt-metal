# usage: zmap.py <csv> <run idx> <zone> [risc] [start|end|dur]: a phys-coordinate grid map of one zone per core (us)
import collections, csv, sys

path, idx, zone = sys.argv[1], int(sys.argv[2]), sys.argv[3]
want_risc = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
what = sys.argv[5] if len(sys.argv) > 5 else "end"
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
t0 = min(t for c, r, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
us = lambda c: (c - t0) / freq
first, last, dur = {}, {}, collections.defaultdict(float)
stack = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if z != zone or (want_risc and risc != want_risc):
        continue
    if kind == "ZONE_START":
        stack[core].append(t)
        first[core] = min(first.get(core, t), t)
    elif stack[core]:
        s = stack[core].pop()
        last[core] = max(last.get(core, t), t)
        dur[core] += (t - s) / freq
xs = sorted({c[0] for c in last})
ys = sorted({c[1] for c in last})
print(f"{zone} {what} (us), run {big[idx-1]}")
print("  y\\x " + "".join(f"{x:>6}" for x in xs))
for y in ys:
    vals = []
    for x in xs:
        c = (x, y)
        if c not in last:
            vals.append("     .")
        else:
            v = us(last[c]) if what == "end" else us(first[c]) if what == "start" else dur[c]
            vals.append(f"{v:6.2f}")
    print(f"  {y:>3} " + "".join(vals))
