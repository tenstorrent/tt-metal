# usage: zone_dump.py <device csv> <run_index> <x,y> <zone-substring> [risc]: start/end (us from the run's first kernel
# start) of every instance of the matching zones on one core, in order.
import collections, csv, sys

path, idx, core, sub = sys.argv[1], int(sys.argv[2]), tuple(int(v) for v in sys.argv[3].split(",")), sys.argv[4]
risc_f = sys.argv[5] if len(sys.argv) > 5 else None
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
t0 = min(t for c, rk, t, z, k in ev if z.endswith("-KERNEL") and k == "ZONE_START")
stack = collections.defaultdict(list)
out = []
for c, rk, t, z, k in ev:
    if c != core or sub not in z or (risc_f and rk != risc_f):
        continue
    if k == "ZONE_START":
        stack[(rk, z)].append(t)
    elif stack[(rk, z)]:
        s = stack[(rk, z)].pop()
        out.append((s, t, z, rk))
for s, t, z, rk in sorted(out):
    print(f"{z:<24} {rk:<8} {(s - t0) / freq:7.3f} {(t - t0) / freq:7.3f}")
