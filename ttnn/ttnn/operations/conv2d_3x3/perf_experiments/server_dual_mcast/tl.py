# usage: tl.py <profile_log_device.csv[.gz]> <run_index (1-based among runs with >= 8 cores)> [mode]
#   mode: "servers" (every server core's zones), "core:x,y", "dist:<zone>:<risc>" (per-core first start / first end
#   distribution of one zone), "crit" (the last-ending core). Times in us from the run's earliest kernel start.
import collections, csv, gzip, sys

path, idx = sys.argv[1], int(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 else "crit"
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
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
zones = collections.defaultdict(list)  # core -> (risc, s, e, z)
stack = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if kind == "ZONE_START":
        stack[(core, risc, z)].append(t)
    elif stack[(core, risc, z)]:
        s = stack[(core, risc, z)].pop()
        zones[core].append((risc, us(s), us(t), z))
print(f"run {big[idx-1]}: {len({c for c, _ in kst})} cores, end max {max(us(v) for v in ken.values()):.2f}")


def dump(core):
    print(f"core {core}")
    for risc in ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"):
        items = sorted([o for o in zones[core] if o[0] == risc], key=lambda o: o[1])
        print(
            f"  {risc}: "
            + "  ".join(
                f"{z.replace('compute_','c.').replace('writer_','w.').replace('reader_','r.')}[{s:.2f}-{e:.2f}]"
                for _, s, e, z in items
            )
        )


if mode == "servers":
    for core in sorted(zones):
        if any(z.startswith("server_") for _, _, _, z in zones[core]):
            dump(core)
elif mode.startswith("core:"):
    dump(tuple(int(v) for v in mode[5:].split(",")))
elif mode.startswith("dist:"):
    _, zname, risc = mode.split(":")
    firsts = []
    for core, items in zones.items():
        it = sorted([o for o in items if o[3] == zname and o[0] == risc], key=lambda o: o[1])
        if it:
            firsts.append((it[0][1], it[0][2], core))
    firsts.sort(key=lambda v: v[1])
    n = len(firsts)
    for q in (0, 0.1, 0.5, 0.9, 1.0):
        s, e, c = firsts[min(n - 1, int(q * n))]
        print(f"  q{q:.1f} first {zname}: start {s:.2f} end {e:.2f} core {c}")
else:
    dump(max(ken, key=lambda k: ken[k])[0])
