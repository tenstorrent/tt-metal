# usage: weight_slack.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)>
# Per compute core: when each weight round landed (end of the writer's k-th writer_weight_fetch) vs when compute's
# unpacker was ready for it (end of the k-th compute_tilize on TRISC_0, after which the matmul waits cb_weights).
# A positive wait means the matmul of that K-block started late because of its weights. Also prints the kernel end.
import collections, csv, gzip, sys

path, idx = sys.argv[1], int(sys.argv[2])
with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
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
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL"))
us = lambda c: (c - t0) / freq
zones = collections.defaultdict(list)
stack = collections.defaultdict(list)
kend = {}
for core, risc, t, z, kind in ev:
    if z == "TRISC-KERNEL" and kind != "ZONE_START":
        kend[core] = max(kend.get(core, 0), us(t))
    if kind == "ZONE_START":
        stack[(core, risc, z)].append(t)
    elif stack[(core, risc, z)]:
        s = stack[(core, risc, z)].pop()
        zones[(core, risc, z)].append((us(s), us(t)))
waits = collections.defaultdict(list)
for core in sorted(kend):
    fetch = sorted(zones.get((core, "BRISC", "writer_weight_fetch"), []))
    til = sorted(zones.get((core, "TRISC_0", "compute_tilize"), []))
    if not fetch:
        continue
    w = [round(f[1] - t[1], 2) for f, t in zip(fetch, til)]
    for k, v in enumerate(w):
        waits[k].append(v)
    print(
        core,
        "end %.2f" % kend[core],
        "land",
        [round(f[1], 1) for f in fetch],
        "tilize_end",
        [round(t[1], 1) for t in til],
        "wait",
        w,
    )
kw = collections.defaultdict(float)
for (core, risc, z), v in zones.items():
    if z == "compute_kwf_weight_wait" and risc == "TRISC_0":
        kw[core] = sum(b - a for a, b in v)
if kw:
    v = sorted(kw.values())
    print(f"kw-fold weight wait (TRISC_0, per-core sum): p50 {v[len(v)//2]:.2f} max {v[-1]:.2f}")
ends = sorted(kend.values())
print(f"compute end: max {ends[-1]:.2f} p50 {ends[len(ends)//2]:.2f}")
for k in sorted(waits):
    v = sorted(waits[k])
    print(
        f"K-block {k}: wait p50 {v[len(v)//2]:.2f} max {v[-1]:.2f}  cores waiting >0.1us: {sum(x > 0.1 for x in v)}/{len(v)}"
    )
