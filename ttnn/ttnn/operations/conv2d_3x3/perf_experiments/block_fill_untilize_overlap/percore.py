# usage: percore.py <profile_log_device.csv> <run index (1-based among runs with >= 8 cores)> zone[:RISC][,zone...]
# Per core (sorted by y, x): each named zone's first start / last end (us from the run's earliest kernel start), its
# executions, and the KERNEL end. Plus min / p50 / max over cores of each zone's span.
import collections, csv, statistics, sys

path, idx = sys.argv[1], int(sys.argv[2])
EACH = "--each" in sys.argv
want = [w.split(":") for w in sys.argv[3].split(",")]
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
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL"))
us = lambda c: (c - t0) / freq
spans = collections.defaultdict(list)  # (core, risc, zone) -> [(s, e)]
stack = collections.defaultdict(list)
kend = {}
for core, risc, t, z, kind in ev:
    if z.endswith("-KERNEL") and kind != "ZONE_START":
        kend[core] = max(kend.get(core, 0), us(t))
    if kind == "ZONE_START":
        stack[(core, risc, z)].append(t)
    elif stack[(core, risc, z)]:
        spans[(core, risc, z)].append((us(stack[(core, risc, z)].pop()), us(t)))
cores = sorted({e[0] for e in ev}, key=lambda c: (c[1], c[0]))
agg = collections.defaultdict(list)
for core in cores:
    parts = []
    for w in want:
        z = w[0]
        rs = [w[1]] if len(w) > 1 else ["NCRISC", "BRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
        for risc in rs:
            sp = spans.get((core, risc, z))
            if sp and EACH:
                parts.append(z + "".join(f"[{a:.1f}-{b:.1f}]" for a, b in sp))
                agg[z].append(max(e for _, e in sp) - min(s for s, _ in sp))
                break
            if sp:
                parts.append(f"{z}[{min(s for s, _ in sp):.1f}-{max(e for _, e in sp):.1f} x{len(sp)}]")
                agg[z].append(max(e for _, e in sp) - min(s for s, _ in sp))
                break
    print(f"{core} end {kend.get(core, 0):.1f}  " + "  ".join(parts))
for z, v in agg.items():
    print(f"{z}: span min {min(v):.2f} p50 {statistics.median(v):.2f} max {max(v):.2f} (n {len(v)})")
