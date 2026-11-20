# usage: percore.py <profile_log_device.csv[.gz]> <run_index (1-based among runs with >= 8 cores)> [top]
# Per compute core (sorted by TRISC kernel end): TRISC_0 tilize end, walk [start-end], every writer_weight_fetch end
# (round landed), writer_weight_reserve spans; plus the servers' zones. Times in us after the run's earliest kernel start.
import collections, csv, gzip, sys

path, idx = sys.argv[1], int(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 12
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
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
us = lambda c: (c - t0) / freq
zones = collections.defaultdict(list)
stack = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if kind == "ZONE_START":
        stack[(core, risc, z)].append(t)
    elif kind == "ZONE_END" and stack[(core, risc, z)]:
        s = stack[(core, risc, z)].pop()
        zones[(core, risc, z)].append((us(s), us(t)))
cores = sorted({k[0] for k in zones})
info = []
for c in cores:
    tk = zones.get((c, "TRISC_0", "TRISC-KERNEL"), [])
    walk = zones.get((c, "TRISC_0", "compute_matmul_walk"), [])
    if not walk:
        continue
    til = zones.get((c, "TRISC_0", "compute_tilize"), [])
    fetch = sorted(zones.get((c, "BRISC", "writer_weight_fetch"), []))
    res = sorted(zones.get((c, "BRISC", "writer_weight_reserve"), []))
    end = max(e for _, e in zones.get((c, "TRISC_2", "TRISC-KERNEL"), tk))
    info.append(
        (
            end,
            c,
            max(e for _, e in til) if til else 0,
            walk[0],
            [round(e, 1) for _, e in fetch],
            [(round(s, 1), round(e, 1)) for s, e in res if e - s > 0.2],
        )
    )
info.sort(reverse=True)
print("cores", len(info), "end max %.2f p50 %.2f" % (info[0][0], info[len(info) // 2][0]))
for end, c, te, w, f, r in info[:top]:
    print(
        c,
        "end %.2f" % end,
        "tilize_end %.2f" % te,
        "walk %.2f-%.2f (%.2f)" % (w[0], w[1], w[1] - w[0]),
        "land",
        f,
        "reserve",
        r,
    )
# distribution: walk start, round landing per round
import statistics as st

for name, vals in (
    ("tilize_end", [i[2] for i in info]),
    ("walk_start", [i[3][0] for i in info]),
    ("walk_len", [i[3][1] - i[3][0] for i in info]),
):
    vals = sorted(vals)
    print(name, "min %.2f p50 %.2f max %.2f" % (vals[0], vals[len(vals) // 2], vals[-1]))
nr = max(len(i[4]) for i in info)
for k in range(nr):
    v = sorted(i[4][k] for i in info if len(i[4]) > k)
    print("land round", k, "min %.1f p50 %.1f max %.1f" % (v[0], v[len(v) // 2], v[-1]))
for c in cores:
    zs = {k[2]: v for k, v in zones.items() if k[0] == c and k[2].startswith("server")}
    if zs:
        print(
            "server",
            c,
            {z: [(round(s, 1), round(e, 1)) for s, e in v] for z, v in sorted(zs.items()) if z != "server_wait_read"},
        )
