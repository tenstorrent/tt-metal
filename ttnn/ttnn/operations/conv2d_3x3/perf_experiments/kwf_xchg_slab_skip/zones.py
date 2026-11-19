# usage: zones.py <profile_log_device.csv> <run_index (1-based among runs with >= 8 cores)> <x,y> [risc...]
# Every zone (nested included) of one core, start/end in us from the run's earliest kernel start.
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
want = tuple(int(v) for v in sys.argv[3].split(","))
riscs = sys.argv[4:] or ["BRISC", "NCRISC", "TRISC_0"]
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
t0 = min(e[2] for e in ev if e[3].endswith("-KERNEL") and e[4] == "ZONE_START")
stack = collections.defaultdict(list)
out = collections.defaultdict(list)
for core, risc, t, z, kind in ev:
    if core != want or risc not in riscs:
        continue
    if kind == "ZONE_START":
        stack[(risc, z)].append(t)
    elif stack[(risc, z)]:
        s = stack[(risc, z)].pop()
        out[risc].append(((s - t0) / freq, (t - t0) / freq, z))
for risc in riscs:
    print(
        risc
        + ": "
        + "  ".join(
            f"{z.replace('reader_','r.').replace('writer_','w.').replace('compute_','c.')}[{a:.2f}-{b:.2f}]"
            for a, b, z in sorted(out[risc])
            if not z.endswith("-FW")
        )
    )
