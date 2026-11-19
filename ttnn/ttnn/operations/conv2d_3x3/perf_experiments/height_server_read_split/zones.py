# usage: zones.py <profile_log_device.csv> <case index (1-based)> <zone prefix>[,<prefix>...]
# Every zone instance whose name starts with one of the prefixes, per (core, zone), in us from the run's first kernel.
import collections, csv, sys

path, idx, prefixes = sys.argv[1], int(sys.argv[2]), tuple(sys.argv[3].split(","))
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
st, out = {}, collections.defaultdict(list)
for c, risc, t, z, k in ev:
    if not z.startswith(prefixes):
        continue
    if k == "ZONE_START":
        st[(c, risc, z)] = t
    elif (c, risc, z) in st:
        out[(c, z)].append(((st.pop((c, risc, z)) - t0) / freq, (t - t0) / freq))
for k, v in sorted(out.items()):
    print(k, len(v), " ".join(f"[{a:.2f}-{b:.2f}]" for a, b in v))
