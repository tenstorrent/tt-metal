# usage: crit.py <profile_log_device.csv> <case index (1-based)> [zone,...]
# Per core: kernel end (max over RISCs, us from the run's first kernel start); prints the 5 latest cores with the given
# zones' instances, and the server cores' read/send zones.
import collections, csv, sys

path, idx = sys.argv[1], int(sys.argv[2])
zones = (
    tuple(sys.argv[3].split(","))
    if len(sys.argv) > 3
    else ("compute_kwf_weight_wait", "compute_kwf_final", "writer_weight_fetch")
)
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
end = collections.defaultdict(float)
st, out = {}, collections.defaultdict(list)
for c, risc, t, z, k in ev:
    if z.endswith("-KERNEL") and k == "ZONE_END":
        end[c] = max(end[c], (t - t0) / freq)
    if k == "ZONE_START":
        st[(c, risc, z)] = t
    elif (c, risc, z) in st:
        out[(c, z)].append(((st.pop((c, risc, z)) - t0) / freq, (t - t0) / freq))
print(f"span {max(end.values()):.2f} us over {len(end)} cores")
for c in sorted(end, key=lambda c: -end[c])[:4]:
    print(f"  core {c} end {end[c]:.2f}")
    for z in zones:
        if (c, z) in out:
            print("     ", z, " ".join(f"[{a:.2f}-{b:.2f}]" for a, b in out[(c, z)]))
for (c, z), v in sorted(out.items()):
    if (
        z.startswith("server_read")
        or z.startswith("server_send")
        or z.startswith("server_flush")
        or z.startswith("hsr_")
    ):
        print(f"  server {c} {z}: " + " ".join(f"[{a:.2f}-{b:.2f}]" for a, b in v))
