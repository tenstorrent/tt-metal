# usage: fwstart.py <dev csv[.gz]> <run idx>: per core BRISC-FW start (launch seen), kernel start (go), in logical
# row-major order (virtual x 1..6 -> 0..5, 11..15 -> 6..10; y 2..11 -> 0..9). us from the run's earliest kernel start.
import collections, csv, gzip, sys

path, idx = sys.argv[1], int(sys.argv[2])
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
st = {}
for core, risc, t, z, kind in runs[big[idx - 1]]:
    if risc == "BRISC" and kind == "ZONE_START" and z in ("BRISC-FW", "BRISC-KERNEL"):
        st.setdefault((core, z), t)
t0 = min(v for (c, z), v in st.items() if z == "BRISC-KERNEL")
lx = lambda x: x - 1 if x <= 6 else x - 5
out = []
for (c, z), v in st.items():
    if z == "BRISC-FW":
        out.append(((c[1] - 2) * 11 + lx(c[0]), c, (v - t0) / freq, (st.get((c, "BRISC-KERNEL"), t0) - t0) / freq))
out.sort()
print(" ".join(f"{i}:{fw:.2f}/{k:.2f}" for i, c, fw, k in out))
