import csv, collections, sys

path = sys.argv[1]
zones = sys.argv[2].split(",")
with open(path) as fh:
    freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
ns = lambda c: c * 1000.0 / freq
st = {}
d = collections.defaultdict(float)
end = collections.defaultdict(int)
t0 = None
for r in rows:
    k = (r["core_x"], r["core_y"], r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        st[k] = t
        if r["zone name"].endswith("-FW"):
            t0 = t if t0 is None else min(t0, t)
    elif r["type"] == "ZONE_END" and k in st:
        d[k] += ns(t - st.pop(k))
        end[k] = max(end[k], t)
cores = sorted({(k[0], k[1]) for k in d}, key=lambda c: (int(c[1]), int(c[0])))
hdr = ["core"] + zones + ["NC_end", "T0_end", "BR_end"]
print(" ".join(f"{h[:14]:>14s}" for h in hdr))
out = []
for c in cores:
    vals = []
    for z in zones:
        want = "NCRISC" if z.startswith("reader") else ("TRISC_0" if z.startswith("compute") else "BRISC")
        vals.append(d.get((c[0], c[1], want, z), 0.0))
    e = [
        ns(
            end.get(
                (c[0], c[1], rk, rk.split("_")[0] + "-KERNEL" if not rk.startswith("TRISC") else "TRISC-KERNEL"), t0
            )
            - t0
        )
        for rk in ("NCRISC", "TRISC_0", "BRISC")
    ]
    out.append((c, vals, e))
for c, vals, e in sorted(out, key=lambda o: -o[2][2]):
    print(f"{c[0]+','+c[1]:>14s} " + " ".join(f"{v:14.0f}" for v in vals + e))
