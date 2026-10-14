"""Per-physical-core map of the last run: BRISC-KERNEL end (ns from cluster start) and a zone sum.
usage: python3 grid.py [zone=writer_flush]"""
import collections, csv, sys

zone_sel = sys.argv[1] if len(sys.argv) > 1 else "writer_flush"
path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/geo_write_noc_prof/.logs/profile_log_device.csv"
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))[1:]
run = max(int(r[7]) for r in rows)
rows = [r for r in rows if int(r[7]) == run]
op, kend, kstart, zs = {}, {}, {}, collections.defaultdict(int)
for r in rows:
    core, risc, t, zone, typ = (int(r[1]), int(r[2])), r[3], int(r[5]), r[10], r[11]
    k = (core, risc, zone)
    if typ == "ZONE_START":
        op[k] = t
    elif typ == "ZONE_END" and k in op:
        d = t - op.pop(k)
        if zone == f"{risc}-KERNEL":
            kend.setdefault(core, {})[risc] = t
            kstart.setdefault(core, {})[risc] = t - d
        if zone == zone_sel:
            zs[core] += d
ns = lambda c: c * 1000.0 / freq
# timer clusters differ per core group: use per-core start of earliest RISC as reference within clusters
cores = sorted(kend)
starts = {c: min(kstart[c].values()) for c in cores}
# cluster by start
GAP = 1_000_000
order = sorted(cores, key=lambda c: starts[c])
cl, ref = {}, {}
cur = None
for c in order:
    if cur is None or starts[c] - cur[-1] > GAP:
        cur = [starts[c]]
    else:
        cur.append(starts[c])
    cl[c] = cur
for c in cores:
    ref[c] = cl[c][0]
xs = sorted({c[0] for c in cores})
ys = sorted({c[1] for c in cores})
for title, val in (
    ("end(max risc) us", lambda c: ns(max(kend[c].values()) - ref[c]) / 1000),
    (f"{zone_sel} us", lambda c: ns(zs[c]) / 1000),
):
    print(title)
    print("   y\\x " + " ".join(f"{x:5d}" for x in xs))
    for y in ys:
        print(f"   {y:3d} " + " ".join(f"{val((x, y)):5.1f}" if (x, y) in kend else "    ." for x in xs))
