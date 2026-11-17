"""Per-core grid of the LAST run in a device log: end time (us from the earliest kernel start) of the last
data-movement kernel (max of BRISC-KERNEL / NCRISC-KERNEL end), and the write span start = cmp_matmul end.
Usage: python3 coremap.py profile_log_device.csv   (host-side CSV parsing only)"""
import csv, sys
from collections import defaultdict

f = open(sys.argv[1])
hdr = f.readline()
freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rows = list(csv.reader(f))[1:]
# keep only the last run host id
last = max(int(r[7]) for r in rows if r[7].strip().isdigit())
st, en = {}, defaultdict(dict)
for r in rows:
    if not r[7].strip().isdigit() or int(r[7]) != last:
        continue
    name, typ, t = r[10].strip(), r[11].strip(), int(r[5])
    core = (int(r[1]), int(r[2]))
    if name in ("BRISC-KERNEL", "NCRISC-KERNEL", "cmp_matmul") and typ == "ZONE_END":
        en[core][name] = max(en[core].get(name, 0), t)
    if name == "BRISC-KERNEL" and typ == "ZONE_START":
        st[core] = t
t0 = min(st.values())
xs = sorted({c[0] for c in en})
ys = sorted({c[1] for c in en})
print("run", last, "end of DM (us) per core; rows = y, cols = x", xs)
for y in ys:
    line = []
    for x in xs:
        d = en.get((x, y))
        if not d:
            line.append("   .  ")
            continue
        e = max(d.get("BRISC-KERNEL", 0), d.get("NCRISC-KERNEL", 0))
        line.append(f"{(e - t0) / freq:6.1f}")
    print(f"y={y:2d} " + " ".join(line))
ends = sorted(max(d.get("BRISC-KERNEL", 0), d.get("NCRISC-KERNEL", 0)) for d in en.values())
mm = sorted(d.get("cmp_matmul", 0) for d in en.values() if d.get("cmp_matmul"))
print(
    f"matmul end p50 {(mm[len(mm)//2]-t0)/freq:.1f}; DM end p10 {(ends[len(ends)//10]-t0)/freq:.1f} p50 {(ends[len(ends)//2]-t0)/freq:.1f} p90 {(ends[9*len(ends)//10]-t0)/freq:.1f} max {(ends[-1]-t0)/freq:.1f}"
)
