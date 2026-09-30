"""Per-core zone timeline (all zones, optional window / RISC filter).
usage: python tl.py profile_log_device.csv <run_host_id> <x> <y> [t0_us t1_us] [min_us]"""
import csv, sys

path, run, cx, cy = sys.argv[1:5]
lo = float(sys.argv[5]) if len(sys.argv) > 5 else -1
hi = float(sys.argv[6]) if len(sys.argv) > 6 else 1e9
mn = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
MHZ = 1350.0
ev = []
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for row in r:
        row = [x.strip() for x in row]
        if row[7] != run or row[1] != cx or row[2] != cy:
            continue
        ev.append((int(row[5]), row[3], row[10], row[11]))
ev.sort()
t0 = ev[0][0]
st = {}
out = []
for t, risc, z, ty in ev:
    if ty == "ZONE_START":
        st[(risc, z)] = t
    elif (risc, z) in st:
        s = st.pop((risc, z))
        a, b = (s - t0) / MHZ, (t - t0) / MHZ
        if b >= lo and a <= hi and b - a >= mn:
            out.append((a, b, risc, z))
for a, b, risc, z in sorted(out):
    print(f"{a:7.1f} {b:7.1f} {b-a:6.1f} {risc:7s} {z}")
