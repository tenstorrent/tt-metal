import csv, sys

path, run, cx, cy = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
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
for t, risc, z, ty in ev:
    if risc not in ("NCRISC", "BRISC", "TRISC_0", "TRISC_1"):
        continue
    if ty == "ZONE_START":
        st[(risc, z)] = t
    elif (risc, z) in st:
        s = st.pop((risc, z))
        d = (t - s) / MHZ
        if d > 0.8 or z.endswith("KERNEL"):
            print(f"{(s-t0)/MHZ:7.1f} {(t-t0)/MHZ:7.1f} {d:6.1f} {risc:7s} {z}")
