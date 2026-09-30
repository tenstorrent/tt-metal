import csv, sys
from collections import defaultdict

path, run = sys.argv[1], sys.argv[2]
MHZ = 1350.0
k0 = {}
k1 = {}
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for row in r:
        row = [x.strip() for x in row]
        if row[7] != run:
            continue
        c = (int(row[1]), int(row[2]))
        z = row[10]
        t = int(row[5])
        if z.endswith("-KERNEL"):
            if row[11] == "ZONE_START":
                k0[c] = min(k0.get(c, t), t)
            else:
                k1[c] = max(k1.get(c, t), t)
t0 = min(k0.values())
xs = sorted({c[0] for c in k1})
ys = sorted({c[1] for c in k1})
print("y\\x " + " ".join(f"{x:4d}" for x in xs))
for y in ys:
    print(f"{y:3d} " + " ".join(f"{(k1[(x,y)]-t0)/MHZ:4.0f}" if (x, y) in k1 else "   ." for x in xs))
