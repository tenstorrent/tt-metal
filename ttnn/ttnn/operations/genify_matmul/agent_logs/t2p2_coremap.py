import csv, collections, sys

f, zone = sys.argv[1], sys.argv[2]
rows = list(csv.reader(open(f)))[2:]
by = collections.defaultdict(list)
for r in rows:
    by[r[7].strip()].append(r)
R = by[max(by, key=lambda k: len(by[k]))]
t0 = min(int(r[5]) for r in R)
last = collections.defaultdict(int)
for r in R:
    if r[10].strip() == zone and r[11].strip() == "ZONE_END":
        k = (int(r[1]), int(r[2]))
        last[k] = max(last[k], int(r[5]) - t0)
grid = collections.defaultdict(dict)
for (x, y), t in last.items():
    grid[y][x] = t / 1350
xs = sorted({x for x, y in last})
print(zone, "last end us; y\\x " + " ".join(f"{x:5d}" for x in xs))
for y in sorted(grid):
    print(f"{y:3d} " + " ".join(f"{grid[y].get(x,0):5.1f}" for x in xs))
