"""List cores by walk end for one run (no device): python3 late_cores.py <dev csv> <run> [threshold_us]"""
import sys, io, contextlib
import timeline

freq, runs = timeline.load(sys.argv[1])
big = [r for r in runs if len({c for c, *_ in r}) >= 60]
with contextlib.redirect_stdout(io.StringIO()):
    info = timeline.analyze(big[int(sys.argv[2])], freq)
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0
grid = {}
for c, d in info.items():
    if d["walk"]:
        grid[c] = d["walk"][1]
xs = sorted({c[0] for c in grid})
ys = sorted({c[1] for c in grid})
print("walk end (us) by core; rows = vy, cols = vx")
print("     " + " ".join(f"{x:5d}" for x in xs))
for y in ys:
    print(f"{y:3d}  " + " ".join(f"{grid.get((x, y), 0):5.1f}" for x in xs))
print("walk start")
for y in ys:
    print(
        f"{y:3d}  "
        + " ".join(f"{info[(x, y)]['walk'][0] if (x, y) in info and info[(x, y)]['walk'] else 0:5.1f}" for x in xs)
    )
print("weight wait sum")
for y in ys:
    print(f"{y:3d}  " + " ".join(f"{sum(info[(x, y)]['wwait']) if (x, y) in info else 0:5.1f}" for x in xs))
