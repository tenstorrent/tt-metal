import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/distributed_rms_norm")
from perf_zones import _parse, CLOCK_GHZ

z = _parse(sys.argv[1])
chip = int(sys.argv[2])
name = sys.argv[3]
k = int(sys.argv[4])
runs = [
    (int(r) if r.isdigit() else 0, ev)
    for (c, r), ev in z.items()
    if c == chip and any(n.startswith("rt_") for _, _, n, _, _ in ev)
]
run, ev = max(runs, key=lambda x: x[0])
t0 = min(s for _, r, n, s, _ in ev if n == "BRISC-KERNEL")
per = {}
for c, r, n, s, e in ev:
    if n == name:
        per.setdefault(c, []).append(s)
ys = sorted({c[1] for c in per})
xs = sorted({c[0] for c in per})
print("y\\x " + " ".join(f"{x:5d}" for x in xs))
for y in ys:
    print(
        f"{y:3d} "
        + " ".join(
            f"{(sorted(per[(x,y)])[k]-t0)/CLOCK_GHZ/1000:5.1f}" if (x, y) in per and len(per[(x, y)]) > k else "   - "
            for x in xs
        )
    )
