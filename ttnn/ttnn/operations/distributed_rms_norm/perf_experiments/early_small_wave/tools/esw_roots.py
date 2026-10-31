# per-root event sequences (chip given), us from the chip's first BRISC-KERNEL start
import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/distributed_rms_norm")
from perf_zones import _parse, CLOCK_GHZ

z = _parse(sys.argv[1])
chip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
runs = [
    (int(r) if r.isdigit() else 0, ev)
    for (c, r), ev in z.items()
    if c == chip and any(n.startswith("rt_") for _, _, n, _, _ in ev)
]
run, ev = max(runs, key=lambda x: x[0])
t0 = min(s for _, r, n, s, _ in ev if n == "BRISC-KERNEL")
us = lambda t: (t - t0) / CLOCK_GHZ / 1000
roots = sorted({c for c, r, n, s, e in ev if n == "rt_publish"})
for c in roots:
    out = []
    for name in ("rd_land", "rt_gather", "rt_publish", "cp_fin", "rt_distribute"):
        v = sorted((us(s), us(e)) for cc, r, n, s, e in ev if cc == c and n == name)
        out.append(
            name.split("_")[1][:4]
            + ":"
            + ",".join(f"{a:.1f}" + (f"+{b-a:.2f}" if name == "rt_publish" else "") for a, b in v)
        )
    print(c, " ".join(out))
