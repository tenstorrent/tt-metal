# early_small_wave zone timeline: per chip (last traced call), us from the chip's first BRISC-KERNEL start.
# usage (repo root): python .../tools/esw_tl.py <profile_log_device.csv> [chips]
import sys, statistics as st

sys.path.insert(0, "tests/ttnn/unit_tests/operations/distributed_rms_norm")
from perf_zones import _parse, CLOCK_GHZ

z = _parse(sys.argv[1])
chips = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3]


def q(v):
    v = sorted(v)
    if not v:
        return "-"
    p = lambda f: v[min(len(v) - 1, int(f * (len(v) - 1) + 0.5))]
    return f"n={len(v):3d} min {v[0]:6.2f} p10 {p(.1):6.2f} p50 {p(.5):6.2f} p90 {p(.9):6.2f} max {v[-1]:6.2f}"


for chip in chips:
    runs = [
        (int(r) if r.isdigit() else 0, ev)
        for (c, r), ev in z.items()
        if c == chip and any(n.startswith("rt_") for _, _, n, _, _ in ev)
    ]
    if not runs:
        continue
    run, ev = max(runs, key=lambda x: x[0])
    t0 = min(s for _, r, n, s, _ in ev if n == "BRISC-KERNEL")
    tend = max(e for _, r, n, _, e in ev if n.endswith("KERNEL"))
    us = lambda t: (t - t0) / CLOCK_GHZ / 1000
    print(f"=== chip {chip}: span {us(tend):.2f} us")
    by = {}
    for c, r, n, s, e in ev:
        by.setdefault(n, []).append((us(s), us(e), c))
    for name in (
        "rd_load",
        "rd_land",
        "mb_send",
        "rt_gather",
        "rt_publish",
        "rt_collect",
        "cp_fin",
        "rt_distribute",
        "mb_recv",
        "wr_store",
        "rd_help",
    ):
        if name not in by:
            continue
        v = by[name]
        print(f"  {name:13s} start {q([a for a, _, _ in v])}")
        print(f"  {'':13s} end   {q([b for _, b, _ in v])}")
    # ports
    for name in ("prt_send", "prt_notify"):
        if name in by:
            print(f"  {name:13s} start {q([a for a, _, _ in by[name]])}")
    # rt_distribute starts sorted (one per root per non-empty wave) and the output burst
    if "rt_distribute" in by:
        print("  distribute starts:", " ".join(f"{a:.1f}" for a in sorted(a for a, _, _ in by["rt_distribute"])))
    if "wr_store" in by:
        ends = sorted(b for _, b, _ in by["wr_store"])
        print(f"  write burst: first store {min(a for a, _, _ in by['wr_store']):.2f}, last end {ends[-1]:.2f}")
    if "rd_land" in by:
        print(f"  reads: last land {max(b for _, b, _ in by['rd_land']):.2f}")
