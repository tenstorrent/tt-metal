# per chip timeline (us from the chip's first BRISC-KERNEL start), last traced call
import sys, statistics as st

sys.path.insert(0, "tests/ttnn/unit_tests/operations/distributed_rms_norm")
from perf_zones import _parse, CLOCK_GHZ

z = _parse(sys.argv[1])
for chip in range(4):
    runs = [
        (int(r) if r.isdigit() else 0, ev)
        for (c, r), ev in z.items()
        if c == chip and any(n.startswith("rt_") for _, _, n, _, _ in ev)
    ]
    run, ev = max(runs, key=lambda x: x[0])
    t0 = min(s for _, r, n, s, _ in ev if n == "BRISC-KERNEL")
    tend = max(e for _, r, n, _, e in ev if n.endswith("KERNEL"))
    us = lambda t: (t - t0) / CLOCK_GHZ / 1000
    print(f"chip {chip}: span {us(tend):.2f}")
    ports = sorted({c for c, _, n, _, _ in ev if n.startswith("prt_")})
    for core in ports:
        rows = sorted(
            (s, e, n)
            for c, r, n, s, e in ev
            if c == core and n in ("prt_ready_wait", "prt_send", "prt_push", "prt_notify")
        )
        print("  port", core, " ".join(f"{n[4:]}:{us(s):.2f}-{us(e):.2f}" for s, e, n in rows))
    # per-core per-occurrence of key zones: k-th occurrence -> min/median/max of start/end
    for name in sys.argv[2].split(","):
        per = {}
        for c, r, n, s, e in ev:
            if n == name:
                per.setdefault(c, []).append((s, e))
        if not per:
            continue
        k = max(len(v) for v in per.values())
        for i in range(k):
            S = [sorted(v)[i][0] for v in per.values() if len(v) > i]
            E = [sorted(v)[i][1] for v in per.values() if len(v) > i]
            print(
                f"  {name}[{i}] n={len(S)} start {us(min(S)):.2f}/{us(st.median(S)):.2f}/{us(max(S)):.2f} end {us(min(E)):.2f}/{us(st.median(E)):.2f}/{us(max(E)):.2f}"
            )
