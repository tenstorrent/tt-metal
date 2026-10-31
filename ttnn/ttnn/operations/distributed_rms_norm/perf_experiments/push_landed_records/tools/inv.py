# per chip: median / max over compute cores of the wave-k inv_rms ready time (end of cp_invwait, max over TRISCs),
# and the last fabric landing (last prt_notify end on any port), us from the chip's first BRISC-KERNEL start
import sys, statistics as st

sys.path.insert(0, "tests/ttnn/unit_tests/operations/distributed_rms_norm")
from perf_zones import _parse, CLOCK_GHZ

for path in sys.argv[1:]:
    z = _parse(path)
    print(path)
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
        per = {}
        for c, r, n, s, e in ev:
            if n == "cp_invwait":
                per.setdefault((c, r), []).append(e)
        cores = {}
        for (c, r), v in per.items():
            for k, e in enumerate(sorted(v)):
                cores.setdefault(c, {}).setdefault(k, []).append(e)
        out = []
        for k in range(2):
            v = [us(max(d[k])) for d in cores.values() if k in d]
            out.append(f"w{k} inv med {st.median(v):6.2f} max {max(v):6.2f}")
        land = max(us(e) for c, r, n, s, e in ev if n == "prt_notify")
        print(f"  chip {chip}: span {us(tend):6.2f}  last landing {land:6.2f}  " + "  ".join(out))
