"""Per-call distribution of a ccl_perf JSON: percentiles of the fastest / slowest / mean chip per call (host only)."""
import json, sys

for p in [a for a in sys.argv[1:] if "coremap" not in a]:
    d = json.load(open(p))
    calls = [c for exp in d["ns"] for c in exp] if isinstance(d["ns"][0][0], list) else d["ns"]
    q = lambda v, f: sorted(v)[int(f * (len(v) - 1))] / 1000
    mn = [min(c) for c in calls]
    mx = [max(c) for c in calls]
    av = [sum(c) / len(c) for c in calls]
    print(
        f"{p.split('/')[-1]:55s} n={len(calls):5d} fastest p10/50/90 {q(mn,.1):6.1f} {q(mn,.5):6.1f} {q(mn,.9):6.1f} | "
        f"slowest p10/50/90 {q(mx,.1):6.1f} {q(mx,.5):6.1f} {q(mx,.9):6.1f} | mean p50 {q(av,.5):6.1f}"
    )
