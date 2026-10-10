exec(open("/tmp/v2an.py").read().split("if __name__")[0])
import collections, sys
base = "base"
for n in sys.argv[1:]:
    for mk in ("TILE_LOOP", "INIT"):
        A, B = L(base, mk), L(n, mk)
        ks = [k for k in A if k in B and A[k]]
        for thr in (0.005, 0.02):
            mv = [k for k in ks if abs(B[k] / A[k] - 1) > thr]
            by = collections.Counter(rt(k) for k in mv)
            mx = max((abs(B[k] / A[k] - 1) for k in ks), default=0)
            cyc = max((abs(B[k] - A[k]) for k in ks), default=0)
            print(f"{n:10s} {mk:9s} >{thr*100:.1f}% {len(mv):4d}/{len(ks)} max {mx*100:5.1f}% maxcyc {cyc:.0f} " + " ".join(f"{r}:{c}" for r, c in sorted(by.items())))
