import csv, glob, os, sys, collections
D = "/tmp/v2c"
def L(n, marker):
    out = {}
    for f in glob.glob(f"{D}/{n}/*.csv"):
        if f.endswith(".post.csv"): continue
        m = os.path.basename(f)[:-4]
        for r in csv.DictReader(open(f)):
            if r.get("marker") != marker: continue
            key = (m,) + tuple((k, v) for k, v in r.items() if not k.startswith(("mean(", "std(", "TEXT_SIZE")) and k != "marker")
            for c, x in r.items():
                if c.startswith("mean(") and x:
                    out[key + (c,)] = float(x)
    return out
def rt(k):
    return k[-1][5:-1]
def cmp(a, b, marker="TILE_LOOP", thr=0.005, lim=6):
    A, B = L(a, marker), L(b, marker)
    ks = [k for k in A if k in B and A[k]]
    mv = [(B[k] / A[k] - 1, k) for k in ks if abs(B[k] / A[k] - 1) > thr]
    by = collections.Counter(rt(k) for _, k in mv); tot = collections.Counter(rt(k) for k in ks)
    ex = sum(1 for k in ks if B[k] != A[k])
    mx = max((abs(d) for d, _ in mv), default=0)
    print(f"{marker:9s} {a:>8s} -> {b:<10s} values {len(ks):5d}  !=:{ex:4d}  moved>{thr*100:.1f}%: {len(mv):4d}  max {mx*100:5.1f}%  " + " ".join(f"{r}:{by[r]}/{tot[r]}" for r in sorted(tot)))
    for d, k in sorted(mv, key=lambda x: -abs(x[0]))[:lim]:
        cfg = " ".join(f"{x}={y}" for x, y in k[1:-1] if x not in ("run_type",))[:150]
        print(f"    {d*100:+6.1f}% {k[0]} {rt(k)} {k[-1]} {A[k]:.0f}->{B[k]:.0f} {cfg}")
if __name__ == "__main__":
    base = sys.argv[1]; mk = sys.argv[2] if len(sys.argv) > 2 else "TILE_LOOP"
    for n in sys.argv[3:]:
        cmp(base, n, mk)
