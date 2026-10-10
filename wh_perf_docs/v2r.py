import csv, glob, os, sys, collections
def L(d, marker="TILE_LOOP"):
    out = {}
    for f in glob.glob(f"{d}/*.csv"):
        if f.endswith(".post.csv"): continue
        m = os.path.basename(f)[:-4]
        for r in csv.DictReader(open(f)):
            if r.get("marker") != marker: continue
            key = (m,) + tuple((k, v) for k, v in r.items() if not k.startswith(("mean(", "std(", "TEXT_SIZE")) and k != "marker")
            for c, x in r.items():
                if c.startswith("mean(") and x and ("L1_" in c):
                    out[key + (c,)] = float(x)
    return out
def cmp(a, b, la=None, lb=None):
    A, B = L(a), L(b)
    ks = [k for k in A if k in B and A[k]]
    d = [B[k] / A[k] - 1 for k in ks]
    by = collections.Counter(k[-1][5:-1] for k in ks if abs(B[k] / A[k] - 1) > 0.02)
    print(f"{la or a:>12s} -> {lb or b:<12s} n {len(ks):4d} !=:{sum(x != 0 for x in d):4d} >0.5%:{sum(abs(x) > .005 for x in d):4d} >2%:{sum(abs(x) > .02 for x in d):4d} max {max(map(abs, d), default=0)*100:5.1f}% median {sorted(d)[len(d)//2]*100:+.2f}% " + " ".join(f"{r}:{c}" for r, c in sorted(by.items())))
R = "/proj_sw/user_dev/nstojic/v2r"
for a, b in [x.split(":") for x in sys.argv[1:]]:
    da = a if a.startswith("/") else f"{R}/{a}"; db = b if b.startswith("/") else f"{R}/{b}"
    cmp(da, db, os.path.basename(a), os.path.basename(b))
