import json, sys, collections
# initcmp.py base.jsonl leg.jsonl : per launch (node, run type) and thread, INIT cycles (end - start) and the delta
def load(f):
    d = {}
    for l in open(f):
        r = json.loads(l)
        k = (r["node"], r["run_type"])
        d[k] = {t: z[1][0] - z[0][0] for t, z in r["zones"].items()}
    return d
A, B = load(sys.argv[1]), load(sys.argv[2])
ks = [k for k in A if k in B]
for t in ("unpack", "math", "pack"):
    c = collections.Counter(B[k][t] - A[k][t] for k in ks if t in A[k] and t in B[k])
    print(f"  {t:6s} delta histogram over {sum(c.values())} launches: {dict(sorted(c.items()))}")
if len(sys.argv) > 3:
    for k in ks:
        print("   ", k[1], k[0].split("[")[1][:110], {t: (A[k][t], B[k][t] - A[k][t]) for t in A[k]})
