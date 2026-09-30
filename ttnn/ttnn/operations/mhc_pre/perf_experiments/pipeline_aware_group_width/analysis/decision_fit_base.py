import json, numpy as np, itertools
from collections import defaultdict

P = json.load(open("/tmp/plans.json"))
DEF = {}
for l in open("/tmp/gw_all.txt"):
    p = l.split()
    if p[3] == "default":
        kv = dict(x.split("=") for x in p[4:11])
        DEF[tuple(map(int, p[1].split("x")))] = (int(kv["w"]), int(kv["d"]))
s = defaultdict(list)
for p in P:
    s[(p["T"], p["C"])].append(p)


def score(p, h0, h1, lam, wr, only_d2=True):
    B, d, k, G, gx, Kt, act = p["B"], p["d"], p["kmax"], p["w"], p["gx"], p["Kt"], p["act"]
    if d != 2 and d != 1:
        return None  # candidates as the op would build them (depth default 2, fallback 1)
    rows = act // G
    c = B * (k + h0 + h1 * G) + lam * k + wr * gx * Kt / rows / G  # W share tiles per core (fp32 W = 2 bf16 tiles)
    if d == 1 and B > 1:
        c += 1e6
    return c


def evaluate(h0, h1, lam, wr, verbose=False):
    tot = 0
    worst = 0
    out = []
    for key in sorted(s):
        L = [p for p in s[key] if p["d"] in (1, 2)]
        # op-realizable plans: depth = default fit (d2 if fits else d1): drop d1 plan if a d2 plan of same w exists
        ws = {p["w"] for p in L if p["d"] == 2}
        L = [p for p in L if not (p["d"] == 1 and p["w"] in ws)]
        best = min(p["us"] for p in L)
        sc = [(score(p, h0, h1, lam, wr), p) for p in L]
        pick = min(sc, key=lambda t: t[0])[1]
        dflt = [p for p in L if (p["w"], p["d"]) == DEF[key]][0]
        loss = pick["us"] / best - 1
        vd = pick["us"] / dflt["us"] - 1
        tot += loss
        worst = max(worst, vd)
        out.append((key, pick, best, dflt, loss, vd))
    if verbose:
        for key, pick, best, dflt, loss, vd in out:
            print(
                key,
                f"pick w{pick['w']}d{pick['d']}B{pick['B']}k{pick['kmax']} {pick['us']:.0f} best {best:.0f} def w{dflt['w']}d{dflt['d']} {dflt['us']:.0f} vsbest {loss*100:+.1f}% vsdef {vd*100:+.1f}%",
            )
    return tot, worst


res = []
for h0 in range(0, 61, 2):
    for h1 in (0, 0.5, 1, 1.5, 2, 3, 4):
        for lam in (0, 0.25, 0.5, 0.75, 1, 1.5, 2):
            for wr in (0, 1, 2, 4):
                t, w = evaluate(h0, h1, lam, wr)
                res.append((t, w, h0, h1, lam, wr))
res.sort()
for r in res[:15]:
    print(np.round(r, 3))
print("--- best by worst-vs-default then total")
res.sort(key=lambda r: (round(r[1], 2), r[0]))
for r in res[:10]:
    print(np.round(r, 3))
import sys

print("=== verbose", sys.argv[1:])
a = [float(x) for x in sys.argv[1:5]]
evaluate(*a, verbose=True)
# plateau for h1=0, lam=0, wr=0
print([(h0, round(evaluate(h0, 0, 0, 0)[0], 3), round(evaluate(h0, 0, 0, 0)[1], 3)) for h0 in range(0, 40, 2)])
