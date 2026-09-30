import re, numpy as np, itertools

rows = []
for l in open("/tmp/gw_all.txt"):
    p = l.split()
    T, C = map(int, p[1].split("x"))
    v = p[3]
    kv = dict(x.split("=") for x in p[4:11])
    us = float(p[11].replace("us", ""))
    w, gx, B, d, kmax = int(kv["w"]), int(kv["gx"]), int(kv["B"]), int(kv["d"]), int(kv["kmax"])
    Mt = T // 32
    Kt = 4 * C // 32
    act = w * min(gx * 10, Mt)
    rows.append(dict(T=T, C=C, v=v, w=w, gx=gx, B=B, d=d, kmax=kmax, us=us, act=act, Kt=Kt, Mt=Mt))
# dedupe by plan (T,C,w,d): median
from collections import defaultdict

g = defaultdict(list)
for r in rows:
    g[(r["T"], r["C"], r["w"], r["d"])].append(r)
plans = []
for k, rs in g.items():
    r = dict(rs[0])
    r["us"] = float(np.median([x["us"] for x in rs]))
    r["n"] = len(rs)
    plans.append(r)
import json

json.dump(plans, open("/tmp/plans.json", "w"))
print(len(plans))
