# usage: med.py <label> [label...] : per (case, variant) the runs and median from results.txt
import collections, statistics, sys

d = collections.defaultdict(list)
for l in open(__file__.replace("med.py", "results.txt")):
    p = l.split()
    if len(p) == 5 and p[0] in sys.argv[1:]:
        d[(p[1], p[2])].append(float(p[4]))
base = {}
for (c, v), t in sorted(d.items()):
    m = statistics.median(t)
    if v in ("base", "base_unz"):
        base[c] = m
for (c, v), t in sorted(d.items(), key=lambda kv: (kv[0][0], kv[0][1] != "base")):
    m = statistics.median(t)
    rel = f"{100 * (m / base[c] - 1):+.2f}%" if c in base else ""
    print(f"{c:7s} {v:16s} {' '.join(f'{x:7.3f}' for x in t)}  median {m:7.3f} {rel}")
