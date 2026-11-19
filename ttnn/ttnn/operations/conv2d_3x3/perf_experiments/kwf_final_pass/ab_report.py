# usage: ab_report.py <label_prefix> <base> <cand>: medians of runs/results.txt entries whose label starts with the prefix
import collections, statistics, sys

pre, base, cand = sys.argv[1:4]
d = collections.defaultdict(lambda: collections.defaultdict(list))
order = []
for line in open(__file__.rsplit("/", 1)[0] + "/runs/results.txt"):
    lab, case, var, v = line.split()
    if not lab.startswith(pre):
        continue
    if case not in order:
        order.append(case)
    d[case][var].append(float(v))
print(f"{'case':20s} {'base med':>9s} {'cand med':>9s} {'delta':>7s}  runs(base | cand)")
for c in order:
    b, k = d[c][base], d[c][cand]
    mb, mk = statistics.median(b), statistics.median(k)
    print(f"{c:20s} {mb:9.2f} {mk:9.2f} {100*(mk-mb)/mb:+6.1f}%  {b} | {k}")
