"""Last-dispatch timeline: sender zones (first 4 instances) + worker medians of first instance of key zones."""
import csv, sys, statistics as st
from collections import defaultdict

F = 1.35
rows = list(csv.reader(open(sys.argv[1])))[2:]
last = max(int(r[7]) for r in rows)
rows = [r for r in rows if int(r[7]) == last]
ks = [int(r[5]) for r in rows if r[10].strip().endswith("-KERNEL")]
med = sorted(ks)[len(ks) // 2]
t0 = min(t for t in ks if abs(t - med) < 1e6)
inst = defaultdict(list)
op = {}
for r in rows:
    if abs(int(r[5]) - med) > 1e6:
        continue
    k = ((r[1], r[2]), r[3].strip(), r[10].strip())
    if r[11].strip() == "ZONE_START":
        op[k] = int(r[5])
    elif k in op:
        inst[k].append(((op.pop(k) - t0) / F, (int(r[5]) - t0) / F))
cores = defaultdict(set)
for c, risc, n in inst:
    cores[c].add(n)
sender = [c for c in cores if "x_mcast_issue" in cores[c] or "x_read_issue" in cores[c]]
for (c, risc, n), v in sorted(inst.items()):
    if c in sender and not n.endswith("-FW"):
        print("sender", risc, n, [(round(a), round(b)) for a, b in v[:6]])
agg = defaultdict(list)
for (c, risc, n), v in inst.items():
    if c not in sender and not n.endswith("-FW"):
        for i in range(min(3, len(v))):
            agg[(risc, n, i)].append(v[i])
for (risc, n, i), v in sorted(agg.items(), key=lambda kv: st.median(x[1] for x in kv[1])):
    if st.median(x[1] for x in v) < 6000:
        print(
            f"worker {risc:8s} {n:18s} #{i} start {st.median(x[0] for x in v):6.0f} end med {st.median(x[1] for x in v):6.0f} max {max(x[1] for x in v):6.0f}"
        )
