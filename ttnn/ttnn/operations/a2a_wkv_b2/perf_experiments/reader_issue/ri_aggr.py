"""Aggregate rate estimate: sum over cores of bytes/end (constant per-core rate), per chip, last replay."""
import csv, sys, bisect
from collections import defaultdict

FREQ = 1350.0
B = 160 * 1024 + 64 * 1088
for path in sys.argv[1:]:
    rows = list(csv.reader(open(path)))[2:]
    kst = defaultdict(list)
    for r in rows:
        if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START":
            kst[r[0]].append(int(r[5]))
    inst = {}
    for chip, ts in kst.items():
        ts.sort()
        s = [ts[0]]
        for a, b in zip(ts, ts[1:]):
            if b - a > 5 * FREQ:
                s.append(b)
        inst[chip] = s[-1]
    readers, end = set(), {}
    for r in rows:
        chip = r[0]
        t = int(r[5])
        if t < inst[chip]:
            continue
        z, typ = r[10].strip(), r[11].strip()
        if r[3].strip() == "NCRISC" and z.startswith("reader_"):
            readers.add((chip, r[1], r[2]))
        if r[3].strip() == "NCRISC" and z == "NCRISC-KERNEL" and typ == "ZONE_END":
            end[(chip, r[1], r[2])] = (t - inst[chip]) / FREQ
    per = defaultdict(list)
    for k, v in end.items():
        if k in readers:
            per[k[0]].append(v)
    agg = [sum(B / e for e in v) / 1e3 for v in per.values()]
    mx = [max(v) for v in per.values()]
    print(
        f"{path.split('/')[-1]:28s} initial aggregate {min(agg):.0f}-{max(agg):.0f} GB/s; total/max-end {96*B/max(mx)/1e3:.0f} GB/s"
    )
