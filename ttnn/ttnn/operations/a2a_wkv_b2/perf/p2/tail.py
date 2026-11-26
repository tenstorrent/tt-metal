"""Perf 2: per-row worker end times + per-tx-core timeline for one trace-replay instance (default last).
Usage: tail.py csv [--replay i]"""
import csv, sys, bisect
from collections import defaultdict

F = 1350.0
path = sys.argv[1]
which = int(sys.argv[sys.argv.index("--replay") + 1]) if "--replay" in sys.argv else -1
rows = list(csv.reader(open(path)))[2:]
kst = defaultdict(list)
for r in rows:
    if r[10].strip().endswith("-KERNEL") and r[11].strip() == "ZONE_START":
        kst[r[0]].append(int(r[5]))
inst = {}
for c, ts in kst.items():
    ts.sort()
    s = [ts[0]]
    for a, b in zip(ts, ts[1:]):
        if b - a > 5 * F:
            s.append(b)
    inst[c] = s
sel = {c: (len(s) - 1 if which < 0 else which) for c, s in inst.items()}
ends = defaultdict(int)
starts = {}
for r in rows:
    c = r[0]
    t = int(r[5])
    i = bisect.bisect_right(inst[c], t) - 1
    if i != sel[c]:
        continue
    z, typ = r[10].strip(), r[11].strip()
    k = (c, int(r[1]), int(r[2]), r[3].strip(), z)
    if typ == "ZONE_START":
        starts.setdefault(k, t)
    else:
        ends[k] = max(ends[k], t)


def us(c, t):
    return (t - inst[c][sel[c]]) / F


# worker rows: end of last matmul (TRISC_2) and last read (NCRISC reader_attn_block)
rowmm, rowrd = defaultdict(list), defaultdict(list)
for (c, x, y, risc, z), t in ends.items():
    if z == "compute_matmul" and risc == "TRISC_2":
        rowmm[y].append(us(c, t))
    if z == "reader_attn_block" and risc == "NCRISC":
        rowrd[y].append(us(c, t))
print("phys_y  read_end(min/mean/max)  matmul_end(min/mean/max)")
for y in sorted(rowmm):
    a, b = rowrd[y], rowmm[y]
    print(
        f"{y:5d}  {min(a):5.1f} {sum(a)/len(a):5.1f} {max(a):5.1f}    {min(b):5.1f} {sum(b)/len(b):5.1f} {max(b):5.1f}"
    )
print("tx cores: chip x y | ready_end stream_start stream_end close_end done_start done_end")
for c, x, y, risc, z in sorted(k for k in ends if k[4] == "tx_writer_stream"):
    g = lambda zz, d: us(c, d.get((c, x, y, "BRISC", zz), inst[c][sel[c]]))
    print(
        f"{c} {x:2d} {y:2d} | {g('tx_writer_await_ready', ends):5.1f} {g('tx_writer_stream', starts):5.1f} {g('tx_writer_stream', ends):5.1f} {g('tx_writer_close', ends):5.1f} {g('tx_writer_await_done', starts):5.1f} {g('tx_writer_await_done', ends):5.1f}"
    )
