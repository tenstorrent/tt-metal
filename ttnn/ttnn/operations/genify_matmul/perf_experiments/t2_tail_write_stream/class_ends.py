"""Per bank class: the end of the core's tail writes (max of BRISC wr_write / NCRISC rd_tail_write end) for the te_gate
bank-local layout. usage: class_ends.py <profile_log_device.csv>"""
import csv, sys, collections

sys.path.insert(0, sys.path[0])
from noc_model import layout, PX

rows = list(csv.reader(open(sys.argv[1])))[2:]
by_run = collections.defaultdict(list)
for r in rows:
    by_run[r[7].strip()].append(r)
run = max(by_run, key=lambda k: len(by_run[k]))
t0 = min(int(r[5]) for r in by_run[run])
end = collections.defaultdict(float)
for r in by_run[run]:
    if r[10].strip() in ("wr_write", "rd_tail_write") and r[11].strip() == "ZONE_END":
        c = (int(r[1]), int(r[2]))
        end[c] = max(end[c], (int(r[5]) - t0) / 1350)
lay = layout()
cls = collections.defaultdict(list)
for (lx, ly), (u, t0_, w) in lay.items():
    if w:
        cls[u].append(((PX[lx], ly + 2), w, end[(PX[lx], ly + 2)]))
for u in range(8):
    v = cls[u]
    rows_ = sorted({c[1] for c, _, _ in v})
    print(
        f"class {u}: cores {len(v):2d} tiles {sum(w for _, w, _ in v)} rows {rows_} end min {min(e for *_, e in v):.1f} max {max(e for *_, e in v):.1f}  slowest {max(v, key=lambda t: t[2])[0]}"
    )
