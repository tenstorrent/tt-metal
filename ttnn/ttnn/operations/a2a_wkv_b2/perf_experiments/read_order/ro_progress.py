"""read_order: per phys row, mean time (us from instance start) of reader milestones: W issue end, end of each
attn tile row (reader_attn_block #0..4), defer end. Usage: ro_progress.py csv [--replay i]"""
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
ev = defaultdict(list)  # (chip,x,y,zone) -> sorted end times
st = defaultdict(list)
for r in rows:
    c = r[0]
    t = int(r[5])
    i = bisect.bisect_right(inst[c], t) - 1
    if i != sel[c] or r[3].strip() != "NCRISC":
        continue
    z, typ = r[10].strip(), r[11].strip()
    if z not in ("reader_attn_block", "reader_w_issue", "reader_defer"):
        continue
    k = (c, int(r[1]), int(r[2]), z)
    (ev if typ == "ZONE_END" else st)[k].append((t - inst[c][sel[c]]) / F)
byrow = defaultdict(lambda: defaultdict(list))
for (c, x, y, z), ts in ev.items():
    ts.sort()
    if z == "reader_attn_block":
        for j, t in enumerate(ts):
            byrow[y][f"r{j}"].append(t)
    else:
        byrow[y][z].append(ts[-1])
for (c, x, y, z), ts in st.items():
    if z == "reader_w_issue":
        byrow[y]["w_start"].append(min(ts))
cols = ["reader_defer", "w_start", "reader_w_issue", "r0", "r1", "r2", "r3", "r4"]
print("phys_y " + " ".join(f"{c[:8]:>8s}" for c in cols))
for y in sorted(byrow):
    d = byrow[y]
    print(f"{y:5d}  " + " ".join(f"{(sum(d[c])/len(d[c]) if d[c] else float('nan')):8.1f}" for c in cols))
