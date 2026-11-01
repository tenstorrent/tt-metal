"""timeline.py <tag> <generic-op index> [core_x,core_y] [riscs]: zone start/end (ns from the program's first marker)."""
import csv, sys, collections
from pathlib import Path

H = Path(__file__).parent / "runs"
tag, idx = sys.argv[1], int(sys.argv[2])
core = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
riscs = sys.argv[4].split(",") if len(sys.argv) > 4 else ["BRISC", "TRISC_0", "TRISC_2"]
ops = [r for r in csv.DictReader(open(H / f"{tag}_ops.csv")) if r["OP CODE"].startswith("GenericOp")]
rid = ops[idx]["GLOBAL CALL COUNT"]
f = open(H / f"{tag}_dev.csv")
head = f.readline()
freq = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rd = csv.reader(f)
hdr = [h.strip() for h in next(rd)]
I = {h: i for i, h in enumerate(hdr)}
rows = [[x.strip() for x in r] for r in rd]
rows = [r for r in rows if r[I["run host ID"]] == rid]
t0 = min(int(r[I["time[cycles since reset]"]]) for r in rows)
if core is None:
    core = (rows[0][I["core_x"]], rows[0][I["core_y"]])
st = {}
ev = []
for r in rows:
    if (r[I["core_x"]], r[I["core_y"]]) != core or r[I["RISC processor type"]] not in riscs:
        continue
    z = r[I["zone name"]]
    t = (int(r[I["time[cycles since reset]"]]) - t0) / freq * 1000
    k = (r[I["RISC processor type"]], z)
    if r[I["type"]] == "ZONE_START":
        st.setdefault(k, []).append(t)
    elif r[I["type"]] == "ZONE_END" and st.get(k):
        s = st[k].pop()
        ev.append((s, t, k[0], z))
for s, e, rv, z in sorted(ev):
    if z.endswith("-FW"):
        continue
    print("%8.0f %8.0f %6.0f  %-8s %s" % (s, e, e - s, rv, z))
