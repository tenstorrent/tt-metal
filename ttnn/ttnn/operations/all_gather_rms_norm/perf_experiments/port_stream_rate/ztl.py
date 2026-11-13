"""Timeline of one port (chip, x, y): ztl.py <csv> <chip> <x> <y>. Sender per-packet CBW wait, reader zones."""
import csv, sys, collections

rows = list(csv.reader(open(sys.argv[1])))[2:]
chip, X, Y = sys.argv[2], sys.argv[3], sys.argv[4]
F = 1350.0
cr = [r for r in rows if r[0] == chip]
last = max(int(r[7]) for r in cr)
sel = sorted([r for r in cr if int(r[7]) == last], key=lambda r: int(r[5]))
t0 = min(int(r[5]) for r in sel)
st = collections.defaultdict(list)
ev = []
for r in sel:
    if r[1] != X or r[2] != Y or not r[10].startswith("AG_"):
        continue
    k = (r[3], r[10])
    if r[11] == "ZONE_START":
        st[k].append(int(r[5]))
    elif st[k]:
        s = st[k].pop()
        ev.append(((s - t0) / F, (int(r[5]) - t0) / F, r[3][:2], r[10]))
for s, e, risc, n in sorted(ev):
    if n in ("AG_FENCE", "AG_SEND"):
        continue
    ind = "" if risc == "BR" else "                                        "
    print(f"{ind}{risc} {n:16s} {s:7.2f} {e:7.2f} ({e-s:5.2f})")
