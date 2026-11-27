"""Per tx core: TXD_* (writer) / TXR_* (reader) accumulators from a diag profile (cycles at 1350 MHz).
Replay 0 dropped, the rest averaged. Prints per core and the mean over cores by packet count.
Usage: python3 txd.py <csv>"""
import csv, sys
from collections import defaultdict

F = 1350.0
rows = list(csv.reader(open(sys.argv[1])))[2:]
seq = defaultdict(list)
for r in rows:
    name = r[10].strip()
    if not (name.startswith("TXD_") or name.startswith("TXR_")):
        continue
    pre = name[:4]
    k = (int(r[0]), int(r[1]), int(r[2]), pre)
    if name in ("TXD_PACKETS", "TXR_BATCHES"):
        seq[k].append({})
    seq[k][-1][name[4:]] = int(r[6])
CNT = {"PACKETS", "BATCHES", "SLOTSTALLS", "NBUF", "PKTS", "POLLS"}
groups = defaultdict(lambda: defaultdict(list))
for k in sorted(seq):
    calls = seq[k][1:] or seq[k]
    keys = sorted({c for d in calls for c in d})
    m = {c: sum(d.get(c, 0) for d in calls) / len(calls) for c in keys}
    npk = m.get("PACKETS", m.get("PKTS", 0))
    print(
        f"{k[3]} chip{k[0]} ({k[1]},{k[2]}) "
        + " ".join(f"{c}={m[c]:.0f}" if c in CNT else f"{c}={m[c]/F:.2f}us" for c in keys)
    )
    g = "link0" if npk >= 100 else "link1"
    for c in keys:
        groups[(k[3], g)][c].append(m[c])
for (pre, g), d in sorted(groups.items()):
    n = sum(d.get("PACKETS", d.get("PKTS", [1]))) / len(d.get("PACKETS", d.get("PKTS", [1])))
    print(
        f"MEAN {pre} {g}: "
        + " ".join(
            f"{c}={sum(v)/len(v):.0f}" if c in CNT else f"{c}={sum(v)/len(v)/F:.2f}us({sum(v)/len(v)/n:.0f}cyc/pkt)"
            for c, v in d.items()
        )
    )
