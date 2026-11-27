"""Per tx core: TXD_* accumulators (cycles -> us at 1350 MHz) from a tx_writer_v TXW_DIAG profile; replay 0
(eager warm-up) dropped, the rest averaged. Usage: python3 txd.py <csv>"""
import csv, sys
from collections import defaultdict

FREQ = 1350.0
rows = list(csv.reader(open(sys.argv[1])))[2:]
seq = defaultdict(list)  # (chip,x,y) -> list of dict per call
for r in rows:
    name = r[10].strip()
    if not name.startswith("TXD_"):
        continue
    k = (int(r[0]), int(r[1]), int(r[2]))
    if name == "TXD_PACKETS":
        seq[k].append({})
    seq[k][-1][name[4:]] = int(r[6])
cols = ["PACKETS", "BATCHES", "WAIT", "HDR", "SLOT", "SLOTSTALLS", "PAY", "HW", "FLUSH", "NBUF"]
print("chip x y  " + " ".join(f"{c:>9}" for c in cols) + "   (cycle fields in us; mean over calls 1..)")
agg = defaultdict(list)
for k in sorted(seq):
    calls = seq[k][1:] or seq[k]
    m = {c: sum(d.get(c, 0) for d in calls) / len(calls) for c in cols}
    vals = []
    for c in cols:
        v = m[c]
        vals.append(f"{v / FREQ:9.2f}" if c in ("WAIT", "HDR", "SLOT", "PAY", "HW", "FLUSH") else f"{v:9.0f}")
    print(f"{k[0]:>4} {k[1]:>2} {k[2]:>2} " + " ".join(vals))
    if m["PACKETS"] >= 100:
        for c in cols:
            agg[c].append(m[c])
if agg:
    print(
        "link-0 (>=100 pkt) mean: "
        + " ".join(
            f"{c}={sum(v)/len(v)/(FREQ if c in ('WAIT','HDR','SLOT','PAY','HW','FLUSH') else 1):.2f}"
            for c, v in agg.items()
        )
    )
    n = sum(agg["PACKETS"]) / len(agg["PACKETS"])
    print(
        "per packet (cycles): "
        + " ".join(
            f"{c}={sum(v)/len(v)/n:.0f}" for c, v in agg.items() if c in ("WAIT", "HDR", "SLOT", "PAY", "HW", "FLUSH")
        )
    )
