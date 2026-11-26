"""tx_rebalance_direct: from a trace-replay device profile (probe_variant_trace_profile.py), per replay
instance (kernel starts jumping by > 5 us), per chip: device span, launch gap before it, and per tx core
the tx_writer_stream / tx_writer_addr_table end (us from the instance start). tx cores are the 4 cores
with a tx_writer_stream zone; sorted by (y, x) they are (+1,link0), (+1,link1), (-1,link0), (-1,link1)
(the descriptor's stream order on the last 4 logical cores). Summary = median over chips and replays
1..N-1 (replay 0 = eager warm-up call). Usage: python3 parse_links.py csv"""
import csv, sys, statistics
from collections import defaultdict

F = 1350.0
rows = list(csv.reader(open(sys.argv[1])))[2:]
kz = defaultdict(list)  # chip -> [(start, end)] kernel zones
open_ = {}
zones = defaultdict(list)  # (chip, x, y, zone) -> [(start, end)]
for r in rows:
    chip, x, y, risc = r[0], int(r[1]), int(r[2]), r[3].strip()
    t, zone, typ = int(r[5]), r[10].strip(), r[11].strip()
    key = (chip, x, y, risc, zone)
    if typ == "ZONE_START":
        open_[key] = t
    elif typ == "ZONE_END" and key in open_:
        s = open_.pop(key)
        if zone.endswith("-KERNEL"):
            kz[chip].append((s, t))
        elif zone.startswith("tx_writer"):
            zones[(chip, x, y, zone)].append((s, t))
res = defaultdict(list)
for chip in sorted(kz, key=int):
    z = sorted(kz[chip])
    groups, cur = [], [z[0]]
    for a in z[1:]:
        if a[0] - cur[-1][0] > 5 * F:
            groups.append(cur)
            cur = []
        cur.append(a)
    groups.append(cur)
    inst = [(min(a for a, _ in g), max(b for _, b in g)) for g in groups]
    txc = sorted({(y, x) for (c, x, y, zn) in zones if c == chip and zn == "tx_writer_stream"})
    names = ["+l0", "+l1", "-l0", "-l1"][: len(txc)]
    line = []
    for i, (s, e) in enumerate(inst):
        if i == 0:
            continue
        res["span"].append((e - s) / F)
        res["gap"].append((s - inst[i - 1][1]) / F)
        for nm, (y, x) in zip(names, txc):
            for zn, lab in (("tx_writer_stream", "end"), ("tx_writer_addr_table", "tbl")):
                for a, b in zones.get((chip, x, y, zn), []):
                    if s <= a <= e:
                        res[f"{nm}_{lab}"].append((b - s) / F)
                        if zn == "tx_writer_stream":
                            res[f"{nm}_busy"].append((b - a) / F)
    print(
        f"chip {chip}: spans "
        + " ".join(f"{(e-s)/F:.1f}" for s, e in inst)
        + " gaps "
        + " ".join(f"{(inst[i][0]-inst[i-1][1])/F:.1f}" for i in range(1, len(inst)))
    )
print("median over chips x replays 1..:")
for k in sorted(res):
    v = res[k]
    print(f"  {k:12s} median {statistics.median(v):6.1f}  min {min(v):6.1f}  max {max(v):6.1f}  (n={len(v)})")
