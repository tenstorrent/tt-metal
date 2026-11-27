"""Per chip (median over trace replays 1..): device span, and per tx core (+l0,+l1,-l0,-l1) the
tx_writer_stream end, close end and await_done end (us from the chip's instance start)."""
import csv, statistics, sys
from collections import defaultdict

F = 1350.0
rows = list(csv.reader(open(sys.argv[1])))[2:]
open_ = {}
Z = defaultdict(list)
K = defaultdict(list)
for r in rows:
    chip, x, y, risc = int(r[0]), int(r[1]), int(r[2]), r[3].strip()
    t = int(r[5])
    zn = r[10].strip()
    ty = r[11].strip()
    k = (chip, x, y, risc, zn)
    if ty == "ZONE_START":
        open_[k] = t
    elif ty == "ZONE_END" and k in open_:
        s = open_.pop(k)
        if zn.endswith("-KERNEL"):
            K[chip].append((s, t))
        elif zn.startswith("tx_writer"):
            Z[chip].append((zn, s, t, x, y))
for chip in sorted(K):
    ks = sorted(K[chip])
    g = [[ks[0]]]
    for a in ks[1:]:
        if a[0] - g[-1][-1][0] > 5 * F:
            g.append([])
        g[-1].append(a)
    txc = sorted({(y, x) for (zn, a, b, x, y) in Z[chip] if zn == "tx_writer_stream"})
    acc = defaultdict(list)
    for grp in g[1:]:
        s = min(a for a, _ in grp)
        e = max(b for _, b in grp)
        acc["span"].append((e - s) / F)
        for nm, (y, x) in zip(["+l0", "+l1", "-l0", "-l1"], txc):
            for zn, lab in (
                ("tx_writer_stream", "st"),
                ("tx_writer_close", "cl"),
                ("tx_writer_await_done", "dn"),
                ("tx_writer_await_ready", "rdy"),
            ):
                for z, a, b, xx, yy in Z[chip]:
                    if z == zn and (xx, yy) == (x, y) and s <= a <= e:
                        acc[f"{nm}.{lab}"].append((b - s) / F)
    m = {k: statistics.median(v) for k, v in acc.items()}
    print(
        f"chip {chip}: span {m['span']:5.1f} | "
        + " | ".join(
            f"{nm} st {m.get(nm+'.st',0):5.1f} dn {m.get(nm+'.dn',0):5.1f} rdy {m.get(nm+'.rdy',0):4.1f}"
            for nm in ["+l0", "+l1", "-l0", "-l1"]
        )
    )
