"""Stage-3 tail per core (host script, no device): last call per chip of a zone CSV + results/coremap.json.

    python3 tail_table.py zones.csv [coremap.json] [--all]

Per chip: last landing (RD_S3_LAND end), program end, and the cores whose WR_S3_OUT ends last, with their heads
(source chip of each head; 'L' = local when the chip position is known = PCIe-slot order), S1 end, last landing,
S3 start / end, writer end and tail = writer end - last landing.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

path = sys.argv[1]
mp = (
    Path(sys.argv[2])
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--")
    else Path(path).parent / "results/coremap.json"
)
show_all = "--all" in sys.argv
cm = json.loads(mp.read_text())
p2l = {tuple(int(v) for v in k.split(",")): tuple(v) for k, v in cm["map"].items()}
heads = {tuple(c): hs for c, hs in zip(cm["cores"], cm["heads_of"])}
Hl = cm["Hl"]
f = open(path)
f.readline()
rows = list(csv.DictReader(f, skipinitialspace=True))
MHZ = 1350.0
last = {}
for r in rows:
    last[r["PCIe slot"]] = max(last.get(r["PCIe slot"], 0), int(r["run host ID"]))
for pos, chip in enumerate(sorted(last, key=int)):
    st = defaultdict(dict)
    # per-chip physical -> logical (harvested columns differ per chip; BH logical order = sorted physical columns)
    xs = sorted({int(r["core_x"]) for r in rows if r["PCIe slot"] == chip})
    ys = sorted({int(r["core_y"]) for r in rows if r["PCIe slot"] == chip})
    for r in rows:
        if r["PCIe slot"] != chip or int(r["run host ID"]) != last[chip]:
            continue
        c = (xs.index(int(r["core_x"])), ys.index(int(r["core_y"])))
        key = (c, r["RISC processor type"], r["zone name"])
        t = int(r["time[cycles since reset]"])
        d = st[key]
        if r["type"] == "ZONE_START":
            d["s"] = min(d.get("s", t), t)
        elif r["type"] == "ZONE_END":
            d["e"] = max(d.get("e", t), t)
    t0 = min(d["s"] for k, d in st.items() if k[2].endswith("KERNEL") and "s" in d)
    per = defaultdict(dict)
    for (c, risc, z), d in st.items():
        if "s" in d and "e" in d:
            per[c][(z, risc)] = ((d["s"] - t0) / MHZ, (d["e"] - t0) / MHZ)
    end = max(v[1] for c in per for k, v in per[c].items() if k[0].endswith("KERNEL"))
    lands = [per[c][("RD_S3_LAND", "NCRISC")][1] for c in per if ("RD_S3_LAND", "NCRISC") in per[c]]
    out = []
    for c in per:
        if ("WR_S3_OUT", "BRISC") not in per[c]:
            continue
        g = lambda z, r, i: per[c].get((z, r), (float("nan"),) * 2)[i]
        hs = heads.get(c, [])
        tag = ",".join(("L" if h // Hl == pos else str(h // Hl)) for h in hs)
        out.append(
            (
                g("WR_S3_OUT", "BRISC", 1),
                c,
                tag,
                g("S1", "TRISC_1", 1),
                g("RD_S3_LAND", "NCRISC", 1),
                g("S3", "TRISC_1", 0),
                g("S3", "TRISC_1", 1),
            )
        )
    out.sort()
    n = len(out)
    print(
        f"chip {chip} (pos {pos}): end {end:6.1f}  last landing {max(lands):6.1f}  out_end med {out[n // 2][0]:6.1f} "
        f"p90 {out[int(0.9 * n)][0]:6.1f} max {out[-1][0]:6.1f}  tail(max out - last land) {out[-1][0] - max(lands):5.1f}"
    )
    for e in out if show_all else out[-8:]:
        print(
            f"   core {str(e[1]):8s} src {e[2]:6s} s1_end {e[3]:6.1f} land {e[4]:6.1f} s3 {e[5]:6.1f}-{e[6]:6.1f}"
            f" wr_end {e[0]:6.1f} tail {e[0] - e[4]:5.1f}"
        )
