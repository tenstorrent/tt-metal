# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-port timeline of the last q_stem call per chip (bench.py QSTEM_MODE=zone + results/plan.json).
Host-side, no device.

    python3 ports.py [plan.json] [profile_log_device.csv]

Per chip (device id, ring position p, antipodal route direction): per port (dir, link, 'late' = carries the antipodal
quarter): PORT_FENCE end, PORT_SEND start/end, PORT_PULL end, PS_BLK block ends (if the variant emits them); then
stage-1 end (median S1 TRISC_1), last landing (max RD_S3_LAND end), program end.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
plan = json.load(open(sys.argv[1] if len(sys.argv) > 1 else HERE / "results" / "plan.json"))
path = sys.argv[2] if len(sys.argv) > 2 else "generated/profiler/.logs/profile_log_device.csv"
f = open(path)
f.readline()
rows = list(csv.DictReader(f, skipinitialspace=True))
MHZ = 1350.0
last = {}
for r in rows:
    last[r["PCIe slot"]] = max(last.get(r["PCIe slot"], 0), int(r["run host ID"]))

summary = {}
for chip in sorted(last):
    pl = plan.get(chip)
    st = defaultdict(lambda: {"s": [], "e": []})
    for r in rows:
        if r["PCIe slot"] != chip or int(r["run host ID"]) != last[chip]:
            continue
        c = f'{r["core_x"]},{r["core_y"]}'
        key = (c, r["RISC processor type"], r["zone name"])
        t = int(r["time[cycles since reset]"])
        st[key]["s" if r["type"] == "ZONE_START" else "e"].append(t)
    t0 = min(min(d["s"]) for k, d in st.items() if k[2].endswith("KERNEL") and d["s"])
    us = lambda t: (t - t0) / MHZ
    end = max(max(d["e"]) for k, d in st.items() if k[2].endswith("KERNEL") and d["e"])
    s1 = sorted(us(max(d["e"])) for k, d in st.items() if k[2] == "S1" and k[1] == "TRISC_1" and d["e"])
    land = [us(max(d["e"])) for k, d in st.items() if k[2] == "RD_S3_LAND" and d["e"]]
    p = pl["p"] if pl else "?"
    G = 4
    anti_dir = pl["route"][f"{p}>{(p + 2) % G}"] if pl else None
    print(
        f"chip dev{chip} p={p} antipodal dir={'FWD' if anti_dir == 0 else 'BWD'}: S1 end med {s1[len(s1) // 2]:.1f}"
        f"  last landing {max(land):.1f}  program end {us(end):.1f}"
    )
    ports = []
    # the profiler reports physical coords (harvesting differs per chip): map the PORT_SEND cores by x rank to the
    # logical port order (x0 + i): (FWD,0), (BWD,0), (FWD,1), (BWD,1)
    pcs = sorted({k[0] for k in st if k[2] == "PORT_SEND"}, key=lambda c: int(c.split(",")[0]))
    order = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for c, (d, l) in sorted(zip(pcs, order), key=lambda kv: kv[1]):
        g = lambda z, risc, w: st.get((c, risc, z), {}).get(w, [])
        fe = g("PORT_FENCE", "BRISC", "e")
        ss, se = g("PORT_SEND", "BRISC", "s"), g("PORT_SEND", "BRISC", "e")
        pe = g("PORT_PULL", "NCRISC", "e")
        blk = sorted(g("PS_BLK", "BRISC", "e"))
        late = d == anti_dir
        tag = f"{'FWD' if d == 0 else 'BWD'} L{l} {'late ' if late else 'early'}"
        line = f"  {tag}  fence {us(max(fe)) if fe else -1:6.1f}  send {us(min(ss)) if ss else -1:6.1f} -> "
        line += f"{us(max(se)) if se else -1:6.1f}  pull end {us(max(pe)) if pe else -1:6.1f}"
        if blk:
            line += "  blk " + " ".join(f"{us(t):.0f}" for t in blk)
        print(line)
        ports.append((tag, us(max(se)) if se else None))
    summary[chip] = dict(p=p, land=max(land), end=us(end), ports=ports)
print("PORTS_SUMMARY", json.dumps(summary))
