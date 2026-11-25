# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-port timeline of the last q_stem call per chip (host-side, no device).

PORT_FENCE / PORT_PULL / PORT_SEND zones per port core, plus (QSTEM_ABLATE=PF_DIAG) the port's wait split:
  reader  PR_T_READY (block-ready waits = starved by stage 1), PR_T_RESERVE (cb_port_stage full = sender behind),
          PR_T_BARRIER (pull read barriers = NoC0 pull latency), PR_BLK (time each block became ready)
  sender  PS_T_CBWAIT (staging empty = puller behind), PS_T_SLOT (router slot wait = fabric backpressure),
          PS_T_FLUSH / PS_T_HFLUSH (writes-flushed waits = local NoC1 port -> eth-core write), PS_BLK (block sent)
Also the last S1 end and the last RD_S3_LAND / kernel end per chip.

    python3 pf_summary.py [profile_log_device.csv]
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
f = open(path)
f.readline()
rows = list(csv.DictReader(f, skipinitialspace=True))
MHZ = 1350.0
last = {}
for r in rows:
    last[r["PCIe slot"]] = max(last.get(r["PCIe slot"], 0), int(r["run host ID"]))

for chip in sorted(last):
    zs = defaultdict(dict)
    data = defaultdict(list)
    for r in rows:
        if r["PCIe slot"] != chip or int(r["run host ID"]) != last[chip]:
            continue
        c = (int(r["core_x"]), int(r["core_y"]))
        t = int(r["time[cycles since reset]"])
        z = r["zone name"]
        if r["type"] == "ZONE_START":
            d = zs[(c, r["RISC processor type"], z)]
            d["s"] = min(d.get("s", t), t)
        elif r["type"] == "ZONE_END":
            d = zs[(c, r["RISC processor type"], z)]
            d["e"] = max(d.get("e", t), t)
        else:
            data[(c, z)].append((t, int(r["data"] or 0)))
    t0 = min(d["s"] for k, d in zs.items() if k[2].endswith("KERNEL") and "s" in d)
    end = max(d["e"] for k, d in zs.items() if k[2].endswith("KERNEL") and "e" in d)
    us = lambda t: (t - t0) / MHZ
    s1 = [d["e"] for k, d in zs.items() if k[2] == "S1" and "e" in d]
    land = [d["e"] for k, d in zs.items() if k[2] == "RD_S3_LAND" and "e" in d]
    print(
        f"chip {chip}: end {us(end):6.1f}  S1 end max {us(max(s1)) if s1 else 0:6.1f}  "
        f"last landing {us(max(land)) if land else 0:6.1f}"
    )
    ports = sorted({k[0] for k in zs if k[2] in ("PORT_PULL", "PORT_SEND")})
    for c in ports:
        g = lambda z, risc: zs.get((c, risc, z), {})
        pull, send, fence = g("PORT_PULL", "NCRISC"), g("PORT_SEND", "BRISC"), g("PORT_FENCE", "BRISC")
        line = (
            f"  port {c}: fence {us(fence.get('s', t0)):5.1f}-{us(fence.get('e', t0)):5.1f}"
            f"  pull end {us(pull.get('e', t0)):6.1f}  send {us(send.get('s', t0)):5.1f}-{us(send.get('e', t0)):6.1f}"
        )
        for z in (
            "PR_T_READY",
            "PR_T_RESERVE",
            "PR_T_BARRIER",
            "PS_T_CBWAIT",
            "PS_T_SLOT",
            "PS_T_FLUSH",
            "PS_T_HFLUSH",
        ):
            if data.get((c, z)):
                line += f"  {z[5:]} {data[(c, z)][-1][1] / MHZ:5.1f}"
        print(line)
        lab = {"PR_BLK": "ready@ (res/bar/ready per blk)", "PS_BLK": "sent@ (cb/slot/flush per blk)"}
        for z in ("PR_BLK", "PS_BLK"):
            if data.get((c, z)):
                prev = (0, 0, 0)
                cells = []
                for t, v in sorted(data[(c, z)]):
                    cur = (v & 0x1FFFFF, (v >> 21) & 0x1FFFFF, (v >> 42) & 0x1FFFFF)
                    dd = [(a - b) / MHZ for a, b in zip(cur, prev)]
                    prev = cur
                    cells.append(f"{us(t):5.1f}[{dd[0]:.1f}/{dd[1]:.1f}/{dd[2]:.1f}]")
                print(f"      {z} {lab[z]}: " + " ".join(cells))
