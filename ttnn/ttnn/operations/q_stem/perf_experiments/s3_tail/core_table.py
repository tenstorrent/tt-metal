"""Per-core end times of the last call per chip, split by core group (host script, no device).

    python3 core_table.py zones.csv [W1 R1]
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
W1 = int(sys.argv[2]) if len(sys.argv) > 2 else 9
R1 = int(sys.argv[3]) if len(sys.argv) > 3 else 8
f = open(path)
f.readline()
rows = list(csv.DictReader(f, skipinitialspace=True))
MHZ = 1350.0
last = {}
for r in rows:
    last[r["PCIe slot"]] = max(last.get(r["PCIe slot"], 0), int(r["run host ID"]))
for chip in sorted(last):
    st = defaultdict(dict)
    for r in rows:
        if r["PCIe slot"] != chip or int(r["run host ID"]) != last[chip]:
            continue
        c = (int(r["core_x"]), int(r["core_y"]))
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
    # physical coords; the stage-1 cores are those with an S1 zone
    s1 = [c for c in per if ("S1", "TRISC_1") in per[c]]
    s3 = [c for c in per if ("S3", "TRISC_1") in per[c] and c not in s1]
    print(f"chip {chip}")
    for name, grp in (("stage-1", s1), ("s3-only", s3)):
        ends = sorted(
            (per[c][("WR_S3_OUT", "BRISC")][1], per[c][("RD_S3_LAND", "NCRISC")][1], per[c][("S3", "TRISC_1")][0], c)
            for c in grp
            if ("WR_S3_OUT", "BRISC") in per[c]
        )
        if not ends:
            continue
        n = len(ends)
        q = lambda i: ends[min(n - 1, int(i * n))]
        print(
            f"  {name:8s} n={n:3d} out_end min {ends[0][0]:6.1f} med {q(0.5)[0]:6.1f} p90 {q(0.9)[0]:6.1f} max {ends[-1][0]:6.1f}"
        )
        for e in ends[-6:]:
            print(
                f"     core {e[3]} out_end {e[0]:6.1f} last_land {e[1]:6.1f} s3_start {e[2]:6.1f} s3dur {e[0]-max(e[1],e[2]):.1f}"
            )
