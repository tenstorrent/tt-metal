"""usage: python analyze.py <report_dir> [label1,label2,...] [--grid N]
Per profiled op (collection order): DEVICE KERNEL DURATION [ns], DEVICE FW DURATION [ns], and per RISC-V (BRISC writer /
NCRISC reader) the *-KERNEL end time (cycles from the op's first kernel start) p50 / max across
Tensix cores. --grid N prints the BRISC/NCRISC end-time grid (NoC0 coords) of op N (0-based)."""
import csv, glob, os, statistics, sys
from collections import defaultdict

rep = sys.argv[1]
labels = sys.argv[2].split(",") if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else []
grid_op = int(sys.argv[sys.argv.index("--grid") + 1]) if "--grid" in sys.argv else None
ops = list(csv.DictReader(open(glob.glob(os.path.join(rep, "ops_perf_results*.csv"))[0])))
with open(os.path.join(rep, "profile_log_device.csv")) as f:
    next(f)
    rd = csv.reader(f)
    hdr = [h.strip() for h in next(rd)]
    data = [dict(zip(hdr, [c.strip() for c in r])) for r in rd]
by = defaultdict(lambda: ({}, {}))
for d in data:
    if not d["zone name"].endswith("-KERNEL"):
        continue
    st, en = by[int(d["run host ID"])]
    k = (int(d["core_x"]), int(d["core_y"]), d["RISC processor type"])
    (st if d["type"] == "ZONE_START" else en)[k] = int(d["time[cycles since reset]"])
ids = sorted(by)
tilize_ops = [r for r in ops if "tilize" in r["OP CODE"].lower() or "generic" in r["OP CODE"].lower()]
print(
    f"{'#':>2} {'label':18s} {'dev ns':>7s} {'FW ns':>7s} {'BR p50':>7s} {'BR max':>7s} {'NC p50':>7s} {'NC max':>7s} {'st p50':>6s} {'st max':>6s} {'BRdur p50':>9s} {'BRdur max':>9s}"
)
for i, r in enumerate(tilize_ops):
    hid = int(r["GLOBAL CALL COUNT"])
    st, en = by.get(hid, ({}, {}))
    lab = labels[i] if i < len(labels) else ""
    if not st:
        print(f"{i:2d} {lab:18s} {r['DEVICE KERNEL DURATION [ns]']:>7s} {r['DEVICE FW DURATION [ns]']:>7s}")
        continue
    t0 = min(st.values())
    ends = defaultdict(list)
    for k, v in en.items():
        ends[k[2]].append(v - t0)
    b, n = ends.get("BRISC", [0]), ends.get("NCRISC", [0])
    starts = [v - t0 for k, v in st.items()]
    bdur = [en[k] - st[k] for k in st if k[2] == "BRISC" and k in en]
    print(
        f"{i:2d} {lab:18s} {r['DEVICE KERNEL DURATION [ns]']:>7s} {r['DEVICE FW DURATION [ns]']:>7s} {statistics.median(b):7.0f} {max(b):7d} "
        f"{statistics.median(n):7.0f} {max(n):7d} {statistics.median(starts):6.0f} {max(starts):6d} "
        f"{statistics.median(bdur):9.0f} {max(bdur):9d}"
    )
    if grid_op == i:
        for risc in ("NCRISC", "BRISC"):
            g = defaultdict(dict)
            for (x, y, rr), v in en.items():
                if rr == risc:
                    g[y][x] = v - t0
            xs = sorted({x for y in g for x in g[y]})
            print(f"  {risc} end (cycles), rows = NoC0 y, cols = NoC0 x")
            print("  y\\x " + " ".join(f"{x:>6d}" for x in xs))
            for y in sorted(g):
                print(f"  {y:3d} " + " ".join(f"{g[y].get(x, 0):6d}" for x in xs))
