"""Usage: tl.py <profile_log_device.csv> <run index (0-based, op order)> [full]
Busiest core: per-zone timeline (us from kernel start); all cores: W delivery end, compute_wait_x max."""
import csv, sys
from collections import defaultdict

f = open(sys.argv[1])
first = f.readline()
freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rows = list(csv.DictReader(f, skipinitialspace=True))
ids = sorted({int(r["run host ID"]) for r in rows})
rid = ids[int(sys.argv[2])]
rows = [r for r in rows if int(r["run host ID"]) == rid]
t0 = min(int(r["time[cycles since reset]"]) for r in rows)
ev = defaultdict(list)
op = {}
for r in rows:
    k = ((r["core_x"], r["core_y"]), r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        op[k] = t
    elif k in op:
        ev[k[0]].append((k[1], k[2], (op.pop(k) - t0) / freq, (t - t0) / freq))
end = {
    c: max(e[3] for e in v if e[1] == "TRISC-KERNEL") for c, v in ev.items() if any(e[1] == "TRISC-KERNEL" for e in v)
}
c = max(end, key=end.get)
print("run", rid, "busiest", c, "end", round(end[c], 2))
wend = {cc: max([e[3] for e in v if e[1] in ("reader_w_deliver",)] or [0]) for cc, v in ev.items()}
wx = {cc: sum(e[3] - e[2] for e in v if e[1] == "compute_wait_x") for cc, v in ev.items()}
print(
    " W deliver end: max %.2f  min %.2f | compute_wait_x per core max %.2f"
    % (max(wend.values()), min(v for v in wend.values() if v), max(wx.values()))
)
for e in sorted(ev[c], key=lambda e: (e[0], e[2])):
    if "FW" in e[1] or "KERNEL" in e[1]:
        continue
    if len(sys.argv) < 4 and e[1] in ("compute_matmul_gelu",) and e[0] != "TRISC_0":
        continue
    print(f"  {e[0]:8s} {e[1]:22s} {e[2]:7.2f} -> {e[3]:7.2f} ({e[3]-e[2]:6.2f})")
