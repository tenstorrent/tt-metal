"""Per-entry fuse census of a fused_finish run: for the LAST program run of each device (TRISC_0), per entry index:
how many matmul cores fused (no FIN zone between this MM and the next), and the mean / max time (us from the
program start) at which the entry's finished slot was complete (MM end if fused, else the following FIN end).
Usage: python entry_zones.py <profile_log_device.csv[.gz]> [device]"""
import csv
import gzip
import sys
from collections import defaultdict

path = sys.argv[1]
dev_f = int(sys.argv[2]) if len(sys.argv) > 2 else None
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    first = f.readline()
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
hdr = [h.strip() for h in rows[0]]
ix = {h: i for i, h in enumerate(hdr)}
by_dev = defaultdict(list)
for r in rows[1:]:
    r = [c.strip() for c in r]
    by_dev[int(r[ix["PCIe slot"]])].append(r)
for dev in sorted(by_dev):
    if dev_f is not None and dev != dev_f:
        continue
    rs = by_dev[dev]
    key = lambda r: (r[ix["run host ID"]], r[ix["trace id counter"]])
    last = key(rs[-1])
    rs = [r for r in rs if key(r) == last and r[ix["RISC processor type"]] == "TRISC_0"]
    t0 = min(int(r[ix["time[cycles since reset]"]]) for r in rs)
    per_core = defaultdict(lambda: defaultdict(list))
    open_z = {}
    for r in rs:
        core = (r[ix["core_x"]], r[ix["core_y"]])
        z, ty, t = r[ix["zone name"]], r[ix["type"]], (int(r[ix["time[cycles since reset]"]]) - t0) / freq
        if ty == "ZONE_START":
            open_z[(core, z)] = t
        elif ty == "ZONE_END" and (core, z) in open_z:
            per_core[core][z].append((open_z.pop((core, z)), t))
    ent = defaultdict(lambda: {"n": 0, "fused": 0, "done": []})
    for core, zs in per_core.items():
        mms, fus, fins = zs.get("MM", []), zs.get("MM_FUSED", []), zs.get("FIN", [])
        for e, (s, t) in enumerate(mms):
            nxt = mms[e + 1][0] if e + 1 < len(mms) else 1e9
            fin_after = [b for a, b in fins if a >= t - 1e-3 and a <= nxt]
            f = not fin_after  # fused: no separate FIN pass behind this MM (a lagged entry's FIN runs later)
            done = t if f else fin_after[0]
            d = ent[e]
            d["n"] += 1
            d["fused"] += f
            d["done"].append(done)
    print(f"device {dev}:")
    for e in sorted(ent):
        d = ent[e]
        print(
            f"  entry {e}: fused {d['fused']}/{d['n']}  slot complete mean {sum(d['done'])/len(d['done']):.1f} max {max(d['done']):.1f} us"
        )
