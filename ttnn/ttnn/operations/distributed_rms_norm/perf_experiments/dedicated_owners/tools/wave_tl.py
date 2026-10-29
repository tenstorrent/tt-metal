"""Per chip, per (zone, RISC, instance index): start min / end median / end max over cores (us from the chip's
earliest kernel start), for the last run per chip of a profile_log_device.csv.
usage: wave_tl.py <csv> [chip] [zone_prefixes,comma] [run_index_from_end=1]"""
import csv, sys, statistics
from collections import defaultdict

log = sys.argv[1]
chip_sel = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "all" else None
pref = [p for p in (sys.argv[3] if len(sys.argv) > 3 else "").split(",") if p]
back = int(sys.argv[4]) if len(sys.argv) > 4 else 1
with open(log) as fh:
    mhz = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
by = defaultdict(list)
for r in rows:
    by[(int(r["PCIe slot"]), int(r["run host ID"]))].append(r)
for chip in sorted({c for c, _ in by}):
    if chip_sel is not None and chip != chip_sel:
        continue
    rid = sorted(rr for c, rr in by if c == chip)[-back]
    stack = defaultdict(list)
    inst = defaultdict(list)
    for r in by[(chip, rid)]:
        key = (r["core_x"], r["core_y"], r["RISC processor type"], r["zone name"])
        t = int(r["time[cycles since reset]"])
        if r["type"] == "ZONE_START":
            stack[key].append(t)
        elif r["type"] == "ZONE_END" and stack[key]:
            inst[key].append((stack[key].pop(), t))
    t0 = min(s for k, v in inst.items() if k[3].endswith("-KERNEL") for s, _ in v)
    ns = lambda c: (c - t0) * 1000 / mhz
    agg = defaultdict(list)
    for (x, y, risc, z), v in inst.items():
        if z.endswith("-FW") or (pref and not any(z.startswith(p) for p in pref)):
            continue
        v.sort()
        for i, (s, e) in enumerate(v):
            agg[(z, risc, i)].append((ns(s), ns(e), e - s))
    print(f"=== chip {chip} (run {rid})")
    for (z, risc, i), v in sorted(agg.items(), key=lambda kv: statistics.median(e for _, e, _ in kv[1])):
        print(
            f"{z:16s} {risc:7s} #{i} n={len(v):3d} start[min {min(s for s,_,_ in v):7.0f}] "
            f"end[med {statistics.median(e for _,e,_ in v):7.0f} max {max(e for _,e,_ in v):7.0f}] "
            f"dur[med {statistics.median(d for *_,d in v)*1000/mhz:6.0f}]"
        )
