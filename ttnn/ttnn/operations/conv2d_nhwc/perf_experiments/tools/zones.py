"""Summarize generated/profiler/.logs/profile_log_device.csv per program (run host ID).

Usage: python3 -I zones.py [csv] [--last]
Prints, per program: op span (min KERNEL start -> max KERNEL end over all cores/RISCs, cycles and ns),
per-core-max KERNEL duration per RISC, and per user zone x RISC: cores, mean count/core,
mean and max total cycles/core, plus marker counts per (core,RISC) max (budget check: 250).
"""
import csv, sys, collections

path = "generated/profiler/.logs/profile_log_device.csv"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if args:
    path = args[0]
last_only = "--last" in sys.argv
with open(path) as fh:
    hdr = fh.readline()
    freq = 1350.0
    if "CHIP_FREQ[MHz]:" in hdr:
        freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(fh))
cols = [c.strip() for c in rows[0]]
idx = {c: i for i, c in enumerate(cols)}
prog = collections.defaultdict(list)
for r in rows[1:]:
    if len(r) < len(cols):
        continue
    prog[r[idx["run host ID"]].strip()].append(r)
keys = sorted(prog, key=lambda k: int(k) if k.isdigit() else 0)
if last_only:
    keys = keys[-1:]
for pid in keys:
    rs = prog[pid]
    stacks = collections.defaultdict(list)
    zone_tot = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))  # (zone,risc)->core->[cnt,cyc]
    kern = {}
    markers = collections.Counter()
    for r in rs:
        core = (r[idx["core_x"]].strip(), r[idx["core_y"]].strip())
        risc = r[idx["RISC processor type"]].strip()
        name = r[idx["zone name"]].strip()
        typ = r[idx["type"]].strip()
        t = int(r[idx["time[cycles since reset]"]])
        markers[(core, risc)] += 1
        key = (core, risc, name)
        if typ == "ZONE_START":
            stacks[key].append(t)
        elif typ == "ZONE_END" and stacks[key]:
            t0 = stacks[key].pop()
            if name.endswith("-KERNEL"):
                kern[(core, risc)] = (t0, t)
            elif not name.endswith("-FW"):
                z = zone_tot[(name, risc)][core]
                z[0] += 1
                z[1] += t - t0
    if not kern:
        continue
    s0 = min(v[0] for v in kern.values())
    s1 = max(v[1] for v in kern.values())
    print(f"== program {pid}: cores={len(set(c for c,_ in kern))} span={s1-s0} cyc = {(s1-s0)/freq*1000:.0f} ns")
    per_risc = collections.defaultdict(list)
    for (core, risc), (a, b) in kern.items():
        per_risc[risc].append(b - a)
    for risc, v in sorted(per_risc.items()):
        v.sort()
        print(f"  KERNEL {risc:7s} max={v[-1]} p50={v[len(v)//2]} min={v[0]}")
    for (name, risc), d in sorted(zone_tot.items(), key=lambda kv: -max(x[1] for x in kv[1].values())):
        tots = [x[1] for x in d.values()]
        cnts = [x[0] for x in d.values()]
        print(
            f"  {name:28s} {risc:7s} cores={len(d):3d} cnt/core={sum(cnts)/len(cnts):6.1f} mean={sum(tots)/len(tots):9.0f} max={max(tots):9d}"
        )
    print(f"  max markers per (core,RISC) = {max(markers.values())}")
