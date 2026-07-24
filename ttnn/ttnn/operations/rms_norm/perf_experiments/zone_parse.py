#!/usr/bin/env python3
"""Parse profile_log_device.csv -> per-zone device-ns, per core, aggregated.

For each (core, RISC, zone name) we pair ZONE_START/ZONE_END in order and sum the
per-invocation cycle deltas into a per-trial total. We report, per zone:
  - the CRITICAL core (max total cycles across cores) — the whole-op critical path
  - median per-invocation ns and total ns
Cycles -> ns via CHIP_FREQ (line 1 of the CSV).

Usage: python3 zone_parse.py <profile_log_device.csv> [zone_substr_filter]
"""
import sys
import csv
from collections import defaultdict

path = sys.argv[1]
zfilter = sys.argv[2] if len(sys.argv) > 2 else ""

with open(path) as f:
    first = f.readline()
    freq_mhz = 1350.0
    for tok in first.split(","):
        if "CHIP_FREQ" in tok:
            try:
                freq_mhz = float(tok.split(":")[1].strip())
            except Exception:
                pass
    reader = csv.reader(f)
    header = next(reader)  # column header row
    rows = list(reader)

# columns (0-based): 1=core_x 2=core_y 3=risc 4=timer_id 5=cycles 10=zone name 11=type
# pair START/END per (core,risc,zone) as a stack
open_stack = defaultdict(list)
# per (core,risc,zone) -> list of per-invocation cycle deltas
invos = defaultdict(list)

for r in rows:
    if len(r) < 12:
        continue
    core = (r[1].strip(), r[2].strip())
    risc = r[3].strip()
    try:
        cyc = int(r[5].strip())
    except Exception:
        continue
    zname = r[10].strip()
    ztype = r[11].strip()
    key = (core, risc, zname)
    if ztype == "ZONE_START":
        open_stack[key].append(cyc)
    elif ztype == "ZONE_END":
        if open_stack[key]:
            start = open_stack[key].pop()
            invos[key].append(cyc - start)

# Aggregate per zone across cores: for each zone, per core sum the cycle deltas
# (that core's total time in the zone across all invocations in this run), pick
# the max-core = critical path.
zone_core_total = defaultdict(dict)  # zname -> core -> total cycles
zone_core_count = defaultdict(dict)
for (core, risc, zname), deltas in invos.items():
    if zfilter and zfilter not in zname:
        continue
    zone_core_total[zname][core] = sum(deltas)
    zone_core_count[zname][core] = len(deltas)


def c2ns(c):
    return c / freq_mhz * 1000.0


print(f"CHIP_FREQ = {freq_mhz} MHz")
print(f"{'zone':<16} {'crit_core':<10} {'crit_total_ns':>14} {'n_invos':>8} {'ns/invo':>10}")
for zname in sorted(zone_core_total):
    cores = zone_core_total[zname]
    crit_core = max(cores, key=cores.get)
    total_c = cores[crit_core]
    n = zone_core_count[zname][crit_core]
    per = c2ns(total_c) / n if n else 0.0
    print(f"{zname:<16} {str(crit_core):<10} {c2ns(total_c):>14.0f} {n:>8} {per:>10.1f}")
