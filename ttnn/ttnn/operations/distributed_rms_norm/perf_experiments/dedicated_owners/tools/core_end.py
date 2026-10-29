"""Per core of one chip (last run): end (us) of the last instance of each listed zone, sorted by the first zone's end.
usage: core_end.py <csv> <chip> <zones,comma> [top_n]"""
import csv, sys
from collections import defaultdict

log, chip, zones = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
top = int(sys.argv[4]) if len(sys.argv) > 4 else 15
with open(log) as fh:
    mhz = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = [r for r in csv.DictReader(fh, skipinitialspace=True) if int(r["PCIe slot"]) == chip]
rid = max(int(r["run host ID"]) for r in rows)
rows = [r for r in rows if int(r["run host ID"]) == rid]
t0 = min(int(r["time[cycles since reset]"]) for r in rows if r["zone name"].endswith("-KERNEL"))
ends = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["type"] == "ZONE_END" and r["zone name"] in zones:
        ends[(int(r["core_x"]), int(r["core_y"]))][r["zone name"]].append(
            (int(r["time[cycles since reset]"]) - t0) / mhz
        )
order = sorted(ends, key=lambda c: -max(ends[c].get(zones[0], [0])))
print("core      " + " ".join(f"{z[:24]:>24s}" for z in zones))
for c in order[:top]:
    print(f"{str(c):9s} " + " ".join(f"{','.join(f'{v:.1f}' for v in sorted(ends[c].get(z, []))):>24s}" for z in zones))
