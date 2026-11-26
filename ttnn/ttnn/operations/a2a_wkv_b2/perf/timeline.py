import csv, sys
from collections import defaultdict

rows = list(csv.reader(open("generated/profiler/.logs/profile_log_device.csv")))[2:]
runs = defaultdict(int)
for r in rows:
    runs[r[0]] = max(runs[r[0]], int(r[7]))
chip = sys.argv[1]
st = min(int(r[5]) for r in rows if r[0] == chip and int(r[7]) == runs[chip])
for core in [("12", "11"), ("13", "11")]:
    ev = [
        (int(r[5]), r[10].strip(), r[11].strip(), r[3].strip())
        for r in rows
        if r[0] == chip
        and int(r[7]) == runs[chip]
        and (r[1].strip(), r[2].strip()) == core
        and r[10].strip().startswith("TX")
    ]
    ev.sort()
    print(
        core,
        " ".join(
            f"{z}{'[' if t=='ZONE_START' else ']'}{(c-st)/1350:.1f}" for c, z, t, _ in ev if z in ("TX_SEND", "TX_POLL")
        ),
    )
