"""Per-core combine-tail timeline from generated/profiler/.logs/profile_log_device.csv (last program only).
Prints, per core that ran a combine zone, the zone [start,end] in us from the program's first FW start,
plus the last writer_ship_partial end over all cores and the kernel end. Usage: timeline.py [csv] [--all]"""
import collections, csv, sys

path = next((a for a in sys.argv[1:] if not a.startswith("--")), "generated/profiler/.logs/profile_log_device.csv")
with open(path) as fh:
    freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
# keep only the last run id
rid = max(int(r["run host ID"]) for r in rows) if "run host ID" in rows[0] else None
if rid is not None:
    rows = [r for r in rows if int(r["run host ID"]) == rid]
us = lambda c: c / freq
starts = {}
z = collections.defaultdict(list)  # (core, zone) -> [(s,e)]
t0 = None
for r in rows:
    core = (int(r["core_x"]), int(r["core_y"]))
    key = (core, r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        starts[key] = t
        if r["zone name"].endswith("-FW"):
            t0 = t if t0 is None else min(t0, t)
    elif r["type"] == "ZONE_END" and key in starts:
        z[(core, r["zone name"])].append((starts.pop(key), t))
f = lambda c: f"{us(c - t0):7.2f}"
ship = {c: v[-1][1] for (c, n), v in z.items() if n == "writer_ship_partial"}
kend = max(e for (c, n), v in z.items() if n.endswith("-FW") for s, e in v)
print(f"kernel end {f(kend)} us; last ship {f(max(ship.values()))} us; ship spread {f(min(ship.values()))}..")
interesting = [
    "writer_ship_partial",
    "compute_lead_combine",
    "writer_lead_wait",
    "writer_lead_send",
    "reader_collect",
    "compute_root_combine",
    "writer_stats_wait",
    "writer_stats",
    "writer_hc_wait",
    "writer_hc_out",
    "writer_hc_tail",
    "BRISC-FW",
    "TRISC-FW",
    "NCRISC-FW",
]
cores = sorted({c for (c, n) in z if n in ("compute_lead_combine", "compute_root_combine")})
for c in cores:
    parts = []
    for n in interesting:
        for s, e in z.get((c, n), [])[-3:]:
            parts.append(f"{n}[{f(s)},{f(e)}]")
    print(c, " ".join(parts))
if "--all" in sys.argv:
    for c, t in sorted(ship.items(), key=lambda kv: kv[1]):
        print("ship", c, f(t))
