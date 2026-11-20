# usage: zone_hash_check.py : hash every MaybeDeviceZoneScope location of this experiment's op copies
# as the profiler does; report collisions.
import glob
import os
import re
import sys


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


root = "/localdev/pmilojevic/workspaces/opgen/"
# The op copies compiled in one session (SDM_ZONE_DIRS, default op_base,op_copy): copies never compiled together
# cannot collide in the profiler's per-session map.
pats = [
    root + f"ttnn/ttnn/operations/conv2d_3x3/perf_experiments/server_dual_mcast/{d}/kernels/*"
    for d in os.environ.get("SDM_ZONE_DIRS", "op_base,op_copy").split(",")
]
locs = set()
for f in [f for p in pats for f in glob.glob(p)]:
    for i, l in enumerate(open(f), 1):
        m = re.search(r'(?:MaybeDeviceZoneScope|DeviceZoneScopedN)\("([^"]+)"\)', l)
        if m:
            locs.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
_logs = [
    root + "generated/profiler/.logs/new_zone_src_locations.log",
    root + "generated/profiler/.logs/zone_src_locations.log",
]
for line in [l for p in _logs if os.path.exists(p) for l in open(p, errors="replace").readlines()]:
    p = line.find("'#pragma message: ")
    # Firmware / dispatch zones of the last session; other op kernels (a sibling's session) are never compiled in ours
    # (python -m tracy clears the logs at session start).
    if p >= 0 and "/operations/" not in line:
        locs.add(line[p + 18 :].rstrip("\n")[:-1])
seen = {}
for s in locs:
    seen.setdefault(h16(s), []).append(s)
bad = [v for v in seen.values() if len(v) > 1]
for v in bad:
    print("COLLIDE", v)
print(len(locs), "locations,", len(bad), "collisions")
sys.exit(1 if bad else 0)
