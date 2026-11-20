# usage: zone_hash_check.py [kernel dirs...]: hash every MaybeDeviceZoneScope location of the op's kernels as the profiler does; report collisions
import re, glob, sys


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


root = "/localdev/pmilojevic/workspaces/opgen/"
locs = set()
dirs = sys.argv[1:] or [root + "ttnn/ttnn/operations/conv2d_3x3/kernels"]
for f in [f for d in dirs for f in glob.glob(d.rstrip("/") + "/*")]:
    for i, l in enumerate(open(f), 1):
        m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', l)
        if m:
            locs.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
# plus whatever else got compiled (firmware / dispatch) from the last log
import os

logs = []  # this dir runs under its own TT_METAL_PROFILER_DIR: only its own logs (run.sh copies them) matter
logs += glob.glob("/tmp/r0_zone_logs/*.log")
for line in [l for p in logs if os.path.exists(p) for l in open(p, errors="replace")]:
    p = line.find("'#pragma message: ")
    if p >= 0:
        locs.add(line[p + 18 :].rstrip("\n")[:-1])
seen = {}
for s in locs:
    seen.setdefault(h16(s), []).append(s)
bad = [v for v in seen.values() if len(v) > 1]
for v in bad:
    print("COLLIDE", v)
print(len(locs), "locations,", len(bad), "collisions")
