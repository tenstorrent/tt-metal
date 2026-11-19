# usage: zone_hash_check_real.py [pkg]: hash every zone location of <pkg>'s kernels (default op_copy) as the profiler
# would under the REAL op's kernel paths (ttnn/ttnn/operations/conv2d_3x3/kernels/<file>, i.e. after graduation.patch);
# report 16-bit hash collisions (plus firmware locations from the default profiler logs, if present).
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
pkg = sys.argv[1] if len(sys.argv) > 1 else "op_copy"
here = os.path.dirname(os.path.abspath(__file__))
locs = set()
for f in glob.glob(os.path.join(here, pkg, "kernels", "*")):
    real = root + "ttnn/ttnn/operations/conv2d_3x3/kernels/" + os.path.basename(f)
    for i, l in enumerate(open(f), 1):
        m = re.search(r'(?:MaybeDeviceZoneScope|DeviceZoneScopedN)\("([^"]+)"\)', l)
        if m:
            locs.add(f"{m.group(1)},{real},{i},KERNEL_PROFILER")
for log in (
    root + "generated/profiler/.logs/new_zone_src_locations.log",
    root + "generated/profiler/.logs/zone_src_locations.log",
):
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            p = line.find("'#pragma message: ")
            if p >= 0 and "conv2d_3x3/kernels" not in line:
                locs.add(line[p + 18 :].rstrip("\n")[:-1])
seen = {}
for s in locs:
    seen.setdefault(h16(s), []).append(s)
bad = [v for v in seen.values() if len(v) > 1]
for v in bad:
    print("COLLIDE", v)
print(len(locs), "locations,", len(bad), "collisions")
