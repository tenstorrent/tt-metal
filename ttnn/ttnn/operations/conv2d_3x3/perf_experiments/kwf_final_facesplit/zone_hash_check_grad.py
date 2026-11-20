# usage: zone_hash_check_grad.py : the graduation's zone source locations, hashed as the profiler does, at the REAL op's
# kernel paths (op_grad/ contents for the files graduation.patch touches, the real files otherwise), plus the
# firmware / dispatch locations of the last profiled run; reports 16-bit collisions.
import glob
import os
import re

root = "/localdev/pmilojevic/workspaces/opgen/"
real = root + "ttnn/ttnn/operations/conv2d_3x3/kernels/"
grad = root + "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_final_facesplit/op_grad/kernels/"


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


locs = set()
for f in glob.glob(real + "*"):
    name = os.path.basename(f)
    src = grad + name if os.path.exists(grad + name) else f
    for i, l in enumerate(open(src), 1):
        m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', l)
        if m:
            locs.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
for n in ("new_zone_src_locations.log", "zone_src_locations.log"):
    p = root + "generated/profiler/.logs/" + n
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            k = line.find("'#pragma message: ")
            if k >= 0:
                s = line[k + 18 :].rstrip("\n")[:-1]
                if "/conv2d_3x3/kernels/" not in s:  # the real op's own (pre-patch) locations are replaced above
                    locs.add(s)
seen = {}
for s in locs:
    seen.setdefault(h16(s), []).append(s)
bad = [v for v in seen.values() if len(v) > 1]
for v in bad:
    print("COLLIDE", v)
print(len(locs), "locations,", len(bad), "collisions")
