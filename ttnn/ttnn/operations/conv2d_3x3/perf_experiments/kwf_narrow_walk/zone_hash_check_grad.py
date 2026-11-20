# usage: zone_hash_check_grad.py : the real op's zone locations with graduation.patch applied (op_grad's kw_fold.hpp at
# the real path) plus the logged firmware / dispatch locations; reports 16-bit hash collisions.
import glob, os, re


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


root = "/localdev/pmilojevic/workspaces/opgen/"
real = root + "ttnn/ttnn/operations/conv2d_3x3/kernels/"
grad = root + "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_narrow_walk/op_grad/kernels/"
locs = set()
for f in glob.glob(real + "*"):
    src = grad + os.path.basename(f)
    for i, l in enumerate(open(src), 1):
        m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', l)
        if m:
            locs.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
for n in ("new_zone_src_locations.log", "zone_src_locations.log"):
    p = root + "generated/profiler/.logs/" + n
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            q = line.find("'#pragma message: ")
            if q >= 0:
                s = line[q + 18 :].rstrip("\n")[:-1]
                if "/perf_experiments/" not in s and "/conv2d_3x3/kernels/" not in s:
                    locs.add(s)
seen = {}
for s in locs:
    seen.setdefault(h16(s), []).append(s)
bad = [v for v in seen.values() if len(v) > 1]
for v in bad:
    print("COLLIDE", v)
print(len(locs), "locations (real op with graduation.patch),", len(bad), "collisions")
