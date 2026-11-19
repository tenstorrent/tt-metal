# usage: zone_hash_check_grad.py : the zone-location hash check of ablation_r12/zone_hash_check.py, for op_copy's kernels
# as they would sit in the real op dir after graduation.patch (the profiler hashes "name,file,line").
import glob, os, re, sys


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


root = "/localdev/pmilojevic/workspaces/opgen/"
real = root + "ttnn/ttnn/operations/conv2d_3x3/kernels/"
here = os.path.dirname(os.path.abspath(__file__))
locs = set()
for f in glob.glob(here + "/op_copy/kernels/*"):
    for i, l in enumerate(open(f), 1):
        m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', l)
        if m:
            locs.add(f"{m.group(1)},{real + os.path.basename(f)},{i},KERNEL_PROFILER")
_logs = [
    root + "generated/profiler/.logs/new_zone_src_locations.log",
    root + "generated/profiler/.logs/zone_src_locations.log",
]
for line in [l for p in _logs if os.path.exists(p) for l in open(p, errors="replace").readlines()]:
    p = line.find("'#pragma message: ")
    if p >= 0 and "conv2d_3x3/kernels" not in line and "kwf_first_subblock" not in line:
        locs.add(line[p + 18 :].rstrip("\n")[:-1])
seen = {}
for s in locs:
    seen.setdefault(h16(s), []).append(s)
bad = [v for v in seen.values() if len(v) > 1]
for v in bad:
    print("COLLIDE", v)
print(len(locs), "locations,", len(bad), "collisions")
sys.exit(1 if bad else 0)
