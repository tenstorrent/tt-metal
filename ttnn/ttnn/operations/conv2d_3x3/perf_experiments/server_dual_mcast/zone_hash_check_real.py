# usage: zone_hash_check_real.py : the profiler's 16-bit zone hashes of the real op's kernels as they will read after
# graduation.patch (op_grad/'s line numbers under the real kernel paths); reports collisions.
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
here = root + "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/server_dual_mcast/op_grad/kernels/"
real = root + "ttnn/ttnn/operations/conv2d_3x3/kernels/"
locs = set()
for f in glob.glob(here + "*"):
    for i, l in enumerate(open(f), 1):
        m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', l)
        if m:
            locs.add(f"{m.group(1)},{real + os.path.basename(f)},{i},KERNEL_PROFILER")
seen = {}
for s in locs:
    seen.setdefault(h16(s), []).append(s)
bad = [v for v in seen.values() if len(v) > 1]
for v in bad:
    print("COLLIDE", v)
print(len(locs), "locations,", len(bad), "collisions")
sys.exit(1 if bad else 0)
