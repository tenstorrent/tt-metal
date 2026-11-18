"""Check the op's kernel zone source locations against every location the profiler logs know (16-bit hash collisions
make the profiler throw at teardown: profiler.cpp populateZoneSrcLocations). usage: python r16_zone_hash_check.py"""
import collections, glob, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
KDIR = os.path.join(ROOT, "ttnn/ttnn/operations/genify_matmul/kernels")


def h16(s):
    v = 2166136261
    for c in s.encode():
        v = ((v ^ c) * 16777619) & 0xFFFFFFFF
    return (v & 0xFFFF) ^ (v >> 16)


locs = set()
for f in glob.glob(os.path.join(ROOT, "generated/profiler/.logs/*zone_src_locations.log")):
    for line in open(f):
        p = line.find("'#pragma message: ")
        if p >= 0:
            l = line[p + len("'#pragma message: ") :].rstrip("\n")[:-1]
            if "/genify_matmul/kernels/" not in l:
                locs.add(l)
for path in glob.glob(os.path.join(KDIR, "*")):
    for i, ln in enumerate(open(path).read().split("\n"), 1):
        m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', ln)
        if m:
            locs.add(f"{m.group(1)},{path},{i},KERNEL_PROFILER")
by = collections.defaultdict(list)
for l in locs:
    by[h16(l)].append(l)
bad = [v for v in by.values() if len(v) > 1]
print("collisions:", bad if bad else "none")
