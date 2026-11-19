# usage: zone_hash_check.py <op dir>...   The device profiler keys each zone by a 16-bit hash of
# "<name>,<abs path>,<line>,KERNEL_PROFILER" and throws on a collision; this lists the zones of the given op copies
# (plus the persistent zone log) and prints any colliding pair, so a copy directory can be renamed before a run.
import collections, re, sys
from pathlib import Path


def h16(s):
    b = 2166136261
    for ch in s.encode():
        b = ((b ^ ch) * 16777619) & 0xFFFFFFFF
    return ((b >> 16) ^ b) & 0xFFFF


locs = set()
import os

log = (
    Path(os.environ.get("TT_METAL_PROFILER_DIR", "/localdev/pmilojevic/workspaces/opgen/generated/profiler"))
    / ".logs/zone_src_locations.log"
)
for line in log.read_text().splitlines() if log.exists() else []:
    p = line.find("'#pragma message: ")
    if p >= 0:
        locs.add(line[p + len("'#pragma message: ") :][:-1])
for d in sys.argv[1:]:
    for f in sorted(Path(d).resolve().glob("kernels/*")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in re.finditer(r'(?:MaybeDeviceZoneScope|DeviceZoneScopedN)\("([^"]+)"\)', line):
                locs.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
by = collections.defaultdict(list)
for l in locs:
    by[h16(l)].append(l)
bad = [v for v in by.values() if len(v) > 1]
print(len(locs), "locations,", len(bad), "collisions")
for v in bad:
    print("  ", v)
