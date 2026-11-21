"""Host-only: find 16-bit profiler zone-hash collisions (tt_metal profiler.cpp hash16CT) among the zone source locations
of the given kernel dirs (MaybeDeviceZoneScope / DeviceZoneScopedN call sites), so two op copies can share one
profiled session. Usage: python3 -I zone_hash_check.py <kernel_dir> ..."""
import re
import sys
from pathlib import Path


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


locs = {}
for d in sys.argv[1:]:
    for f in sorted(Path(d).resolve().glob("*.[ch]pp")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in re.finditer(r'(?:MaybeDeviceZoneScope|DeviceZoneScopedN)\("([^"]+)"\)', line):
                s = f"{m.group(1)},{f},{i},KERNEL_PROFILER"
                locs[s] = h16(s)
by = {}
for s, h in locs.items():
    by.setdefault(h, []).append(s)
bad = {h: v for h, v in by.items() if len(v) > 1}
print(f"{len(locs)} locations, {len(bad)} colliding hashes")
for h, v in bad.items():
    print(hex(h), *v, sep="\n  ")
