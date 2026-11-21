# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device profiler keys zones by a 16-bit hash of "name,path,line,KERNEL_PROFILER" and throws on a collision among
every location compiled in one session: list the kernel zones of the given op copies and report colliding pairs.
zone_hash_check.py <op dir> [<op dir> ...]"""
import collections
import pathlib
import re
import sys


def h16(s):
    b = 2166136261
    for ch in s.encode():
        b = ((b ^ ch) * 16777619) & 0xFFFFFFFF
    return ((b & 0xFFFF) ^ (b >> 16)) & 0xFFFF


pat = re.compile(r'(?:MaybeDeviceZoneScope|DeviceZoneScopedN)\("([^"]+)"\)')
locs = set()
for d in sys.argv[1:]:
    for f in sorted(pathlib.Path(d).resolve().glob("kernels/*")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in pat.finditer(line):
                locs.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
by = collections.defaultdict(list)
for loc in locs:
    by[h16(loc)].append(loc)
bad = [v for v in by.values() if len(v) > 1]
print(len(locs), "zone locations,", len(bad), "collisions")
for v in bad:
    print("  " + "\n  ".join(v))
