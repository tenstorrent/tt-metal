# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device profiler keys zones by a 16-bit hash of "name,path,line,KERNEL_PROFILER" and throws on a collision among
every location compiled in one session, firmware included. Lists the kernel zones of the given op copies, adds every
location the last profiled session's compiles reported (generated/profiler/.logs/*zone_src_locations.log: firmware,
dispatch, other ops; their zone names often come from macros a source scan misses), and reports colliding pairs that
involve an op zone.
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
mine = set()
for d in sys.argv[1:]:
    for f in sorted(pathlib.Path(d).resolve().glob("kernels/*")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in pat.finditer(line):
                mine.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
logs = pathlib.Path(__file__).resolve().parents[6] / "generated/profiler/.logs"
other = set()
for log in logs.glob("*zone_src_locations.log"):
    for line in log.read_text(errors="ignore").splitlines():
        p = line.find("'#pragma message: ")
        if p >= 0:
            loc = line[p + len("'#pragma message: ") : -1]
            if "/conv2d_3x3/" not in loc:
                other.add(loc)
by = collections.defaultdict(list)
for loc in mine | other:
    by[h16(loc)].append(loc)
bad = [v for v in by.values() if len(v) > 1 and any(x in mine for x in v)]
print(len(mine), "op zone locations,", len(other), "others from the last session's logs,", len(bad), "collisions")
for v in bad:
    for loc in v:
        print(" ", loc)
