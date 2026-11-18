# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Profiler zone-location hash check (no ttnn): every MaybeDeviceZoneScope / DeviceZoneScopedN in the given kernel dirs,
as the profiler names it ("zone,abs file,line,KERNEL_PROFILER"), against each other and against the locations of a
previous run's new_zone_src_locations.log (firmware zones). A 16-bit collision aborts a profiled run.

    python3 zone_hash_check.py <new_zone_src_locations.log> <kernel dir>[=<path the profiler will see>] [...]
"""
import collections
import re
import sys
from pathlib import Path


def h16(s):
    r = 2166136261
    for ch in s.encode():
        r = ((r ^ ch) * 16777619) & 0xFFFFFFFF
    return ((r & 0xFFFF) ^ (r >> 16)) & 0xFFFF


locs = set()
for line in open(sys.argv[1]):
    d = "'#pragma message: "
    p = line.find(d)
    if p >= 0:
        loc = line[p + len(d) :].rstrip("\n").rstrip("'")
        if "/kernels/" not in loc:
            locs.add(loc)
pat = re.compile(r'(?:MaybeDeviceZoneScope|DeviceZoneScopedN)\("([^"]+)"\)')
for arg in sys.argv[2:]:
    kdir, _, as_dir = arg.partition("=")
    for f in sorted(Path(kdir).resolve().glob("*.[ch]pp")):
        name = Path(as_dir).resolve() / f.name if as_dir else f
        for i, text in enumerate(f.read_text().splitlines(), 1):
            for m in pat.finditer(text):
                locs.add(f"{m.group(1)},{name},{i},KERNEL_PROFILER")
by = collections.defaultdict(set)
for loc in locs:
    by[h16(loc)].add(loc)
bad = {k: v for k, v in by.items() if len(v) > 1}
for k, v in bad.items():
    print(hex(k), sorted(v))
print(f"{len(locs)} locations, {len(bad)} collisions")
sys.exit(1 if bad else 0)
