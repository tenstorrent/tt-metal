# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: 16-bit profiler zone-hash collisions between the op_base and op_copy kernels (the device profiler throws
"Source location hashes are colliding" when two zone source locations of one run share a hash). Scans every
MaybeDeviceZoneScope / DeviceZoneScopedN in both kernel dirs, plus the locations of a run log (`python3 -I
zone_hash_check.py [run.log]`)."""

import collections
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent


def h16(s):
    r = 2166136261
    for c in s.encode():
        r = ((r ^ c) * 16777619) & 0xFFFFFFFF
    return ((r & 0xFFFF) ^ (r >> 16)) & 0xFFFF


locs = set()
pat = re.compile(r'(?:MaybeDeviceZoneScope|DeviceZoneScopedN)\("([^"]+)"\)')
for d in sorted(p.name for p in HERE.glob("op_*") if (p / "kernels").is_dir()):
    for f in sorted((HERE / d / "kernels").iterdir()):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "#define" in line:
                continue
            for m in pat.finditer(line):
                locs.add(f"{m.group(1)},{f.resolve()},{i},KERNEL_PROFILER")
for log in sys.argv[1:]:
    for line in open(log, errors="ignore"):
        p = line.find("'#pragma message: ")
        if p >= 0 and "/kh_tile_reuse/" not in line:
            locs.add(line[p + len("'#pragma message: ") :].rstrip("\n")[:-1])
by = collections.defaultdict(list)
for loc in locs:
    by[h16(loc)].append(loc)
bad = {k: v for k, v in by.items() if len(v) > 1}
for k, v in bad.items():
    print(hex(k), v)
print(f"{len(locs)} locations, {len(bad)} collisions")
sys.exit(1 if bad else 0)
