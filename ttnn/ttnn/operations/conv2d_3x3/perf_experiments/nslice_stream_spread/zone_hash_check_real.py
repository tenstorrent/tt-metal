# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the 16-bit zone hashes of grad/'s kernels as they will compile at the real op's path (graduation), plus
the firmware / kernel_lib zones of the shared profiler logs. Exit 1 on a collision."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
REAL = ROOT / "ttnn/ttnn/operations/conv2d_3x3/kernels"


def h16(s):
    r = 2166136261
    for c in s.encode():
        r = ((r ^ c) * 16777619) & 0xFFFFFFFF
    return ((r & 0xFFFF) ^ (r >> 16)) & 0xFFFF


pat = re.compile(r'MaybeDeviceZoneScope\("([^"]+)"\)')
locs = set()
for f in sorted((HERE / "grad" / "kernels").iterdir()):
    for i, line in enumerate(f.read_text().splitlines(), 1):
        if line.lstrip().startswith("#define"):
            continue
        for m in pat.finditer(line):
            locs.add(f"{m.group(1)},{REAL / f.name},{i},KERNEL_PROFILER")
for name in ("zone_src_locations.log", "new_zone_src_locations.log"):
    log = ROOT / "generated/profiler/.logs" / name
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            p = line.find("#pragma message: ")
            if p >= 0 and "conv2d_3x3" not in line:
                locs.add(line[p + 17 :].rstrip("'"))
seen, bad = {}, 0
for s in sorted(locs):
    h = h16(s)
    if h in seen:
        print("COLLISION", hex(h), seen[h], "|", s)
        bad = 1
    seen[h] = s
print(f"{len(locs)} zone locations, collisions: {bad}")
sys.exit(bad)
