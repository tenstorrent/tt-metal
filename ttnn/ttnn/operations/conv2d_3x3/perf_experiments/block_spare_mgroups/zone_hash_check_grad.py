# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: zone hash collisions of the graduated kernels (grad/kernels, at the line numbers graduation.patch gives
them) located at the real op's kernel path, as the profiler hashes them once the patch is applied to HEAD."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = Path("/localdev/pmilojevic/workspaces/opgen/ttnn/ttnn/operations/conv2d_3x3/kernels")


def h16(s):
    r = 2166136261
    for c in s.encode():
        r = ((r ^ c) * 16777619) & 0xFFFFFFFF
    return ((r & 0xFFFF) ^ (r >> 16)) & 0xFFFF


locs = set()
pat = re.compile(r'(?:MaybeDeviceZoneScope|DeviceZoneScopedN)\("([^"]+)"\)')
for f in sorted((HERE / "grad" / "kernels").iterdir()):
    for i, line in enumerate(f.read_text().splitlines(), 1):
        if line.lstrip().startswith("#define"):
            continue
        for m in pat.finditer(line):
            locs.add(f"{m.group(1)},{REAL / f.name},{i},KERNEL_PROFILER")
log = Path("/localdev/pmilojevic/workspaces/opgen/generated/profiler/.logs/zone_src_locations.log")
if log.exists():
    for line in log.read_text().splitlines():
        if "#pragma message: " in line and "conv2d_3x3" not in line:
            locs.add(line.split("#pragma message: ", 1)[1].rstrip("'"))
seen = {}
bad = 0
for s in sorted(locs):
    h = h16(s)
    if h in seen:
        print("COLLISION", hex(h), seen[h], "|", s)
        bad = 1
    seen[h] = s
print(f"{len(locs)} zone locations, collisions: {bad}")
sys.exit(bad)
