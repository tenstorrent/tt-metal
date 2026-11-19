# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: 16-bit profiler zone hash collisions among the zones of both op copies profiled in one process (op_base,
op_copy), plus the non-conv2d zones of the shared log (firmware / kernel_lib). Prints colliding pairs; exit 1 on one."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def h16(s):
    r = 2166136261
    for c in s.encode():
        r = ((r ^ c) * 16777619) & 0xFFFFFFFF
    return ((r & 0xFFFF) ^ (r >> 16)) & 0xFFFF


locs = set()
pat = re.compile(r'(?:MaybeDeviceZoneScope|DeviceZoneScopedN)\("([^"]+)"\)')
for d in sorted(p.name for p in HERE.iterdir() if (p / "kernels").is_dir()):
    if not (HERE / d / "kernels").exists():
        continue
    for f in sorted((HERE / d / "kernels").iterdir()):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if line.lstrip().startswith("#define"):
                continue
            for m in pat.finditer(line):
                locs.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
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
