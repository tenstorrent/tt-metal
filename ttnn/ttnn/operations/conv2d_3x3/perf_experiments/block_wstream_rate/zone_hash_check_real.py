# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the profiler's 16-bit zone-hash check (as ablation_r12/zone_hash_check.py) on the real op's kernel paths
with grad2/'s kernels in place, i.e. the tree after graduation.patch, plus the non-conv2d zones of the profiler logs."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/localdev/pmilojevic/workspaces/opgen")
REAL = ROOT / "ttnn/ttnn/operations/conv2d_3x3/kernels"


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


locs = set()
for f in sorted(REAL.iterdir()):
    text = (HERE / "grad2" / "kernels" / f.name).read_text()
    for i, line in enumerate(text.splitlines(), 1):
        m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', line)
        if m:
            locs.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
for log in ("new_zone_src_locations.log", "zone_src_locations.log"):
    p = ROOT / "generated/profiler/.logs" / log
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            k = line.find("'#pragma message: ")
            if k >= 0 and "conv2d_3x3" not in line:
                locs.add(line[k + 18 :].rstrip("\n")[:-1])
seen = {}
for s in locs:
    seen.setdefault(h16(s), []).append(s)
bad = [v for v in seen.values() if len(v) > 1]
for v in bad:
    print("COLLIDE", v)
print(len(locs), "locations,", len(bad), "collisions")
