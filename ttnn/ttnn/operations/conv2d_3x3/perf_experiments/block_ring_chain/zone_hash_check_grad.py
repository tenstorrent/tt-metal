# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""perf_experiments/ablation_r12/zone_hash_check.py for grad/ (HEAD + graduation.patch) as it will sit in the real op:
every zone location of grad/kernels hashed under the op's kernels/ path, plus the firmware / dispatch zones of the last
profiler log. Prints colliding pairs; exit 1 on one."""
import glob
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = "/localdev/pmilojevic/workspaces/opgen/"
real = root + "ttnn/ttnn/operations/conv2d_3x3/kernels/"


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


locs = set()
for f in glob.glob(str(HERE / "grad" / "kernels" / "*")):
    for i, l in enumerate(open(f), 1):
        m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', l)
        if m:
            locs.add(f"{m.group(1)},{real + Path(f).name},{i},KERNEL_PROFILER")
for name in ("new_zone_src_locations.log", "zone_src_locations.log"):
    p = Path(root + "generated/profiler/.logs/" + name)
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            q = line.find("'#pragma message: ")
            if q >= 0 and "conv2d_3x3" not in line:
                locs.add(line[q + 18 :][:-1])
seen = {}
for s in locs:
    seen.setdefault(h16(s), []).append(s)
bad = [v for v in seen.values() if len(v) > 1]
for v in bad:
    print("COLLIDE", v)
print(len(locs), "locations,", len(bad), "collisions")
sys.exit(1 if bad else 0)
