# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The device profiler keys zones by a 16-bit hash of "name,file,line,KERNEL_PROFILER" and throws on a collision
between any two locations of one session. Profiling the committed op next to this dir's copy doubles the zone
locations, so check the two kernel sets (plus firmware zones from the last build log) before a profiled run."""

import collections
import glob
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))


def h16(s):
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


def main():
    locs = set()
    log = os.path.join(ROOT, "generated/profiler/.logs/new_zone_src_locations.log")
    if os.path.exists(log):
        for line in open(log):
            pos = line.find("'#pragma message: ")
            if pos >= 0:
                locs.add(line[pos + len("'#pragma message: ") :].rstrip("\n")[:-1])
    for d in [
        "ttnn/ttnn/operations/conv2d_3x3/kernels",
        os.path.relpath(os.path.dirname(__file__), ROOT) + "/op/kernels",
    ]:
        for f in glob.glob(os.path.join(ROOT, d, "*")):
            for i, line in enumerate(open(f), 1):
                m = re.search(r'MaybeDeviceZoneScope\("([^"]+)"\)', line)
                if m:
                    locs.add(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
    by = collections.defaultdict(list)
    for loc in locs:
        by[h16(loc)].append(loc)
    bad = {k: v for k, v in by.items() if len(v) > 1}
    for k, v in bad.items():
        print("COLLISION", k, v)
    print(f"{len(locs)} zone locations, {len(bad)} collisions")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
