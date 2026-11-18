# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-core start / end of one profiled run as a grid (no device, no ttnn).

    python3 core_grid.py <profile_log_device.csv> <run id> [zone]

Prints, per physical core (x, y), the kernel end (max over RISCs of the *-KERNEL end) relative to the run's first kernel
start, in us; with a zone name, the end of that zone instead (last execution, any RISC)."""

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import zone_report as zr  # noqa: E402

path, run = sys.argv[1], int(sys.argv[2])
zone = sys.argv[3] if len(sys.argv) > 3 else None
freq, runs = zr.load(path)
ev = runs[run]
t0 = min(e[2] for e in ev)
start, end = {}, collections.defaultdict(int)
for core, risc, t, z, kind in ev:
    if zone is None and z.endswith("-KERNEL") or z == zone:
        if kind == "ZONE_START":
            start[core] = min(start.get(core, t), t)
        elif kind == "ZONE_END":
            end[core] = max(end[core], t)
xs = sorted({c[0] for c in end})
ys = sorted({c[1] for c in end})
print("end (us)   " + " ".join(f"x{x:<5}" for x in xs))
for y in ys:
    print(f"y{y:<9}" + " ".join(f"{(end[(x, y)] - t0) / freq:6.2f}" if (x, y) in end else "   -  " for x in xs))
print("start (us) " + " ".join(f"x{x:<5}" for x in xs))
for y in ys:
    print(f"y{y:<9}" + " ".join(f"{(start[(x, y)] - t0) / freq:6.2f}" if (x, y) in start else "   -  " for x in xs))
