# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-profiler zone hashes are 16-bit (name, file, line); two zones of one profiled run colliding make it throw.
Checks op_copy's kernels against op_base's (both compile in one run) and the real op's, and shifts a colliding op_copy
file by inserting `// zone-hash pad` lines after its license header (make_patch.sh strips them). op_base stays an exact
copy of the real op.   python3 -I zone_pad.py"""

import glob
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP = HERE.parents[1]
PAD = "// zone-hash pad\n"


def h16(s):
    h = 2166136261
    for c in s.encode():
        h = ((h ^ c) * 16777619) & 0xFFFFFFFF
    return ((h & 0xFFFF) ^ (h >> 16)) & 0xFFFF


def zones(f):
    out = []
    for i, line in enumerate(open(f, errors="replace"), 1):
        for m in re.finditer(r'(?:MaybeDeviceZoneScope|DeviceZoneScopedN)\("([^"]+)"\)', line):
            out.append(f"{m.group(1)},{f},{i},KERNEL_PROFILER")
    return out


def main():
    mine = sorted(glob.glob(str(HERE / "op_copy" / "kernels" / "*")))
    others = sorted(glob.glob(str(HERE / "op_base" / "kernels" / "*"))) + sorted(glob.glob(str(OP / "kernels" / "*")))
    base_table = {}
    for f in others:
        for z in zones(f):
            base_table.setdefault(h16(z), set()).add(z)
    for _ in range(200):
        table = {k: set(v) for k, v in base_table.items()}
        bad = None
        for f in mine:
            for z in zones(f):
                hit = table.setdefault(h16(z), set())
                if hit - {z}:
                    bad = f
                    break
                hit.add(z)
            if bad:
                break
        if bad is None:
            print("no zone-hash collisions")
            return
        lines = open(bad).read().splitlines(keepends=True)
        at = next(i for i, l in enumerate(lines) if not l.startswith("//")) if lines else 0
        lines.insert(at, PAD)
        open(bad, "w").write("".join(lines))
        print("padded", bad)
    raise SystemExit("still colliding")


if __name__ == "__main__":
    main()
