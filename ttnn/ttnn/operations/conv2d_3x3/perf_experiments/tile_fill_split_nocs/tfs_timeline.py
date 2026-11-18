# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zone timeline of one core from a profile_log_device.csv (no device):
    tfs_timeline.py <report dir> X Y [zone,...]    (TFS_RUN=<n>: the n-th run host ID of the log, 0-based)"""

import csv
import os
import sys


def main(path, x, y, zones=None):
    log = os.path.join(path, "profile_log_device.csv") if os.path.isdir(path) else path
    with open(log) as f:
        freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = csv.reader(f)
        cols = [c.strip() for c in next(rows)]
        ix = {c: i for i, c in enumerate(cols)}
        ev = []
        t0 = None
        all_rows = [r for r in rows if len(r) >= len(cols)]
        run_ids = sorted({int(r[ix["run host ID"]]) for r in all_rows})
        want = run_ids[int(os.environ.get("TFS_RUN", "0"))]
        for r in all_rows:
            if int(r[ix["run host ID"]]) != want:
                continue
            t = int(r[ix["time[cycles since reset]"]])
            z = r[ix["zone name"]].strip()
            if z.endswith("-KERNEL"):
                t0 = t if t0 is None else min(t0, t)
            if (int(r[ix["core_x"]]), int(r[ix["core_y"]])) != (x, y):
                continue
            if zones and z not in zones:
                continue
            ev.append((t, r[ix["RISC processor type"]].strip(), z, r[ix["type"]].strip()))
    stacks = {}
    out = []
    for t, risc, z, kind in sorted(ev):
        if kind == "ZONE_START":
            stacks.setdefault((risc, z), []).append(t)
        elif kind == "ZONE_END" and stacks.get((risc, z)):
            s = stacks[(risc, z)].pop()
            out.append(((s - t0) / freq, (t - t0) / freq, risc, z))
    for s, e, risc, z in sorted(out):
        print(f"{s:9.2f} {e:9.2f} {e - s:8.2f}  {risc:<7} {z}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(",") if len(sys.argv) > 4 else None)
