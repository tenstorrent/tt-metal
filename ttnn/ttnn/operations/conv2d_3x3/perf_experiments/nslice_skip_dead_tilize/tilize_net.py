# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per run: the prelude tilize net of its wait (compute_tilize - compute_tilize_wait, per core, per TRISC), p50 / max
over cores, from a device profile log (no ttnn).  python3 tilize_net.py <profile_log_device.csv[.gz]>"""

import collections
import csv
import gzip
import statistics
import sys


def main():
    path = sys.argv[1]
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rows = csv.reader(f)
        cols = [c.strip() for c in next(rows)]
        ix = {c: i for i, c in enumerate(cols)}
        runs = collections.defaultdict(list)
        for r in rows:
            if len(r) < len(cols):
                continue
            runs[int(r[ix["run host ID"]])].append(
                (
                    (int(r[ix["core_x"]]), int(r[ix["core_y"]])),
                    r[ix["RISC processor type"]].strip(),
                    int(r[ix["time[cycles since reset]"]]),
                    r[ix["zone name"]].strip(),
                    r[ix["type"]].strip(),
                )
            )
    for run_id in sorted(runs):
        ev = sorted(runs[run_id], key=lambda e: e[2])
        if len({e[0] for e in ev}) < 50:
            continue
        st = {}
        sums = collections.defaultdict(float)
        for core, risc, t, zone, kind in ev:
            if zone in ("compute_tilize", "compute_tilize_wait"):
                if kind == "ZONE_START":
                    st[(core, risc, zone)] = t
                elif kind == "ZONE_END":
                    sums[(core, risc, zone)] += t - st.pop((core, risc, zone))
        out = []
        for risc in ("TRISC_0", "TRISC_1", "TRISC_2"):
            net = [
                (sums[(c, r, "compute_tilize")] - sums[(c, r, "compute_tilize_wait")]) / freq
                for (c, r, z) in list(sums)
                if r == risc and z == "compute_tilize"
            ]
            out.append(f"{risc} p50 {statistics.median(net):5.2f} max {max(net):5.2f}")
        print(f"run {run_id:>6}: tilize net of wait (us)  " + " | ".join(out))


if __name__ == "__main__":
    main()
