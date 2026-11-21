# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core NoC0 share of the output store (the writer's `store_noc0_tiles` / `store_all_tiles` data markers).

    share_grid.py <ops_perf_results.csv[.gz]> <profile_log_device.csv[.gz]>

Prints, per signposted call (`cmp|<case>|<variant>|<i>`), a y-by-x grid of the percentage of output tiles the writer
issued on NoC0, plus the chip-wide share."""
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ring_fill_noc_split"))
from span_grid import _open, calls  # noqa: E402


def main():
    ops_path, dev_path = sys.argv[1], sys.argv[2]
    ids = calls(ops_path)
    data = collections.defaultdict(dict)
    with _open(dev_path) as f:
        f.readline()
        r = csv.reader(f)
        cols = [c.strip() for c in next(r)]
        ix = {c: i for i, c in enumerate(cols)}
        for row in r:
            if len(row) < len(cols):
                continue
            run = int(row[ix["run host ID"]])
            if run not in ids:
                continue
            zone = row[ix["zone name"]].strip()
            if zone not in ("store_noc0_tiles", "store_all_tiles", "store_mask0"):
                continue
            data[(run, int(row[ix["core_x"]]), int(row[ix["core_y"]]))][zone] = int(row[ix["data"]])
    for run, label in sorted(ids.items(), key=lambda kv: kv[1]):
        grid = {(x, y): d for (r, x, y), d in data.items() if r == run and "store_all_tiles" in d}
        if not grid:
            continue
        n0 = sum(d.get("store_noc0_tiles", 0) for d in grid.values())
        na = sum(d["store_all_tiles"] for d in grid.values())
        print(f"== {label} (run {run}) NoC0 share of output tiles: {100.0 * n0 / max(na, 1):.1f}% of {na}")
        xs = sorted({x for x, _ in grid})
        ys = sorted({y for _, y in grid})
        print("     " + " ".join(f"x{x:<4}" for x in xs))
        for y in ys:
            cells = []
            for x in xs:
                d = grid.get((x, y))
                cells.append(
                    f"{100.0 * d.get('store_noc0_tiles', 0) / max(d['store_all_tiles'], 1):5.0f}" if d else "    ."
                )
            print(f"y{y:<3} " + " ".join(cells))
        if any("store_mask0" in d for d in grid.values()):
            print("     NoC0-preferred bank mask (hex)")
            for y in ys:
                print(
                    f"y{y:<3} "
                    + " ".join(f"{grid[(x, y)].get('store_mask0', 0):5x}" if (x, y) in grid else "    ." for x in xs)
                )


if __name__ == "__main__":
    main()
