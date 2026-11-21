# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core kernel span grid (us) of every signposted call of a run.

    span_grid.py <ops_perf_results.csv[.gz]> <profile_log_device.csv[.gz]> [zone ...]

The call of window `cmp|<case>|<variant>|<i>` is the op after that signpost in the ops CSV (its GLOBAL CALL COUNT is the
device log's run host ID). Zones default to NCRISC-KERNEL and BRISC-KERNEL; a zone that repeats on a core is summed.
Prints, per call and zone, a y-by-x grid of spans plus max / mean / min and max/p50 (the grid-row imbalance)."""
import collections
import csv
import gzip
import statistics
import sys


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def calls(ops_path):
    out, label = {}, None
    with _open(ops_path) as f:
        for row in csv.DictReader(f):
            if row.get("OP TYPE") == "signpost":
                head = row["OP CODE"]
                label = head[4:] if head.startswith("cmp|") else None
                continue
            if label is not None and row.get("GLOBAL CALL COUNT"):
                out[int(row["GLOBAL CALL COUNT"])] = label
                label = None
    return out


def spans(dev_path, run_ids, zones):
    st, dur = {}, collections.defaultdict(float)
    with _open(dev_path) as f:
        f.readline()
        r = csv.reader(f)
        cols = [c.strip() for c in next(r)]
        ix = {c: i for i, c in enumerate(cols)}
        i_run, i_zone, i_x, i_y = ix["run host ID"], ix["zone name"], ix["core_x"], ix["core_y"]
        i_t, i_type = ix["time[cycles since reset]"], ix["type"]
        for row in r:
            if len(row) < len(cols):
                continue
            run = int(row[i_run])
            if run not in run_ids:
                continue
            zone = row[i_zone].strip()
            if zone not in zones:
                continue
            k = (run, zone, int(row[i_x]), int(row[i_y]))
            t = int(row[i_t])
            if row[i_type].strip() == "ZONE_START":
                st[k] = t
            elif k in st:
                dur[k] += (t - st.pop(k)) / 1350.0
    return dur


def main():
    ops_path, dev_path = sys.argv[1], sys.argv[2]
    zones = sys.argv[3:] or ["NCRISC-KERNEL", "BRISC-KERNEL"]
    ids = calls(ops_path)
    dur = spans(dev_path, set(ids), set(zones))
    for run, label in sorted(ids.items(), key=lambda kv: kv[1]):
        for zone in zones:
            grid = {(x, y): d for (r, z, x, y), d in dur.items() if r == run and z == zone}
            if not grid:
                continue
            xs = sorted({x for x, _ in grid})
            ys = sorted({y for _, y in grid})
            vals = list(grid.values())
            p50 = statistics.median(vals)
            print(
                f"== {label} (run {run}) {zone}: max {max(vals):.0f} mean {sum(vals) / len(vals):.0f} "
                f"min {min(vals):.0f} p50 {p50:.0f} max/p50 {max(vals) / p50:.3f}"
            )
            print("     " + " ".join(f"x{x:<4}" for x in xs) + "  rowmax")
            for y in ys:
                row = [grid.get((x, y)) for x in xs]
                cells = " ".join(f"{v:5.0f}" if v is not None else "    -" for v in row)
                print(f"y{y:<3} {cells}  {max(v for v in row if v is not None):5.0f}")


if __name__ == "__main__":
    main()
