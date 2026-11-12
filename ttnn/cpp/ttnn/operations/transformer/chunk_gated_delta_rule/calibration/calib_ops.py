# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op device durations of a capture, from the profiler's ops report (ops_perf_results*.csv).

  python calib_ops.py <capture dir or csv> [--json out.json] [--keep-first]

Groups the rows by (op code, core count), drops the first launch of each group (JIT compile and warm-up) and prints
n, median, min and max of the device kernel duration in microseconds. The batch runner stores this as
ops_summary.txt in the capture directory; the collector reads the medians from it.
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd


def find_csv(path):
    if os.path.isfile(path):
        return path
    cands = glob.glob(os.path.join(path, "**", "ops_perf_results*.csv"), recursive=True)
    if not cands:
        sys.exit(f"no ops_perf_results*.csv under {path}")
    return max(cands, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep-first", action="store_true")
    args = ap.parse_args()
    path = find_csv(args.path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    code = "OP CODE"
    dur = next((c for c in df.columns if c.startswith("DEVICE KERNEL DURATION")), None)
    fw = next((c for c in df.columns if c.startswith("DEVICE FW DURATION")), None)
    cores = "CORE COUNT" if "CORE COUNT" in df.columns else None
    if dur is None:
        sys.exit(f"no DEVICE KERNEL DURATION column in {path}; columns: {list(df.columns)}")
    df = df[pd.to_numeric(df[dur], errors="coerce").notna()].copy()
    df[dur] = df[dur].astype(float) / 1000.0
    if fw:
        df[fw] = pd.to_numeric(df[fw], errors="coerce") / 1000.0
    keys = [code] + ([cores] if cores else [])
    print(
        f"report: {path}\n{'op':44s} {'cores':>5s} {'n':>4s} {'median':>9s} {'min':>9s} {'max':>9s}  kernel [us]"
        + (f" {'fw med':>9s}" if fw else "")
    )
    out = []
    for key, g in df.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        if not args.keep_first and len(g) > 1:
            g = g.iloc[1:]
        row = dict(
            op=key[0],
            cores=int(key[1]) if cores else None,
            n=len(g),
            median=float(g[dur].median()),
            min=float(g[dur].min()),
            max=float(g[dur].max()),
            fw_median=float(g[fw].median()) if fw else None,
        )
        out.append(row)
        print(
            f"{row['op'][:44]:44s} {row['cores'] if cores else '':>5} {row['n']:4d} {row['median']:9.1f} "
            f"{row['min']:9.1f} {row['max']:9.1f}" + (f" {row['fw_median']:9.1f}" if fw else "")
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
