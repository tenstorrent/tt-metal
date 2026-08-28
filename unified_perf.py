# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Record and compare device-time baselines for the unified kernels.

    python unified_perf.py record  <label>          # run the benchmarks, append to the CSV
    python unified_perf.py compare <old> <new>      # print the deltas between two labels
    python unified_perf.py labels                   # what has been recorded

Both benchmarks measure DEVICE time through metal's real-time profiler (see unified_bench.py),
not wall clock -- these kernels run in tens of microseconds and host dispatch is tens of
microseconds, so a wall-clock number measures the dispatcher.

THE CONFIGURATIONS ARE FIXED HERE ON PURPOSE. A baseline is only worth keeping if the next run
measures the same thing, so the sweeps below are arguments to this script and not to the
caller. Widening one means starting a new baseline, not comparing across a change of shape.

A run takes about fifteen minutes and needs a quiet device: nothing else holding it, and a
`tt-smi -r` first if anything has hung, or the numbers are contention rather than compute.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import date

CSV = "unified_perf_baseline.csv"
FIELDS = ["label", "commit", "date", "benchmark", "config", "us"]

# The matmul sweep. Two accumulator modes over a 3x3 shape grid at two k depths: enough to
# separate a per-launch cost from a per-MAC one, small enough to run often.
MATMUL_ARGS = ["--rt", "1", "4", "8", "--ct", "1", "4", "8", "--kt", "8", "32", "--modes", "dst", "l1"]
# The llama-prefill sweep: real model attention shapes through the flash kernel, per head,
# causal, summed over q-chunks. One sequence length, because the shape table is the axis.
MODELS_ARGS = ["--seq", "512"]

# Each benchmark logs its table through loguru at a known line number; these pick the rows out.
MATMUL_ROW = re.compile(r"main:214 - \s*(\w+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+([\d.]+)us")
MODELS_ROW = re.compile(r"main:206 - \s*(.+?)\s+(\d+)\s+(\d+)\s+\d+\s+([\d.]+)us")


def _run(script, args):
    env = dict(os.environ, TT_METAL_HOME=os.getcwd())
    out = subprocess.run([sys.executable, script, *args], env=env, capture_output=True, text=True, timeout=3600)
    return out.stdout + out.stderr


def record(label):
    rows = []
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    today = date.today().isoformat()

    log = _run("bench_matmul.py", MATMUL_ARGS)
    for mode, kb, kt, rt, ct, us in MATMUL_ROW.findall(log):
        rows.append([label, commit, today, "matmul", f"{mode}/kb{kb}/kt{kt}/rt{rt}/ct{ct}", us])

    log = _run("bench_models.py", MODELS_ARGS)
    for model, d, seq, us in MODELS_ROW.findall(log):
        rows.append([label, commit, today, "llama_prefill", f"{model.strip()}/d{d}/S{seq}", us])

    if not rows:
        # A benchmark that produced no rows is a failed run, not a fast one. Recording it would
        # put an empty label in the baseline and make the next comparison silently vacuous.
        raise SystemExit("no rows parsed -- the benchmarks did not produce a table; check the device")

    exists = os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(FIELDS)
        w.writerows(rows)
    print(f"recorded {len(rows)} rows as {label!r} at {commit}")


def _load():
    if not os.path.exists(CSV):
        raise SystemExit(f"no {CSV} yet -- run `record` first")
    with open(CSV) as f:
        return list(csv.DictReader(f))


def compare(old, new):
    data = _load()

    def by(label):
        return {(r["benchmark"], r["config"]): float(r["us"]) for r in data if r["label"] == label}

    a, b = by(old), by(new)
    if not a or not b:
        raise SystemExit(f"missing label: {old if not a else new}")

    shared = sorted(set(a) & set(b))
    print(f"{old} -> {new}   ({len(shared)} shared cells of {len(a)} / {len(b)})")
    for missing, side in ((set(a) - set(b), new), (set(b) - set(a), old)):
        for m in sorted(missing):
            print(f"  only in the other run, not {side}: {m[0]} {m[1]}")

    worst = (0.0, None)
    total = 0.0
    for bench in sorted({k[0] for k in shared}):
        print(f"\n{bench}")
        print(f"  {'config':<34}{'old':>9}{'new':>9}{'delta':>9}")
        for key in [k for k in shared if k[0] == bench]:
            d = (b[key] / a[key] - 1) * 100
            total += d
            if abs(d) > abs(worst[0]):
                worst = (d, key)
            print(f"  {key[1]:<34}{a[key]:>8.2f}{b[key]:>9.2f}{d:>+8.1f}%")
    print(f"\nmean {total / len(shared):+.2f}%   worst {worst[0]:+.1f}% at {worst[1][0]} {worst[1][1]}")
    # Negative is faster. A threshold is deliberately not enforced here: what counts as a
    # regression depends on the change, and a script that exits nonzero on noise gets ignored.


def labels():
    seen = {}
    for r in _load():
        seen.setdefault(r["label"], (r["commit"], r["date"], 0))
        c, d, n = seen[r["label"]]
        seen[r["label"]] = (c, d, n + 1)
    for label, (commit, d, n) in seen.items():
        print(f"  {label:<24} {commit}  {d}  {n} rows")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="run the benchmarks and append a labelled baseline")
    r.add_argument("label")
    c = sub.add_parser("compare", help="print deltas between two labels")
    c.add_argument("old")
    c.add_argument("new")
    sub.add_parser("labels", help="list recorded baselines")
    args = p.parse_args()

    if args.cmd == "record":
        record(args.label)
    elif args.cmd == "compare":
        compare(args.old, args.new)
    else:
        labels()


if __name__ == "__main__":
    main()
