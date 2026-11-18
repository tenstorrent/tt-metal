# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zone timeline of one core of one run (us from the earliest kernel start on that core).

    python3 timeline.py <profile_log_device.csv> --run RUN_ID [--core X,Y]
"""

import argparse
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("zone_report", Path(__file__).parent.parent / "zone_report.py")
zr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--core", default=None)
    args = ap.parse_args()
    freq, runs = zr.load(args.log)
    ev = runs[args.run]
    cores = sorted({e[0] for e in ev})
    core = tuple(int(v) for v in args.core.split(",")) if args.core else cores[len(cores) // 2]
    ev = sorted((e for e in ev if e[0] == core), key=lambda e: e[2])
    t0 = min(e[2] for e in ev)
    stacks = {}
    spans = []
    for _, risc, t, zone, kind in ev:
        key = (risc, zone)
        if kind == "ZONE_START":
            stacks.setdefault(key, []).append(t)
        elif kind == "ZONE_END" and stacks.get(key):
            s = stacks[key].pop()
            spans.append(((s - t0) / freq, (t - t0) / freq, risc, zone))
    print(f"core {core}")
    for s, e, risc, zone in sorted(spans):
        print(f"  {s:8.2f} -> {e:8.2f}  ({e - s:6.2f})  {risc:<8} {zone}")


if __name__ == "__main__":
    main()
