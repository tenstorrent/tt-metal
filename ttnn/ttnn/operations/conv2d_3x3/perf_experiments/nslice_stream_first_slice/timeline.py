# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-core zone timelines of one run (no ttnn): us from the run's earliest kernel start (over all cores).

    python3 timeline.py <profile_log_device.csv> --run RUN_ID [--core X,Y] [--summary]

--summary prints, per core, when the gather ends, when weight rounds 0..2 land (writer_weight_fetch ends), when the
first stream matmul starts/ends and when the TRISC kernel ends, sorted by kernel end.
"""

import argparse
import collections
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("zone_report", Path(__file__).parent.parent / "zone_report.py")
zr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zr)


def spans_of(ev, freq, t0):
    stacks = {}
    spans = collections.defaultdict(list)
    for core, risc, t, zone, kind in sorted(ev, key=lambda e: e[2]):
        key = (core, risc, zone)
        if kind == "ZONE_START":
            stacks.setdefault(key, []).append(t)
        elif kind == "ZONE_END" and stacks.get(key):
            s = stacks[key].pop()
            spans[core].append(((s - t0) / freq, (t - t0) / freq, risc, zone))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--core", default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    freq, runs = zr.load(args.log)
    ev = runs[args.run]
    t0 = min(e[2] for e in ev)
    spans = spans_of(ev, freq, t0)
    if args.summary:
        rows = []
        for core, sp in spans.items():

            def first(zone, risc=None, what=1, n=0):
                xs = sorted(s for s in sp if s[3] == zone and (risc is None or s[2] == risc))
                return xs[n][what] if len(xs) > n else float("nan")

            kend = max((s[1] for s in sp if s[3].endswith("KERNEL") and s[2].startswith("TRISC")), default=float("nan"))
            rows.append(
                (
                    kend,
                    core,
                    first("compute_untilize", "TRISC_0"),
                    first("reader_gather"),
                    first("writer_weight_fetch", n=0),
                    first("writer_weight_fetch", n=1),
                    first("writer_weight_fetch", n=2),
                    first("compute_stream_mm", "TRISC_0", what=0),
                    first("compute_stream_mm", "TRISC_0", what=1),
                    first("compute_matmul_walk", "TRISC_0", what=0),
                )
            )
        print("core     kend   until_end gath_end  w0     w1     w2    smm0_s smm0_e walk_s")
        for r in sorted(rows):
            print(f"{str(r[1]):<8} " + " ".join(f"{v:6.1f}" for v in (r[0],) + r[2:]))
        return
    cores = sorted(spans)
    core = tuple(int(v) for v in args.core.split(",")) if args.core else cores[len(cores) // 2]
    print(f"core {core}")
    for s, e, risc, zone in sorted(spans[core]):
        print(f"  {s:8.2f} -> {e:8.2f}  ({e - s:6.2f})  {risc:<8} {zone}")


if __name__ == "__main__":
    main()
