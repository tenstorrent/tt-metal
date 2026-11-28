#!/usr/bin/env python3
"""Summarize test_tensix_tilize.py::test_bench logs (in-process device profiler reads).

usage: summarize.py [--log runs/bench.jsonl] [--since EPOCH]
Per (case, variant) record: median / min / max DEVICE KERNEL DURATION [ns] over launches 2..N (launch 1
is the warm-up), plus the median NCRISC / TRISC1 / BRISC spans. A launch that ran several programs
(stock may) sums them.
"""
import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(HERE / "runs/bench.jsonl"))
    ap.add_argument("--since", type=float, default=0.0)
    args = ap.parse_args()
    for line in open(args.log):
        rec = json.loads(line)
        if rec["t"] < args.since or "programs" not in rec:
            continue
        launches = rec["programs"][1:]
        if not launches or not all(launches):
            print(f"{rec['case']:>6} {rec['variant']:<32} NO PROFILER DATA")
            continue
        tot = lambda key: [sum(p.get(key, 0.0) for p in progs) for progs in launches]  # noqa: E731
        ks = tot("kernel")
        plan = rec.get("plan") or {}
        print(
            f"{rec['case']:>6} {rec['variant']:<32} median {statistics.median(ks):>9.0f} ns  "
            f"min {min(ks):>8.0f} max {max(ks):>8.0f} n={len(ks)} progs={len(launches[0])} exact={rec['exact']} "
            f"ncrisc {statistics.median(tot('ncrisc')):>8.0f} trisc1 {statistics.median(tot('trisc1')):>8.0f} "
            f"brisc {statistics.median(tot('brisc')):>8.0f} "
            f"block={plan.get('block_rows_tiles')}x{plan.get('block_cols_tiles')}"
        )


if __name__ == "__main__":
    main()
