# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prelude timeline of one profiled run (no ttnn): per core, when the gather / tilize start and end, and the per-unit
split of the fine zones (reader rg_reserve / rg_work, unpack ct_wait / ct_unit). Times in us from the run's earliest
kernel start.

    python3 prelude.py <profile_log_device.csv[.gz]> [--run RUN_ID] [--core X,Y] [--units]

Without --run every run on >= 8 cores is summarized (one line per core, sorted by core y then x, plus per-row medians).
--core prints that core's full zone list; --units prints its per-unit sequence.
"""

import argparse
import collections
import importlib.util
import statistics
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


def summarize(spans):
    rows = []
    for core, sp in spans.items():

        def sel(zone, risc=None):
            return sorted(s for s in sp if s[3] == zone and (risc is None or s[2] == risc))

        def total(zone, risc=None):
            return sum(e - s for s, e, _, _ in sel(zone, risc))

        g = sel("reader_gather")
        t0u = sel("compute_tilize", "TRISC_0")
        t2u = sel("compute_tilize", "TRISC_2")
        un = sel("compute_untilize", "TRISC_0")
        bw = sel("reader_band_wait")
        kend = max((s[1] for s in sp if s[3].endswith("KERNEL")), default=float("nan"))
        rows.append(
            dict(
                core=core,
                kend=kend,
                untl_end=un[-1][1] if un else float("nan"),
                bw_s=bw[0][0] if bw else float("nan"),
                g_s=g[0][0] if g else float("nan"),
                g_e=g[0][1] if g else float("nan"),
                t0_s=t0u[0][0] if t0u else float("nan"),
                t0_e=t0u[0][1] if t0u else float("nan"),
                t2_e=t2u[0][1] if t2u else float("nan"),
                rres=total("rg_reserve"),
                rwork=total("rg_work"),
                cwait=total("ct_wait", "TRISC_0"),
                cunit=total("ct_unit", "TRISC_0"),
                nunits=len(sel("rg_work")),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--core", default=None)
    ap.add_argument("--units", action="store_true")
    ap.add_argument("--per-core", action="store_true")
    args = ap.parse_args()
    log = args.log
    if log.endswith(".gz"):
        import gzip
        import shutil
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        with gzip.open(log, "rb") as f:
            shutil.copyfileobj(f, tmp)
        tmp.close()
        log = tmp.name
    freq, runs = zr.load(log)
    for run_id in sorted(runs):
        if args.run is not None and run_id != args.run:
            continue
        ev = runs[run_id]
        if len({e[0] for e in ev}) < 8:
            continue
        t0 = min(e[2] for e in ev)
        spans = spans_of(ev, freq, t0)
        if args.core:
            core = tuple(int(v) for v in args.core.split(","))
            print(f"=== run {run_id} core {core}")
            if args.units:
                rr = sorted(s for s in spans[core] if s[3] in ("rg_reserve", "rg_work"))
                cc = sorted(s for s in spans[core] if s[3] in ("ct_wait", "ct_unit") and s[2] == "TRISC_0")
                for lst, name in ((rr, "reader"), (cc, "unpack")):
                    print(f"  {name}:")
                    for s, e, risc, zone in lst:
                        print(f"    {s:8.3f} -> {e:8.3f} ({(e - s) * 1000:7.0f} ns) {zone}")
            else:
                for s, e, risc, zone in sorted(spans[core]):
                    print(f"  {s:8.2f} -> {e:8.2f}  ({e - s:6.2f})  {risc:<8} {zone}")
            continue
        rows = summarize(spans)
        cols = ["kend", "untl_end", "bw_s", "g_s", "g_e", "t0_s", "t0_e", "t2_e", "rres", "rwork", "cwait", "cunit"]
        print(f"=== run {run_id}: {len(rows)} cores")
        print("core      " + " ".join(f"{c:>8}" for c in cols) + "  units")
        by_row = collections.defaultdict(list)
        for r in sorted(rows, key=lambda r: (r["core"][1], r["core"][0])):
            by_row[r["core"][1]].append(r)
            if args.per_core:
                print(f"{str(r['core']):<9} " + " ".join(f"{r[c]:8.2f}" for c in cols) + f"  {r['nunits']}")
        for y, rs in sorted(by_row.items()):
            med = {c: statistics.median(r[c] for r in rs) for c in cols}
            print(f"row y={y:<3} " + " ".join(f"{med[c]:8.2f}" for c in cols) + f"  {rs[0]['nunits']}")
        g = [r["g_e"] - r["g_s"] for r in rows]
        print(f"reader_gather span p50 {statistics.median(g):.2f} max {max(g):.2f}")


if __name__ == "__main__":
    main()
