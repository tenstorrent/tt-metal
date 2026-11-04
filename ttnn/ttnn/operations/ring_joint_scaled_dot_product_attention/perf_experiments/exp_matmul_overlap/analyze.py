# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side analysis of a test_emo.py::test_perf session (run after the mesh closed, the profiler CSV is complete).

  python analyze.py [out.json]

Per op call (runs with a compute_unit zone, in order, mapped to perf_order.json): per chip op span (first kernel
start -> last kernel end), and the per-key-block `emo_block` zone (mean / median ns on TRISC_1, 16 sampled blocks
per compute core of unit 0)."""
import json
import sys
from pathlib import Path

import pandas as pd

import os

LOG = os.path.join(os.environ.get("TT_METAL_PROFILER_DIR", "generated/profiler"), ".logs/profile_log_device.csv")
FREQ_GHZ = 1.35
HERE = Path(__file__).parent


def main(out=None):
    order = json.load(open(HERE / "perf_order.json"))
    df = pd.read_csv(LOG, skiprows=1, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    run_col = "run host ID" if "run host ID" in df.columns else "run ID"
    tcol = "time[cycles since reset]"
    calls = {}  # call index -> {chip: {...}}
    for dev, d in df.groupby("PCIe slot"):
        runs = [r for r, g in d.groupby(run_col) if (g["zone name"] == "compute_unit").any()]
        runs = sorted(runs)
        for ci, r in enumerate(runs):
            g = d[d[run_col] == r]
            k = g[g["zone name"].astype(str).str.contains("KERNEL")]
            span = (k[tcol].max() - k[tcol].min()) / FREQ_GHZ
            # unnamed zones = emo_block compiled in an earlier session (name log elsewhere); the only other TRISC_1 zones are named
            z = g[(g["zone name"].isna() | (g["zone name"] == "emo_block")) & (g["RISC processor type"] == "TRISC_1")]
            durs = []
            for _, gc in z.groupby(["core_x", "core_y"]):
                gc = gc.sort_values(tcol)
                st = gc[gc["type"] == "ZONE_START"][tcol].values
                en = gc[gc["type"] == "ZONE_END"][tcol].values
                n = min(len(st), len(en))
                durs += list((en[:n] - st[:n]) / FREQ_GHZ)
            s = pd.Series(durs, dtype=float)
            calls.setdefault(ci, {})[int(dev)] = {
                "span_us": round(span / 1e3, 2),
                "blk_mean_ns": round(float(s.mean()), 0) if len(s) else None,
                "blk_med_ns": round(float(s.median()), 0) if len(s) else None,
                "blk_n": int(len(s)),
            }
    res = []
    for ci in sorted(calls):
        tag = order[ci] if ci < len(order) else f"call{ci}"
        chips = calls[ci]
        spans = [c["span_us"] for c in chips.values()]
        blk = [c["blk_med_ns"] for c in chips.values() if c["blk_med_ns"] is not None]
        row = {
            "call": tag,
            "span_max_us": max(spans),
            "span_min_us": min(spans),
            "blk_med_ns": round(sum(blk) / len(blk), 0) if blk else None,
            "chips": chips,
        }
        res.append(row)
        print(
            f"{tag:58s} span max {row['span_max_us']:9.1f} us  min {row['span_min_us']:9.1f} us  block median {row['blk_med_ns']} ns"
        )
    if out:
        json.dump(res, open(out, "w"), indent=1)


def summary(path):
    """Median over measured calls per (case, variant): span max / min (us), and cand/base ratio per case."""
    import statistics as st

    rows = json.load(open(path))
    agg = {}
    for r in rows:
        tag = r["call"]
        if not tag.endswith(":meas"):
            continue
        case, var = tag[: -len(":meas")].split("|") if "|" in tag else ("focus", tag[: -len(":meas")])
        agg.setdefault(case, {}).setdefault(var, []).append((r["span_max_us"], r["span_min_us"], r["blk_med_ns"]))
    for case, vs in agg.items():
        med = {
            v: (
                st.median(x[0] for x in xs),
                st.median(x[1] for x in xs),
                len(xs),
                st.median(x[2] for x in xs) if all(x[2] is not None for x in xs) else None,
            )
            for v, xs in vs.items()
        }
        base = med.get("base")
        line = f"{case:40s}"
        for v, (mx, mn, n, blk) in med.items():
            rel = f" ({mx / base[0]:.3f}x max, {mn / base[1]:.3f}x min" if base and v != "base" else ""
            if rel and blk and base[3]:
                rel += f", {blk / base[3]:.3f}x blk"
            rel += ")" if rel else ""
            line += f" | {v}: max {mx:.1f} min {mn:.1f} blk {blk} n={n}{rel}"
        print(line)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    main(out)
    if out:
        summary(out)
