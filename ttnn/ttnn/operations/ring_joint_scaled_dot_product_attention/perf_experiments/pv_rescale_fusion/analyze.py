# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side analysis of the last profiled call per chip (copied from probes/perf_breakdown.py)."""
import json
import pandas as pd

FREQ_GHZ = 1.35
LOG = "generated/profiler/.logs/profile_log_device.csv"


def _analyze(chips=None):
    df = pd.read_csv(LOG, skiprows=1, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    run_col = "run host ID" if "run host ID" in df.columns else "run ID"
    df = df[df[run_col] == df.groupby("PCIe slot")[run_col].transform("max")]
    tcol = "time[cycles since reset]"
    port_cores = {}
    for coord, ch in chips.items():
        port_cores[coord] = {(c.x, c.y) for c in ch["ports"].values()}
    out = {}
    for dev, d in df.groupby("PCIe slot"):
        k = d[d["zone name"].astype(str).str.contains("KERNEL")]
        span = k.groupby(["core_x", "core_y"])[tcol].agg(["min", "max"])
        t0 = span["min"].min()
        rel = (span - t0) / FREQ_GHZ
        rel["dur"] = rel["max"] - rel["min"]
        res = {
            "op_span_ns": float(rel["max"].max()),
            "cores": int(len(rel)),
            "core_end_q_ns": rel["max"].quantile([0.0, 0.1, 0.5, 0.9, 1.0]).round(0).tolist(),
            "core_start_q_ns": rel["min"].quantile([0.0, 0.5, 1.0]).round(0).tolist(),
        }
        z = d[~d["zone name"].astype(str).str.contains("KERNEL") & d["type"].isin(["ZONE_START", "ZONE_END"])]
        zones = {}
        for (zn, risc), g in z.groupby(["zone name", "RISC processor type"]):
            g = g.sort_values(["core_x", "core_y", tcol])
            durs, last_end = [], []
            for _, gc in g.groupby(["core_x", "core_y"]):
                st = gc[gc["type"] == "ZONE_START"][tcol].values
                en = gc[gc["type"] == "ZONE_END"][tcol].values
                n = min(len(st), len(en))
                durs += list((en[:n] - st[:n]) / FREQ_GHZ)
                if n:
                    last_end.append((en[n - 1] - t0) / FREQ_GHZ)
            if not durs:
                continue
            s = pd.Series(durs)
            zones[f"{zn}@{risc}"] = {
                "n": int(len(s)),
                "mean_ns": round(float(s.mean()), 0),
                "median_ns": round(float(s.median()), 0),
                "max_ns": round(float(s.max()), 0),
                "sum_per_core_ns": round(
                    float(s.sum()) / max(1, g[["core_x", "core_y"]].drop_duplicates().shape[0]), 0
                ),
                "last_end_max_ns": round(max(last_end), 0) if last_end else None,
            }
        # marker budget check
        cnt = z.groupby(["core_x", "core_y", "RISC processor type"]).size()
        res["max_markers_per_risc"] = int(cnt.max()) if len(cnt) else 0
        res["zones"] = zones
        out[int(dev)] = res
    return out


def report(res):
    for dev, r in sorted(res.items()):
        print(
            f"\nCHIP {dev}: op span {r['op_span_ns']/1e3:.1f} us, cores {r['cores']}, max markers/RISC "
            f"{r['max_markers_per_risc']}"
        )
        print("  core end quantiles (us):", [round(x / 1e3, 1) for x in r["core_end_q_ns"]])
        print("  core start quantiles (us):", [round(x / 1e3, 1) for x in r["core_start_q_ns"]])
        for zn, zs in sorted(r["zones"].items()):
            print(
                f"  {zn:45s} n={zs['n']:6d} mean={zs['mean_ns']:10.0f} med={zs['median_ns']:10.0f} "
                f"max={zs['max_ns']:10.0f} per_core_sum={zs['sum_per_core_ns']:12.0f} last_end={zs['last_end_max_ns']}"
            )


if __name__ == "__main__":
    import sys

    res = _analyze({})
    report(res)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump({str(k_): v_ for k_, v_ in res.items()}, f, indent=1)
