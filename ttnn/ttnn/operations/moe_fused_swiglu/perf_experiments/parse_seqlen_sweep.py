#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Turn one or more profiled sequence-length sweep CHUNKS into a table + a scaling plot.

    perf_experiments/parse_seqlen_sweep.py <out_prefix> <report|csv> <manifest> [<report> <manifest> ...]

`count` is device-resident, so it is NOT recoverable from the profiler CSV — the mapping comes from
the manifest the sweep test wrote in dispatch order. Within each chunk the two are zipped by GLOBAL
CALL COUNT order, and the script REFUSES to report if their lengths disagree, because a silent
off-by-one there would shift every point of the curve onto a neighbouring count and still look like a
plausible graph.
"""

import csv
import glob
import json
import os
import statistics
import sys

DRAM_BW = 512e9  # blackhole p150 peak DRAM bandwidth, B/s
OP_CODES = {"GenericOpDeviceOperation", "MoeFusedSwiGluDeviceOperation"}


def load_rows(path):
    csvs = [path] if path.endswith(".csv") else sorted(glob.glob(os.path.join(path, "ops_perf_results*.csv")))
    if not csvs:
        sys.exit(f"no ops_perf_results*.csv under {path}")
    rows = []
    for p in csvs:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r.get("OP CODE") not in OP_CODES:
                    continue
                rows.append(
                    {
                        "call": int(r["GLOBAL CALL COUNT"]),
                        "ns": int(r["DEVICE KERNEL DURATION [ns]"]),
                        "cores": int(r["CORE COUNT"]),
                    }
                )
    rows.sort(key=lambda r: r["call"])
    return rows, csvs


def main():
    prefix, pairs = sys.argv[1], sys.argv[2:]
    if not pairs or len(pairs) % 2:
        sys.exit(__doc__)

    points = {}
    sources = []
    for report, manifest_path in zip(pairs[0::2], pairs[1::2]):
        manifest = json.load(open(manifest_path))
        rows, csvs = load_rows(report)
        if len(rows) != len(manifest):
            sys.exit(
                f"REFUSING TO REPORT: {len(rows)} moe_fused_swiglu rows in {csvs} but "
                f"{len(manifest)} dispatches in {manifest_path}. The count<->row mapping is "
                f"order-based, so a length mismatch means every point could be attributed to the "
                f"wrong count."
            )
        sources.append({"report": report, "manifest": manifest_path, "dispatches": len(rows)})
        for m, r in zip(manifest, rows):
            if m["warmup"]:
                continue
            # A manifest predating the placement axis is an INTERLEAVED run — that was the only
            # placement the harness could produce, so the default is a fact, not a guess.
            key = (m["format"], m.get("wplace", "interleaved"), m["count"])
            p = points.setdefault(key, {"m": m, "ns": []})
            p["ns"].append(r["ns"])
            p["cores"] = r["cores"]

    recs = []
    for (fmt, wplace, count), p in sorted(points.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        ns = sorted(p["ns"])
        med = statistics.median(ns)
        recs.append(
            {
                "format": fmt,
                "wplace": wplace,
                "grid": p["m"].get("grid", "full"),
                "emb": p["m"]["emb"],
                "capacity": p["m"]["capacity"],
                "count": count,
                "ns_median": med,
                "ns_min": ns[0],
                "ns_max": ns[-1],
                "reps": len(ns),
                "us_median": med / 1e3,
                "read_MB": p["m"]["read_bytes"] / 1e6,
                "dram_util": p["m"]["read_bytes"] / (DRAM_BW * med / 1e9),
                # Tokens per second the block sustains at this sequence length. M=0 is a useful
                # fixed-dispatch-cost measurement, but per-token latency is undefined there.
                "tokens_per_s": count / (med / 1e9),
                "ns_per_token": med / count if count else None,
                "cores": p["cores"],
            }
        )
    if not recs:
        sys.exit("no non-warmup points found")

    with open(f"{prefix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    json.dump({"sources": sources, "points": recs}, open(f"{prefix}.json", "w"), indent=1)

    for fmt, wplace in sorted({(r["format"], r["wplace"]) for r in recs}):
        sub = [r for r in recs if r["format"] == fmt and r["wplace"] == wplace]
        print(
            f"\n=== {fmt} · weights {wplace} (emb {sub[0]['emb']}, capacity {sub[0]['capacity']}, "
            f"{sub[0]['cores']} cores, grid {sub[0]['grid']}) ==="
        )
        print(f"{'count':>6} {'us':>9} {'spread%':>8} {'util':>6} {'ns/token':>9} {'Mtok/s':>8} {'reps':>5}")
        for r in sub:
            spread = 100.0 * (r["ns_max"] - r["ns_min"]) / r["ns_median"]
            ns_per_token = f"{r['ns_per_token']:.1f}" if r["ns_per_token"] is not None else "—"
            print(
                f"{r['count']:>6} {r['us_median']:>9.2f} {spread:>8.2f} {r['dram_util']:>6.3f} "
                f"{ns_per_token:>9} {r['tokens_per_s'] / 1e6:>8.2f} {r['reps']:>5}"
            )
        if len(sub) > 100:  # a full step-32 sweep, where a gap means a lost chunk
            missing = [c for c in range(32, sub[0]["capacity"] + 1, 32) if c not in {r["count"] for r in sub}]
            if missing:
                print(f"MISSING {len(missing)} counts: {missing}")
    print(f"\nwrote {prefix}.csv / {prefix}.json from {len(sources)} chunk(s)")


if __name__ == "__main__":
    main()
