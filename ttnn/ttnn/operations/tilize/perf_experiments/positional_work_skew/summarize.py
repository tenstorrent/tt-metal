#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pair the ops-perf CSV rows with the interleaved dispatch order and report
per-mode DEVICE KERNEL DURATION [ns] (median of the measured reps), the GB/s the
shape implies, and the delta vs `baseline`."""
import argparse
import csv
import json
import statistics
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", required=True)
    ap.add_argument("--dispatches", required=True)
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.ops)) if r.get("OP CODE", "").strip()]
    meta = json.loads(open(a.dispatches).read())
    disp = meta["dispatches"]
    if len(rows) != len(disp):
        print(f"!! {len(rows)} profiler rows vs {len(disp)} dispatches — pairing by order anyway")
    per = {}
    for r, d in zip(rows, disp):
        ns = int(r["DEVICE KERNEL DURATION [ns]"])
        cc = int(r["CORE COUNT"])
        per.setdefault((d["label"], d["mode"]), {"warm": [], "measure": [], "cores": set()})
        per[(d["label"], d["mode"])][d["phase"]].append(ns)
        per[(d["label"], d["mode"])].setdefault("pos", []).append(d.get("pos"))
        per[(d["label"], d["mode"])]["cores"].add(cc)

    plans = {(p["label"], p["mode"]): p for p in meta["plans"]}
    labels = []
    for lbl, _m in per:
        if lbl not in labels:
            labels.append(lbl)
    for lbl in labels:
        shape = next(p["shape"] for (l, _), p in plans.items() if l == lbl)
        elems = 1
        for d in shape:
            elems *= d
        gbytes = 2 * elems * 2 / 1e9  # bf16 in + bf16 out
        base = per.get((lbl, "baseline"), {}).get("measure") or None
        base_med = statistics.median(base) if base else None
        print(f"\n=== {lbl} {shape}  ({gbytes*1e3:.3f} MB moved) ===")
        print(f"{'mode':16s} {'median ns':>10s} {'GB/s':>7s} {'vs base':>8s} {'cores':>6s} spread  reps (sorted)")
        for (l, m), v in per.items():
            if l != lbl:
                continue
            reps = v["measure"]
            if not reps:
                continue
            med = statistics.median(reps)
            rel = f"{base_med/med:.3f}x" if base_med else "-"
            grp = plans.get((l, m), {})
            note = ""
            if grp.get("reason_not_applied"):
                note = "  [SKEW NOT APPLIED: " + grp["reason_not_applied"] + "]"
            spread = (max(reps) - min(reps)) / med * 100
            print(
                f"{m:16s} {med:10.0f} {gbytes/(med/1e9):7.1f} {rel:>8s} "
                f"{sorted(v['cores'])!s:>6s} {spread:5.1f}%  {sorted(reps)}{note}"
            )


if __name__ == "__main__":
    sys.exit(main())
