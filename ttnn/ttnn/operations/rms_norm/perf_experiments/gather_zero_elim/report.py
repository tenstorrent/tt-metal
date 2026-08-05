# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Join a gather_zero_elim launch log with its profiler CSV and print the tables.

    python3 .../gather_zero_elim/report.py <launches_*.jsonl> [ops_perf_results_*.csv]

One `ttnn.generic_op` per logged launch, in launch order, so the CSV rows and the JSONL
lines are 1:1.  Columns:

  total_ns   DEVICE KERNEL DURATION [ns] for the whole launch
  stage_ns   total_ns minus the `none` launch's total for the same geometry -- the
             ablation-subtracted cost of the boot-zeroing stage itself
  txns       NoC transactions the scheme issues (`async_write_zeros` chunks at
             MEM_ZEROS_SIZE == 512 B, so a 1024 B face zero is TWO; a `scratch`
             write of the same face is ONE)
  ns/txn     stage_ns / txns -- the issue cost this stage is actually bound by
"""

import csv
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DUR = "DEVICE KERNEL DURATION [ns]"
ZERO_CHUNK = 512  # MEM_ZEROS_SIZE on Wormhole and Blackhole


def load_csv(path=None):
    if path is None:
        cands = sorted(
            glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=lambda p: Path(p).stat().st_mtime
        )
        if not cands:
            sys.exit("no ops_perf_results_*.csv under generated/profiler/reports/")
        path = cands[-1]
    with open(path) as f:
        rows = list(csv.DictReader(f))
    print(f"# csv: {path}  ({len(rows)} device ops)")
    return rows


def txn_count(r):
    """NoC transactions the scheme issues, from the geometry (not from the device)."""
    v, pages, gp, g, faces = r["variant"], r["pages"], r["gather_slots"], r["group_size"], r["gather_faces"]
    pads = pages - (pages // gp) * g
    live = pages - pads
    face_zeros = 0 if faces == 4 else (2 if faces == 2 else 1)
    if v == "none":
        return 0
    if v == "whole_cb":
        return pages * 4096 // ZERO_CHUNK
    if v == "pad_only":
        return pads * (4096 // ZERO_CHUNK)
    if v == "pad_faces02":
        return pads * 2 * (1024 // ZERO_CHUNK)
    if v in ("faces", "faces_r"):
        return live * face_zeros * (1024 // ZERO_CHUNK) + pads * (4096 // ZERO_CHUNK)
    if v in ("scratch", "scratch_r"):
        return 4096 // ZERO_CHUNK + live * face_zeros + pads * (4096 // ZERO_CHUNK)
    return 0


def main():
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else (HERE / "launches_zero.jsonl")
    if not log.is_absolute():
        log = HERE / log.name
    launches = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    rows = load_csv(sys.argv[2] if len(sys.argv) > 2 else None)
    if len(rows) != len(launches):
        keep = [r for r in rows if "generic" in (r.get("OP CODE") or "").lower()]
        print(f"# {len(rows)} csv rows vs {len(launches)} launches -> filter to OP CODE ~ generic: {len(keep)}")
        rows = keep
    if len(rows) != len(launches):
        sys.exit(f"cannot align: {len(rows)} csv rows vs {len(launches)} launches")

    recs = []
    for lau, row in zip(launches, rows):
        r = dict(lau)
        r["total_ns"] = int(float(row[DUR]))
        recs.append(r)

    if recs and recs[0].get("bench") == "poison":
        print_poison(recs)
    else:
        print_zero(recs)


def print_zero(recs):
    keys = []
    for r in recs:
        k = (r["tag"], r["group_size"], r["rows"], r["gather_faces"])
        if k not in keys:
            keys.append(k)
    hdr = f"{'variant':11s} {'total_ns':>9s} {'stage_ns':>9s} {'txns':>6s} {'ns/txn':>7s} {'bytes':>8s} {'exact':>6s}"
    for k in keys:
        tag, g, rows_, faces = k
        grp = [r for r in recs if (r["tag"], r["group_size"], r["rows"], r["gather_faces"]) == k]
        floor = next((r["total_ns"] for r in grp if r["variant"] == "none"), None)
        print(
            f"\n=== {tag}  GROUP_SIZE={g} rows={rows_} GATHER_FACES={faces} pages={grp[0]['pages']} (none={floor} ns) ==="
        )
        print(hdr)
        for r in grp:
            stage = (r["total_ns"] - floor) if floor is not None else 0
            t = txn_count(r)
            print(
                f"{r['variant']:11s} {r['total_ns']:9d} {stage:9d} {t:6d} "
                f"{(f'{stage/t:.1f}' if t else '-'):>7s} {r['bytes_zeroed']:8d} "
                f"{('OK' if r['byte_exact'] else 'FAIL'):>6s}"
            )


def print_poison(recs):
    hdr = (
        f"{'tag':22s} {'face':9s} {'pad':9s} {'g':>3s} {'r':>3s} {'wt':>3s} "
        f"{'bit_eq':>7s} {'col0_eq':>8s} {'pcc_out':>10s} {'rrms_out':>9s} "
        f"{'stat_hi_nf':>10s} {'stat_hi_max':>12s}"
    )
    print(hdr)
    for r in recs:
        print(
            f"{r['tag']:22s} {r['face_seed']:9s} {r['pad_seed']:9s} {r['group_size']:3d} {r['rows']:3d} {r['wt']:3d} "
            f"{str(r['bit_equal_to_zeroed']):>7s} {str(r.get('stat_col0_bit_equal', '-')):>8s} "
            f"{r['pcc_out']:10.6f} {r['rel_rms_out']:9.2e} "
            f"{r['stat_hi_nonfinite']:10.2f} {r['stat_hi_absmax']:12.2e}"
        )


if __name__ == "__main__":
    main()
