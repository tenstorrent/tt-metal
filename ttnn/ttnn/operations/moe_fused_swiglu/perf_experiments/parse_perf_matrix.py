#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Turn one or more profiled PERF MATRIX chunks into a per-cell table.

    perf_experiments/parse_perf_matrix.py <out_prefix> <report|csv> <manifest> [<report> <manifest> ...]

The matrix is (op x input_format x wplace x weight_dtype) x M. `count` is DEVICE-resident and neither
the weight dtype nor the op is recoverable from a shape, so the whole mapping comes from the manifest
`test_moe_fused_swiglu_perf_matrix.py` writes in dispatch order. Within each chunk the manifest is
zipped against the CSV's op rows (see `OP_CODES` — one code per op on the axis) sorted by GLOBAL CALL
COUNT, and this script REFUSES to report on a length mismatch: the mapping is order-based, so an
off-by-one there would attribute every cell to a neighbouring one and still print a plausible table.

THE `op` AXIS makes this a comparison between implementations: `moe_fused_swiglu` against the
reference `routed_expert` (`UnifiedRoutedExpertFfnDeviceOperation`) on the same cores, tensors and
maths. Because the two ops emit DIFFERENT device-op codes, filtering the CSV by a single code — which
this script used to do — would have silently dropped one column's rows and then refused on the length
check, so `OP_CODES` is a set and must gain an entry alongside any new op on the axis.

This is a SEPARATE parser from `parse_seqlen_sweep.py` rather than a flag on it, because that one
keys points by (format, wplace, count) — it predates the weight-dtype axis and would silently
AVERAGE a bfp4 cell together with the bfp8 cell that shares its layout and placement.
"""

import csv
import glob
import json
import os
import statistics
import sys

DRAM_BW = 512e9  # blackhole p150 peak DRAM bandwidth, B/s

#: The manifest fields that identify a CELL. Everything else in an entry (rep, warmup, read_bytes)
#: is per-dispatch data, not identity.
KEY = ("op", "format", "wplace", "weight_dtype", "grid", "emb", "hidden", "capacity", "count")

#: A CONFIGURATION is a cell minus M — the thing an M-sweep is a curve of. `emb` (K) belongs in here:
#: it is a swept axis, and leaving it out would put two K values in one column.
CONFIG = ("emb", "hidden", "format", "op", "wplace", "weight_dtype", "grid", "capacity")

#: Column order for the placement axis — the op's DESIGNED placement first, so the ratio below reads
#: as "what the uncoalesced stream costs" rather than the other way round.
WPLACE_ORDER = ("nd_shard", "shard_tall", "interleaved")

#: Column order for the OP axis — this repo's op first, so an op ratio reads as "what the reference
#: implementation costs relative to ours".
OP_ORDER = ("moe_fused_swiglu", "routed_expert")

#: Device-op codes the matrix can produce, one per op in the manifest's `op` axis. The mapping is
#: needed only to KNOW WHICH CSV ROWS ARE OURS — the manifest, not the op code, is what attributes a
#: row to a cell (`count` is device-resident and invisible in the CSV). A row whose code is not here
#: is incidental setup and dropped.
OP_CODES = {
    "GenericOpDeviceOperation",
    "MoeFusedSwiGluDeviceOperation",
    "UnifiedRoutedExpertFfnDeviceOperation",
}

#: Column-header abbreviations. The full op names make a 7-column table unreadable in a terminal.
OP_SHORT = {"moe_fused_swiglu": "fused", "routed_expert": "routed"}


def op_short(o):
    return OP_SHORT.get(o, o)


#: Mirrors the op's geometry (`M_BLOCK`, `pow2_ceil(OUT_SUBBLOCK_H_GU)`) so the M-quantization can be
#: shown as a COLUMN instead of appearing as an unexplained kink in the curve. DISPLAY ONLY: drift
#: here mislabels that column, it cannot affect a measurement.
M_BLOCK, M_EFF_MIN = 8, 1


def m_eff_total(m_t, m_block=M_BLOCK, m_min=M_EFF_MIN):
    """Tile-rows the op actually COMPUTES for `m_t` real tile-rows.

    Only the last M-block can shrink, and it shrinks to the next POWER OF TWO at or above the
    remainder (`m_tiles_eff` in moe_fused_swiglu_common.hpp), because the column's all-to-all is
    deadlock-free only while every core agrees to the tile on who owns which slice. So 3 tile-rows
    cost 4 and 6 cost 8 — M 96 is priced as M 128, and M 192 as M 256.
    """
    total, done = 0, 0
    while done < m_t:
        rem = m_t - done
        if rem >= m_block:
            total += m_block
        else:
            p = max(m_min, 1)
            while p < rem:
                p <<= 1
            total += min(p, m_block)
        done += m_block
    return total


def cfg_of(rec):
    return tuple(rec[k] for k in CONFIG)


def wplace_rank(p):
    return WPLACE_ORDER.index(p) if p in WPLACE_ORDER else len(WPLACE_ORDER)


def op_rank(o):
    return OP_ORDER.index(o) if o in OP_ORDER else len(OP_ORDER)


def col_rank(c):
    """Sort key for an `(op, wplace)` COLUMN: op-major, then placement.

    Op-major rather than placement-major so each op's two placements sit next to each other — the
    placement delta is a property of one implementation, while the op delta is the comparison, and
    both stay readable if the pairs are contiguous."""
    op, place = c
    return (op_rank(op), wplace_rank(place))


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


def table_groups(recs, counts, by_cell):
    """One table per (K, N, format, weight dtype): M down the side, (OP x PLACEMENT) across.

    The column axis is the PAIR, not the placement alone, because the matrix now measures two
    implementations of the same maths on the same cores: `moe_fused_swiglu` and the reference
    `routed_expert` (`UnifiedRoutedExpertFfnDeviceOperation`). Two ratios come out of that and they
    answer different questions, so both are reported rather than collapsed:

      * the OP ratio, `routed/fused` at a FIXED placement — which implementation is faster. This is
        in the main table, since it is the comparison the matrix exists to make.
      * the PLACEMENT ratio, `interleaved/nd_shard` within a FIXED op — what the uncoalesced weight
        stream costs that op. Demoted to the details, since each op has its own answer and they are
        not comparable to each other.

    A ratio is emitted only where BOTH of its cells exist, so a skipped or L1-refused cell leaves a
    hole rather than a number computed against a missing measurement.

    Returned as (title, columns, rows, detail_columns, detail_rows) so stdout and Markdown render the
    SAME records — a hand-copied table is a table that silently drifts from the CSV beside it.
    """
    groups = {}
    for r in recs:
        gk = (r["emb"], r["hidden"], r["format"], r["weight_dtype"], r["grid"], r["capacity"])
        groups.setdefault(gk, set()).add((r["op"], r["wplace"]))

    out = []
    for gk in sorted(groups):
        emb, hidden, fmt, wdt, grid, cap = gk
        pairs = sorted(groups[gk], key=col_rank)
        cores = next(
            r["cores"]
            for r in recs
            if (r["emb"], r["hidden"], r["format"], r["weight_dtype"], r["grid"], r["capacity"]) == gk
        )
        title = f"K {emb} · N {hidden} · {fmt} activations · {wdt} weights (capacity {cap}, {cores} cores, grid {grid})"

        #: Placements measured for BOTH ops — the only ones where an op ratio is meaningful.
        ops_present = [o for o in OP_ORDER if any(p[0] == o for p in pairs)]
        ratio_places = (
            [p for p in WPLACE_ORDER if all((o, p) in pairs for o in OP_ORDER)] if len(ops_present) == 2 else []
        )
        #: Ops measured at both placements — where a placement ratio is meaningful.
        ratio_ops = [o for o in ops_present if sum(1 for p in pairs if p[0] == o) == 2]

        cols = ["M"] + [f"{op_short(o)}/{p} us" for o, p in pairs]
        cols += [f"routed/fused @{p}" for p in ratio_places]
        det_cols = (
            ["M", "m_t", "m_eff"]
            + [f"{op_short(o)} read_MB" for o in ops_present]
            + [f"{op_short(o)}/{p} util" for o, p in pairs]
            + [f"{op_short(o)}/{p} spread%" for o, p in pairs]
            + [f"{op_short(o)} intlv/nd" for o in ratio_ops]
        )

        def cell(op, place, c):
            return by_cell.get((emb, hidden, fmt, op, place, wdt, grid, cap, c))

        rows, det_rows = [], []
        for c in counts:
            cells = [cell(o, p, c) for o, p in pairs]
            if not any(cells):
                continue
            row = [str(c)] + [f"{x['us_median']:.2f}" if x else "-" for x in cells]
            for p in ratio_places:
                a, b = cell("moe_fused_swiglu", p, c), cell("routed_expert", p, c)
                row.append(f"{b['ns_median'] / a['ns_median']:.3f}" if a and b else "-")
            rows.append(row)

            m_t = -(-c // 32)
            det = [str(c), str(m_t), str(m_eff_total(m_t))]
            # read_MB is PER OP: moe_fused_swiglu holds the weights L1-resident and reads them once,
            # while the routed expert re-reads the full set per M-chunk, so one shared column would
            # misattribute the util denominator. It does not depend on placement.
            for o in ops_present:
                v = next((cell(o, p, c) for _, p in pairs if cell(o, p, c)), None)
                det.append(f"{v['read_MB']:.2f}" if v else "-")
            det += [f"{x['dram_util']:.3f}" if x else "-" for x in cells]
            det += [f"{x['spread_pct']:.2f}" if x else "-" for x in cells]
            for o in ratio_ops:
                a, b = cell(o, "nd_shard", c), cell(o, "interleaved", c)
                det.append(f"{b['ns_median'] / a['ns_median']:.3f}" if a and b else "-")
            det_rows.append(det)
        out.append((title, cols, rows, det_cols, det_rows))
    return out


def render_text(title, cols, rows):
    widths = [max(len(c), max((len(r[i]) for r in rows), default=0)) for i, c in enumerate(cols)]
    lines = ["  ".join(c.rjust(w) for c, w in zip(cols, widths))]
    lines += ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]
    return lines


def write_markdown(path, tables, recs, sources):
    dispatches = sum(s["dispatches"] for s in sources)
    reps = sorted({r["reps"] for r in recs})
    rep_text = str(reps[0]) if len(reps) == 1 else f"{min(reps)}–{max(reps)}"
    md = [
        "# moe_fused_swiglu vs routed_expert — op and weight placement vs M",
        "",
        "`DEVICE KERNEL DURATION [ns]` from a Tracy-profiled run, reported as the **median** over "
        f"{rep_text} repetitions per cell "
        f"({dispatches} dispatches total). Measured on **Blackhole p150** at 1.35 GHz.",
        "",
        "`fused` is `moe_fused_swiglu`; `routed` is the reference DeepSeek-prefill "
        "`unified_routed_expert_ffn` (`UnifiedRoutedExpertFfnDeviceOperation`). Both compute "
        "`silu(x@Wg) * (x@Wu) @ Wd` on the same 88 cores from the same tensors, both read the token "
        "count device-side, and both write bfp8 TILE DRAM — so `routed/fused` is an "
        "implementation-to-implementation ratio, not a configuration difference.",
        "",
        "`util` is `dram_read_bytes / (512e9 * device_kernel_time_s)`. The bytes do not depend on "
        "placement — ND sharding changes how many NoC transactions carry the same bytes, not how many "
        "bytes there are — but they DO depend on the op, which is why `read_MB` is reported per op: "
        "`fused` holds all three weight sets L1-resident and reads them **once** per dispatch, while "
        "`routed` chunks M to 32 tile-rows and **re-reads** the full set per chunk (5x at M 5120).",
        "",
        "Generated by `perf_experiments/parse_perf_matrix.py` from "
        "`tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_perf_matrix.py` — "
        "regenerate rather than hand-edit.",
        "",
    ]
    for title, cols, rows, det_cols, det_rows in tables:
        md += [f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---:"] * len(cols)) + "|"]
        md += ["| " + " | ".join(r) + " |" for r in rows]
        md += ["", "<details><summary>bytes read, DRAM utilisation and run-to-run spread</summary>", ""]
        md += ["| " + " | ".join(det_cols) + " |", "|" + "|".join(["---:"] * len(det_cols)) + "|"]
        md += ["| " + " | ".join(r) + " |" for r in det_rows]
        md += ["", "</details>", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(md) + "\n")


def main():
    if len(sys.argv) < 4 or len(sys.argv) % 2:
        sys.exit(__doc__)
    prefix, pairs = sys.argv[1], sys.argv[2:]

    points, sources = {}, []
    for report, manifest_path in zip(pairs[0::2], pairs[1::2]):
        manifest = json.load(open(manifest_path))
        rows, csvs = load_rows(report)
        if len(rows) != len(manifest):
            sys.exit(
                f"REFUSING TO REPORT: {len(rows)} GenericOpDeviceOperation rows in {csvs} but "
                f"{len(manifest)} dispatches in {manifest_path}. The cell<->row mapping is "
                f"order-based, so a length mismatch means every measurement could be attributed to "
                f"the wrong cell."
            )
        sources.append({"report": report, "manifest": manifest_path, "dispatches": len(rows)})
        for m, r in zip(manifest, rows):
            if m["warmup"]:
                continue
            p = points.setdefault(tuple(m[k] for k in KEY), {"m": m, "ns": []})
            p["ns"].append(r["ns"])
            p["cores"] = r["cores"]

    if not points:
        sys.exit("no non-warmup points found")

    recs = []
    for key, p in sorted(points.items(), key=lambda kv: kv[0]):
        ns = sorted(p["ns"])
        med = statistics.median(ns)
        rec = dict(zip(KEY, key))
        rec.update(
            {
                "ns_median": med,
                "ns_min": ns[0],
                "ns_max": ns[-1],
                "reps": len(ns),
                "us_median": med / 1e3,
                "spread_pct": 100.0 * (ns[-1] - ns[0]) / med,
                "read_MB": p["m"]["read_bytes"] / 1e6,
                "dram_util": p["m"]["read_bytes"] / (DRAM_BW * med / 1e9),
                "tokens_per_s": rec["count"] / (med / 1e9) if rec["count"] else 0.0,
                "ns_per_token": med / rec["count"] if rec["count"] else None,
                "cores": p["cores"],
            }
        )
        recs.append(rec)

    with open(f"{prefix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    json.dump({"sources": sources, "points": recs}, open(f"{prefix}.json", "w"), indent=1)

    counts = sorted({r["count"] for r in recs})
    by_cell = {(*cfg_of(r), r["count"]): r for r in recs}
    configs = sorted({cfg_of(r) for r in recs}, key=lambda c: (c[0], c[1], c[2], col_rank((c[3], c[4])), c[5]))

    # ---- per-configuration tables --------------------------------------------------------------
    for cfg in configs:
        sub = sorted((r for r in recs if cfg_of(r) == cfg), key=lambda r: r["count"])
        h = sub[0]
        print(
            f"\n=== {h['op']} · K {h['emb']} · N {h['hidden']} · {h['format']} · weights {h['wplace']} "
            f"{h['weight_dtype']} (capacity {h['capacity']}, {h['cores']} cores, grid {h['grid']}) ==="
        )
        print(f"{'M':>6} {'us':>9} {'spread%':>8} {'read_MB':>8} {'util':>6} {'ns/token':>9} {'Mtok/s':>8} {'reps':>5}")
        for r in sub:
            ns_per_token = f"{r['ns_per_token']:.1f}" if r["ns_per_token"] is not None else "-"
            print(
                f"{r['count']:>6} {r['us_median']:>9.2f} {r['spread_pct']:>8.2f} {r['read_MB']:>8.2f} "
                f"{r['dram_util']:>6.3f} {ns_per_token:>9} {r['tokens_per_s'] / 1e6:>8.2f} {r['reps']:>5}"
            )

    # ---- the requested shape: one table per (K, N, format, weight dtype), placement as COLUMNS ---
    # This is the comparison the matrix exists for, so it is emitted to stdout and to Markdown from
    # the SAME records — a hand-copied table is a table that drifts from the CSV beside it.
    tables = table_groups(recs, counts, by_cell)
    for title, cols, rows, _, _ in tables:
        print(f"\n=== {title} ===")
        for line in render_text(title, cols, rows):
            print(line)

    # A/B on each axis in isolation, holding every other axis fixed: the ratio is the whole reason the
    # axis is in the matrix, and eyeballing it off a stack of tables is where mistakes get made.
    for name, field, base in (
        ("op", "op", "moe_fused_swiglu"),
        ("weight placement", "wplace", "nd_shard"),
        ("weight dtype", "weight_dtype", "bfp4"),
        ("input format", "format", "bf16_rm"),
        ("K (emb)", "emb", 6144),
    ):
        idx = CONFIG.index(field)
        pairs_seen = []
        for r in recs:
            k = [*cfg_of(r), r["count"]]
            if k[idx] == base:
                continue
            k[idx] = base
            ref = by_cell.get(tuple(k))
            if ref:
                pairs_seen.append(r["ns_median"] / ref["ns_median"])
        if pairs_seen:
            ratios = sorted(pairs_seen)
            med = statistics.median(ratios)
            print(
                f"\n{name}: {len(ratios)} paired cells vs {base} — "
                f"ratio min {ratios[0]:.3f} median {med:.3f} max {ratios[-1]:.3f} "
                f"({'slower' if med > 1 else 'faster'} than {base})"
            )

    write_markdown(f"{prefix}.md", tables, recs, sources)
    print(f"\nwrote {prefix}.csv / {prefix}.json / {prefix}.md from {len(sources)} chunk(s), {len(recs)} cells")


if __name__ == "__main__":
    main()
