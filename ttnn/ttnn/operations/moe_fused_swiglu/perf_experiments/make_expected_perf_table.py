#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure and render the 8-row x 11-column moe_fused_swiglu perf table.

One-command reproduction (Blackhole p150, Tracy-enabled build):

    python ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/make_expected_perf_table.py \\
        --measure --output-prefix perf_table_8x11 \\
        --reference /path/to/perf_table_upstream.txt

``--measure`` runs the existing pytest device sweep in four bounded Tracy sessions. Splitting the
run keeps each trace small enough for the device profiler and host report generator. The renderer
then joins each profiler CSV to its manifest by GLOBAL CALL COUNT, refuses a length mismatch, and
reports the median DEVICE KERNEL DURATION over three repetitions.

The renderer can also be rerun without hardware from saved CSV/manifest pairs:

    python .../make_expected_perf_table.py --output-prefix perf_table_8x11 \\
        --reference perf_table_upstream.txt \\
        report_1.csv manifest_1.json report_2.csv manifest_2.json ...

Outputs are ``<prefix>.txt`` in the EXPECTED_NS dictionary format, ``<prefix>.md`` with readable
tables, and ``<prefix>.json`` with medians, raw samples, and source paths. The human grid name is
8x11 (rows x columns); the op API and manifest spell it ``core_grid=(11, 8)`` (x, y).
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


OP_CODE = "GenericOpDeviceOperation"
EXPECTED_GRID = "11x8"
COUNTS = (0, 64, 128, 256, 512, 1024, 2048, 4096, 5120)
FORMATS = ("x_rm", "x_tile")
PLACEMENTS = ("w_interleaved", "w_ndshard")
MODELS = ("kimi_k26", "glm_51")

FORMAT_FROM_MANIFEST = {"bf16_rm": "x_rm", "bfp8_tile": "x_tile"}
FORMAT_TO_MANIFEST = {v: k for k, v in FORMAT_FROM_MANIFEST.items()}
PLACEMENT_FROM_MANIFEST = {"interleaved": "w_interleaved", "nd_shard": "w_ndshard"}
MODEL_FROM_EMB = {7168: "kimi_k26", 6144: "glm_51"}
MODEL_SHAPE = {"kimi_k26": "K=7168, N=2048", "glm_51": "K=6144, N=2048"}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "run_safe_pytest.sh").is_file() and (parent / "ttnn").is_dir():
            return parent
    raise RuntimeError(f"could not find repository root above {here}")


def _report_csvs(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return [Path(p) for p in sorted(glob.glob(str(path / "ops_perf_results*.csv")))]


def _load_op_rows(report: str | Path) -> tuple[list[dict[str, str]], list[Path]]:
    csvs = _report_csvs(report)
    if not csvs:
        raise ValueError(f"no ops_perf_results*.csv found at {report}")
    rows: list[dict[str, str]] = []
    for path in csvs:
        with path.open(newline="") as fh:
            rows.extend(row for row in csv.DictReader(fh) if row.get("OP CODE") == OP_CODE)
    rows.sort(key=lambda row: int(row["GLOBAL CALL COUNT"]))
    if not rows:
        raise ValueError(f"no {OP_CODE} rows found in {csvs}")
    return rows, csvs


def _key_from_manifest(entry: dict) -> tuple[str, str, str, int]:
    if entry.get("op") != "moe_fused_swiglu":
        raise ValueError(f"unexpected op in manifest: {entry.get('op')!r}")
    if entry.get("grid") != EXPECTED_GRID:
        raise ValueError(f"expected API grid {EXPECTED_GRID} (the 8-row x 11-column grid), got {entry.get('grid')!r}")
    if int(entry.get("hidden", -1)) != 2048:
        raise ValueError(f"expected hidden N=2048, got {entry.get('hidden')!r}")
    try:
        return (
            FORMAT_FROM_MANIFEST[entry["format"]],
            PLACEMENT_FROM_MANIFEST[entry["wplace"]],
            MODEL_FROM_EMB[int(entry["emb"])],
            int(entry["count"]),
        )
    except KeyError as exc:
        raise ValueError(f"unsupported table axis in manifest entry: {entry}") from exc


def collect_measurements(source_pairs: list[tuple[str | Path, str | Path]]) -> tuple[dict, list[dict]]:
    samples: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
    sources: list[dict] = []
    core_counts: set[int] = set()

    for report, manifest_path in source_pairs:
        rows, csvs = _load_op_rows(report)
        with Path(manifest_path).open() as fh:
            manifest = json.load(fh)
        if len(rows) != len(manifest):
            raise ValueError(
                f"REFUSING TO REPORT: {len(rows)} {OP_CODE} rows in {csvs}, but "
                f"{len(manifest)} dispatches in {manifest_path}; attribution is order-based"
            )
        sources.append(
            {
                "report": [str(path.resolve()) for path in csvs],
                "manifest": str(Path(manifest_path).resolve()),
                "dispatches": len(rows),
            }
        )
        for entry, row in zip(manifest, rows):
            key = _key_from_manifest(entry)
            core_counts.add(int(row["CORE COUNT"]))
            if not entry["warmup"]:
                samples[key].append(int(row["DEVICE KERNEL DURATION [ns]"]))

    if core_counts != {88}:
        raise ValueError(f"expected every measurement to use 88 cores, got {sorted(core_counts)}")
    required = {(fmt, place, model, m) for fmt in FORMATS for place in PLACEMENTS for model in MODELS for m in COUNTS}
    missing, extra = required - samples.keys(), samples.keys() - required
    if missing or extra:
        raise ValueError(f"table universe mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    bad_reps = {key: len(values) for key, values in samples.items() if len(values) != 3}
    if bad_reps:
        raise ValueError(f"expected exactly three measured repetitions per cell, got {bad_reps}")

    points = {}
    for key, values in samples.items():
        ordered = sorted(values)
        points[key] = {
            "median_ns": int(statistics.median(ordered)),
            "samples_ns": ordered,
            "spread_pct": 100.0 * (ordered[-1] - ordered[0]) / statistics.median(ordered),
        }
    return points, sources


def load_expected_ns(path: str | Path) -> dict[tuple[str, str, str, int], int]:
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "EXPECTED_NS" for target in targets):
                result = ast.literal_eval(value)
                if not isinstance(result, dict):
                    break
                return result
    raise ValueError(f"could not find a literal EXPECTED_NS dictionary in {path}")


def canonical_keys():
    for fmt in FORMATS:
        for placement in PLACEMENTS:
            for model in MODELS:
                for count in COUNTS:
                    yield fmt, placement, model, count


def expected_ns_text(points: dict) -> str:
    lines = ["EXPECTED_NS: dict[tuple[str, str, str, int], int] = {"]
    for fmt in FORMATS:
        for placement in PLACEMENTS:
            for model in MODELS:
                lines.append(f"    # ---- {fmt}, {placement}, {model} ----")
                for count in COUNTS:
                    ns = points[(fmt, placement, model, count)]["median_ns"]
                    lines.append(f'    ("{fmt}", "{placement}", "{model}", {count}): {ns:_},')
    lines.append("}")
    return "\n".join(lines) + "\n"


def markdown_text(points: dict, sources: list[dict], reference: dict | None) -> str:
    lines = [
        "# moe_fused_swiglu performance — 8 rows × 11 columns",
        "",
        "Device-kernel duration on Blackhole p150 at 1.35 GHz. Each value is the median of three "
        "Tracy-profiled dispatches. The op API spells this grid `core_grid=(11, 8)` because its tuple "
        "is `(columns, rows)`; every CSV row reports 88 cores.",
        "",
        "Inputs: `x_rm` = BF16 row-major; `x_tile` = BFP8 tiled. Weights are BFP4 tiled. "
        "`w_ndshard` is the op-aware DRAM ND-sharded placement and `w_interleaved` is DRAM interleaved.",
        "",
    ]
    if reference is not None:
        lines += [
            "The upstream columns come from the supplied `perf_table_upstream.txt`. "
            "Delta is `(this kernel / upstream - 1)`: negative is faster, positive is slower.",
            "",
        ]

    for fmt in FORMATS:
        for model in MODELS:
            lines += [f"## {fmt} · {model} ({MODEL_SHAPE[model]})", ""]
            if reference is None:
                lines += [
                    "| M | interleaved (us) | ND-shard (us) |",
                    "|---:|---:|---:|",
                ]
            else:
                lines += [
                    "| M | this interleaved (us) | upstream interleaved (us) | delta | "
                    "this ND-shard (us) | upstream ND-shard (us) | delta |",
                    "|---:|---:|---:|---:|---:|---:|---:|",
                ]
            for count in COUNTS:
                int_key = (fmt, "w_interleaved", model, count)
                nd_key = (fmt, "w_ndshard", model, count)
                int_ns = points[int_key]["median_ns"]
                nd_ns = points[nd_key]["median_ns"]
                if reference is None:
                    lines.append(f"| {count} | {int_ns / 1000:.3f} | {nd_ns / 1000:.3f} |")
                else:
                    ref_int, ref_nd = reference[int_key], reference[nd_key]
                    delta_int = 100.0 * (int_ns / ref_int - 1.0)
                    delta_nd = 100.0 * (nd_ns / ref_nd - 1.0)
                    lines.append(
                        f"| {count} | {int_ns / 1000:.3f} | {ref_int / 1000:.3f} | {delta_int:+.1f}% | "
                        f"{nd_ns / 1000:.3f} | {ref_nd / 1000:.3f} | {delta_nd:+.1f}% |"
                    )
            lines.append("")

    lines += ["## Reproduction inputs", ""]
    for source in sources:
        for report in source["report"]:
            lines.append(f"- CSV: `{report}`")
        lines.append(f"- Manifest: `{source['manifest']}`")
    lines.append("")
    return "\n".join(lines)


def write_outputs(prefix: Path, points: dict, sources: list[dict], reference_path: str | None) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    reference = load_expected_ns(reference_path) if reference_path else None
    if reference is not None:
        local_keys, reference_keys = set(points), set(reference)
        if local_keys != reference_keys:
            raise ValueError(
                f"reference universe mismatch: missing={sorted(reference_keys - local_keys)}, "
                f"extra={sorted(local_keys - reference_keys)}"
            )

    txt = expected_ns_text(points)
    md = markdown_text(points, sources, reference)
    prefix.with_suffix(".txt").write_text(txt)
    prefix.with_suffix(".md").write_text(md)
    serial_points = [
        {
            "format": key[0],
            "placement": key[1],
            "model": key[2],
            "count": key[3],
            **points[key],
        }
        for key in canonical_keys()
    ]
    prefix.with_suffix(".json").write_text(json.dumps({"sources": sources, "points": serial_points}, indent=2) + "\n")
    print(txt, end="")
    print(f"wrote {prefix.with_suffix('.txt')}, {prefix.with_suffix('.md')}, and {prefix.with_suffix('.json')}")


def run_measurement(prefix: Path) -> list[tuple[Path, Path]]:
    repo = _repo_root()
    reports_root = Path(os.environ.get("TT_METAL_HOME", repo)) / "generated" / "profiler" / "reports"
    raw = prefix.parent / f"{prefix.name}_raw"
    raw.mkdir(parents=True, exist_ok=True)
    test = "tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_perf_matrix.py"
    source_pairs: list[tuple[Path, Path]] = []

    for emb, fmt in ((7168, "bf16_rm"), (7168, "bfp8_tile"), (6144, "bf16_rm"), (6144, "bfp8_tile")):
        manifest = raw / f"manifest_k{emb}_{fmt}.json"
        before = {path.resolve() for path in reports_root.glob("*/ops_perf_results*.csv")}
        started_ns = time.time_ns()
        env = os.environ.copy()
        env.update(
            {
                "MOE_MATRIX_EMBS": str(emb),
                "MOE_MATRIX_FORMATS": fmt,
                "MOE_MATRIX_WPLACES": "nd_shard,interleaved",
                "MOE_MATRIX_WDTYPES": "bfp4",
                "MOE_MATRIX_OPS": "moe_fused_swiglu",
                "MOE_MATRIX_COUNTS": ",".join(map(str, COUNTS)),
                "MOE_MATRIX_REPS": "3",
                "MOE_MATRIX_WARMUP": "1",
                "MOE_MATRIX_MANIFEST": str(manifest.resolve()),
            }
        )
        subprocess.run(
            [
                str(repo / "scripts" / "run_safe_pytest.sh"),
                "--profile",
                "--no-precompile",
                "--run-all",
                test,
                "-q",
                "-s",
            ],
            cwd=repo,
            env=env,
            check=True,
        )
        candidates = [
            path
            for path in reports_root.glob("*/ops_perf_results*.csv")
            if path.resolve() not in before and path.stat().st_mtime_ns >= started_ns
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one new profiler CSV for K={emb}/{fmt}, found {candidates}")
        saved_csv = raw / f"ops_k{emb}_{fmt}.csv"
        shutil.copy2(candidates[0], saved_csv)
        source_pairs.append((saved_csv, manifest))
    return source_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", help="CSV/report and manifest pairs (omit with --measure)")
    parser.add_argument("--measure", action="store_true", help="run the exact four-session pytest sweep first")
    parser.add_argument("--output-prefix", default="perf_table_8x11", help="path without .txt/.md/.json suffix")
    parser.add_argument("--reference", help="optional upstream EXPECTED_NS file for comparison columns")
    args = parser.parse_args()
    if args.measure and args.sources:
        parser.error("pass either --measure or explicit report/manifest pairs, not both")
    if not args.measure and (not args.sources or len(args.sources) % 2):
        parser.error("without --measure, pass one or more report/manifest pairs")
    return args


def main() -> None:
    args = parse_args()
    prefix = Path(args.output_prefix)
    pairs = run_measurement(prefix) if args.measure else list(zip(args.sources[0::2], args.sources[1::2]))
    points, sources = collect_measurements(pairs)
    write_outputs(prefix, points, sources, args.reference)


if __name__ == "__main__":
    main()
