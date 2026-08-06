#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Plot local and upstream EXPECTED_NS tables for BF16 row-major activations.

    python ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/plot_expected_perf_table.py \\
        perf_table_8x11.txt /path/to/perf_table_upstream.txt \\
        --output-dir generated/perf_table_8x11_graphs

Produces full-range and M<=512 zoom figures in both PNG and SVG formats. Each model panel has four
curves: local/upstream crossed with ND-sharded/interleaved weights.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt


COUNTS = (0, 64, 128, 256, 512, 1024, 2048, 4096, 5120)
MODELS = (("kimi_k26", "Kimi K2.6 · K=7168, N=2048"), ("glm_51", "GLM 5.1 · K=6144, N=2048"))
SERIES = (
    ("Local ND-sharded", "local", "w_ndshard", "#0072B2", "-", "o"),
    ("Local interleaved", "local", "w_interleaved", "#0072B2", "--", "^"),
    ("Upstream ND-sharded", "upstream", "w_ndshard", "#D55E00", "-", "s"),
    ("Upstream interleaved", "upstream", "w_interleaved", "#D55E00", "--", "D"),
)


def load_expected_ns(path: Path) -> dict[tuple[str, str, str, int], int]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "EXPECTED_NS" for target in targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, dict):
                    return value
    raise ValueError(f"could not find a literal EXPECTED_NS dictionary in {path}")


def make_figure(tables: dict, counts: tuple[int, ...], title: str, output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1), sharey=True)
    for axis, (model, model_title) in zip(axes, MODELS):
        for label, source, placement, color, linestyle, marker in SERIES:
            values_us = [tables[source][("x_rm", placement, model, count)] / 1000 for count in counts]
            axis.plot(
                counts,
                values_us,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=2.2,
                markersize=5.5,
            )
        axis.set_title(model_title)
        axis.set_xlabel("M (tokens)")
        axis.set_xticks(counts if max(counts) <= 512 else (0, 256, 512, 1024, 2048, 4096, 5120))
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Device kernel time (µs)")
    fig.suptitle(title, y=0.98)
    fig.text(
        0.5,
        0.925,
        "BF16 row-major activations · BFP4 weights · 8×11 grid (88 cores)",
        ha="center",
        fontsize=12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncols=4, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.72, wspace=0.035)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("local", type=Path, help="local EXPECTED_NS table")
    parser.add_argument("upstream", type=Path, help="upstream EXPECTED_NS table")
    parser.add_argument("--output-dir", type=Path, default=Path("generated/perf_table_8x11_graphs"))
    args = parser.parse_args()

    tables = {"local": load_expected_ns(args.local), "upstream": load_expected_ns(args.upstream)}
    required = {
        ("x_rm", placement, model, count)
        for _, _, placement, _, _, _ in SERIES
        for model, _ in MODELS
        for count in COUNTS
    }
    for name, table in tables.items():
        missing = required - table.keys()
        if missing:
            raise ValueError(f"{name} table is missing required cells: {sorted(missing)}")

    make_figure(
        tables,
        COUNTS,
        "moe_fused_swiglu scaling: local vs upstream",
        args.output_dir / "bf16_rm_m_scaling_full",
    )
    zoom_counts = tuple(count for count in COUNTS if count <= 512)
    make_figure(
        tables,
        zoom_counts,
        "moe_fused_swiglu small-M scaling: local vs upstream",
        args.output_dir / "bf16_rm_m_scaling_zoom_m512",
    )
    print(f"wrote plots under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
