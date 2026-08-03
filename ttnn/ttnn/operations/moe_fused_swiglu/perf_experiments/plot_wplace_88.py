#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Plot the weight-placement A/B at a fixed core count.

    perf_experiments/plot_wplace_88.py <sweep88.json> <out.png>

Hue stays FORMAT (same mapping as `plot_seqlen_sweep.py`, so the two figures read together) and
placement is carried by line style — a secondary encoding, so the placement pair never rests on
colour. x is log-scaled: the count list is geometric, not uniform.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
SERIES = {"bf16_rm": "#2a78d6", "bfp8_tile": "#eb6834"}
LABEL = {"bf16_rm": "bfloat16 · ROW_MAJOR", "bfp8_tile": "bfloat8_b · TILE"}
STYLE = {"interleaved": "-", "nd_shard": (0, (5, 2))}


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="semibold", loc="left", pad=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)


def main():
    src, out = sys.argv[1], sys.argv[2]
    pts = json.load(open(src))["points"]
    fmts = [f for f in ("bf16_rm", "bfp8_tile") if any(p["format"] == f for p in pts)]
    cores = pts[0]["cores"]
    grid = pts[0]["grid"]
    counts = sorted({p["count"] for p in pts})

    def get(f, w, c):
        return next(p for p in pts if p["format"] == f and p["wplace"] == w and p["count"] == c)

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.72, bottom=0.13, wspace=0.20)
    fig.suptitle(
        f"moe_fused_swiglu — DRAM weight placement at {cores} cores",
        x=0.06,
        y=0.965,
        ha="left",
        va="top",
        color=INK,
        fontsize=16,
        fontweight="semibold",
    )
    fig.text(
        0.06,
        0.885,
        f"Grid {grid} ({cores} cores), emb 7168, hidden 2048, capacity 5120, bfloat4_b weights, LoFi. "
        f"Median of 7 reps.\nInterleaved = one NoC request per weight tile; ND shard "
        f"(`weight_memory_configs`) = one request per K-row of a core's N slice.",
        ha="left",
        va="top",
        color=INK2,
        fontsize=9.5,
        linespacing=1.5,
    )

    for f in fmts:
        for w in ("interleaved", "nd_shard"):
            ys = [get(f, w, c)["us_median"] for c in counts]
            a.plot(
                counts,
                ys,
                color=SERIES[f],
                linewidth=2.0,
                linestyle=STYLE[w],
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                zorder=3,
            )
        wins = [
            100
            * (get(f, "nd_shard", c)["ns_median"] - get(f, "interleaved", c)["ns_median"])
            / get(f, "interleaved", c)["ns_median"]
            for c in counts
        ]
        b.plot(
            counts,
            wins,
            color=SERIES[f],
            linewidth=2.0,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            zorder=3,
            label=LABEL[f],
        )
        b.annotate(
            f"{wins[-1]:+.1f}%",
            (counts[-1], wins[-1]),
            xytext=(6, -3),
            textcoords="offset points",
            color=SERIES[f],
            fontsize=8.5,
        )

    style(a, "tokens (count)", "device kernel duration  [us]", "A · Sharding the weights helps only at short sequences")
    a.set_xscale("log")
    a.set_yscale("log")
    a.legend(
        handles=[Line2D([], [], color=SERIES[f], lw=2, label=LABEL[f]) for f in fmts]
        + [Line2D([], [], color=INK2, lw=2, linestyle=STYLE[w], label=w) for w in ("interleaved", "nd_shard")],
        frameon=False,
        fontsize=9,
        labelcolor=INK2,
        loc="upper left",
    )

    # NOT "the win tracks the weight share": the 64- and 384-token points wobble against that story,
    # and the interleaved baseline's own rep spread (3-6 % below 512 tokens) is the same size as the
    # wobble. What the data does support is the two endpoints and the direction between them.
    style(
        b,
        "tokens (count)",
        "ND shard vs interleaved  [%]   (negative = shard faster)",
        "B · ~10 % at short sequences, gone by 2k tokens",
    )
    b.axhline(0, color=INK2, linewidth=1.0, zorder=2)
    b.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")

    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xticks([32, 64, 128, 256, 512, 1024, 2048, 5120])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.minorticks_off()
    a.set_yticks([80, 100, 200, 400, 800, 2000])
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
