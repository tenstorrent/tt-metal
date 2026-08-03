#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Plot the sequence-length scaling of moe_fused_swiglu from a parsed sweep.

    perf_experiments/plot_seqlen_sweep.py <sweep.json> <out.png>

Four panels, one measure each — never two y-scales on one axis. Series identity is carried by a
legend AND a direct end-label, so it never rests on colour alone.
"""

import json
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
# Categorical slots 1 and 2 of the reference palette, in fixed order (validated: adjacent CVD
# dE 24.7 protan, normal-vision 33.6, both >= 3:1 on this surface).
SERIES = {"bf16_rm": "#2a78d6", "bfp8_tile": "#eb6834"}
LABEL = {"bf16_rm": "bfloat16 · ROW_MAJOR", "bfp8_tile": "bfloat8_b · TILE"}

#: Blackhole p150: 110 Tensix @ 1.35 GHz. Used only to label the panels, never to derive a number.
CORES, CLOCK_GHZ = 110, 1.35
HIDDEN = 2048
M_BLOCK = 8  # moe_fused_swiglu_program_descriptor.M_BLOCK default


def work_rows(count):
    """Tile-rows the op actually PROCESSES for `count` tokens.

    Full M-blocks plus a tail block whose work rounds UP to a power of two (the descriptor's
    `m_tiles_eff`). This — not `count` — is what latency is linear in, which is why half the 32-token
    steps of the sweep are free and the other half cost up to a whole M-block.
    """
    mt = count // 32
    full, tail = (mt // M_BLOCK) * M_BLOCK, mt % M_BLOCK
    if tail:
        p = 1
        while p < tail:
            p *= 2
        tail = p
    return full + tail


def flops(count, emb):
    """Three matmuls (gate, up, down), each 2*M*K*N."""
    return 3 * 2 * count * emb * HIDDEN


def fixed_cost(sub):
    """Least-squares intercept over the linear tail — the count-independent floor, in us."""
    tail = [p for p in sub if p["count"] >= 1024]
    xs = [p["count"] for p in tail]
    ys = [p["us_median"] for p in tail]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope * 1000.0  # us, ns/token


def style(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="semibold", loc="left", pad=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)


def endlabel(ax, x, y, text, color, dx=60, va="center"):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", color=color, fontsize=8.5, va=va, ha="left")


def main():
    src, out = sys.argv[1], sys.argv[2]
    doc = json.load(open(src))
    pts = doc["points"]
    fmts = [f for f in ("bf16_rm", "bfp8_tile") if any(p["format"] == f for p in pts)]
    emb = pts[0]["emb"]
    cap = pts[0]["capacity"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.065, right=0.9, top=0.825, bottom=0.065, hspace=0.30, wspace=0.22)
    a, b, c, d = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    fig.suptitle(
        "moe_fused_swiglu — sequence-length scaling",
        x=0.065,
        y=0.977,
        ha="left",
        va="top",
        color=INK,
        fontsize=16,
        fontweight="semibold",
    )
    floors = {f: fixed_cost([p for p in pts if p["format"] == f]) for f in fmts}
    fig.text(
        0.065,
        0.915,
        f"Tokens routed to one local expert, 32 -> {cap} step 32 (all {len(pts) // len(fmts)} counts, median of 3). "
        f"emb {emb}, hidden {HIDDEN}, bfloat4_b weights, LoFi, capacity {cap}.\n"
        f"Blackhole p150, {CORES} Tensix @ {CLOCK_GHZ} GHz. DEVICE KERNEL DURATION via Tracy; "
        f"rep spread <= 2.9 % on every point.",
        ha="left",
        va="top",
        color=INK2,
        fontsize=9.5,
        linespacing=1.5,
    )

    for f in fmts:
        sub = sorted((p for p in pts if p["format"] == f), key=lambda p: p["count"])
        col = SERIES[f]
        xs = [p["count"] for p in sub]

        ys = [p["us_median"] for p in sub]
        a.plot(xs, ys, color=col, linewidth=2.0, zorder=3, label=LABEL[f])
        endlabel(a, xs[-1], ys[-1], f"{ys[-1]:.0f} us", col, dx=8)

        ys = [p["ns_per_token"] for p in sub]
        b.plot(xs, ys, color=col, linewidth=2.0, zorder=3, label=LABEL[f])
        endlabel(b, xs[-1], ys[-1], f"{ys[-1]:.0f}", col, dx=8)

        ys = [p["dram_util"] * 100 for p in sub]
        c.plot(xs, ys, color=col, linewidth=2.0, zorder=3, label=LABEL[f])
        endlabel(c, xs[-1], ys[-1], f"{ys[-1]:.0f} %", col, dx=8)

        ys = [flops(p["count"], p["emb"]) / (p["ns_median"] * 1e-9) / 1e12 for p in sub]
        d.plot(xs, ys, color=col, linewidth=2.0, zorder=3, label=LABEL[f])
        endlabel(d, xs[-1], ys[-1], f"{ys[-1]:.0f}", col, dx=8)

    style(
        a,
        "tokens (count)",
        "device kernel duration  [us]",
        "A · Latency rises in M-block steps above a fixed floor",
    )
    # The count-independent floor: three whole weight sets must be read whatever the token count is.
    lo = min(v[0] for v in floors.values())
    a.axhline(lo, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    a.annotate(
        f"fixed floor ~{lo:.0f} us (weights only)",
        (0, lo),
        xytext=(6, 6),
        textcoords="offset points",
        color=INK2,
        fontsize=8.5,
    )
    a.set_ylim(0, None)
    a.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left", bbox_to_anchor=(0.02, 0.93))

    # ZOOM on the last 512 tokens: at full-sweep scale the steps read as a thick line, and the step
    # structure is the whole point — the curve is a function of work_rows(count), not of count.
    zf = fmts[0]
    zsub = sorted((p for p in pts if p["format"] == zf and p["count"] > cap - 512), key=lambda p: p["count"])
    ins = a.inset_axes([0.55, 0.12, 0.43, 0.33])
    style(ins, "", "", "")
    ins.plot(
        [p["count"] for p in zsub],
        [p["us_median"] for p in zsub],
        color=SERIES[zf],
        linewidth=1.6,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        zorder=3,
    )
    # The WIDEST equal-work run in the zoom window: those counts are free relative to each other.
    runs = {}
    for p in zsub:
        runs.setdefault(work_rows(p["count"]), []).append(p)
    flat = max(runs.values(), key=lambda v: v[-1]["count"] - v[0]["count"])
    if len(flat) > 1:
        ins.axvspan(flat[0]["count"], flat[-1]["count"], color=SERIES[zf], alpha=0.12, zorder=1)
        ins.annotate(
            f"{flat[0]['count']}→{flat[-1]['count']} tokens\ncosts +0 us",
            (flat[0]["count"], flat[0]["us_median"]),
            xytext=(-2, -34),
            textcoords="offset points",
            color=INK2,
            fontsize=7.5,
            linespacing=1.4,
        )
    ins.text(
        0.03,
        0.94,
        f"zoom · last 512 tokens ({LABEL[zf]})",
        transform=ins.transAxes,
        color=INK2,
        fontsize=7.5,
        va="top",
    )
    ins.tick_params(labelsize=7)
    ins.set_xticks([cap - 448, cap - 256, cap - 64])
    ins.margins(y=0.22)

    style(b, "tokens (count)", "cost per token  [ns]", "B · Per-token cost amortises the weight read")
    b.set_yscale("log")
    b.set_yticks([300, 400, 600, 1000, 2000])
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")

    style(
        c, "tokens (count)", "DRAM read utilisation  [% of 512 GB/s]", "C · Bandwidth-limited only at short sequences"
    )
    c.set_ylim(0, None)
    c.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")

    style(d, "tokens (count)", "matmul throughput  [TFLOP/s]", "D · Compute throughput saturates by ~2k tokens")
    d.set_ylim(0, None)
    d.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")

    for ax in (a, b, c, d):
        ax.set_xlim(0, cap * 1.045)
        ax.set_xticks([0, 1024, 2048, 3072, 4096, 5120])

    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    for f in fmts:
        floor, per_tok = floors[f]
        print(f"  {f:10s} tail fit: {per_tok:.1f} ns/token + {floor:.1f} us fixed")


if __name__ == "__main__":
    main()
