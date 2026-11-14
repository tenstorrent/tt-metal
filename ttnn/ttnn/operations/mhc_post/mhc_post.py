# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post — fused post-sublayer half of one mHC wrap (registry model).

    X'_j = post_j * F + sum_i comb[i][j] * X_i     (comb applied transposed)

Phase 0 builds regime R1 (plain mode, see op_design.md): one device program,
work split over token-tile rows x C-column tiles, every coefficient product
in fp32 SFPU lanes.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .mhc_post_program_descriptor import create_program_descriptor

N_STREAMS = 4  # DeepSeek-V4 / V4.1 hc_mult (contract)
NG = N_STREAMS * (N_STREAMS + 2)
FEED_GROUP = 4  # feed_split tagger: C splits into column blocks for a group of up to 4 chips


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs = (F_shape, X_shape, post_shape, comb_shape)
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Token dim (dim -2 of F) tile-aligned or not."""
    f_shape = inputs[0]
    return "tile_aligned" if f_shape[-2] % 32 == 0 else "h_non_aligned"


def tag_feed_split(inputs, axes):
    """C (F's last dim) splits into FEED_GROUP column blocks of whole tiles."""
    C = inputs[0][-1]
    return "block_aligned" if C % (32 * FEED_GROUP) == 0 else "not_block_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "feed_split": tag_feed_split,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0 baseline (R1 plain, fp32 streams / sublayer)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16],
    "sublayer_dtype": [ttnn.float32, ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True],
    "mode": ["plain"],
    "coef_source": ["separate", "packed"],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "norm": ["off"],
    "feed": ["off"],
    "feed_split": ["block_aligned", "not_block_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Compute config
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """HiFi4, fp32 DEST, approximation off."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_call(
    input_tensor,
    residual,
    post,
    comb,
    *,
    post_col,
    comb_col,
    stats_weight,
    collapse_pre,
    hc_scale,
    hc_base,
    norm_weight,
    feed_output,
    feed_semaphores,
    output_tensor,
):
    """Malformed calls (not support refusals): clear ValueErrors."""
    if collapse_pre is not None and stats_weight is None:
        raise ValueError("mhc_post: collapse_pre requires stats_weight")
    if collapse_pre is not None:
        if hc_scale is None or len(hc_scale) != 3:
            raise ValueError("mhc_post: collapse mode needs hc_scale = (a_pre, a_post, a_res)")
        if hc_base is None or len(hc_base) != NG:
            raise ValueError(f"mhc_post: collapse mode needs hc_base of {NG} values")
    if norm_weight is not None and collapse_pre is None:
        raise ValueError("mhc_post: norm_weight requires collapse mode (collapse_pre)")
    if feed_output is not None:
        if collapse_pre is None:
            raise ValueError("mhc_post: feed_output requires collapse mode (collapse_pre)")
        if feed_semaphores is None or len(feed_semaphores) != 2:
            raise ValueError("mhc_post: feed_output requires feed_semaphores = [ready, arrival]")
    f_shape, x_shape = tuple(input_tensor.shape), tuple(residual.shape)
    if len(f_shape) < 2 or len(f_shape) > 4 or len(x_shape) != len(f_shape):
        raise ValueError(f"mhc_post: F {f_shape} and X {x_shape} must have the same rank, 2..4")
    C = f_shape[-1]
    if C % 32 != 0:
        raise ValueError(f"mhc_post: C={C} must be a multiple of 32")
    if f_shape[:-1] != x_shape[:-1] or x_shape[-1] != N_STREAMS * C:
        raise ValueError(f"mhc_post: X {x_shape} must be (..., T, {N_STREAMS}*C) for F {f_shape}")
    for name, t, col, width in (("post", post, post_col, N_STREAMS), ("comb", comb, comb_col, N_STREAMS**2)):
        s = tuple(t.shape)
        if s[:-1] != f_shape[:-1]:
            raise ValueError(f"mhc_post: {name} {s} must share F's leading dims {f_shape[:-1]}")
        if col < 0 or col + width > 32 or col + width > s[-1]:
            raise ValueError(f"mhc_post: {name}_col={col} must keep {width} values inside one tile row of {s}")
        if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"mhc_post: {name} must be float32 TILE")
    if output_tensor is not None:
        if (
            tuple(output_tensor.shape) != x_shape
            or output_tensor.dtype != residual.dtype
            or output_tensor.layout != ttnn.TILE_LAYOUT
            or output_tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("mhc_post: output_tensor must match X' (shape, dtype, TILE, DRAM interleaved)")
        addr = output_tensor.buffer_address()
        for t in (input_tensor, residual, post, comb, stats_weight, collapse_pre, norm_weight):
            if t is not None and t.buffer_address() == addr:
                raise ValueError("mhc_post: output_tensor aliases an input")


def validate(
    input_tensor,
    residual,
    post,
    comb,
    *,
    compute_kernel_config=None,
    post_col=0,
    comb_col=0,
    stats_weight=None,
    collapse_pre=None,
    hc_scale=None,
    hc_base=None,
    norm_weight=None,
    feed_output=None,
    feed_semaphores=None,
    output_tensor=None,
):
    _check_call(
        input_tensor,
        residual,
        post,
        comb,
        post_col=post_col,
        comb_col=comb_col,
        stats_weight=stats_weight,
        collapse_pre=collapse_pre,
        hc_scale=hc_scale,
        hc_base=hc_base,
        norm_weight=norm_weight,
        feed_output=feed_output,
        feed_semaphores=feed_semaphores,
        output_tensor=output_tensor,
    )
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    if collapse_pre is not None:
        mode = "collapse"
    elif stats_weight is not None:
        mode = "stats"
    else:
        mode = "plain"
    layout = (
        ttnn.TILE_LAYOUT
        if all(t.layout == ttnn.TILE_LAYOUT for t in (input_tensor, residual))
        else ttnn.ROW_MAJOR_LAYOUT
    )
    axes = {
        "dtype": residual.dtype,
        "sublayer_dtype": input_tensor.dtype,
        "layout": layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "mode": mode,
        "coef_source": "packed" if post.buffer_address() == comb.buffer_address() else "separate",
        "norm": "on" if norm_weight is not None else "off",
        "feed": "on" if feed_output is not None else "off",
    }
    shapes = (tuple(input_tensor.shape), tuple(residual.shape), tuple(post.shape), tuple(comb.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(shapes, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"mhc_post: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"mhc_post: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def mhc_post(
    input_tensor: ttnn.Tensor,
    residual: ttnn.Tensor,
    post: ttnn.Tensor,
    comb: ttnn.Tensor,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    post_col: int = 0,
    comb_col: int = 0,
    stats_weight: ttnn.Tensor = None,
    collapse_pre: ttnn.Tensor = None,
    hc_scale: list = None,
    hc_base: list = None,
    hc_norm_width: float = None,
    hc_norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    hc_sinkhorn_iters: int = 20,
    output_tensor: ttnn.Tensor = None,
    stats_output: ttnn.Tensor = None,
    hc_output: ttnn.Tensor = None,
    y_output: ttnn.Tensor = None,
    max_grid: ttnn.CoreCoord = None,
    norm_weight: ttnn.Tensor = None,
    norm_eps: float = 1e-6,
    feed_output: ttnn.Tensor = None,
    feed_semaphores: list = None,
    feed_num_links: int = 1,
    feed_cluster_axis: int = 1,
):
    axes = validate(
        input_tensor,
        residual,
        post,
        comb,
        compute_kernel_config=compute_kernel_config,
        post_col=post_col,
        comb_col=comb_col,
        stats_weight=stats_weight,
        collapse_pre=collapse_pre,
        hc_scale=hc_scale,
        hc_base=hc_base,
        norm_weight=norm_weight,
        feed_output=feed_output,
        feed_semaphores=feed_semaphores,
        output_tensor=output_tensor,
    )
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    device = residual.device()

    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(residual.shape)),
            residual.dtype,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

    program_descriptor = create_program_descriptor(
        input_tensor,
        residual,
        post,
        comb,
        output_tensor,
        n=N_STREAMS,
        post_col=post_col,
        comb_col=comb_col,
        math_fidelity=cfg.math_fidelity,
        math_approx_mode=bool(cfg.math_approx_mode),
        max_grid=max_grid,
        mode=axes["mode"],
    )
    # Output tensor must be last.
    return ttnn.generic_op([input_tensor, residual, post, comb, output_tensor], program_descriptor)
