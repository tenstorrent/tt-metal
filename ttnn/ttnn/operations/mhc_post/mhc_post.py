# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post — fused post-sublayer half of one Manifold-Constrained Hyper-Connection.

    X'_j = post_j * F + sum_i comb[i][j] * X_i        (comb applied transposed)

computed in fp32 (SFPU, DEST) and rounded once to X.dtype, in ONE device program.

Regimes built (op_design.md → Blocking Model → Regimes):
  plain-2d       X' only; float32 / bfloat16 streams and sublayer; separate or packed mixes.
  stats-2d-tree  + stats = X' @ W (col 24 = sum of the unrounded mix squared), bf16/bf16,
                 cross-core deterministic two-level combine.
collapse is a later refinement.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .mhc_post_program_descriptor import N_STREAMS, create_program_descriptor


# ---------------------------------------------------------------------------
# Compute config
# ---------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """HiFi4, fp32 DEST, approx off — the only default `compute_kernel_config=None` resolves to."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """`inputs` = (F_shape, X_shape, post_shape, comb_shape). C % 32 == 0 by contract,
    so only the token dim T (dim -2) can break tile alignment."""
    x_shape = inputs[1]
    return "tile_aligned" if x_shape[-2] % 32 == 0 else "h_non_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16],
    "sublayer_dtype": [ttnn.float32, ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True],
    "mode": ["plain", "stats"],
    "coef_source": ["separate", "packed"],
    "alignment": ["tile_aligned", "h_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _mode_of(stats_weight, collapse_pre):
    if collapse_pre is not None:
        return "collapse"
    if stats_weight is not None:
        return "stats"
    return "plain"


def _same_buffer(a, b):
    if a is b:
        return True
    try:
        return a.buffer_address() == b.buffer_address()
    except Exception:  # noqa: BLE001 — host tensors / no buffer
        return False


def _check_caller_buffer(name, buf, shape, dtype, inputs):
    if buf is None:
        return
    if list(buf.shape) != list(shape):
        raise ValueError(f"mhc_post: {name} shape {list(buf.shape)} != expected {list(shape)}")
    if buf.dtype != dtype:
        raise ValueError(f"mhc_post: {name} dtype {buf.dtype} != expected {dtype}")
    if buf.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"mhc_post: {name} must be TILE_LAYOUT")
    if buf.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError(f"mhc_post: {name} must be DRAM interleaved")
    for t in inputs:
        if t is not None and _same_buffer(buf, t):
            raise ValueError(f"mhc_post: {name} aliases an input tensor")


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
    output_tensor=None,
    stats_output=None,
):
    n = N_STREAMS
    # --- (1) structural errors (ValueError) ---
    if collapse_pre is not None and stats_weight is None:
        raise ValueError("mhc_post: collapse_pre requires stats_weight")
    mode = _mode_of(stats_weight, collapse_pre)
    if mode == "collapse":
        if hc_scale is None or len(hc_scale) != 3:
            raise ValueError("mhc_post: collapse mode needs hc_scale with 3 values")
        if hc_base is None or len(hc_base) != n * (n + 2):
            raise ValueError(f"mhc_post: collapse mode needs hc_base with {n * (n + 2)} values")

    f_shape, x_shape = list(input_tensor.shape), list(residual.shape)
    p_shape, c_shape = list(post.shape), list(comb.shape)
    if len(x_shape) < 2 or len(x_shape) > 4:
        raise ValueError(f"mhc_post: residual rank must be 2..4, got {x_shape}")
    C = f_shape[-1]
    if C % 32 != 0:
        raise ValueError(f"mhc_post: C={C} must be a multiple of 32")
    if x_shape[-1] != n * C:
        raise ValueError(f"mhc_post: residual last dim {x_shape[-1]} != n*C = {n}*{C} (n must be {n})")
    lead = x_shape[:-1]
    for name, s in (("input_tensor", f_shape), ("post", p_shape), ("comb", c_shape)):
        if s[:-1] != lead:
            raise ValueError(f"mhc_post: {name} leading dims {s[:-1]} != residual leading dims {lead}")
    for name, col, width, w in (("post", post_col, n, p_shape[-1]), ("comb", comb_col, n * n, c_shape[-1])):
        if col < 0 or col + width > w:
            raise ValueError(f"mhc_post: {name}_col={col} + {width} exceeds width {w}")
        if col % 32 + width > 32:
            raise ValueError(f"mhc_post: {name}_col={col} window crosses a 32-wide tile column")
    for name, t in (("post", post), ("comb", comb)):
        if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"mhc_post: {name} must be float32 TILE")
    # The reader / writer address every tensor through an interleaved TensorAccessor page walk;
    # a sharded tensor would be read as if interleaved (TARGET has no memory_layout axis).
    for name, t in (("input_tensor", input_tensor), ("residual", residual), ("post", post), ("comb", comb)):
        if t.memory_config().is_sharded():
            raise ValueError(f"mhc_post: {name} must be interleaved (sharded inputs are not in the contract)")
    inputs = (input_tensor, residual, post, comb, stats_weight, collapse_pre)
    if stats_weight is not None:
        w_shape = list(stats_weight.shape)
        if len(w_shape) < 2 or w_shape[-2:] != [n * C, 32] or math.prod(w_shape[:-2]) != 1:
            raise ValueError(f"mhc_post: stats_weight shape {w_shape} must be (1.., n*C={n * C}, 32)")
        if stats_weight.dtype != ttnn.float32 or stats_weight.layout != ttnn.TILE_LAYOUT:
            raise ValueError("mhc_post: stats_weight must be float32 TILE")
        if stats_weight.memory_config().is_sharded():
            raise ValueError("mhc_post: stats_weight must be interleaved")
    _check_caller_buffer("output_tensor", output_tensor, x_shape, residual.dtype, inputs)
    if mode != "plain":
        _check_caller_buffer("stats_output", stats_output, x_shape[:-1] + [32], ttnn.float32, inputs)

    # --- (2) per-axis SUPPORTED, in TARGET order ---
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = {
        "dtype": residual.dtype,
        "sublayer_dtype": input_tensor.dtype,
        "layout": residual.layout if residual.layout == input_tensor.layout else input_tensor.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "mode": mode,
        "coef_source": "packed" if _same_buffer(post, comb) else "separate",
    }
    tagger_inputs = (f_shape, x_shape, p_shape, c_shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(tagger_inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"mhc_post: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # --- (3) EXCLUSIONS ---
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
    compute_kernel_config=None,
    post_col: int = 0,
    comb_col: int = 0,
    stats_weight=None,
    collapse_pre=None,
    hc_scale=None,
    hc_base=None,
    hc_norm_width=None,
    hc_norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    hc_sinkhorn_iters: int = 20,
    output_tensor=None,
    stats_output=None,
    hc_output=None,
    y_output=None,
    max_grid=None,
):
    """Write the n = 4 new residual streams X' (plain) and, with `stats_weight`, the next hc
    step's stats row (stats mode). See module docstring."""
    validate(
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
        output_tensor=output_tensor,
        stats_output=stats_output,
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
    fused = stats_weight is not None
    if fused and stats_output is None:
        stats_output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(residual.shape)[:-1] + [32]),
            ttnn.float32,
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
        compute_config=cfg,
        post_col=post_col,
        comb_col=comb_col,
        max_grid=max_grid,
        stats_weight=stats_weight,
        stats_output=stats_output,
    )
    if not fused:
        # Output tensor MUST be last in the list.
        return ttnn.generic_op([input_tensor, residual, post, comb, output_tensor], program_descriptor)
    ttnn.generic_op([input_tensor, residual, post, comb, stats_weight, stats_output, output_tensor], program_descriptor)
    return output_tensor, stats_output
