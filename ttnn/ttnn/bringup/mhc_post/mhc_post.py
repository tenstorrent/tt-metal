# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""mhc_post — fused post-sublayer half of one mHC wrap, ONE device program per call.

For every token t and output stream j:
    X'[t, j*C:(j+1)*C] = post[t, j] * F[t, :] + sum_i comb[t, i*n + j] * X[t, i*C:(i+1)*C]
(comb applied TRANSPOSED). See op_design.md.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .mhc_post_program_descriptor import TILE_HW, create_program_descriptor

MAX_STREAMS = math.isqrt(TILE_HW)  # mechanism cap (= 5): the n*n comb row must sit in one raw tile row (n*n <= 32)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
def tag_alignment(inputs, axes):
    """(F_shape, X_shape, post_shape, comb_shape) -> tile alignment of the token dim T (dim -2)."""
    f_shape = inputs[0]
    return "tile_aligned" if f_shape[-2] % TILE_HW == 0 else "h_non_aligned"


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
    "alignment": ["tile_aligned", "h_non_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """HiFi4, fp32 DEST, approx off."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------
def validate(input_tensor, residual, post, comb, *, compute_kernel_config=None):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = {
        "dtype": residual.dtype,
        "sublayer_dtype": input_tensor.dtype,
        "layout": residual.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
    }
    tagger_inputs = (list(input_tensor.shape), list(residual.shape), list(post.shape), list(comb.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(tagger_inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"mhc_post: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"mhc_post: unsupported combination (refinement candidate): {exc}")

    _validate_shapes(input_tensor, residual, post, comb)
    return cfg


def _validate_shapes(f, x, post, comb):
    shapes = {name: list(t.shape) for name, t in (("input_tensor", f), ("residual", x), ("post", post), ("comb", comb))}
    for name, shape in shapes.items():
        if not 2 <= len(shape) <= 4:
            raise ValueError(f"mhc_post: {name} rank must be 2..4, got {shape}")
    lead = shapes["input_tensor"][:-1]
    for name, shape in shapes.items():
        if shape[:-1] != lead:
            raise ValueError(f"mhc_post: leading dims of {name} {shape[:-1]} != input_tensor's {lead}")
    C = shapes["input_tensor"][-1]
    n = shapes["post"][-1]
    if C % TILE_HW != 0:
        raise ValueError(f"mhc_post: C={C} must be a multiple of {TILE_HW}")
    if not 1 <= n <= MAX_STREAMS:
        raise ValueError(f"mhc_post: n={n} must be in [1, {MAX_STREAMS}]")
    if shapes["residual"][-1] != n * C:
        raise ValueError(f"mhc_post: residual last dim {shapes['residual'][-1]} != n*C = {n * C}")
    if shapes["comb"][-1] != n * n:
        raise ValueError(f"mhc_post: comb last dim {shapes['comb'][-1]} != n*n = {n * n}")
    for name, t in (("post", post), ("comb", comb)):
        if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"mhc_post: {name} must be float32 TILE")
    for name, t in (("input_tensor", f), ("residual", x), ("post", post), ("comb", comb)):
        if t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"mhc_post: {name} must be TILE_LAYOUT")
        mc = t.memory_config()
        if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED or mc.buffer_type != ttnn.BufferType.DRAM:
            raise ValueError(f"mhc_post: {name} must be DRAM interleaved")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def mhc_post(input_tensor, residual, post, comb, *, compute_kernel_config=None) -> ttnn.Tensor:
    cfg = validate(input_tensor, residual, post, comb, compute_kernel_config=compute_kernel_config)

    device = residual.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(residual.shape)),
        residual.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    program_descriptor = create_program_descriptor(input_tensor, residual, post, comb, output_tensor, cfg)
    return ttnn.generic_op([input_tensor, residual, post, comb, output_tensor], program_descriptor)
