# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — LayerNorm op that natively handles both ROW_MAJOR and TILE input layouts.

Implements:
    out[..., i, j] = gamma[j] * (x[..., i, j] - mean_i) / sqrt(var_i + eps) + beta[j]

where mean_i and var_i are computed over the last dimension. gamma and beta are optional
and, when supplied, must be ROW_MAJOR (1, 1, 1, W) tensors whose dtype matches the input.

Algorithm: three-pass streaming (input read three times from DRAM):
  Pass 1: mean    via reduce<SUM, REDUCE_ROW> with scaler 1/W
  Pass 2: variance via sub<COL> + square_in_place + reduce<SUM, REDUCE_ROW>,
                  then transform_in_place to rsqrt(var + eps)
  Pass 3: normalize + optional affine + drain, per-row.

Single-core; loops over `total_tile_rows = ceil(prod(shape[:-1]) / 32)` row-blocks.

This module declares the four registry-model objects expected by
`eval/golden_tests/layer_norm_rm/test_golden.py` and `eval.verify_supported`:

    INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate()

INVALID (structural impossibilities) is **not** declared here — it lives in
`eval/golden_tests/layer_norm_rm/feature_spec.py`.
"""

import ttnn

from .layer_norm_rm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# Project the input shape onto the categorical `alignment` and `rank` axes
# declared in `eval/golden_tests/layer_norm_rm/feature_spec.py`.


def tag_alignment(inputs, axes):
    """Three-way split over (H, W) tile alignment.

    - tile_aligned   : both H (-2) and W (-1) divisible by 32.
    - w_non_aligned  : W not divisible by 32 (regardless of H).
    - h_non_aligned  : W aligned, H not aligned.
    """
    shape = inputs[0]
    H, W = int(shape[-2]), int(shape[-1])
    if W % 32 != 0:
        return "w_non_aligned"
    if H % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    """Rank of the activation tensor (matches feature_spec.TARGET["rank"])."""
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# What the op handles correctly *right now*. Gaps vs. feature_spec.TARGET are
# refinement candidates (see op_requirements.md).
#
# Phase 0 notes:
#   * bfloat8_b activation is NOT supported — the compute kernel produces
#     out-of-range tile values that `to_torch` rejects with "datum for bfp8
#     is invalid". Refinement target.
#   * `affine_layout` lists both layouts because the canonical `no_affine`
#     cell pins affine_layout to TILE_LAYOUT (canonicalization in INVALID);
#     actual gamma/beta in TILE layout is caught by EXCLUSIONS below.
#   * `affine_dtype` lists both supported floats; the cross-axis constraint
#     "affine_dtype must equal dtype" is enforced via EXCLUSIONS.

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "affine": ["gamma_beta", "gamma_only", "no_affine"],
    "affine_dtype": [ttnn.bfloat16, ttnn.float32],
    "affine_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# Cells inside cartesian(SUPPORTED) that we reject for now. Each is a
# refinement candidate; moving one out of EXCLUSIONS opens a new test cell.

EXCLUSIONS = [
    # fp32 + ROW_MAJOR + W-partial: kernel correctness gap (PCC ~ 0.2-0.8).
    # The non-tile-aligned-W reduction path interacts badly with fp32 RM
    # output streaming — likely a partial-scaler or pad-handling issue in
    # the RM-output write path. Refinement target.
    {"dtype": ttnn.float32, "layout": ttnn.ROW_MAJOR_LAYOUT, "alignment": "w_non_aligned"},
    # Cross-axis: gamma/beta dtype must equal input dtype in the current op
    # (validate() rejects mismatched dtypes). The independent-affine-dtype
    # case is a refinement target.
    {"dtype": ttnn.bfloat16, "affine": "gamma_only", "affine_dtype": ttnn.float32},
    {"dtype": ttnn.bfloat16, "affine": "gamma_beta", "affine_dtype": ttnn.float32},
    {"dtype": ttnn.float32, "affine": "gamma_only", "affine_dtype": ttnn.bfloat16},
    {"dtype": ttnn.float32, "affine": "gamma_beta", "affine_dtype": ttnn.bfloat16},
    # Gamma/beta must always be ROW_MAJOR_LAYOUT per the op spec (the in-kernel
    # tilize step expects RM sticks). TILE-layout affine is a refinement target.
    {"affine": "gamma_only", "affine_layout": ttnn.TILE_LAYOUT},
    {"affine": "gamma_beta", "affine_layout": ttnn.TILE_LAYOUT},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------
#
# Runtime gate. Order:
#   1. Basic structural checks (rank, gamma/beta shape match) — ValueError
#      (these are not registry axes; they're per-call invariants).
#   2. Registry checks (SUPPORTED + EXCLUSIONS) — NotImplementedError.
#
# NotImplementedError extends RuntimeError, so the immutable acceptance test
# (which catches `(ValueError, RuntimeError)`) still passes.


def _affine_axes(input_tensor, gamma, beta):
    """Project (gamma, beta) onto (affine, affine_dtype, affine_layout) axes.

    For `no_affine`, use the canonical INVALID-aligned values
    (affine_dtype=float32, affine_layout=TILE_LAYOUT) — matching the
    canonical no-affine cell in feature_spec.INVALID.
    """
    if gamma is None and beta is None:
        return "no_affine", ttnn.float32, ttnn.TILE_LAYOUT

    if beta is not None and gamma is None:
        raise ValueError("layer_norm: beta supplied without gamma (unsupported affine mode)")

    weight = gamma  # gamma is always present when affine is in use
    if beta is None:
        return "gamma_only", weight.dtype, weight.layout
    return "gamma_beta", weight.dtype, weight.layout


def validate(input_tensor, gamma=None, beta=None):
    """Validate inputs against SUPPORTED / EXCLUSIONS plus shape invariants.

    Raises:
        ValueError              — rank/shape mismatches or other per-call invariants.
        NotImplementedError     — axis values or combinations outside the registry.
    """
    # --- Structural shape checks (per-call, not axes) ---
    if len(input_tensor.shape) < 2:
        raise ValueError(f"layer_norm: input must have rank >= 2 (got rank {len(input_tensor.shape)})")

    W = int(input_tensor.shape[-1])
    for name, t in (("gamma", gamma), ("beta", beta)):
        if t is None:
            continue
        if int(t.shape[-1]) != W:
            raise ValueError(f"layer_norm: {name}.shape[-1] ({int(t.shape[-1])}) must equal input W ({W})")

    # --- Build axes dict ---
    affine, affine_dtype, affine_layout = _affine_axes(input_tensor, gamma, beta)
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "affine": affine,
        "affine_dtype": affine_dtype,
        "affine_layout": affine_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

    # --- Registry checks ---
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise NotImplementedError(f"layer_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise NotImplementedError(f"layer_norm: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def layer_norm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor | None = None,
    beta: ttnn.Tensor | None = None,
    *,
    epsilon: float = 1e-5,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor | None" = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Per-last-dim LayerNorm with optional gamma scale and beta shift.

    Args:
        input_tensor: rank>=2 ttnn.Tensor on device. dtype ∈ {bfloat16, float32}.
            layout ∈ {ROW_MAJOR_LAYOUT, TILE_LAYOUT}.
        gamma: optional (1, 1, 1, W) ROW_MAJOR tensor with same dtype as input.
        beta: optional (1, 1, 1, W) ROW_MAJOR tensor with same dtype as input.
        epsilon: small positive constant added to variance before sqrt. Default 1e-5.
        compute_kernel_config: optional ComputeConfigDescriptor controlling math fidelity, etc.
        memory_config: output memory config (default: DRAM interleaved).

    Returns:
        ttnn.Tensor with same shape/dtype/layout as `input_tensor`.
    """
    validate(input_tensor, gamma, beta)

    if not (epsilon > 0):
        raise ValueError(f"layer_norm: epsilon must be > 0 (got {epsilon})")

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    # Build kernel inputs list — order must match the runtime args assumed by the kernels.
    op_inputs: list[ttnn.Tensor] = [input_tensor]
    if gamma is not None:
        op_inputs.append(gamma)
    if beta is not None:
        op_inputs.append(beta)
    # Output tensor MUST be last.
    op_inputs.append(output_tensor)

    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma,
        beta,
        output_tensor,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )

    return ttnn.generic_op(op_inputs, program_descriptor)
