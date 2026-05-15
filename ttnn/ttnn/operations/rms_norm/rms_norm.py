# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
rms_norm — main entry point + registry-model declarations.

Mathematical definition:
    output[..., i, j] = input[..., i, j] / sqrt(mean(input[..., i, :]^2) + epsilon) * gamma[j]

Registry contract (see eval/REGISTRY_MODEL.md):
  - SHAPE_TAGGERS    — project (input_shape, axes) onto categorical axes.
  - SUPPORTED        — per-axis lists of values the kernel handles right now.
  - EXCLUSIONS       — cell-dicts inside cartesian(SUPPORTED) refused for now.
  - validate()       — runtime gate, raises NotImplementedError on miss.

Phase 0 SUPPORTED summary:
  - dtype:        {bfloat16, float32}.
  - layout:       {TILE_LAYOUT, ROW_MAJOR_LAYOUT}.
  - alignment:    {tile_aligned, w_non_aligned, h_non_aligned}.
                  ROW_MAJOR handles all three; TILE handles tile_aligned only
                  (the two non-aligned cells are EXCLUSIONS — gated as
                  NotImplementedError).
  - rank:         {2, 3, 4}.
  - shape_size:   {small}.  Wt <= 32 (W <= 1024). Larger Wt blows the
                  per-core L1 budget — see op_design.md risk 13. Refinement
                  candidate (W-blocking via accumulate_reduce_block).
  - gamma_mode:   {gamma, no_gamma}.
  - gamma_dtype:  {bfloat16, float32}  (independent of input dtype).
  - gamma_layout: {ROW_MAJOR_LAYOUT, TILE_LAYOUT}.
                  TILE_LAYOUT is in SUPPORTED only so the canonical no_gamma
                  cell (per feature_spec.INVALID:
                  gamma_dtype=float32, gamma_layout=TILE_LAYOUT) is accepted.
                  When gamma is actually supplied, gamma_layout=TILE_LAYOUT
                  is forbidden via EXCLUSIONS.
"""

from typing import Optional

import ttnn

from .rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# 1. SHAPE_TAGGERS
# ---------------------------------------------------------------------------
#
# Two taggers expected by eval/golden_tests/rms_norm/feature_spec.py:
#  - alignment:  three-value split (tile_aligned, w_non_aligned, h_non_aligned)
#                so the W-mask and H-mask kernel paths can be tracked
#                independently. W non-alignment exercises the partial-scaler
#                second tile; H non-alignment exercises the writer's
#                partial-H stick-skip path.
#  - rank:       len(input_tensor.shape); rms_norm is rank-agnostic but the
#                axis is tracked so future refinements (e.g. rank-5 inputs)
#                can be expressed as "add 5 to SUPPORTED[rank]".


def tag_alignment(inputs, axes):
    """tile_aligned / w_non_aligned / h_non_aligned, last two dims."""
    shape = inputs[0]
    H, W = shape[-2], shape[-1]
    w_aligned = (W % 32) == 0
    h_aligned = (H % 32) == 0
    if w_aligned and h_aligned:
        return "tile_aligned"
    if not w_aligned:
        # W non-aligned dominates the tag: it triggers the partial-W scaler
        # path regardless of H alignment.
        return "w_non_aligned"
    return "h_non_aligned"


def tag_rank(inputs, axes):
    """Number of dimensions in the input shape."""
    return len(inputs[0])


def tag_shape_size(inputs, axes):
    """small (Wt<=32, W<=1024) or large.

    The kernel holds 4–5 CBs of size Wt*tile_size simultaneously
    (cb_input_tiles, cb_x_sq, cb_x_norm, cb_output_tiles, and for RM I/O
    cb_input_raw_rm + cb_output_rm). With fp32 tiles (4096 B), Wt=128
    pushes the L1 reservation past the 1.5 MB per-core budget. A
    W-blocking refinement (op_design.md risk 13) would unlock 'large'.

    This axis isn't declared in feature_spec.TARGET; it's an op-local
    shape tagger that grafts a new categorical dimension onto every
    case. SUPPORTED["shape_size"]=["small"] gates wide shapes to xfail.
    """
    shape = inputs[0]
    W = shape[-1]
    Wt = (W + 31) // 32
    return "small" if Wt <= 32 else "large"


SHAPE_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
    "shape_size": tag_shape_size,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "shape_size": ["small"],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.bfloat16, ttnn.float32],
    # TILE_LAYOUT is listed only so the canonical no_gamma cell
    # (gamma_layout=TILE_LAYOUT per feature_spec.INVALID canonicalization)
    # passes SUPPORTED. EXCLUSIONS forbids it when gamma is actually
    # supplied — see below.
    "gamma_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# TILE_LAYOUT inputs require H and W divisible by 32 — anything else hits
# a path the kernel doesn't support (W%32!=0 has no partial-W scaler in the
# TILE path; H%32!=0 would require fenced last-row handling). Both cells
# are inside SUPPORTED but explicitly refused.

EXCLUSIONS = [
    {"layout": ttnn.TILE_LAYOUT, "alignment": "w_non_aligned"},
    {"layout": ttnn.TILE_LAYOUT, "alignment": "h_non_aligned"},
    # gamma_layout=TILE_LAYOUT is only valid for the no_gamma canonical
    # cell. When gamma is supplied, the kernel reads a single RM stick and
    # tilizes it in-kernel; a TILE-layout gamma tensor would skip that
    # path and isn't currently supported.
    {"gamma_mode": "gamma", "gamma_layout": ttnn.TILE_LAYOUT},
    # Mixed-precision gamma + TILE input: chain-driven unpack/pack register
    # reconfig doesn't fully restore state across the (Phase 0 gamma tilize
    # → stage A → ... → stage E) sequence when gamma_dtype != input_dtype.
    # The RM input path works because the Phase 1a input-tilize step
    # re-establishes the input dtype's format. Stage A now uses
    # CopyTileReconfig::Input + PackTileReconfig::Output, which fixed the
    # catastrophic blow-ups but a residual systematic amplification remains
    # (~1.27×, consistent with UnpackToDestMode or fp32→bf16 srcB-into-Dest
    # conversion path). Refinement candidate.
    {"layout": ttnn.TILE_LAYOUT, "gamma_mode": "gamma", "dtype": ttnn.float32, "gamma_dtype": ttnn.bfloat16},
    {"layout": ttnn.TILE_LAYOUT, "gamma_mode": "gamma", "dtype": ttnn.bfloat16, "gamma_dtype": ttnn.float32},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, gamma):
    """Per-axis SUPPORTED check + EXCLUSIONS check + a few shape sanity
    checks that don't map cleanly onto axes.

    All gate violations raise NotImplementedError so the golden harness'
    xfail(strict, raises=NotImplementedError) decoration is satisfied.

    Pure shape-validity errors (rank < 2, gamma_shape mismatch) raise
    ValueError — they are caller-bug signals, not feature-gap signals,
    and the test suite asserts ValueError on those paths.
    """
    shape = list(input_tensor.shape)

    # --- Shape sanity (ValueError, not registry-gated) ---
    if len(shape) < 2:
        raise ValueError("rms_norm: input must have at least 2 dimensions")
    if gamma is not None and list(gamma.shape)[-1] != shape[-1]:
        raise ValueError(
            f"rms_norm: gamma last dim must match input last dim "
            f"(got gamma {list(gamma.shape)[-1]} vs input {shape[-1]})"
        )

    # --- Build axes dict the way the test harness does ---
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "rank": len(shape),
        "gamma_mode": "gamma" if gamma is not None else "no_gamma",
        # Use canonical no-gamma cell when gamma absent (matches
        # feature_spec.INVALID canonicalization).
        "gamma_dtype": gamma.dtype if gamma is not None else ttnn.float32,
        "gamma_layout": gamma.layout if gamma is not None else ttnn.TILE_LAYOUT,
    }
    for axis, tagger in SHAPE_TAGGERS.items():
        axes[axis] = tagger((tuple(shape),), axes)

    # --- 4a. SUPPORTED — per-axis ---
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise NotImplementedError(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # --- 4b. EXCLUSIONS — cell-level ---
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise NotImplementedError(f"rms_norm: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
) -> ttnn.Tensor:
    """RMSNorm along the last dimension.

    Args:
        input_tensor: Input tensor on device (rank >= 2). Dtype must be
            bfloat16 or float32. Layout may be ROW_MAJOR_LAYOUT or
            TILE_LAYOUT (TILE requires H, W divisible by 32).
        gamma: Optional scale tensor, shape (1, 1, 1, W), ROW_MAJOR_LAYOUT.
        epsilon: Numerical stabilizer inside the sqrt (default 1e-6).

    Returns:
        Output tensor with same shape, dtype, and layout as input_tensor.
    """
    validate(input_tensor, gamma)

    device = input_tensor.device()

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma,
        output_tensor,
        epsilon=epsilon,
    )

    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)
