# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax — main entry point under the registry model.

Numerically-stable row-wise (dim=-1) or column-wise (dim=-2) softmax for
fp32 TILE-layout 4D tensors.

The op declares the four registry-model artefacts inline:
    1. INPUT_TAGGERS  — categorical projections of input shape.
    2. SUPPORTED      — per-axis values accepted today.
    3. EXCLUSIONS     — cells inside cartesian(SUPPORTED) refused for now.
    4. validate()     — runtime gate; raises NotImplementedError for anything
                        outside SUPPORTED or matching an EXCLUSIONS entry.

INVALID lives test-side (eval/golden_tests/softmax/feature_spec.py); the
op file is intentionally agnostic to it.

Phase-0 supports the single precision name `fp32_hifi4_fp32acc` (input
dtype=float32, math_fidelity=HiFi4, fp32_dest_acc_en=True), TILE layout,
rank-4, tile-aligned H/W, dim ∈ {-1, -2}, numeric_stable ∈ {True, False}.
"""

import ttnn

from .softmax_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Precision-name resolution
# ---------------------------------------------------------------------------
#
# The golden-test universe treats precision as a single bundled axis
# (eval/golden_tests/softmax/feature_spec.py:PRECISION_CONFIG). The op's
# validate() inspects input_tensor.dtype + compute_kernel_config and
# resolves them back to a precision name; SUPPORTED["precision"] is then
# checked against that name. Combinations that don't match any known
# precision-name resolve to None, which will fail the SUPPORTED check.

_PRECISION_NAMES = {
    (ttnn.float32, ttnn.MathFidelity.HiFi4, True): "fp32_hifi4_fp32acc",
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True): "bf16_hifi2_fp32acc",
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, False): "bf16_hifi2_bf16acc",
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, True): "bf16_hifi4_fp32acc",
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, False): "bf16_hifi4_bf16acc",
}


def _resolve_precision_name(input_dtype, math_fidelity, fp32_dest_acc_en):
    return _PRECISION_NAMES.get((input_dtype, math_fidelity, bool(fp32_dest_acc_en)))


def _default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """Phase-0 default — installed when caller passes None."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# Both taggers are pure shape-based — they ignore the `axes` argument.
# Declared with the canonical `(inputs, axes)` signature so they match
# the test-harness contract (feature_matrix.apply_input_taggers).


def tag_alignment(inputs, axes):
    """Three-value alignment bucket.

    - "tile_aligned"   if both H (-2) and W (-1) are tile-aligned.
    - "w_non_aligned"  if W is not tile-aligned (regardless of H).
    - "h_non_aligned"  if W is aligned but H is not.

    This matches the contract described in feature_spec.py.
    """
    shape = inputs[0]
    h = shape[-2]
    w = shape[-1]
    if w % 32 != 0:
        return "w_non_aligned"
    if h % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    """Integer rank of the input tensor."""
    return len(inputs[0])


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# Phase-0 envelope. Refinements expand each axis toward TARGET (see
# eval/golden_tests/softmax/feature_spec.py and op_requirements.md).

SUPPORTED = {
    "precision": ["fp32_hifi4_fp32acc"],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "rank": [4],
    "dim": [-1, -2],
    "numeric_stable": [True, False],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# Cells inside cartesian(SUPPORTED) that the op refuses for now.
# Phase-0 supports the full SUPPORTED rectangle — no exclusions.

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, *, dim=-1, numeric_stable=True, compute_kernel_config=None):
    """Runtime gate.

    Raises NotImplementedError for anything outside the SUPPORTED rectangle
    or matching an EXCLUSIONS entry. INVALID is the test harness's concern
    (cells are skipped before validate() is ever called for them).
    """
    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config()

    # Resolve the precision axis. None means "no known precision name
    # matches this (dtype, fidelity, accumulator) combo"; that will fail
    # the SUPPORTED check below.
    precision = _resolve_precision_name(
        input_tensor.dtype,
        compute_kernel_config.math_fidelity,
        compute_kernel_config.fp32_dest_acc_en,
    )

    axes = {
        "precision": precision,
        "layout": input_tensor.layout,
        "dim": dim,
        "numeric_stable": numeric_stable,
    }
    inputs = (tuple(input_tensor.shape),)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    # 1. SUPPORTED — per-axis values.
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise NotImplementedError(f"softmax: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside the SUPPORTED rectangle.
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise NotImplementedError(f"softmax: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def softmax(
    input_tensor: ttnn.Tensor,
    dim: int = -1,
    numeric_stable: bool = True,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Compute softmax along ``dim`` of a fp32 4D TILE-layout tensor.

    Args:
        input_tensor: Input tensor. Phase-0 requires float32, TILE_LAYOUT,
            rank-4, on device, with H % 32 == 0 and W % 32 == 0.
        dim: Reduction axis. Phase-0 supports only ``-1`` and ``-2``.
        numeric_stable: If True (default), subtract the per-row/column max
            before exponentiating. If False, skip the max subtraction
            (faster, but overflows on large inputs).
        compute_kernel_config: Either ``None`` (entry point installs the
            Phase-0 default of math_fidelity=HiFi4, fp32_dest_acc_en=True),
            or an explicit ``ttnn.ComputeConfigDescriptor``. Configs that
            don't resolve to a precision name in SUPPORTED["precision"] are
            rejected.
        memory_config: Output memory config; default DRAM interleaved.

    Returns:
        Output tensor with the same shape, dtype, and layout as
        ``input_tensor``.
    """
    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config()

    validate(
        input_tensor,
        dim=dim,
        numeric_stable=numeric_stable,
        compute_kernel_config=compute_kernel_config,
    )

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_shape = list(input_tensor.shape)

    # allocate_tensor_on_device requires POSITIONAL args, not keyword args.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        dim=dim,
        numeric_stable=numeric_stable,
        compute_kernel_config=compute_kernel_config,
    )

    # Output tensor MUST be last in the IO tensor list.
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
