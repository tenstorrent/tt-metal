# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Softmax operation entry point.

Numerically-stable softmax along the last (W, dim=-1) or second-to-last
(H, dim=-2) dimension of a 2D/3D/4D tensor.

Math:
    output[n,c,h,w] = exp(x[n,c,h,w] - max(x[n,c,row_or_col]))
                    / sum(exp(x[n,c,h,w] - max(x[n,c,row_or_col])))

Rank 2/3 tensors are internally unsqueezed to 4D before kernel dispatch,
then reshaped back to the original rank on return. The kernels are
rank-agnostic — they only see Ht, Wt, and num_slabs.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from .softmax_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Default compute kernel config (single source of truth)
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """Default ComputeConfigDescriptor for softmax.

    Maxed-out precision corner: fp32_dest_acc_en=True with float32 input.
    """
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Three-value alignment split matching feature_spec.

    - tile_aligned  — both H (-2) and W (-1) divisible by 32.
    - w_non_aligned — W not divisible by 32 (regardless of H).
    - h_non_aligned — W aligned, H not aligned.
    """
    shape = inputs[0]
    w_aligned = shape[-1] % 32 == 0
    h_aligned = shape[-2] % 32 == 0
    if w_aligned and h_aligned:
        return "tile_aligned"
    if not w_aligned:
        return "w_non_aligned"
    return "h_non_aligned"


def tag_rank(inputs, axes):
    """Tensor rank as an integer (2, 3, 4, ...)."""
    return len(inputs[0])


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "dim": [-1, -2],
    "fp32_dest_acc_en": [True],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # This op is fp32-dest-only: fp32_dest_acc_en=False is rejected for
    # every dtype (the golden suite xfails those cells). The exclusion is
    # keyed only on fp32_dest_acc_en=False so it applies regardless of dtype.
    {"fp32_dest_acc_en": False},
    # bfloat8_b is a block format; non-tile-aligned shapes need the masking
    # from Refinement 3 first. These are pre-emptive — they activate when
    # Refinement 3 adds w_non_aligned / h_non_aligned to SUPPORTED.
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
    {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, *, dim=-1, compute_kernel_config=None):
    # Canonicalize dim to negative offset
    ndim = len(input_tensor.shape)
    if dim < 0:
        canonical_dim = dim
    else:
        canonical_dim = dim - ndim

    # Resolve compute_kernel_config
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    # Build the axes dict
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "dim": canonical_dim,
        "fp32_dest_acc_en": cfg.fp32_dest_acc_en,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((input_tensor.shape,), axes)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"softmax: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"softmax: unsupported combination (refinement candidate): {exc}")

    return canonical_dim, cfg


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def softmax(
    input_tensor: ttnn.Tensor,
    dim: int = -1,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.Tensor:
    """Numerically-stable softmax along dim.

    Args:
        input_tensor: Input tensor (float32/bfloat16/bfloat8_b, TILE/ROW_MAJOR,
                      rank 2/3/4, H/W divisible by 32 for tile-aligned cases)
        dim: Dimension along which softmax is computed (-1 = W, -2 = H). Default: -1.
        compute_kernel_config: Compute kernel config. Default: fp32_dest_acc_en=True.

    Returns:
        Output tensor (same shape, dtype, layout as input)
    """
    # Validate and resolve config
    canonical_dim, cfg = validate(input_tensor, dim=dim, compute_kernel_config=compute_kernel_config)

    device = input_tensor.device()

    # Save original shape for output reshape. The program descriptor and
    # kernels operate on 4D internally — lower-rank tensors are unsqueezed
    # to 4D before kernel dispatch and reshaped back on return.
    original_shape = ttnn.Shape(list(input_tensor.shape))
    input_4d = ttnn.unsqueeze_to_4D(input_tensor) if len(input_tensor.shape) < 4 else input_tensor

    # Allocate output tensor (4D shape, same dtype, layout as input)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_4d.shape)),
        input_4d.dtype,
        input_4d.layout,
        device,
        input_4d.memory_config(),
    )

    program_descriptor = create_program_descriptor(
        input_4d,
        output_tensor,
        dim=canonical_dim,
        compute_kernel_config=cfg,
    )

    # Output tensor MUST be last in the list
    result = ttnn.generic_op([input_4d, output_tensor], program_descriptor)

    # Reshape back to the original rank if the input was lower-dimensional
    if len(original_shape) < 4:
        result = ttnn.reshape(result, original_shape)
    return result
