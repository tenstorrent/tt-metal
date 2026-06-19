# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — Root-mean-square normalization over the last dimension.

    out[..., h, w] = x[..., h, w] * rsqrt( mean(x[..., h, :]^2) + eps ) * gamma[w]

Phase 0 SUPPORTED rectangle (registry model):
  - dtype:     bfloat16
  - layout:    TILE_LAYOUT
  - alignment: tile_aligned (H, W both multiples of 32)
  - memory:    interleaved DRAM
  - gamma:     optional; bf16 / TILE_LAYOUT / shape (1, 1, 1, W)

The op picks one of two regimes at host time (see the program descriptor):
  - Regime A: row-parallel, each core holds a full tile-row resident.
  - Regime B: wide-W cross-core W-split with an mcast all-gather of the
              partial sum-of-squares (used when a full row does not fit L1).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
# Each tagger has the (inputs, axes) signature even when it ignores `axes`.
# `inputs` is the per-case shape tuple (1-tuple for this single-input op).
def tag_alignment(inputs, axes):
    """Three-value last-two-dim alignment split (matches feature_spec TARGET):
      - tile_aligned  : both H (-2) and W (-1) divisible by 32.
      - w_non_aligned : W not divisible by 32 (H may or may not be).
      - h_non_aligned : W aligned, H not aligned.
    These hit different kernel paths (masked reduce vs row-padding mask), so
    the verifier can route refinement signal correctly.
    """
    shape = inputs[0]
    w_aligned = shape[-1] % 32 == 0
    h_aligned = shape[-2] % 32 == 0
    if not w_aligned:
        return "w_non_aligned"
    if not h_aligned:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    """Input tensor rank (number of dims)."""
    return int(len(inputs[0]))


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
# Every axis the kernel gates on (incl. every INPUT_TAGGERS key) appears here.
# Phase 0 corner: bf16 / TILE / tile-aligned / fp32_dest_acc_en in {True, False
# accepted at the True-only Phase-0 precision corner per prompt}, ranks 2-4,
# optional bf16/TILE gamma.
SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "rank": [2, 3, 4],
    # Phase 0 maxed-out precision corner. bf16 + fp32_dest_acc_en=False is a
    # later (numeric) refinement per the op prompt.
    "fp32_dest_acc_en": [True],
    "gamma_mode": ["gamma", "no_gamma"],
    # float32 is listed only so the no_gamma canonical cell
    # (gamma_dtype=float32, gamma_layout=TILE) is supported; gamma-present +
    # float32 is refused by the EXCLUSIONS entry below.
    "gamma_dtype": [ttnn.bfloat16, ttnn.float32],
    "gamma_layout": [ttnn.TILE_LAYOUT],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
# gamma-present + float32 gamma is inside SUPPORTED (float32 is only there for
# the no_gamma canonical cell) but the kernel reads gamma with the input's
# (bf16) tile format, so a real fp32 gamma is refused for now.
EXCLUSIONS = [
    {"gamma_mode": "gamma", "gamma_dtype": ttnn.float32},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------
def _classify_axes(input_tensor, gamma, compute_kernel_config):
    """Reconstruct the registry axes cell (mirrors helpers.classify_call)."""
    has_gamma = gamma is not None
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": (
            True if compute_kernel_config is None else bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))
        ),
        "gamma_mode": "gamma" if has_gamma else "no_gamma",
        "gamma_dtype": gamma.dtype if has_gamma else ttnn.float32,
        "gamma_layout": gamma.layout if has_gamma else ttnn.TILE_LAYOUT,
    }
    shape = tuple(input_tensor.shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)
    return axes


def validate(input_tensor, gamma=None, compute_kernel_config=None):
    # --- input-error guards (ValueError, not a support refusal) ---
    shape = tuple(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input rank must be >= 2, got shape {shape}")
    if gamma is not None and int(gamma.shape[-1]) != int(shape[-1]):
        raise ValueError(
            f"rms_norm: gamma last dim ({int(gamma.shape[-1])}) must match input " f"last dim ({int(shape[-1])})"
        )

    # --- registry support gate ---
    axes = _classify_axes(input_tensor, gamma, compute_kernel_config)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    epsilon: float = 1e-6,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """RMSNorm over the last dimension. See module docstring for the contract."""
    validate(input_tensor, gamma=gamma, compute_kernel_config=compute_kernel_config)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_shape = list(input_tensor.shape)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor, inputs = create_program_descriptor(
        input_tensor,
        output_tensor,
        gamma,
        float(epsilon),
        compute_kernel_config,
    )

    # Output tensor MUST be last in the io list.
    return ttnn.generic_op(inputs, program_descriptor)
