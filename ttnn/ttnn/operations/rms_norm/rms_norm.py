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
def tag_alignment(inputs, axes):
    """Last two dims tile-aligned (both multiples of 32)."""
    shape = inputs[0]
    if len(shape) < 2:
        return "non_tile_aligned"
    if shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
# fp32 + fp32_dest_acc_en=False would be the only exclusion, but fp32 input is
# out of Phase 0 scope (gated by SUPPORTED[dtype]), so EXCLUSIONS is empty.
EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------
def validate(input_tensor):
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(input_tensor.shape),), axes)

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
    validate(input_tensor)

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
