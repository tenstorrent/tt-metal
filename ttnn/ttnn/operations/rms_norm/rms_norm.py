# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — registry-model op file (INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate) + entry point.

    RMSNorm(x) = x * rsqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

The kernel design lives in `op_design.md` beside this file; the program descriptor
(`rms_norm_program_descriptor.py`) realizes its Blocking Model. Every call dispatches exactly
one device program.
"""

from __future__ import annotations

import math
from typing import Optional

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_rank(inputs, axes):
    """Shape-derived axis: the input tensor's rank."""
    return int(len(inputs[0]))


def tag_alignment(inputs, axes):
    """Shape-derived axis: whether the last two dims are whole tiles. The kernel has no edge-tile
    mask / pad path, so non-aligned shapes are refused as an unsupported axis value (a refinement
    candidate), not as a shape error."""
    shape = inputs[0]
    if len(shape) >= 2 and shape[-1] % 32 == 0 and shape[-2] % 32 == 0:
        return "tile_aligned"
    return "non_tile_aligned"


INPUT_TAGGERS = {"rank": tag_rank, "alignment": tag_alignment}

# ---------------------------------------------------------------------------
# 2. SUPPORTED — one entry per TARGET axis
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    # Two-axis precision model (dtype x DEST width). Every accumulated-intermediate page format and the
    # collective payload stride follow this flag (rms_norm_program_descriptor.acc_dtype_for).
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "rank": [2, 3, 4],
    "alignment": ["tile_aligned"],
    "gamma_mode": ["gamma", "no_gamma"],
    # "none" is the canonical "no weight tensor" sentinel — always legal.
    "gamma_dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, "none"],
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # fp32 input with 16-bit DEST accumulation is lossy by construction — natively rejected
    # forever (references/precision_convention.md).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


# ---------------------------------------------------------------------------
# Default compute config — the single exported factory (read by the golden axis tagger too)
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _shape_errors(input_tensor, gamma, epsilon, memory_config):
    """Structural (ValueError) checks. Message text: 'rank' for the rank error, 'gamma' for
    every gamma-shape error — the acceptance test matches on those substrings."""
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input rank must be >= 2, got rank {len(shape)} (shape {shape})")
    if gamma is not None:
        gshape = list(gamma.shape)
        if gshape[-1] != shape[-1]:
            raise ValueError(
                f"rms_norm: gamma last dim {gshape[-1]} does not match input last dim {shape[-1]} (gamma shape {gshape})"
            )
        if any(d != 1 for d in gshape[:-1]):
            raise ValueError(f"rms_norm: gamma must have shape (1, ..., 1, W), got gamma shape {gshape}")
        if gamma.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError("rms_norm: gamma must be an interleaved tensor")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"rms_norm: epsilon must be finite and >= 0, got {epsilon}")

    input_mc = input_tensor.memory_config()
    if input_mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError("rms_norm: WIDTH_SHARDED input requires TILE_LAYOUT (ROW_MAJOR + sharded is unsupported)")
        if memory_config is not None and memory_config != input_mc:
            raise ValueError("rms_norm: for a WIDTH_SHARDED input the output memory_config must equal the input's")


def _axes_for(input_tensor, gamma, compute_kernel_config):
    has_gamma = gamma is not None
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(compute_kernel_config.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "gamma_mode": "gamma" if has_gamma else "no_gamma",
        "gamma_dtype": gamma.dtype if has_gamma else "none",
        "gamma_layout": gamma.layout if has_gamma else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((list(input_tensor.shape),), axes)
    return axes


def validate(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.ComputeConfigDescriptor:
    """Runtime gate. Returns the resolved compute config (None -> default_compute_kernel_config())."""
    _shape_errors(input_tensor, gamma, epsilon, memory_config)

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = _axes_for(input_tensor, gamma, cfg)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")
    return cfg


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    program_config=None,  # accepted for signature compatibility; ignored
) -> ttnn.Tensor:
    cfg = validate(
        input_tensor,
        gamma=gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )

    device = input_tensor.device()
    input_mc = input_tensor.memory_config()
    if input_mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        output_memory_config = input_mc  # the output inherits the input's shard spec
    else:
        output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        gamma,
        output_tensor,
        epsilon=epsilon,
        compute_kernel_config=cfg,
    )

    io_tensors = [input_tensor] + ([gamma] if gamma is not None else []) + [output_tensor]
    return ttnn.generic_op(io_tensors, program_descriptor)
