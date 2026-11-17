# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm: out = x * rsqrt(mean(x^2, -1) + eps) [* gamma], one native dispatch.

Registry-model op file: INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate(), entry point.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Compute config (single source of truth for the Phase 0 default)
# ---------------------------------------------------------------------------
def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
def tag_alignment(inputs, axes):
    shape = inputs[0]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
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
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [2, 3, 4],
    "gamma_mode": ["gamma", "no_gamma"],
    "gamma_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, "none"],
    "gamma_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, "none"],
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
}

_SHARDED = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
EXCLUSIONS = [
    # fp32 input requires fp32 accumulation (no intermediate bf16 downcast).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # bfloat8_b is TILE-only and tile-aligned only (block-float + partial tiles / RM untilize).
    {"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT},
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
    {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"},
    # bfloat8_b gamma cannot be an RM stick tensor.
    {"gamma_dtype": ttnn.bfloat8_b, "gamma_layout": ttnn.ROW_MAJOR_LAYOUT},
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["LoFi", "HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------
def validate(input_tensor, *, gamma=None, compute_kernel_config=None, memory_config=None):
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input must have rank >= 2, got shape {shape}")
    if gamma is not None and list(gamma.shape)[-1] != shape[-1]:
        raise ValueError(f"rms_norm: gamma last dim {list(gamma.shape)[-1]} does not match input last dim {shape[-1]}")

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    has_gamma = gamma is not None
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "gamma_mode": "gamma" if has_gamma else "no_gamma",
        "gamma_dtype": gamma.dtype if has_gamma else "none",
        "gamma_layout": gamma.layout if has_gamma else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    # Requested output placement must also be a supported memory_layout.
    if memory_config is not None:
        if memory_config.memory_layout not in SUPPORTED["memory_layout"]:
            raise UnsupportedAxisValue(
                f"rms_norm: output memory_layout={memory_config.memory_layout!r} not in SUPPORTED "
                f"{SUPPORTED['memory_layout']}"
            )
        if memory_config.memory_layout in _SHARDED:
            in_mc = input_tensor.memory_config()
            if in_mc.memory_layout in _SHARDED and (
                in_mc.memory_layout != memory_config.memory_layout or in_mc.shard_spec != memory_config.shard_spec
            ):
                raise UnsupportedAxisValue("rms_norm: sharded input and sharded output must share one shard spec")
    shard_spec = input_tensor.memory_config().shard_spec
    if shard_spec is not None and shard_spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise UnsupportedAxisValue("rms_norm: only ROW_MAJOR shard orientation is supported")

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
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    validate(input_tensor, gamma=gamma, compute_kernel_config=compute_kernel_config, memory_config=memory_config)
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()

    if memory_config is None:  # sharded input -> output inherits its placement; otherwise DRAM interleaved
        in_mc = input_tensor.memory_config()
        memory_config = in_mc if in_mc.memory_layout in _SHARDED else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, gamma, epsilon, compute_kernel_config)

    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    io_tensors.append(output_tensor)  # output MUST be last
    return ttnn.generic_op(io_tensors, program_descriptor)
