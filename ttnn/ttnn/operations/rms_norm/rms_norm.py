# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps) * gamma.

One ttnn.generic_op dispatch per call. Regimes of op_design.md: R1 (interleaved, rows split across the
full compute grid, whole rows resident per core), R6 (HEIGHT_SHARDED / single-slice shards, reduction
local to the core) and R3 (WIDTH/BLOCK_SHARDED, or interleaved rows too wide for one core: W split
across a group of cores, partial sums gathered to a root, rstd multicast back). TILE and ROW_MAJOR inputs are handled natively —
no host-side layout conversion, padding or slicing.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .rms_norm_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Default compute config — the single source of truth for `compute_kernel_config=None`.
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


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
    return int(len(inputs[0]))


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
    "gamma_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "none"],
    "memory_layout": [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # fp32 input with bf16 DEST accumulation is natively rejected (precision convention).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _shape_list(t):
    return [int(d) for d in t.shape]


def validate(input_tensor, *, gamma=None, compute_kernel_config=None, memory_config=None):
    shape = _shape_list(input_tensor)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input must have rank >= 2, got shape {shape}")
    if gamma is not None:
        g_shape = _shape_list(gamma)
        numel = 1
        for d in g_shape:
            numel *= d
        if g_shape[-1] != shape[-1] or numel != shape[-1]:
            raise ValueError(
                f"rms_norm: gamma shape {g_shape} must have last dim == input last dim ({shape[-1]}) "
                f"and exactly {shape[-1]} elements"
            )

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", True)),
        "layout": input_tensor.layout,
        "gamma_mode": "gamma" if gamma is not None else "no_gamma",
        "gamma_dtype": gamma.dtype if gamma is not None else "none",
        "gamma_layout": gamma.layout if gamma is not None else "none",
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination (refinement candidate): {exc}")

    # Output placement follows the input placement: interleaved in -> interleaved out (DRAM or L1);
    # sharded in -> the same shard spec out (the output CB is backed by / written into the output shard).
    if memory_config is not None:
        if axes["memory_layout"] == ttnn.TensorMemoryLayout.INTERLEAVED:
            if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                raise UnsupportedAxisValue("rms_norm: interleaved input requires an interleaved output memory_config")
        elif memory_config.memory_layout != axes["memory_layout"] or memory_config.shard_spec != (
            input_tensor.memory_config().shard_spec
        ):
            raise UnsupportedAxisValue("rms_norm: sharded output memory_config must match the input's shard spec")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    program_config=None,
) -> ttnn.Tensor:
    validate(
        input_tensor,
        gamma=gamma,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    device = input_tensor.device()
    if memory_config is not None:
        output_memory_config = memory_config
    elif input_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        output_memory_config = input_tensor.memory_config()
    else:
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(_shape_list(input_tensor)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, gamma, output_tensor, float(epsilon), cfg)
    io = [input_tensor] + ([gamma] if gamma is not None else []) + [output_tensor]
    return ttnn.generic_op(io, program_descriptor)
