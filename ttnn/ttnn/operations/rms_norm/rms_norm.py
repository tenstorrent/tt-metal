# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — one native generic_op dispatch per call.

    rms_norm(x, gamma=None, epsilon=1e-6) = x / sqrt(mean(x**2, dim=-1) + epsilon) * gamma

computed over the last dimension of `x`. Design: op_design.md (LOCAL and W_SPLIT regimes, both built).

Inputs
    input_tensor  rank >= 2; float32 / bfloat16 / bfloat8_b; TILE or ROW_MAJOR; any H and W (tile-aligned or not);
                  bfloat8_b is TILE + tile-aligned only (block-float); interleaved or L1-sharded
                  (HEIGHT / WIDTH / BLOCK — the shard grid is the core assignment, consumed in place).
    gamma         optional; shape [1, 1, 1, W]; float32 / bfloat16 / bfloat8_b (may differ from x); ROW_MAJOR or
                  TILE (bfloat8_b: TILE only).
    epsilon       added to the mean of squares before the reciprocal square root.
    compute_kernel_config
                  a ttnn.ComputeConfigDescriptor; None means default_compute_kernel_config() (the Phase 0 corner:
                  HiFi4, fp32 DEST accumulation on, approx off). math_fidelity / math_approx_mode are honoured as
                  given; fp32_dest_acc_en=False switches every intermediate CB to Float16_b (refused for float32
                  input — EXCLUSIONS).
    memory_config the output's placement. Interleaved input: None / any interleaved config (default DRAM). Sharded
                  input (HEIGHT / WIDTH / BLOCK, L1): the output inherits the input's shard spec (None or that same
                  config) and is produced in place, shard for shard.

Output
    Same shape, dtype and layout as the input.

Refusals
    A cell the op does not support raises NotImplementedError (UnsupportedAxisValue or ExcludedCell from
    ttnn.operations._op_contract). Structural misuse (rank < 2, gamma width mismatch) raises ValueError.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

TILE_DIM = 32

TARGET = {
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

INPUT_SHAPES = [
    (1, 1, 32, 64), (1, 1, 64, 128), (4, 8, 32, 256), (2, 4, 128, 512), (1, 1, 2048, 256), (4, 1, 512, 512),
    (1, 1, 32, 50), (1, 1, 64, 17), (4, 8, 32, 47), (2, 1, 128, 100), (1, 1, 17, 64), (1, 1, 50, 128),
    (4, 8, 47, 256), (1, 1, 17, 50), (2, 1, 100, 47), (1, 1, 32, 4096), (1, 1, 32, 8192), (1, 1, 128, 4096),
    (2, 1, 64, 4096), (1, 32, 128), (4, 128, 512), (2, 512, 1024), (1, 32, 4096), (1, 32, 8192), (1, 32, 50),
    (4, 128, 47), (1, 17, 128), (32, 64), (128, 512), (1024, 1024), (32, 4096), (128, 8192), (32, 17), (128, 100),
    (17, 64),
]


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS (shape-derived axes only)
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """tile_aligned (H%32==0 and W%32==0) / w_non_aligned (W%32!=0) / h_non_aligned (W aligned, H not)."""
    shape = inputs[0]
    H, W = shape[-2], shape[-1]
    if W % TILE_DIM != 0:
        return "w_non_aligned"
    if H % TILE_DIM != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return int(len(inputs[0]))


INPUT_TAGGERS = {"alignment": tag_alignment, "rank": tag_rank}


# Gamma-format axes with the "none" sentinel. These read the gamma TENSOR, which a registry tagger never sees
# (taggers get only the shape tuple), and registering them in INPUT_TAGGERS would make the golden harness stop
# iterating TARGET["gamma_dtype"/"gamma_layout"] (a tagger axis wins over its TARGET entry). So validate() calls
# them directly; the harness keeps cartesian-iterating the two axes.
def tag_gamma_dtype(gamma):
    return "none" if gamma is None else gamma.dtype


def tag_gamma_layout(gamma):
    return "none" if gamma is None else gamma.layout


# ---------------------------------------------------------------------------
# 2. SUPPORTED / 3. EXCLUSIONS
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

EXCLUSIONS = [
    # Permanent (precision convention): float32 input requires fp32 DEST accumulation.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """The Phase 0 corner. The single place the default lives (the golden tagger reads it too)."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.ComputeConfigDescriptor:
    """Gate the call; return the resolved compute config."""
    shape = tuple(int(d) for d in input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"rms_norm: input must have rank >= 2, got shape {list(shape)}")
    if gamma is not None:
        g_shape = tuple(int(d) for d in gamma.shape)
        g_volume = 1
        for d in g_shape:
            g_volume *= d
        if g_shape[-1] != shape[-1] or g_volume != shape[-1]:
            raise ValueError(
                f"rms_norm: gamma shape {list(g_shape)} must be (1, 1, 1, W) with W == input W ({shape[-1]})"
            )

    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "gamma_mode": "no_gamma" if gamma is None else "gamma",
        "gamma_dtype": tag_gamma_dtype(gamma),
        "gamma_layout": tag_gamma_layout(gamma),
        "memory_layout": input_tensor.memory_config().memory_layout,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((shape,), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"rms_norm: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"rms_norm: unsupported combination: {exc}")

    if gamma is not None and gamma.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise UnsupportedAxisValue("rms_norm: gamma must be interleaved")
    if memory_config is not None and memory_config != _default_output_memory_config(input_tensor):
        in_mc = input_tensor.memory_config()
        if in_mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            # A sharded input produces its output in place, shard for shard: the output must carry the same spec.
            raise UnsupportedAxisValue("rms_norm: a sharded input requires the output to inherit its shard spec")
        if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise UnsupportedAxisValue(f"rms_norm: output memory_config {memory_config.memory_layout} not supported")
    return cfg


def _default_output_memory_config(input_tensor: ttnn.Tensor) -> ttnn.MemoryConfig:
    """Interleaved input -> DRAM interleaved; sharded input -> the input's own shard spec (produced in place)."""
    in_mc = input_tensor.memory_config()
    if in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return ttnn.DRAM_MEMORY_CONFIG
    return in_mc


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
) -> ttnn.Tensor:
    cfg = validate(input_tensor, gamma=gamma, compute_kernel_config=compute_kernel_config, memory_config=memory_config)

    from .rms_norm_program_descriptor import create_program_descriptor

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else _default_output_memory_config(input_tensor)
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )
    program_descriptor = create_program_descriptor(input_tensor, gamma, output_tensor, float(epsilon), cfg)
    io_tensors = [input_tensor] + ([gamma] if gamma is not None else []) + [output_tensor]
    return ttnn.generic_op(io_tensors, program_descriptor)
