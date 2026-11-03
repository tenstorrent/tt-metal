# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""moe_down_proj: y[r] = x[r] @ M_e^T for every real row r of expert region e, one generic_op dispatch per call.

expert_counts is read on the device only; every host-derived quantity depends on shapes, grid and config.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .moe_down_proj_program_descriptor import FFN, HIDDEN, NUM_EXPERTS, TILE, create_program_descriptor

MAX_TOKENS = 4096
MAX_ROWS = TILE * (-(-2 * MAX_TOKENS // TILE) + NUM_EXPERTS - 1)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------

# R is always a tile multiple; no shape-derived axis.
INPUT_TAGGERS = {}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "buffer_type": [ttnn.BufferType.L1, ttnn.BufferType.DRAM],
    "out_buffer_type": [ttnn.BufferType.L1, ttnn.BufferType.DRAM],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _resolve(input_tensor, compute_kernel_config, memory_config):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    out_memory = memory_config if memory_config is not None else input_tensor.memory_config()
    return cfg, out_memory


def _check_structure(input_tensor, weight, expert_counts):
    x_shape = list(input_tensor.shape)
    if len(x_shape) != 4 or x_shape[:2] != [1, 1] or x_shape[-2] % TILE != 0:
        raise ValueError(f"moe_down_proj: input_tensor must be (1, 1, R, F) with R % {TILE} == 0, got {x_shape}")
    c_shape = list(expert_counts.shape)
    if len(c_shape) != 2 or c_shape[0] != 1:
        raise ValueError(f"moe_down_proj: expert_counts must be (1, E), got {c_shape}")
    if expert_counts.dtype != ttnn.uint32:
        raise ValueError(f"moe_down_proj: expert_counts must be uint32, got {expert_counts.dtype}")
    w_shape = list(weight.shape)
    num_experts = c_shape[1]
    if len(w_shape) != 4 or w_shape[:2] != [1, 1] or w_shape[-1] != num_experts * x_shape[-1]:
        raise ValueError(
            f"moe_down_proj: weight must be (1, 1, H, E * F) = (1, 1, H, {num_experts * x_shape[-1]}), got {w_shape}"
        )
    device = input_tensor.device()
    if weight.device() != device or expert_counts.device() != device:
        raise ValueError("moe_down_proj: all tensors must be on the same device")


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, weight, expert_counts, *, compute_kernel_config=None, memory_config=None):
    _check_structure(input_tensor, weight, expert_counts)

    rows, ffn = list(input_tensor.shape)[-2:]
    hidden = list(weight.shape)[-2]
    num_experts = list(expert_counts.shape)[-1]
    if ffn != FFN or hidden != HIDDEN or num_experts != NUM_EXPERTS:
        raise UnsupportedAxisValue(
            f"moe_down_proj: only F={FFN}, H={HIDDEN}, E={NUM_EXPERTS} (got F={ffn}, H={hidden}, E={num_experts})"
        )
    if rows > MAX_ROWS:
        raise UnsupportedAxisValue(f"moe_down_proj: R <= {MAX_ROWS} (t <= {MAX_TOKENS}), got R={rows}")
    if weight.dtype != input_tensor.dtype:
        raise UnsupportedAxisValue(f"moe_down_proj: weight dtype {weight.dtype} != x dtype {input_tensor.dtype}")
    for name, tensor in (("weight", weight), ("expert_counts", expert_counts)):
        if tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise UnsupportedAxisValue(f"moe_down_proj: {name} must be interleaved")
    if weight.layout != ttnn.TILE_LAYOUT or expert_counts.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise UnsupportedAxisValue("moe_down_proj: weight must be TILE and expert_counts ROW_MAJOR")

    cfg, out_memory = _resolve(input_tensor, compute_kernel_config, memory_config)
    if out_memory.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise UnsupportedAxisValue("moe_down_proj: output memory_layout must be INTERLEAVED")
    input_memory = input_tensor.memory_config()
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "memory_layout": input_memory.memory_layout,
        "buffer_type": input_memory.buffer_type,
        "out_buffer_type": out_memory.buffer_type,
    }
    shapes = tuple(tuple(t.shape) for t in (input_tensor, weight, expert_counts))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(shapes, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"moe_down_proj: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"moe_down_proj: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def moe_down_proj(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    expert_counts: ttnn.Tensor,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    validate(
        input_tensor, weight, expert_counts, compute_kernel_config=compute_kernel_config, memory_config=memory_config
    )

    cfg, out_memory = _resolve(input_tensor, compute_kernel_config, memory_config)
    rows = list(input_tensor.shape)[-2]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, HIDDEN]),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor.device(),
        out_memory,
    )
    descriptor = create_program_descriptor(input_tensor, weight, expert_counts, output, cfg)
    return ttnn.generic_op([input_tensor, weight, expert_counts, output], descriptor)
