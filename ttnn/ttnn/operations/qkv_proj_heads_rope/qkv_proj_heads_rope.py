# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""qkv_proj_heads_rope: x @ W + b -> q/k/v head split -> rotary on q, k. One generic_op dispatch per call."""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .qkv_proj_heads_rope_program_descriptor import (
    HIDDEN,
    NUM_HEADS,
    TILE,
    create_program_descriptor,
    select_blocking,
)

MAX_SEQ = 512
MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    return "tile_aligned" if inputs[0][-2] % TILE == 0 else "h_non_aligned"


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
    "fp32_dest_acc_en": [True],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "buffer_type": [ttnn.BufferType.L1],
    "out_buffer_type": [ttnn.BufferType.L1],
    "table_buffer_type": [ttnn.BufferType.L1],
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
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    input_tensor, weight, bias, cos, sin, *, num_heads=NUM_HEADS, compute_kernel_config=None, memory_config=None
):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    input_memory = input_tensor.memory_config()
    out_memory = memory_config if memory_config is not None else input_memory
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", True)),
        "memory_layout": input_memory.memory_layout,
        "buffer_type": input_memory.buffer_type,
        "out_buffer_type": out_memory.buffer_type,
        "table_buffer_type": cos.memory_config().buffer_type,
    }
    shapes = tuple(tuple(t.shape) for t in (input_tensor, weight, bias, cos, sin))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(shapes, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"qkv_proj_heads_rope: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"qkv_proj_heads_rope: unsupported combination (refinement candidate): {exc}")

    # Every other tensor must match the gated properties of x (one cell per call).
    for name, tensor in (("weight", weight), ("bias", bias), ("cos", cos), ("sin", sin)):
        if tensor.dtype != input_tensor.dtype or tensor.layout != input_tensor.layout:
            raise UnsupportedAxisValue(f"qkv_proj_heads_rope: {name} must be {input_tensor.dtype} TILE")
        if tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise UnsupportedAxisValue(f"qkv_proj_heads_rope: {name} must be interleaved")
    if out_memory.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise UnsupportedAxisValue("qkv_proj_heads_rope: output memory_layout must be INTERLEAVED")
    if sin.memory_config().buffer_type != cos.memory_config().buffer_type:
        raise UnsupportedAxisValue("qkv_proj_heads_rope: cos and sin must share a buffer type")
    if not getattr(cfg, "packer_l1_acc", True):
        raise UnsupportedAxisValue("qkv_proj_heads_rope: packer_l1_acc=False")


def _check_shapes(input_tensor, weight, bias, cos, sin, num_heads):
    x_shape = list(input_tensor.shape)
    if len(x_shape) != 4 or x_shape[1] != 1:
        raise ValueError(f"qkv_proj_heads_rope: input_tensor must be (B, 1, S, H), got {x_shape}")
    batch, _, seq, hidden = x_shape
    if num_heads <= 0 or hidden % num_heads != 0:
        raise ValueError(f"qkv_proj_heads_rope: H={hidden} not divisible by num_heads={num_heads}")
    head_dim = hidden // num_heads
    if list(weight.shape) != [hidden, 3 * hidden]:
        raise ValueError(f"qkv_proj_heads_rope: weight shape {list(weight.shape)} != {[hidden, 3 * hidden]}")
    if math.prod(bias.shape) != 3 * hidden:
        raise ValueError(f"qkv_proj_heads_rope: bias must have {3 * hidden} elements, got {list(bias.shape)}")
    for name, table in (("cos", cos), ("sin", sin)):
        t_shape = list(table.shape)
        if len(t_shape) != 4 or t_shape[:2] != [1, 1] or t_shape[3] != head_dim:
            raise ValueError(f"qkv_proj_heads_rope: {name} table must be (1, 1, S_tab, {head_dim}), got {t_shape}")
        if t_shape[2] < seq:
            raise ValueError(f"qkv_proj_heads_rope: {name} table has {t_shape[2]} rows < S={seq}")
    if list(cos.shape) != list(sin.shape):
        raise ValueError("qkv_proj_heads_rope: cos and sin table shapes differ")
    device = input_tensor.device()
    for name, tensor in (("weight", weight), ("bias", bias), ("cos", cos), ("sin", sin)):
        if tensor.device() != device:
            raise ValueError(f"qkv_proj_heads_rope: {name} is on a different device than input_tensor")

    if hidden != HIDDEN:
        raise UnsupportedAxisValue(f"qkv_proj_heads_rope: H={hidden} (only {HIDDEN} supported)")
    if num_heads != NUM_HEADS:
        raise UnsupportedAxisValue(f"qkv_proj_heads_rope: num_heads={num_heads} (only {NUM_HEADS} supported)")
    if not 1 <= seq <= MAX_SEQ:
        raise UnsupportedAxisValue(f"qkv_proj_heads_rope: S={seq} outside [1, {MAX_SEQ}]")
    if batch * seq > MAX_TOKENS:
        raise UnsupportedAxisValue(f"qkv_proj_heads_rope: B*S={batch * seq} > {MAX_TOKENS}")
    return batch, seq, head_dim


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def qkv_proj_heads_rope(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    *,
    num_heads: int = NUM_HEADS,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig = None,
):
    validate(
        input_tensor,
        weight,
        bias,
        cos,
        sin,
        num_heads=num_heads,
        compute_kernel_config=compute_kernel_config,
        memory_config=memory_config,
    )
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    batch, seq, head_dim = _check_shapes(input_tensor, weight, bias, cos, sin, num_heads)

    device = input_tensor.device()
    out_memory = memory_config if memory_config is not None else input_tensor.memory_config()
    out_shape = ttnn.Shape([batch, num_heads, seq, head_dim])
    outputs = tuple(
        ttnn.allocate_tensor_on_device(out_shape, input_tensor.dtype, ttnn.TILE_LAYOUT, device, out_memory)
        for _ in range(3)
    )

    blocking = select_blocking(device, batch, seq, compute_kernel_config)
    program_descriptor = create_program_descriptor(
        input_tensor, weight, bias, cos, sin, outputs, blocking, compute_kernel_config
    )
    ttnn.generic_op([input_tensor, weight, bias, cos, sin, *outputs], program_descriptor)
    return outputs
