# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""linear_gelu: y = gelu_tanh(x @ W + b), one generic_op dispatch per call."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .linear_gelu_program_descriptor import FFN, HIDDEN, TILE, create_program_descriptor

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
    "alignment": ["tile_aligned", "h_non_aligned"],
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
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _check_structure(input_tensor, weight, bias):
    x_shape = list(input_tensor.shape)
    if len(x_shape) != 4 or x_shape[1] != 1:
        raise ValueError(f"linear_gelu: input_tensor must be (B, 1, S, K), got {x_shape}")
    w_shape = list(weight.shape)
    if len(w_shape) != 2 or w_shape[0] != x_shape[-1]:
        raise ValueError(f"linear_gelu: weight must be (K={x_shape[-1]}, N), got {w_shape}")
    n = w_shape[1]
    b_shape = list(bias.shape)
    if b_shape not in ([n], [1, n]):
        raise ValueError(f"linear_gelu: bias must be ({n},) or (1, {n}), got {b_shape}")
    device = input_tensor.device()
    if weight.device() != device or bias.device() != device:
        raise ValueError("linear_gelu: all tensors must be on the same device")


def _resolve(input_tensor, compute_kernel_config, memory_config):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    out_memory = memory_config if memory_config is not None else input_tensor.memory_config()
    return cfg, out_memory


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, weight, bias, *, compute_kernel_config=None, memory_config=None):
    _check_structure(input_tensor, weight, bias)

    batch, _, seq, hidden = list(input_tensor.shape)
    ffn = list(weight.shape)[1]
    if hidden != HIDDEN or ffn != FFN:
        raise UnsupportedAxisValue(f"linear_gelu: only K={HIDDEN}, N={FFN} (got K={hidden}, N={ffn})")
    if seq > MAX_SEQ or batch * seq > MAX_TOKENS:
        raise UnsupportedAxisValue(f"linear_gelu: S <= {MAX_SEQ} and B*S <= {MAX_TOKENS} (got B={batch}, S={seq})")
    for name, tensor in (("weight", weight), ("bias", bias)):
        if (
            tensor.dtype != ttnn.bfloat16
            or tensor.layout != ttnn.TILE_LAYOUT
            or tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
        ):
            raise UnsupportedAxisValue(f"linear_gelu: {name} must be bfloat16 TILE interleaved")

    cfg, out_memory = _resolve(input_tensor, compute_kernel_config, memory_config)
    if out_memory.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise UnsupportedAxisValue("linear_gelu: output memory_layout must be INTERLEAVED")
    input_memory = input_tensor.memory_config()
    axes = {
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "memory_layout": input_memory.memory_layout,
        "buffer_type": input_memory.buffer_type,
        "out_buffer_type": out_memory.buffer_type,
    }
    shapes = tuple(tuple(t.shape) for t in (input_tensor, weight, bias))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(shapes, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"linear_gelu: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"linear_gelu: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def linear_gelu(
    input_tensor: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    validate(input_tensor, weight, bias, compute_kernel_config=compute_kernel_config, memory_config=memory_config)

    cfg, out_memory = _resolve(input_tensor, compute_kernel_config, memory_config)
    batch, _, seq, _ = list(input_tensor.shape)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, 1, seq, FFN]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        out_memory,
    )
    descriptor = create_program_descriptor(input_tensor, weight, bias, output, cfg)
    return ttnn.generic_op([input_tensor, weight, bias, output], descriptor)
