# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gated_delta_net: Gated DeltaNet token mixer as one device program (registry model)."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .gated_delta_net_program_descriptor import (
    HISTORY,
    L1_BUDGET,
    TILE,
    create_program_descriptor,
    l1_footprint_bytes,
)


def default_compute_kernel_config():
    """Phase 0 default (single source of truth; the golden axis tagger imports it)."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  (inputs = (mixed_qkv_shape, recurrent_state_shape))
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    rows = inputs[0][-2]
    key_dim, value_dim = inputs[1][-2], inputs[1][-1]
    if key_dim % TILE or value_dim % TILE:
        return "w_non_aligned"
    if rows % TILE:
        return "h_non_aligned"
    return "tile_aligned"


def tag_sequence_mode(inputs, axes):
    batch = inputs[1][0]
    tokens = inputs[0][-2] // batch
    if tokens == 1:
        return "decode"
    return "prefill" if batch == 1 else "batched_prefill"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "sequence_mode": tag_sequence_mode,
}

# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "sequence_mode": ["decode", "prefill", "batched_prefill"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _shape(t):
    return tuple(int(d) for d in t.shape)


def _check_shapes(mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight, epsilon):
    if len(recurrent_state.shape) != 4:
        raise ValueError("gated_delta_net: recurrent_state must be (B, Hv, K, V)")
    batch, value_heads, key_dim, value_dim = _shape(recurrent_state)
    x_shape = _shape(mixed_qkv)
    if len(x_shape) != 3 or x_shape[0] != 1:
        raise ValueError("gated_delta_net: mixed_qkv must be (1, B*T, C)")
    rows, channels = x_shape[1], x_shape[2]
    if batch < 1 or rows < 1 or rows % batch:
        raise ValueError("gated_delta_net: rows of mixed_qkv must be a positive multiple of B")
    qk_channels = channels - value_heads * value_dim
    if qk_channels <= 0 or qk_channels % (2 * key_dim):
        raise ValueError("gated_delta_net: C - Hv*V must be a positive multiple of 2*K")
    key_heads = qk_channels // (2 * key_dim)
    if value_heads % key_heads:
        raise ValueError("gated_delta_net: Hv must be a multiple of Hk")
    if _shape(z) != (1, rows, value_heads * value_dim):
        raise ValueError("gated_delta_net: z must be (1, B*T, Hv*V)")
    for name, t in (("b", b), ("a", a)):
        if _shape(t) != (1, rows, value_heads):
            raise ValueError(f"gated_delta_net: {name} must be (1, B*T, Hv)")
    if _shape(conv_state) != (1, HISTORY * batch, channels):
        raise ValueError("gated_delta_net: conv_state must be (1, 3*B, C)")
    if _shape(conv_weight) != (HISTORY + 1, channels):
        raise ValueError("gated_delta_net: conv_weight must be (4, C)")
    for name, t in (("A_log", A_log), ("dt_bias", dt_bias)):
        if _shape(t) != (value_heads,):
            raise ValueError(f"gated_delta_net: {name} must be (Hv,)")
    if _shape(norm_weight) != (value_dim,):
        raise ValueError("gated_delta_net: norm_weight must be (V,)")
    if recurrent_state.dtype != ttnn.float32:
        raise ValueError("gated_delta_net: recurrent_state must be float32")
    others = (mixed_qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias, norm_weight)
    if any(t.dtype != mixed_qkv.dtype for t in others):
        raise ValueError("gated_delta_net: mixed_qkv, z, b, a, conv_state and the weights must share one dtype")
    if not epsilon > 0:
        raise ValueError("gated_delta_net: epsilon must be > 0")
    return key_dim, value_dim


def validate(
    mixed_qkv,
    z,
    b,
    a,
    conv_state,
    recurrent_state,
    conv_weight,
    A_log,
    dt_bias,
    norm_weight,
    *,
    epsilon=1e-6,
    compute_kernel_config=None,
):
    key_dim, value_dim = _check_shapes(
        mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight, epsilon
    )
    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    axes = {
        "dtype": mixed_qkv.dtype,
        "layout": mixed_qkv.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
    }
    inputs = (_shape(mixed_qkv), _shape(recurrent_state))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"gated_delta_net: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"gated_delta_net: unsupported combination (refinement candidate): {exc}")

    tensors = (mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight)
    if any(t.layout != ttnn.TILE_LAYOUT for t in tensors):
        raise UnsupportedAxisValue("gated_delta_net: every tensor must be TILE_LAYOUT")

    footprint = l1_footprint_bytes(key_dim // TILE, value_dim // TILE)
    if footprint > L1_BUDGET:
        raise RuntimeError(f"gated_delta_net: per-core L1 footprint {footprint} B exceeds {L1_BUDGET} B")
    return config


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def gated_delta_net(
    mixed_qkv: ttnn.Tensor,
    z: ttnn.Tensor,
    b: ttnn.Tensor,
    a: ttnn.Tensor,
    conv_state: ttnn.Tensor,
    recurrent_state: ttnn.Tensor,
    conv_weight: ttnn.Tensor,
    A_log: ttnn.Tensor,
    dt_bias: ttnn.Tensor,
    norm_weight: ttnn.Tensor,
    *,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
):
    config = validate(
        mixed_qkv,
        z,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config,
    )
    device = mixed_qkv.device()
    batch, value_heads, key_dim, value_dim = _shape(recurrent_state)
    rows, channels = _shape(mixed_qkv)[1], _shape(mixed_qkv)[2]

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, rows, value_heads * value_dim]),
        mixed_qkv.dtype,
        ttnn.TILE_LAYOUT,
        device,
        mixed_qkv.memory_config(),
    )
    conv_state_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, HISTORY * batch, channels]),
        conv_state.dtype,
        ttnn.TILE_LAYOUT,
        device,
        conv_state.memory_config(),
    )
    recurrent_state_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, value_heads, key_dim, value_dim]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
        recurrent_state.memory_config(),
    )

    inputs = (mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight)
    outputs = (output, conv_state_out, recurrent_state_out)
    descriptor = create_program_descriptor(inputs, outputs, epsilon=epsilon, compute_kernel_config=config)
    ttnn.generic_op([*inputs, *outputs], descriptor)
    return output, conv_state_out, recurrent_state_out
