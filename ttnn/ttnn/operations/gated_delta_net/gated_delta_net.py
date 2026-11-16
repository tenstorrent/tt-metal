# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gated_delta_net: Gated DeltaNet token mixer as one device program (registry model)."""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .gated_delta_net_program_descriptor import (
    HISTORY,
    TILE,
    cached_program_descriptor,
    check_l1_footprint,
    compute_config_key,
    knob_values,
    l1_cb_budget,
    l1_footprint_bytes,
    l1_region_end,
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
    "dtype": [ttnn.float32, ttnn.bfloat16],
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


def _check_shapes(
    mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight, epsilon, packed
):
    if len(recurrent_state.shape) != 4:
        raise ValueError("gated_delta_net: recurrent_state must be (B, Hv, K, V)")
    batch, value_heads, key_dim, value_dim = _shape(recurrent_state)
    x_shape = _shape(mixed_qkv)
    if len(x_shape) != 3 or x_shape[0] != 1:
        raise ValueError("gated_delta_net: mixed_qkv must be (1, B*T, C)")
    rows, channels = x_shape[1], x_shape[2]
    if packed:
        # Rows [x (C) | z (Hv*V) | a (Hv) | b (Hv)]: z, a and b are read from them (z, b, a are mixed_qkv here).
        channels -= value_heads * value_dim + 2 * value_heads
        if 2 * value_heads > TILE:
            raise ValueError("gated_delta_net: the packed form needs Hv <= 16 (a and b in one tile column)")
    if batch < 1 or rows < 1 or rows % batch:
        raise ValueError("gated_delta_net: rows of mixed_qkv must be a positive multiple of B")
    qk_channels = channels - value_heads * value_dim
    if qk_channels <= 0 or qk_channels % (2 * key_dim):
        raise ValueError(
            "gated_delta_net: C - Hv*V must be a positive multiple of 2*K"
            + (" (packed form: mixed_qkv is (1, B*T, C + Hv*V + 2*Hv))" if packed else "")
        )
    key_heads = qk_channels // (2 * key_dim)
    if value_heads % key_heads:
        raise ValueError("gated_delta_net: Hv must be a multiple of Hk")
    if not packed and _shape(z) != (1, rows, value_heads * value_dim):
        raise ValueError("gated_delta_net: z must be (1, B*T, Hv*V)")
    for name, t in (("b", b), ("a", a)):
        if not packed and _shape(t) != (1, rows, value_heads):
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


def _check_state_out(name, out, state):
    """A caller-provided state output: the state input itself (updated in place) or a distinct tensor of its form."""
    if out is not None and (
        _shape(out) != _shape(state)
        or out.dtype != state.dtype
        or out.layout != state.layout
        or out.memory_config() != state.memory_config()
    ):
        raise ValueError(f"gated_delta_net: {name} must match the shape, dtype, layout and memory config of its input")


def _check_state_out_aliasing(name, out, state, addresses):
    """Per call (addresses are not in the cache key): a distinct state output must not be another input's buffer."""
    if out is None:
        return
    address = out.buffer_address()
    if address != state.buffer_address() and address in addresses:
        raise ValueError(f"gated_delta_net: {name} must be its own state input or a tensor no other input uses")


# Cache keys whose structural checks passed (validate() then re-checks only the per-call buffer aliasing).
_validated_keys = set()


def _tensor_key(t):
    # TensorSpec and Shape hash by identity; repr(shape) is the cheapest value form.
    return (repr(t.shape), t.dtype, t.layout, t.memory_config())


def _cache_key(tensors, conv_state_out, recurrent_state_out, epsilon, compute_kernel_config, packed, memory_config):
    """Everything the checks and the program descriptor read from a call except buffer addresses: the device, every
    tensor's shape, dtype, layout and memory config, which state outputs are given and which update in place,
    epsilon, the compute config, the descriptor knobs, the packed form and the output's memory config (the L1 bound
    of a call with L1 tensors is appended per call)."""
    conv_state, recurrent_state = tensors[4], tensors[5]
    outs = []
    for out, state in ((conv_state_out, conv_state), (recurrent_state_out, recurrent_state)):
        outs += [None, None] if out is None else [_tensor_key(out), out.buffer_address() == state.buffer_address()]
    return (
        tensors[0].device().id(),
        *(_tensor_key(t) for t in tensors),
        *outs,
        float(epsilon),
        compute_config_key(compute_kernel_config),
        knob_values(),
        packed,
        memory_config,
    )


_l1_keys = {}  # cache key -> a tensor of the call is in L1


def _any_l1(key, memory_config):
    known = _l1_keys.get(key)
    if known is None:
        if len(_l1_keys) > 1024:
            _l1_keys.clear()
        configs = [spec[3] for spec in key[1:11]] + [memory_config]
        known = _l1_keys[key] = any(c is not None and c.buffer_type == ttnn.BufferType.L1 for c in configs)
    return known


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
    conv_state_out=None,
    recurrent_state_out=None,
    memory_config=None,
):
    inputs, packed = _inputs(mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight)
    return _validate(
        inputs, epsilon, compute_kernel_config, conv_state_out, recurrent_state_out, packed, memory_config
    )[0]


def _inputs(mixed_qkv, z, b, a, *rest):
    """(the ten input tensors, packed): the packed form (z, b and a all None) reads them from mixed_qkv."""
    given = [t is not None for t in (z, b, a)]
    if any(given) != all(given):
        raise ValueError(
            "gated_delta_net: pass z, b and a as tensors (split form) or all three as None (packed form: mixed_qkv "
            "rows hold [x | z | a | b]); got "
            + ", ".join(f"{name}={'tensor' if g else 'None'}" for name, g in zip("zba", given))
        )
    if not given[0]:
        z = b = a = mixed_qkv
    return (mixed_qkv, z, b, a, *rest), not given[0]


def _validate(tensors, epsilon, compute_kernel_config, conv_state_out, recurrent_state_out, packed, memory_config):
    """validate(), returning (compute config, program cache key)."""
    config = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    key = _cache_key(
        tensors, conv_state_out, recurrent_state_out, epsilon, compute_kernel_config, packed, memory_config
    )
    if key not in _validated_keys:
        _validate_structure(tensors, epsilon, config, conv_state_out, recurrent_state_out, packed)
        if len(_validated_keys) > 1024:
            _validated_keys.clear()
        _validated_keys.add(key)
    addresses = [t.buffer_address() for t in tensors]
    _check_state_out_aliasing("conv_state_out", conv_state_out, tensors[4], addresses)
    _check_state_out_aliasing("recurrent_state_out", recurrent_state_out, tensors[5], addresses)
    return config, key


def _validate_structure(tensors, epsilon, config, conv_state_out, recurrent_state_out, packed):
    mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight = tensors
    key_dim, value_dim = _check_shapes(
        mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight, epsilon, packed
    )
    _check_state_out("conv_state_out", conv_state_out, conv_state)
    _check_state_out("recurrent_state_out", recurrent_state_out, recurrent_state)
    axes = {
        "dtype": mixed_qkv.dtype,
        "layout": mixed_qkv.layout,
        "fp32_dest_acc_en": bool(config.fp32_dest_acc_en),
    }
    inputs = ((1, _shape(mixed_qkv)[1], _shape(conv_state)[2]), _shape(recurrent_state))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"gated_delta_net: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"gated_delta_net: unsupported combination (refinement candidate): {exc}")

    if any(t.layout != ttnn.TILE_LAYOUT for t in tensors):
        raise UnsupportedAxisValue("gated_delta_net: every tensor must be TILE_LAYOUT")

    footprint = l1_footprint_bytes(key_dim // TILE, value_dim // TILE, mixed_qkv.dtype)
    check_l1_footprint("R-STEP pair", footprint, l1_cb_budget(mixed_qkv.device()))


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
    conv_state_out: ttnn.Tensor = None,
    recurrent_state_out: ttnn.Tensor = None,
    memory_config: ttnn.MemoryConfig = None,
):
    """Returns (output, conv_state_out, recurrent_state_out).

    Packed form: with z, b and a all None, mixed_qkv is (1, B*T, C + Hv*V + 2*Hv) and holds per row x in columns
    [0, C), z in [C, C + Hv*V), a in the next Hv columns and b in the last Hv (the model's fused projection output);
    the tile padding beyond the last column is never read into a result. Hv <= 16.
    mixed_qkv, z, b and a may be interleaved L1 tensors: they are read where they live, and the static buffers of the
    program are sized to end below them (l1_region_end).
    memory_config: of the output (default: mixed_qkv's).

    conv_state_out / recurrent_state_out: optional caller-provided state outputs, each either a tensor of the same
    shape, dtype, layout and memory config as its state input, or that input itself (the state is then updated in
    place; every byte of it is read before any core overwrites it). None allocates a new tensor. No other input is
    written.
    """
    inputs, packed = _inputs(mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight)
    config, key = _validate(
        inputs, epsilon, compute_kernel_config, conv_state_out, recurrent_state_out, packed, memory_config
    )
    device = mixed_qkv.device()
    batch, value_heads, key_dim, value_dim = _shape(recurrent_state)
    rows, channels = _shape(mixed_qkv)[1], _shape(conv_state)[2]

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, rows, value_heads * value_dim]),
        mixed_qkv.dtype,
        ttnn.TILE_LAYOUT,
        device,
        memory_config if memory_config is not None else mixed_qkv.memory_config(),
    )
    if conv_state_out is None:
        conv_state_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, HISTORY * batch, channels]),
            conv_state.dtype,
            ttnn.TILE_LAYOUT,
            device,
            conv_state.memory_config(),
        )
    if recurrent_state_out is None:
        recurrent_state_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batch, value_heads, key_dim, value_dim]),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            device,
            recurrent_state.memory_config(),
        )

    outputs = (output, conv_state_out, recurrent_state_out)
    # Any tensor of the call in L1 (key: every memory config, in _tensor_key order): the CB region ends below them.
    l1_end = l1_region_end((*inputs, *outputs)) if _any_l1(key, memory_config) else None
    descriptor = cached_program_descriptor(
        (key, l1_end), inputs, outputs, epsilon=epsilon, compute_kernel_config=config, packed=packed, l1_end=l1_end
    )
    ttnn.generic_op([*inputs, *outputs], descriptor)
    return output, conv_state_out, recurrent_state_out
