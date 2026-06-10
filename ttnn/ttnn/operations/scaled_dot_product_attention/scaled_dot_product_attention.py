# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
scaled_dot_product_attention — Flash Attention for TT hardware.

O = softmax(Q @ K^T * scale + mask, dim=-1) @ V, computed with the online
softmax recurrence so the full S_q x S_kv score matrix is never materialized.

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate().
"""

from __future__ import annotations

import math

import ttnn

from .scaled_dot_product_attention_program_descriptor import create_program_descriptor

TILE = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# inputs = (q_shape, k_shape, v_shape) — mask is carried in axes["mask"].


def tag_alignment(inputs, axes):
    """S_q, S_kv and D all tile-aligned (Phase 0 requirement)."""
    q, k, _v = inputs
    if q[-1] % TILE == 0 and q[-2] % TILE == 0 and k[-2] % TILE == 0:
        return "tile_aligned"
    return "non_tile_aligned"


def tag_gqa(inputs, axes):
    """Phase 0: H_kv must equal H (no grouped-query mapping yet)."""
    q, k, _v = inputs
    return "mha" if q[1] == k[1] else "gqa"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "gqa": tag_gqa,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "gqa": ["mha"],
    "mask": ["none", "additive"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# Structural validation — caller errors (ValueError), not NotImplementedError
# ---------------------------------------------------------------------------


def _validate_shapes(q, k, v, attention_mask):
    qs, ks, vs = list(q.shape), list(k.shape), list(v.shape)
    if len(qs) != 4 or len(ks) != 4 or len(vs) != 4:
        raise ValueError(f"sdpa: expected 4D tensors, got q={qs} k={ks} v={vs}")
    if ks != vs:
        raise ValueError(f"sdpa: K and V shapes must match, got k={ks} v={vs}")
    if qs[0] != ks[0]:
        raise ValueError(f"sdpa: batch mismatch q={qs[0]} k={ks[0]}")
    if qs[3] != ks[3]:
        raise ValueError(f"sdpa: head_dim mismatch q={qs[3]} k={ks[3]}")
    if ks[1] != 0 and qs[1] % ks[1] != 0:
        raise ValueError(f"sdpa: H ({qs[1]}) must be divisible by H_kv ({ks[1]})")
    if attention_mask is not None:
        ms = list(attention_mask.shape)
        if len(ms) != 4 or ms[0] != qs[0] or ms[1] not in (1, qs[1]) or ms[2] != qs[2] or ms[3] != ks[2]:
            raise ValueError(
                f"sdpa: mask shape {ms} incompatible with q={qs}, kv={ks} "
                f"(expected ({qs[0]}, 1 or {qs[1]}, {qs[2]}, {ks[2]}))"
            )


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(q, k, v, *, attention_mask=None, scale=None):
    _validate_shapes(q, k, v, attention_mask)

    axes = {
        "dtype": q.dtype,
        "layout": q.layout,
        "mask": "none" if attention_mask is None else "additive",
    }
    inputs = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for t in (k, v) + ((attention_mask,) if attention_mask is not None else ()):
        if t.dtype != q.dtype:
            raise NotImplementedError(f"sdpa: mixed dtypes not supported ({t.dtype} vs {q.dtype})")
        if t.layout != ttnn.TILE_LAYOUT:
            raise NotImplementedError("sdpa: all tensors must be TILE_LAYOUT")

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise NotImplementedError(f"sdpa: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(kk) == vv for kk, vv in exc.items()):
            raise NotImplementedError(f"sdpa: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    *,
    attention_mask: ttnn.Tensor = None,
    scale: float = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Flash-Attention SDPA. Output shape = query shape."""
    validate(query, key, value, attention_mask=attention_mask, scale=scale)

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    device = query.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(query.shape)),
        query.dtype,
        query.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(query, key, value, attention_mask, output_tensor, scale=float(scale))

    io_tensors = [query, key, value]
    if attention_mask is not None:
        io_tensors.append(attention_mask)
    io_tensors.append(output_tensor)  # output last
    return ttnn.generic_op(io_tensors, program_descriptor)
