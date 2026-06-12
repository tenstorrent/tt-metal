# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""scaled_dot_product_attention (Flash Attention) — registry-model op.

Computes softmax(Q . Kᵀ * scale + mask) . V via the Flash Attention
algorithm (tiled, online softmax, O(S) memory). The full S_q x S_kv score
matrix is never materialized — see kernels/ and the program descriptor.

Public entry point signature mirrors torch.nn.functional.scaled_dot_product_attention:

    scaled_dot_product_attention(query, key, value, *,
                                 attn_mask=None, is_causal=False, scale=None)
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .scaled_dot_product_attention_program_descriptor import create_program_descriptor

TILE_DIM = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — project input shapes onto categorical axes
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Q's last two dims (S_q, D). w_non_aligned if D % 32; else h_non_aligned
    if S_q % 32; else tile_aligned."""
    s_q, d = inputs[0][-2], inputs[0][-1]
    if d % TILE_DIM != 0:
        return "w_non_aligned"
    if s_q % TILE_DIM != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_attention_kind(inputs, axes):
    """self when Q.S_q == K.S_kv, else cross."""
    return "self" if inputs[0][-2] == inputs[1][-2] else "cross"


def tag_kv_heads(inputs, axes):
    """mha when H_q == H_kv; mqa when H_kv == 1; else gqa (H_q % H_kv == 0)."""
    h_q, h_kv = inputs[0][1], inputs[1][1]
    if h_q == h_kv:
        return "mha"
    if h_kv == 1:
        return "mqa"
    return "gqa"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "attention_kind": tag_attention_kind,
    "kv_heads_mode": tag_kv_heads,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — currently-accepted per-axis values (Phase 0)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "attention_kind": ["self", "cross"],
    "kv_heads_mode": ["mha"],
    "mask_mode": ["none", "custom"],
    "scale_mode": ["auto", "explicit"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — cells inside cartesian(SUPPORTED) refused for now
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate() — runtime gate (first line of the entry point)
# ---------------------------------------------------------------------------


def validate(query, key, value, *, attn_mask=None, is_causal=False, scale=None):
    # --- tensor-shape contract violations (ValueError / RuntimeError) ---
    for name, t in (("query", query), ("key", key), ("value", value)):
        if len(t.shape) != 4:
            raise ValueError(f"scaled_dot_product_attention: {name} must be 4D (B,H,S,D); got {tuple(t.shape)}")

    q_shape = tuple(int(x) for x in query.shape)
    k_shape = tuple(int(x) for x in key.shape)
    v_shape = tuple(int(x) for x in value.shape)

    if q_shape[-1] != k_shape[-1]:
        raise ValueError(f"scaled_dot_product_attention: head_dim mismatch Q.D={q_shape[-1]} vs K.D={k_shape[-1]}")
    if k_shape[-2] != v_shape[-2]:
        raise ValueError(f"scaled_dot_product_attention: seq_len mismatch K.S_kv={k_shape[-2]} vs V.S_kv={v_shape[-2]}")
    if k_shape[-1] != v_shape[-1]:
        raise ValueError(f"scaled_dot_product_attention: head_dim mismatch K.D={k_shape[-1]} vs V.D={v_shape[-1]}")
    if q_shape[0] != k_shape[0] or k_shape[0] != v_shape[0]:
        raise ValueError(
            f"scaled_dot_product_attention: batch mismatch Q.B={q_shape[0]} K.B={k_shape[0]} V.B={v_shape[0]}"
        )
    if k_shape[1] != v_shape[1]:
        raise ValueError(f"scaled_dot_product_attention: K/V num_heads mismatch K.H={k_shape[1]} V.H={v_shape[1]}")

    if is_causal and attn_mask is not None:
        raise ValueError("scaled_dot_product_attention: is_causal and attn_mask are mutually exclusive")

    if attn_mask is not None:
        m_shape = tuple(int(x) for x in attn_mask.shape)
        if len(m_shape) != 4:
            raise ValueError(f"scaled_dot_product_attention: attn_mask must be 4D; got {m_shape}")
        if m_shape[0] != q_shape[0]:
            raise ValueError(f"scaled_dot_product_attention: mask batch {m_shape[0]} != Q batch {q_shape[0]}")
        if m_shape[1] not in (1, q_shape[1]):
            raise ValueError(f"scaled_dot_product_attention: mask num_heads {m_shape[1]} must be 1 or {q_shape[1]}")
        if m_shape[-2] != q_shape[-2] or m_shape[-1] != k_shape[-2]:
            raise ValueError(
                f"scaled_dot_product_attention: mask (S_q,S_kv)=({m_shape[-2]},{m_shape[-1]}) "
                f"!= ({q_shape[-2]},{k_shape[-2]})"
            )

    # --- axes dict (tensor props + kwargs + taggers) ---
    if is_causal:
        mask_mode = "causal"
    elif attn_mask is not None:
        mask_mode = "custom"
    else:
        mask_mode = "none"
    scale_mode = "explicit" if scale is not None else "auto"

    axes = {
        "dtype": query.dtype,
        "layout": query.layout,
        "mask_mode": mask_mode,
        "scale_mode": scale_mode,
    }
    tagger_inputs = (q_shape, k_shape, v_shape)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(tagger_inputs, axes)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: {axis}={axes[axis]!r} not in SUPPORTED {allowed}"
            )

    # 2. EXCLUSIONS — cell-level
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"scaled_dot_product_attention: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    *,
    attn_mask: ttnn.Tensor = None,
    is_causal: bool = False,
    scale: float = None,
) -> ttnn.Tensor:
    validate(query, key, value, attn_mask=attn_mask, is_causal=is_causal, scale=scale)

    if scale is None:
        scale = 1.0 / math.sqrt(int(query.shape[-1]))

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(query.shape)),
        query.dtype,
        query.layout,
        query.device(),
        query.memory_config(),
    )

    program_descriptor = create_program_descriptor(query, key, value, attn_mask, output_tensor, scale=scale)

    io_tensors = [query, key, value]
    if attn_mask is not None:
        io_tensors.append(attn_mask)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)
