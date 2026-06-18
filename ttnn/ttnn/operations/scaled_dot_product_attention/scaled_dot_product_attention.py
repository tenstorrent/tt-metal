# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Flash-Attention scaled_dot_product_attention — registry-model op file.

Computes ``softmax(Q @ K^T * scale + mask) @ V`` with the Flash Attention
(online-softmax) algorithm: the full ``S_q x S_kv`` score matrix is never
materialized — every score-bearing CB is sized to one ``B_q x B_kv`` block.

This file holds the four registry declarations (INPUT_TAGGERS, SUPPORTED,
EXCLUSIONS, validate) inline, mirroring eval/op_template.py, plus the public
entry point. The kernel work happens in the generic-op program descriptor.

INVALID is intentionally NOT declared here — it lives in
eval/golden_tests/scaled_dot_product_attention/feature_spec.py (SDPA is
TILE-only, so the canonical bf8b+ROW_MAJOR rule is vacuous; INVALID == []).
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .scaled_dot_product_attention_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------
#
# inputs is (Q_shape, K_shape, V_shape); each is (B, H, S, D). Taggers project
# shape facets the kernel cares about onto categorical axes.


def tag_alignment(inputs, axes):
    """Q's last two dims (S_q, D): both %32==0 -> tile_aligned; D not aligned ->
    w_non_aligned; else (S_q not aligned, D aligned) -> h_non_aligned."""
    s_q, d = inputs[0][-2], inputs[0][-1]
    if d % 32 != 0:
        return "w_non_aligned"
    if s_q % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_attention_kind(inputs, axes):
    """Q.S_q vs K.S_kv: equal -> self, differ -> cross."""
    return "self" if inputs[0][-2] == inputs[1][-2] else "cross"


def tag_kv_heads(inputs, axes):
    """Q.H_q vs K.H_kv: equal -> mha, H_kv==1 -> mqa, else gqa."""
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
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# Phase 0 baseline. GQA/MQA require nothing beyond the reader's head remap
# (identical compute kernel), so all three kv_heads_mode values are supported.
# The mask is read as a generic additive block, so causal+cross needs no
# special handling (no S_q==S_kv assumption) — hence no EXCLUSION.

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "attention_kind": ["self", "cross"],
    "kv_heads_mode": ["mha", "gqa", "mqa"],
    "mask_mode": ["none", "causal"],
    "scale_mode": ["auto", "explicit"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# The generic additive-mask path handles causal+cross with no S_q==S_kv
# assumption, so there is nothing to exclude inside the SUPPORTED rectangle.

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _shape_contract(Q, K, V, attention_mask):
    """Tensor-shape contract violations -> ValueError (raised BEFORE the axis
    gate). These are not 'unsupported axis values' — they are malformed calls."""
    q, k, v = tuple(Q.shape), tuple(K.shape), tuple(V.shape)

    if len(q) != 4 or len(k) != 4 or len(v) != 4:
        raise ValueError(f"scaled_dot_product_attention: Q/K/V must be 4D (B,H,S,D); " f"got Q={q}, K={k}, V={v}")
    if q[-1] != k[-1]:
        raise ValueError(f"scaled_dot_product_attention: head_dim mismatch Q.D={q[-1]} vs K.D={k[-1]}")
    if k != v:
        raise ValueError(f"scaled_dot_product_attention: K and V must share shape; got K={k}, V={v}")
    if q[0] != k[0]:
        raise ValueError(f"scaled_dot_product_attention: batch mismatch Q.B={q[0]} vs K.B={k[0]}")
    h_q, h_kv = q[1], k[1]
    if h_kv == 0 or h_q % h_kv != 0:
        raise ValueError(f"scaled_dot_product_attention: H_q ({h_q}) must be a multiple of H_kv ({h_kv})")
    if attention_mask is not None:
        m = tuple(attention_mask.shape)
        if len(m) != 4:
            raise ValueError(f"scaled_dot_product_attention: mask must be 4D; got {m}")
        if m[-2] != q[-2] or m[-1] != k[-2]:
            raise ValueError(
                f"scaled_dot_product_attention: mask last two dims {m[-2:]} must equal "
                f"(S_q, S_kv) = ({q[-2]}, {k[-2]})"
            )
        if m[1] not in (1, h_q):
            raise ValueError(f"scaled_dot_product_attention: mask num_heads {m[1]} must be 1 or H_q={h_q}")
        if m[0] != q[0]:
            raise ValueError(f"scaled_dot_product_attention: mask batch {m[0]} must equal B={q[0]}")


def validate(Q, K, V, *, attention_mask=None, scale=None):
    # Shape contract first (malformed call -> ValueError, distinct from a
    # deliberate support refusal).
    _shape_contract(Q, K, V, attention_mask)

    inputs = (tuple(Q.shape), tuple(K.shape), tuple(V.shape))

    axes = {
        "dtype": Q.dtype,
        "layout": Q.layout,
        "mask_mode": "causal" if attention_mask is not None else "none",
        "scale_mode": "explicit" if scale is not None else "auto",
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    # 1. SUPPORTED — per axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: {axis}={axes[axis]!r} not in SUPPORTED {allowed}"
            )

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == val for k, val in exc.items()):
            raise ExcludedCell(
                f"scaled_dot_product_attention: unsupported combination " f"(refinement candidate): {exc}"
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    Q: ttnn.Tensor,
    K: ttnn.Tensor,
    V: ttnn.Tensor,
    *,
    attention_mask: ttnn.Tensor = None,
    scale: float = None,
) -> ttnn.Tensor:
    """Flash-Attention SDPA: ``softmax(Q @ K^T * scale + mask) @ V``.

    Args:
        Q: ``(B, H_q, S_q, D)`` bf16, TILE_LAYOUT.
        K, V: ``(B, H_kv, S_kv, D)`` bf16, TILE_LAYOUT. ``H_q % H_kv == 0``.
        attention_mask: optional additive mask ``(B, 1|H_q, S_q, S_kv)``
            (0=attend, -inf=mask).
        scale: optional float; ``1/sqrt(D)`` when None.

    Returns:
        ``(B, H_q, S_q, D)`` bf16, TILE_LAYOUT.
    """
    validate(Q, K, V, attention_mask=attention_mask, scale=scale)

    device = Q.device()
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    D = int(Q.shape[-1])
    resolved_scale = float(scale) if scale is not None else 1.0 / math.sqrt(D)

    output_shape = list(Q.shape)  # same as Q
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        Q.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(Q, K, V, attention_mask, output_tensor, scale=resolved_scale)

    io_tensors = [Q, K, V]
    if attention_mask is not None:
        io_tensors.append(attention_mask)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)
