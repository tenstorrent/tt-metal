# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""scaled_dot_product_attention (Flash Attention) — registry-model op.

Computes ``O = softmax(Q·Kᵀ·scale [+ mask]) · V`` per (batch, head) with the
online-softmax recurrence (the S_q×S_kv score matrix is never materialized).

Four registry declarations (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate)
plus the public entry point. See eval/op_template.py for the skeleton and
ttnn/ttnn/operations/scaled_dot_product_attention/op_design.md for the design.
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
# inputs = (Q_shape, K_shape, V_shape). Each is (B, H, S, D).


def tag_alignment(inputs, axes):
    """Q's last two dims (S_q, D). tile_aligned when both %32==0;
    w_non_aligned when D not aligned; h_non_aligned when D aligned but S_q not."""
    q_shape = inputs[0]
    s_q, d = q_shape[-2], q_shape[-1]
    if d % 32 != 0:
        return "w_non_aligned"
    if s_q % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_attention_kind(inputs, axes):
    """self when Q.S_q == K.S_kv, cross otherwise."""
    return "self" if inputs[0][-2] == inputs[1][-2] else "cross"


def tag_kv_heads(inputs, axes):
    """mha when H_q == H_kv, mqa when H_kv == 1, gqa otherwise."""
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
# 2. SUPPORTED  (Phase-1: narrow, complete in axes)
# ---------------------------------------------------------------------------
#
# Every axis the golden feature_spec TARGET enumerates gets an entry.
# Phase-1 ships the maxed corner: bf16 @ fp32_dest_acc_en=True, TILE, tile-
# aligned, self/cross, mha/gqa/mqa, mask none/custom, scale auto/explicit.

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "attention_kind": ["self", "cross"],
    "kv_heads_mode": ["mha", "gqa", "mqa"],
    "mask_mode": ["none", "custom"],
    "scale_mode": ["auto", "explicit"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# causal is not in SUPPORTED[mask_mode], so {causal, cross} is already refused
# at the axis level; no cell-level EXCLUSION is required for it.
#
# Refinement 1 (numerical configurability):
#   * {float32, fp32_dest_acc_en=False} — the maxed input dtype paired with the
#     16-bit-DEST accumulator is legal-but-lossy (fp32 precision is thrown away
#     mid-reduce). Refused, mirroring softmax.
#   * {bfloat8_b, fp32_dest_acc_en=False} — kept SUPPORTED: it clears the golden
#     (bf8b, False) tolerance (0.99 / 0.12); block-float already dominates the
#     error budget, so the DEST width is second-order.

EXCLUSIONS = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """Phase-1 default compute-kernel config (single source of truth).

    HiFi2 + fp32 DEST accumulation + no approx. The golden harness (axes.py)
    reads this to tag the ``fp32_dest_acc_en`` axis; the entry point resolves a
    ``None`` ``compute_kernel_config`` through it. Never hardcode the default
    elsewhere — if it moves, both the op and the tag follow together.
    """
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def _mask_mode(attn_mask, is_causal):
    if is_causal:
        return "causal"
    return "custom" if attn_mask is not None else "none"


def validate(query, key, value, *, attn_mask=None, is_causal=False, scale=None, fp32_dest_acc_en=True):
    # ----- Minimal rank guard (needed to build the axes dict / run taggers) -----
    q_shape = tuple(query.shape)
    k_shape = tuple(key.shape)
    v_shape = tuple(value.shape)

    for name, shape in (("query", q_shape), ("key", k_shape), ("value", v_shape)):
        if len(shape) != 4:
            raise ValueError(f"scaled_dot_product_attention: {name} must be 4D (B,H,S,D); got {shape}")

    b_q, h_q, s_q, d_q = q_shape
    b_k, h_kv, s_kv, d_k = k_shape

    # ----- Axis dict (mirrors the golden harness) -----
    #
    # The registry support gate takes precedence over the finer tensor-shape
    # contract: a cell whose axes fall outside SUPPORTED (or match EXCLUSIONS)
    # must raise the support-refusal (NotImplementedError) REGARDLESS of any
    # accompanying shape issue. Running SUPPORTED/EXCLUSIONS *before* the
    # detailed shape checks keeps an unsupported cell (e.g. fp32_dest_acc_en=
    # False) from being misclassified when it also carries a shape-contract
    # violation (e.g. a batch-broadcast mask) — the axis refusal wins.
    inputs = (q_shape, k_shape, v_shape)
    axes = {
        "dtype": query.dtype,
        "fp32_dest_acc_en": bool(fp32_dest_acc_en),
        "layout": query.layout,
        "mask_mode": _mask_mode(attn_mask, is_causal),
        "scale_mode": "explicit" if scale is not None else "auto",
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

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

    # ----- Tensor-shape contract (ValueError) — only for SUPPORTED cells -----
    if d_q != d_k:
        raise ValueError(f"scaled_dot_product_attention: head_dim mismatch Q({d_q}) vs K({d_k})")
    if k_shape != v_shape:
        raise ValueError(f"scaled_dot_product_attention: K shape {k_shape} must equal V shape {v_shape}")
    if b_q != b_k:
        raise ValueError(f"scaled_dot_product_attention: batch mismatch Q({b_q}) vs K/V({b_k})")
    if h_kv == 0 or h_q % h_kv != 0:
        raise ValueError(f"scaled_dot_product_attention: H_q({h_q}) must be a multiple of H_kv({h_kv})")

    if is_causal and attn_mask is not None:
        raise ValueError("scaled_dot_product_attention: is_causal and attn_mask are mutually exclusive")

    if attn_mask is not None:
        m_shape = tuple(attn_mask.shape)
        if len(m_shape) != 4:
            raise ValueError(f"scaled_dot_product_attention: attn_mask must be 4D; got {m_shape}")
        m_b, m_h, m_sq, m_skv = m_shape
        if m_b != b_q or m_sq != s_q or m_skv != s_kv:
            raise ValueError(
                f"scaled_dot_product_attention: attn_mask {m_shape} incompatible with "
                f"(B={b_q}, S_q={s_q}, S_kv={s_kv})"
            )
        if m_h not in (1, h_q):
            raise ValueError(f"scaled_dot_product_attention: attn_mask head dim {m_h} must be 1 or H_q({h_q})")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    query,
    key,
    value,
    *,
    attn_mask=None,
    attention_mask=None,
    is_causal: bool = False,
    scale: float = None,
    compute_kernel_config=None,
):
    """Flash-attention SDPA. Output shape == query, bfloat16, TILE_LAYOUT."""
    # Accept the prompt's `attention_mask` spelling as an alias for `attn_mask`.
    if attention_mask is not None:
        if attn_mask is not None:
            raise ValueError("scaled_dot_product_attention: pass only one of attn_mask / attention_mask")
        attn_mask = attention_mask

    # Resolve the compute-kernel config (Phase-1 default: HiFi2 + fp32 DEST acc).
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    fp32_dest_acc_en = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))

    validate(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )

    d = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    device = query.device()
    # Output dtype follows the input dtype (fp32→fp32, bf16→bf16, bf8b→bf8b) —
    # the golden contract checks got.dtype == input dtype, and a downstream fp32
    # consumer must not be silently narrowed to bf16.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(query.shape)),
        query.dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program_descriptor, ordered_tensors = create_program_descriptor(
        query,
        key,
        value,
        attn_mask,
        output_tensor,
        scale=float(scale),
        compute_kernel_config=compute_kernel_config,
    )

    return ttnn.generic_op(ordered_tensors, program_descriptor)
