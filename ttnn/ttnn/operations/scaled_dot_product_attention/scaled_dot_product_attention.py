# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""scaled_dot_product_attention (Flash Attention) — registry-model op.

Fused, on-device FlashAttention-2: output = softmax(Q·Kᵀ·scale + mask)·V,
computed with tiled online-softmax over KV blocks so the S_q×S_kv score matrix
is never materialized. See op_design.md for the binding blocking model.

Four registry declarations (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate) plus
the public entry point, mirroring eval/op_template.py.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .scaled_dot_product_attention_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Compute-config default — single source of truth for the maxed-out corner.
# ---------------------------------------------------------------------------
def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    """HiFi4 + fp32 DEST accumulation (mirrors rms_norm/softmax).

    `compute_kernel_config=None` resolves through this in BOTH validate() and the
    entry point; the golden axis-tagger reads the same factory so the
    `fp32_dest_acc_en` axis never drifts from the op's real default.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — shape facets the kernel gates on.
#    inputs = (Q_shape, K_shape, V_shape); dims are (B, H, S, D).
# ---------------------------------------------------------------------------
def tag_alignment(inputs, axes):
    """Q's last two dims (S_q, D): tile_aligned / w_non_aligned / h_non_aligned."""
    q_shape = inputs[0]
    s_q = q_shape[-2]
    d = q_shape[-1]
    if d % 32 != 0:
        return "w_non_aligned"
    if s_q % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_attention_kind(inputs, axes):
    """self when Q.S_q == K.S_kv, else cross."""
    s_q = inputs[0][-2]
    s_kv = inputs[1][-2]
    return "self" if s_q == s_kv else "cross"


def tag_kv_heads(inputs, axes):
    """mha when H_q == H_kv; mqa when H_kv == 1; else gqa."""
    h_q = inputs[0][1]
    h_kv = inputs[1][1]
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
# 2. SUPPORTED — Phase-0 claim (what the kernel actually handles now).
# ---------------------------------------------------------------------------
SUPPORTED = {
    # R2: full float precision surface. float32 + bfloat8_b ride the same helper
    # pipeline as bf16 (helpers carry data-format reconfig); input/output CB
    # formats derive from the tensor dtype in the descriptor, and the softmax
    # intermediates (cb_scores/cb_exp) are promoted to fp32 under fp32-DEST
    # accumulation (bf16 otherwise, keeping the perf regime byte-identical).
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    # R2: expose the 16-bit-DEST accumulation regime (the perf-flagged shape pins
    # fp32_dest_acc_en=False). {float32, False} is refused in EXCLUSIONS below.
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    # R1: non-tile-aligned shapes handled natively in the kernel. w_non_aligned
    # (D%32≠0) rides the input tile zero-padding through the QKᵀ contraction and
    # PV free dim; h_non_aligned (S_q or S_kv %32≠0) writes only valid Q rows and
    # masks the last KV tile's padding columns to −∞ before the softmax reduce.
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "attention_kind": ["self", "cross"],
    "kv_heads_mode": ["mha", "gqa", "mqa"],
    "mask_mode": ["none", "custom"],
    "scale_mode": ["auto", "explicit"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — R2 arms the maxed-input + non-maxed-accumulation corner.
#    (R4 will arm {mask_mode:causal, attention_kind:cross}.)
# ---------------------------------------------------------------------------
EXCLUSIONS = [
    # R2: float32 input with 16-bit DEST accumulation. A maxed-precision input fed
    # through a non-maxed accumulator is lossy for no benefit — refuse it, honoring
    # the caller's fp32_dest_acc_en=False rather than silently forcing True. The
    # golden TOLERANCES map omits this exact combo for the same reason.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # R2: bfloat8_b on non-tile-aligned shapes — the canonical /numeric-formats-metal
    # block-float × partial-tile incompatibility. When S_kv is not a multiple of 32
    # the softmax reduce spans an additive-−∞ KV-padding mask; feeding that boundary
    # through bfloat8_b's shared-exponent blocks corrupts the padded columns'
    # contribution catastrophically (measured PCC ≈ 0.2–0.5 across the non-aligned
    # golden cells — not a near-miss). The failure tracks the S_kv%32≠0 path, which
    # appears under BOTH alignment tags, so both are refused. bf8b + tile_aligned is
    # fully supported; bf16 / float32 on non-aligned shapes (R1) are unaffected.
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
    {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"},
]


# ---------------------------------------------------------------------------
# 3b. PROPERTIES — non-axis capabilities for the Features tab.
# ---------------------------------------------------------------------------
PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
    # R2: the caller's compute_kernel_config (math_fidelity / fp32_dest_acc_en /
    # math_approx_mode / dst_full_sync_en) is threaded end-to-end; all four
    # fidelities are honored (default HiFi4). Characterized in
    # test_scaled_dot_product_attention_precision_matrix.
    "math_fidelity": {"value": ["HiFi4", "HiFi3", "HiFi2", "LoFi"], "source": "measured"},
}


# ---------------------------------------------------------------------------
# helper: derive mask_mode / scale_mode exactly as the golden harness does.
# ---------------------------------------------------------------------------
def _derive_mask_mode(attn_mask, is_causal):
    if is_causal:
        return "causal"
    if attn_mask is not None:
        return "custom"
    return "none"


def _build_axes(query, key, value, attn_mask, is_causal, scale, compute_kernel_config):
    cfg = compute_kernel_config or default_compute_kernel_config()
    axes = {
        "dtype": query.dtype,
        "fp32_dest_acc_en": bool(getattr(cfg, "fp32_dest_acc_en", True)),
        "layout": query.layout,
        "mask_mode": _derive_mask_mode(attn_mask, is_causal),
        "scale_mode": "explicit" if scale is not None else "auto",
    }
    tagger_inputs = (list(query.shape), list(key.shape), list(value.shape))
    ctx = dict(axes)
    for axis, tagger in INPUT_TAGGERS.items():
        val = tagger(tagger_inputs, ctx)
        axes[axis] = val
        ctx[axis] = val
    return axes


# ---------------------------------------------------------------------------
# 4. validate() — structural checks, then SUPPORTED, then EXCLUSIONS.
# ---------------------------------------------------------------------------
def validate(query, key, value, *, attn_mask=None, is_causal=False, scale=None, compute_kernel_config=None):
    # --- structural contract violations (raise before axis gating) ---
    for name, t in (("query", query), ("key", key), ("value", value)):
        if len(t.shape) != 4:
            raise ValueError(f"scaled_dot_product_attention: {name} must be rank 4 (B,H,S,D), got {list(t.shape)}")

    q_shape = list(query.shape)
    k_shape = list(key.shape)
    v_shape = list(value.shape)

    if q_shape[-1] != k_shape[-1]:
        raise ValueError(f"scaled_dot_product_attention: head_dim mismatch Q.D={q_shape[-1]} vs K.D={k_shape[-1]}")
    if k_shape != v_shape:
        raise ValueError(f"scaled_dot_product_attention: key/value shape mismatch {k_shape} vs {v_shape}")
    if q_shape[0] != k_shape[0]:
        raise ValueError(f"scaled_dot_product_attention: batch mismatch Q.B={q_shape[0]} vs K.B={k_shape[0]}")
    h_q, h_kv = q_shape[1], k_shape[1]
    if h_kv == 0 or h_q % h_kv != 0:
        raise ValueError(
            f"scaled_dot_product_attention: illegal GQA/MQA ratio H_q={h_q}, H_kv={h_kv} (need H_q % H_kv == 0)"
        )

    if is_causal and attn_mask is not None:
        raise ValueError("scaled_dot_product_attention: is_causal=True and attn_mask are mutually exclusive")

    if attn_mask is not None:
        m_shape = list(attn_mask.shape)
        if len(m_shape) != 4:
            raise ValueError(f"scaled_dot_product_attention: attn_mask must be rank 4, got {m_shape}")
        if m_shape[-2] != q_shape[-2]:
            raise ValueError(f"scaled_dot_product_attention: mask S_q={m_shape[-2]} != Q.S_q={q_shape[-2]}")
        if m_shape[-1] != k_shape[-2]:
            raise ValueError(f"scaled_dot_product_attention: mask S_kv={m_shape[-1]} != K.S_kv={k_shape[-2]}")
        if m_shape[1] not in (1, h_q):
            raise ValueError(f"scaled_dot_product_attention: mask head dim {m_shape[1]} not in {{1, {h_q}}}")

    # --- axis gating ---
    axes = _build_axes(query, key, value, attn_mask, is_causal, scale, compute_kernel_config)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: {axis}={axes[axis]!r} not in SUPPORTED {allowed}"
            )

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"scaled_dot_product_attention: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    query,
    key,
    value,
    *,
    attn_mask=None,
    is_causal=False,
    scale=None,
    compute_kernel_config=None,
    memory_config=None,
):
    validate(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
    )

    cfg = compute_kernel_config or default_compute_kernel_config()
    d = int(query.shape[-1])
    resolved_scale = float(scale) if scale is not None else 1.0 / math.sqrt(d)

    device = query.device()
    out_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(query.shape)),
        query.dtype,
        query.layout,
        device,
        out_memory_config,
    )

    program_descriptor, ordered_inputs = create_program_descriptor(
        query,
        key,
        value,
        output_tensor,
        attn_mask=attn_mask,
        scale=resolved_scale,
        compute_kernel_config=cfg,
    )

    return ttnn.generic_op(ordered_inputs + [output_tensor], program_descriptor)
