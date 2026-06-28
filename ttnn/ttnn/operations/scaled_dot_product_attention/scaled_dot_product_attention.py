# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""scaled_dot_product_attention — Flash Attention (online softmax).

Computes softmax(Q @ K^T * scale + mask) @ V using tiled, online-softmax with
O(S) memory: the full S_q × S_kv score matrix is NEVER materialized. Uses the
Flash Attention v2 recurrence (running max/sum/output per Q-block).

Phase 0 baseline: bfloat16 + fp32_dest_acc_en=True (maxed-out corner).
See op_design.md for the full design and the registry-model contract.
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
# Multi-input op: inputs_tuple is (Q_shape, K_shape, V_shape). K and V share
# shape; Q may differ in seq_len (cross-attention) or num_heads (GQA/MQA).
# Taggers project shape facets the kernel cares about onto categorical
# labels. See eval/golden_tests/scaled_dot_product_attention/feature_spec.py.


def tag_alignment(inputs, axes):
    """Q last-two-dims (S_q, D) tile alignment.

    - "tile_aligned": both S_q and D divisible by 32.
    - "w_non_aligned": D (last dim) not divisible by 32.
    - "h_non_aligned": D aligned, S_q (second-to-last) not divisible by 32.
    """
    q_shape = inputs[0]
    S_q, D = q_shape[-2], q_shape[-1]
    if D % 32 != 0:
        return "w_non_aligned"
    if S_q % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_attention_kind(inputs, axes):
    """Self vs cross attention: Q seq_len vs K seq_len."""
    q_S = inputs[0][-2]
    k_S = inputs[1][-2]
    return "self" if q_S == k_S else "cross"


def tag_kv_heads_mode(inputs, axes):
    """MHA / GQA / MQA from Q vs K head counts.

    - "mha": H_q == H_kv (multi-head attention).
    - "mqa": H_kv == 1 (multi-query attention).
    - "gqa": 1 < H_kv < H_q (grouped-query attention).
    Assumes H_q % H_kv == 0 (the GQA invariant; INPUTS never breaks it).
    """
    H_q = inputs[0][1]
    H_kv = inputs[1][1]
    if H_q == H_kv:
        return "mha"
    if H_kv == 1:
        return "mqa"
    return "gqa"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "attention_kind": tag_attention_kind,
    "kv_heads_mode": tag_kv_heads_mode,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# Phase 0 baseline: bfloat16, TILE_LAYOUT, tile_aligned, fp32_dest_acc_en=True.
# mask_mode: none + custom (causal is a refinement). scale_mode: auto + explicit.
# attention_kind / kv_heads_mode: all three values are accepted by the kernel
# (the reader handles GQA/MQA head replication), so all are in SUPPORTED.

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
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
# Cells inside the SUPPORTED rectangle that are refused for now:
# - is_causal + cross-attention: causal masking requires S_q == S_kv.
#   (causal mask_mode is not yet in SUPPORTED; this exclusion is listed
#   forward-looking for when causal lands via refinement. For Phase 0 it
#   is inert since "causal" ∉ SUPPORTED["mask_mode"].)

EXCLUSIONS = [
    # causal + cross-attention is structurally invalid for the triangular mask.
    # Kept commented until causal mask_mode enters SUPPORTED via refinement.
    # {"mask_mode": "causal", "attention_kind": "cross"},
]


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _resolve_axes(query, key, value, *, attn_mask, is_causal, scale, compute_kernel_config):
    """Build the axes dict exactly as the test harness / classify_call do."""
    # --- kwarg-derived axes ---
    if is_causal:
        mask_mode = "causal"
    elif attn_mask is not None:
        mask_mode = "custom"
    else:
        mask_mode = "none"
    scale_mode = "explicit" if scale is not None else "auto"

    # Precision axis: read from the resolved config (single source of truth).
    resolved_cfg = compute_kernel_config or default_compute_kernel_config()
    fp32_dest_acc_en = bool(getattr(resolved_cfg, "fp32_dest_acc_en", True))

    axes = {
        "dtype": query.dtype,
        "layout": query.layout,
        "mask_mode": mask_mode,
        "scale_mode": scale_mode,
        "fp32_dest_acc_en": fp32_dest_acc_en,
    }
    # --- shape-derived axes via the op's own taggers ---
    tagger_inputs = (list(query.shape), list(key.shape), list(value.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(tagger_inputs, axes)
    return axes


def validate(query, key, value, *, attn_mask=None, is_causal=False, scale=None, compute_kernel_config=None):
    """Runtime gate. Raises UnsupportedAxisValue / ExcludedCell for refusals.

    Order: (1) per-axis SUPPORTED check, (2) cell-level EXCLUSIONS check.
    INVALID (structurally-impossible cells) is the test harness's concern —
    it skips those before calling the op.
    """
    # is_causal + attn_mask is mutually exclusive (ValueError, not a support refusal).
    if is_causal and attn_mask is not None:
        raise ValueError("is_causal and attn_mask are mutually exclusive")

    axes = _resolve_axes(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
    )

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: {axis}={axes[axis]!r} not in SUPPORTED {allowed}"
            )

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(
                f"scaled_dot_product_attention: unsupported combination " f"(refinement candidate): {exc}"
            )


# ---------------------------------------------------------------------------
# default_compute_kernel_config — single source of truth for None resolution
# ---------------------------------------------------------------------------


def default_compute_kernel_config():
    """Return the default compute kernel config for scaled_dot_product_attention.

    Phase 0 baseline: HiFi4, fp32_dest_acc_en=True, math_approx_mode=False.
    `None` passed as compute_kernel_config resolves through this factory.
    """
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    *,
    attn_mask: ttnn.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Flash Attention: softmax(Q @ K^T * scale + mask) @ V.

    Args:
        query: (B, H_q, S_q, D) bf16, TILE_LAYOUT.
        key:   (B, H_kv, S_kv, D) bf16, TILE_LAYOUT.
        value: (B, H_kv, S_kv, D) bf16, TILE_LAYOUT.
        attn_mask: optional (B, 1, S_q, S_kv) or (B, H_q, S_q, S_kv) additive mask.
        is_causal: if True, generate a triangular -inf mask on-device. Mutually
            exclusive with attn_mask.
        scale: if None, 1/sqrt(D).
        compute_kernel_config: ttnn.ComputeConfigDescriptor or None (→
            default_compute_kernel_config()).
        memory_config: output memory config (default: DRAM interleaved).
    """
    validate(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
    )

    device = query.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape matches Q: (B, H_q, S_q, D).
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(query.shape)),
        query.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    resolved_cfg = compute_kernel_config or default_compute_kernel_config()

    program_descriptor = create_program_descriptor(
        query,
        key,
        value,
        output_tensor,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        math_fidelity=getattr(resolved_cfg, "math_fidelity", ttnn.MathFidelity.HiFi4),
        fp32_dest_acc_en=bool(getattr(resolved_cfg, "fp32_dest_acc_en", True)),
        math_approx_mode=bool(getattr(resolved_cfg, "math_approx_mode", False)),
    )
    inputs = [query, key, value, output_tensor]
    if attn_mask is not None:
        inputs.append(attn_mask)
    return ttnn.generic_op(inputs, program_descriptor)
