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
# Refinement 1: dtype expanded to [bfloat16, float32, bfloat8_b] and
# fp32_dest_acc_en expanded to [True, False].
# Intermediate CB formats are fp32 when fp32_dest_acc_en=True (accumulation
# crosses the CB), input dtype when False.  See /numeric-formats-metal.
# The {dtype: float32, fp32_dest_acc_en: False} cell is in EXCLUSIONS
# (maxed input + non-maxed acc is rejected — mirrors softmax convention).

SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "attention_kind": ["self", "cross"],
    "kv_heads_mode": ["mha", "gqa", "mqa"],
    "mask_mode": ["none", "custom", "causal"],
    "scale_mode": ["auto", "explicit"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# Cells inside the SUPPORTED rectangle that are refused for now:
# - {dtype: float32, fp32_dest_acc_en: False}: maxed input + non-maxed acc
#   is rejected. fp32 input through the 16-bit DEST path loses 13 mantissa
#   bits at every FPU phase (matmul, mul, add), eroding the precision that
#   float32 input was chosen to provide. Mirrors softmax convention per
#   feature_spec.py.
# - is_causal + cross-attention: causal masking requires S_q == S_kv.
#   (causal mask_mode is not yet in SUPPORTED; this exclusion is listed
#   forward-looking for when causal lands via refinement. For Phase 0 it
#   is inert since "causal" ∉ SUPPORTED["mask_mode"].)

EXCLUSIONS = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # causal + cross-attention is structurally invalid for the triangular mask.
    # Causal masking requires S_q == S_kv (decoder self-attention); cross-attn
    # has S_q != S_kv by definition, so the triangular mask is undefined.
    {"mask_mode": "causal", "attention_kind": "cross"},
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
# Padding mask for non-tile-aligned S_kv
# ---------------------------------------------------------------------------
#
# When S_kv is not tile-aligned, from_torch zero-pads K/V to the next tile
# boundary.  The zero rows in K produce score = Q @ 0 = 0 for those padded
# positions.  After the online-softmax subtract-max step, exp(0 - m) ≠ 0,
# so the padded S_kv columns contribute non-zero probability mass and
# corrupt the softmax denominator (sum_l) by ~exp(-m) per padded column.
# With ~22 padded columns (S_kv=47 → 17 padded, or S_kv=50 → 14 padded),
# the error is ~14-22% — far exceeding the 0.05 RMS tolerance.
#
# Fix: generate a (B, 1, S_q, S_kv) additive mask with -inf in the padded
# S_kv columns, combine with any user-supplied mask, and pass as attn_mask
# to the kernel.  The existing mask-add path handles it — no kernel changes.
#
# Similarly, when S_q is not tile-aligned, the padded Q rows produce output
# tiles whose valid region is stripped by to_torch on the way back.  The
# padded Q rows' scores are garbage (Q is zero-padded, so Q@K^T = 0 for
# all KV positions — uniform scores → uniform softmax → mean of V rows).
# This does NOT contaminate the valid Q rows because the row-max, row-sum,
# and PV matmul all operate independently per Q row.  No S_q padding mask
# is needed.
#
# D non-aligned is likewise safe: the padded D columns in K are zeros, so
# Q@K^T is correct.  The padded D columns in V are zeros, so P@V produces
# zeros in the padded output positions — stripped by to_torch.  No D
# padding mask needed.


def _make_padding_mask(query, key, attn_mask, *, is_causal):
    """Create or extend an additive mask that masks out padded S_kv positions.

    Returns (combined_mask, combined_mask_is_per_head) where combined_mask
    is a ttnn.Tensor or None.  When S_kv is tile-aligned, no padding mask
    is needed and the original attn_mask is returned unchanged.

    When is_causal=True: the causal mask is generated on-device and naturally
    masks out all positions where col > row. For self-attention (S_q == S_kv),
    padded KV positions always have col > row (since they're past the valid
    S_kv range), so the causal mask subsumes the padding mask. Return None
    to let the reader's causal path handle everything.
    """
    if is_causal:
        # Causal mask generated on-device naturally handles padded KV positions.
        # For self-attention, padded columns have col > row → -inf in the causal
        # pattern, which is exactly what the padding mask would add.
        return None, False
    import torch

    B = list(query.shape)[0]
    H_q = list(query.shape)[1]
    S_q = list(query.shape)[2]
    S_kv = list(key.shape)[2]
    S_kv_tiles = (S_kv + 31) // 32
    S_kv_padded = S_kv_tiles * 32

    if S_kv_padded == S_kv:
        # S_kv is tile-aligned — no padding mask needed
        return attn_mask, attn_mask is not None and list(attn_mask.shape)[1] == H_q

    device = query.device()

    # Build padding mask: (B, 1, S_q, S_kv_padded) with 0 in the valid
    # region [0, S_kv) and -inf in the padded region [S_kv, S_kv_padded).
    pad_mask = torch.zeros(B, 1, S_q, S_kv_padded, dtype=torch.bfloat16)
    pad_mask[:, :, :, S_kv:] = float("-inf")

    if attn_mask is None:
        # No user mask — just use the padding mask
        ttnn_mask = ttnn.from_torch(
            pad_mask,
            dtype=query.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn_mask, False

    # Combine with user-supplied mask on the host side.
    # The user mask may be (B, 1, S_q, S_kv) or (B, H_q, S_q, S_kv).
    user_shape = list(attn_mask.shape)
    user_is_per_head = user_shape[1] == H_q
    user_S_q = user_shape[2]
    user_S_kv = user_shape[3]
    torch_user = ttnn.to_torch(attn_mask)

    if user_is_per_head:
        # Pad user mask to S_kv_padded, then add padding mask (broadcast to H_q)
        torch_user_padded = torch.zeros(B, H_q, S_q, S_kv_padded, dtype=torch.bfloat16)
        torch_user_padded[:, :, :user_S_q, :user_S_kv] = torch_user[:, :, :user_S_q, :user_S_kv]
        combined = torch_user_padded + pad_mask.expand(B, H_q, S_q, S_kv_padded)
    else:
        # User mask is (B, 1, S_q, S_kv) — pad and combine
        torch_user_padded = torch.zeros(B, 1, S_q, S_kv_padded, dtype=torch.bfloat16)
        torch_user_padded[:, :, :user_S_q, :user_S_kv] = torch_user[:, :, :user_S_q, :user_S_kv]
        combined = torch_user_padded + pad_mask

    ttnn_combined = ttnn.from_torch(
        combined,
        dtype=query.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn_combined, user_is_per_head


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

    # For non-tile-aligned S_kv, generate a padding mask with -inf in the
    # padded S_kv columns and combine with any user-supplied mask.  This
    # prevents the zero-padded K rows from contaminating the softmax
    # denominator.  See _make_padding_mask docstring for details.
    effective_mask, effective_mask_is_per_head = _make_padding_mask(
        query,
        key,
        attn_mask,
        is_causal=is_causal,
    )

    program_descriptor = create_program_descriptor(
        query,
        key,
        value,
        output_tensor,
        attn_mask=effective_mask,
        is_causal=is_causal,
        scale=scale,
        math_fidelity=getattr(resolved_cfg, "math_fidelity", ttnn.MathFidelity.HiFi4),
        fp32_dest_acc_en=bool(getattr(resolved_cfg, "fp32_dest_acc_en", True)),
        math_approx_mode=bool(getattr(resolved_cfg, "math_approx_mode", False)),
    )

    # The mask tensor (whether user-supplied or generated by _make_padding_mask)
    # stays alive as a local variable during the synchronous generic_op call.
    # It is NOT added to the operands list — the original code never included
    # the user mask in operands either. The program descriptor references
    # mask.buffer_address() at descriptor-creation time, and the buffer stays
    # valid because the tensor is alive in this scope.
    return ttnn.generic_op([query, key, value, output_tensor], program_descriptor)
