# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""FlashAttention scaled_dot_product_attention — registry-model op.

Computes softmax(Q . K^T * scale + mask) . V via the FlashAttention online
softmax recurrence; the S_q x S_kv score matrix is never materialized.

Four registry declarations (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate)
plus the public entry point. INVALID is NOT declared here — it lives in
eval/golden_tests/scaled_dot_product_attention/feature_spec.py.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .scaled_dot_product_attention_program_descriptor import create_program_descriptor

TILE_DIM = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — inputs = (Q_shape, K_shape, V_shape)
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """Q's last two dims (S_q, D): w_non_aligned if D%32; else h_non_aligned if
    S_q%32; else tile_aligned."""
    q = inputs[0]
    if q[-1] % TILE_DIM != 0:
        return "w_non_aligned"
    if q[-2] % TILE_DIM != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_attention_kind(inputs, axes):
    """self if S_q == S_kv else cross."""
    return "self" if inputs[0][-2] == inputs[1][-2] else "cross"


def tag_kv_heads(inputs, axes):
    """mha if H_q == H_kv; mqa if H_kv == 1; else gqa."""
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
# 2. SUPPORTED  (Phase 0)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "fp32_dest_acc_en": [True, False],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "attention_kind": ["self", "cross"],
    "kv_heads_mode": ["mha"],
    "mask_mode": ["none", "custom"],
    "scale_mode": ["auto", "explicit"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS  (cell-shaped refusals inside cartesian(SUPPORTED))
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # float32 input (maxed precision) with 16-bit DEST accumulation is
    # legal-but-lossy and refused — a maxed input demands the maxed DEST
    # accumulator. (mirrors softmax; R1 numeric-formats contract).
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_fp32_dest_acc_en(compute_kernel_config) -> bool:
    if compute_kernel_config is None:
        return True
    return bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))


def _build_axes(query, key, value, *, is_causal, attn_mask, scale, compute_kernel_config):
    if is_causal and attn_mask is not None:
        raise ValueError("scaled_dot_product_attention: is_causal and attn_mask are mutually exclusive")

    if is_causal:
        mask_mode = "causal"
    elif attn_mask is not None:
        mask_mode = "custom"
    else:
        mask_mode = "none"

    scale_mode = "auto" if scale is None else "explicit"

    inputs = (tuple(query.shape), tuple(key.shape), tuple(value.shape))
    axes = {
        "dtype": query.dtype,
        "fp32_dest_acc_en": _resolve_fp32_dest_acc_en(compute_kernel_config),
        "layout": query.layout,
        "mask_mode": mask_mode,
        "scale_mode": scale_mode,
    }
    for name, tagger in INPUT_TAGGERS.items():
        axes[name] = tagger(inputs, axes)
    return axes


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(query, key, value, *, is_causal=False, attn_mask=None, scale=None, compute_kernel_config=None):
    # Non-axis shape-contract checks (ValueError) ---------------------------
    if len(query.shape) != 4 or len(key.shape) != 4 or len(value.shape) != 4:
        raise ValueError("scaled_dot_product_attention: Q, K, V must all be rank-4")
    if list(key.shape) != list(value.shape):
        raise ValueError("scaled_dot_product_attention: K and V must have identical shapes")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("scaled_dot_product_attention: Q and K head_dim (D) must match")
    if query.shape[0] != key.shape[0]:
        raise ValueError("scaled_dot_product_attention: Q and K batch (B) must match")
    h_q, h_kv = int(query.shape[1]), int(key.shape[1])
    if h_kv == 0 or h_q % h_kv != 0:
        raise ValueError("scaled_dot_product_attention: H_q must be a multiple of H_kv")
    if attn_mask is not None:
        B, S_q, S_kv = int(query.shape[0]), int(query.shape[-2]), int(key.shape[-2])
        m = [int(x) for x in attn_mask.shape]
        # Batch dim may be B or 1 (batch-broadcast); head dim may be H_q or 1
        # (head-broadcast). Trailing two dims must match (S_q, S_kv).
        ok = len(m) == 4 and m[0] in (1, B) and m[1] in (1, h_q) and m[2] == S_q and m[3] == S_kv
        if not ok:
            raise ValueError(
                "scaled_dot_product_attention: attn_mask shape must be "
                f"({{1 or B}},{{1 or H_q}},S_q,S_kv); got {m} "
                f"for B={B}, H_q={h_q}, S_q={S_q}, S_kv={S_kv}"
            )

    axes = _build_axes(
        query,
        key,
        value,
        is_causal=is_causal,
        attn_mask=attn_mask,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
    )

    # 1. SUPPORTED — per axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: {axis}={axes[axis]!r} not in SUPPORTED {allowed}"
            )

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
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
    compute_kernel_config=None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """FlashAttention scaled dot-product attention.

    Args:
        query: (B, H_q, S_q, D), TILE_LAYOUT, bfloat16.
        key:   (B, H_kv, S_kv, D), TILE_LAYOUT, bfloat16.
        value: (B, H_kv, S_kv, D), TILE_LAYOUT, bfloat16.
        attn_mask: optional additive mask (B,1,S_q,S_kv) or (B,H_q,S_q,S_kv).
        is_causal: native causal masking (Phase 0: unsupported).
        scale: softmax scale; None -> 1/sqrt(D).
        compute_kernel_config: exposes fp32_dest_acc_en / math_fidelity.
        memory_config: output memory config (default DRAM interleaved).

    Returns:
        Output tensor (B, H_q, S_q, D), bfloat16, TILE_LAYOUT.
    """
    validate(
        query,
        key,
        value,
        is_causal=is_causal,
        attn_mask=attn_mask,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
    )

    device = query.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    D = int(query.shape[-1])
    resolved_scale = scale if scale is not None else 1.0 / math.sqrt(D)

    fp32_dest_acc_en = _resolve_fp32_dest_acc_en(compute_kernel_config)
    if compute_kernel_config is not None:
        math_fidelity = getattr(compute_kernel_config, "math_fidelity", ttnn.MathFidelity.HiFi4)
    else:
        # R1: default to HiFi4 (+ fp32 DEST acc). SDPA chains two matmuls
        # (QK^T and PV) whose operands unpack to TF32; under HiFi2 the matmul
        # truncation dominates and long-context mask=none rows (a near-uniform
        # softmax over thousands of keys -> tiny-magnitude output) miss the
        # relative-RMS gate. HiFi4 drops rel_rms ~10x (0.146 -> 0.0145 at
        # S=2048) and is the lever that clears the Phase-0 long-context
        # `supported_fail` set; the fp32 accumulator (below) compounds it.
        math_fidelity = ttnn.MathFidelity.HiFi4

    output_shape = list(query.shape)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        query.dtype,
        ttnn.TILE_LAYOUT,
        device,
        out_mem,
    )

    program_descriptor = create_program_descriptor(
        query,
        key,
        value,
        attn_mask,
        output,
        scale=resolved_scale,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_fidelity=math_fidelity,
    )

    inputs = [query, key, value]
    if attn_mask is not None:
        inputs.append(attn_mask)
    inputs.append(output)
    return ttnn.generic_op(inputs, program_descriptor)
