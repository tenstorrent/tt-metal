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
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "attention_kind": ["self", "cross"],
    "kv_heads_mode": ["mha", "gqa", "mqa"],
    "mask_mode": ["none", "custom", "causal"],
    "scale_mode": ["auto", "explicit"],
    # Precision axis #2, independent of dtype: fp32 vs 16-bit DEST-register
    # accumulation (caller's compute_kernel_config.fp32_dest_acc_en; True when
    # no config is passed — the Phase-0 default HiFi2 + fp32 DEST acc). bf16 and
    # bf8b support BOTH modes (the bf8b fp16-DEST QK-matmul interm-format defect
    # was fixed in Refinement 7); fp32 + 16-bit DEST is legal-but-lossy and is
    # refused via EXCLUSIONS below (mirrors the softmax precedent).
    "fp32_dest_acc_en": [True, False],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — cells inside cartesian(SUPPORTED) refused for now
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # Causal masking generates an on-device upper-triangular bias from the
    # diagonal-block position (element (r,c) masked iff c > r) and skips the
    # whole-future KV blocks. Both assume a square score matrix (S_q == S_kv),
    # i.e. decoder self-attention. A rectangular S_q x S_kv causal mask is
    # mathematically well-defined but corresponds to no real workload and the
    # on-device generation path would be wrong for it — refuse it explicitly.
    {"mask_mode": "causal", "attention_kind": "cross"},
    # bfloat8_b on a partial last tile (Refinement 4). Block-float shares one
    # exponent across each 16x16 face; from_torch's tilization computes that
    # shared exponent from ALL values in the face INCLUDING the non-zero
    # padding, so the padded columns/rows can dominate the exponent and crush
    # the live values' mantissa. The reader's post-read zeroing cannot fix this
    # — the tile arrives already packed with the corrupted exponent. bf16/fp32
    # have no shared exponent and are zeroed cleanly; bf8b non-aligned is
    # refused for now (canonical EXCLUSION flagged in Refinement 1).
    {"dtype": ttnn.bfloat8_b, "alignment": "w_non_aligned"},
    {"dtype": ttnn.bfloat8_b, "alignment": "h_non_aligned"},
    # fp32 input + 16-bit DEST accumulation (Refinement 7). Truncating an fp32
    # tensor's running softmax stats / score accumulators through a 16-bit DEST
    # register defeats the entire point of an fp32 input — it cannot reach
    # fp32's tight golden RMS target (0.02), so it is legal-but-lossy and we
    # refuse it op-side rather than ship a guaranteed near-miss (mirrors the
    # softmax precedent). bf16 and bf8b support BOTH DEST modes.
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


# ---------------------------------------------------------------------------
# 4. validate() — runtime gate (first line of the entry point)
# ---------------------------------------------------------------------------


def validate(query, key, value, *, attn_mask=None, is_causal=False, scale=None, compute_kernel_config=None):
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
    # GQA/MQA head broadcast requires H_q to be an integer multiple of H_kv
    # (each KV head is shared by exactly H_q / H_kv query heads).
    if q_shape[1] % k_shape[1] != 0:
        raise ValueError(
            f"scaled_dot_product_attention: Q num_heads ({q_shape[1]}) must be a multiple of "
            f"K/V num_heads ({k_shape[1]}) for grouped/multi-query attention"
        )

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

    # fp32_dest_acc_en axis: True when no compute_kernel_config is passed (the
    # Phase-0 default — HiFi2 + fp32 DEST accumulation), else the caller's
    # explicit setting. Mirrors how the program descriptor resolves the config.
    fp32_dest_acc_en = True if compute_kernel_config is None else bool(compute_kernel_config.fp32_dest_acc_en)

    axes = {
        "dtype": query.dtype,
        "layout": query.layout,
        "mask_mode": mask_mode,
        "scale_mode": scale_mode,
        "fp32_dest_acc_en": fp32_dest_acc_en,
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
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """Flash Attention.

    `compute_kernel_config` is an optional `ttnn.*ComputeKernelConfig`
    (Wormhole/Blackhole) controlling `math_fidelity`, `fp32_dest_acc_en`,
    `math_approx_mode`, `dst_full_sync_en`. When omitted, defaults reproduce the
    Phase-0 behavior exactly (HiFi2 + fp32 DEST accumulation).
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

    if scale is None:
        scale = 1.0 / math.sqrt(int(query.shape[-1]))

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(query.shape)),
        query.dtype,
        query.layout,
        query.device(),
        query.memory_config(),
    )

    program_descriptor = create_program_descriptor(
        query,
        key,
        value,
        attn_mask,
        output_tensor,
        scale=scale,
        is_causal=is_causal,
        compute_kernel_config=compute_kernel_config,
    )

    io_tensors = [query, key, value]
    if attn_mask is not None:
        io_tensors.append(attn_mask)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)
