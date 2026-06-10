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
# inputs = (q_shape, k_shape, v_shape). Axis names match feature_spec.TARGET
# (alignment / attention_kind / kv_heads_mode) so the golden cartesian derives
# them from shape instead of iterating them as free axes.


def tag_alignment(inputs, axes):
    """Examines Q's last two dims (S_q, D); W takes priority when both off."""
    q, _k, _v = inputs
    if q[-1] % TILE != 0:
        return "w_non_aligned"
    if q[-2] % TILE != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_attention_kind(inputs, axes):
    """self when S_q == S_kv, cross otherwise."""
    q, k, _v = inputs
    return "self" if q[-2] == k[-2] else "cross"


def tag_kv_heads(inputs, axes):
    """mha when H_q == H_kv, mqa when H_kv == 1, gqa otherwise."""
    q, k, _v = inputs
    if q[1] == k[1]:
        return "mha"
    if k[1] == 1:
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

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
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

    # mask_mode: any additive mask (causal included) takes the same kernel
    # path, so presence maps to the "causal" axis value.
    axes = {
        "dtype": q.dtype,
        "layout": q.layout,
        "mask_mode": "none" if attention_mask is None else "causal",
        "scale_mode": "auto" if scale is None else "explicit",
    }
    inputs = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    # The alignment tagger examines Q only (feature-spec contract); S_kv
    # alignment is gated here for external callers.
    if k.shape[-2] % TILE != 0:
        raise NotImplementedError(f"sdpa: S_kv ({k.shape[-2]}) must be tile-aligned")

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


def _default_compute_kernel_config(dtype):
    """Defaults preserve Phase 0 behavior exactly for bfloat16 (HiFi2 +
    fp32 DEST acc, no approx, half-sync). float32 inputs default to HiFi4 +
    fp32 DEST — per matmul_block_helpers.hpp, the only correct pairing for
    fp32 matmul inputs (no prior behavior existed for fp32, so no drift).
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4 if dtype == ttnn.float32 else ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
        dst_full_sync_en=False,
    )


def scaled_dot_product_attention(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    *,
    attention_mask: ttnn.Tensor = None,
    scale: float = None,
    memory_config: ttnn.MemoryConfig = None,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """Flash-Attention SDPA. Output shape = query shape."""
    validate(query, key, value, attention_mask=attention_mask, scale=scale)

    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config(query.dtype)

    # Documented known-bad on Wormhole B0 (issue #38306, matmul_block_helpers.hpp):
    # HiFi4 + fp32 DEST with bf16-family inputs silently corrupts the K-accumulator.
    if (
        compute_kernel_config.math_fidelity == ttnn.MathFidelity.HiFi4
        and compute_kernel_config.fp32_dest_acc_en
        and query.dtype != ttnn.float32
    ):
        raise ValueError(
            "sdpa: HiFi4 + fp32_dest_acc_en with bf16/bf8b inputs is known-bad on "
            "Wormhole B0 (issue #38306) — use HiFi2/HiFi3, or float32 inputs"
        )

    # fp32 / bf8b inputs through a 16-bit DEST corrupt values (probe_008: fp32
    # pcc 0.008, bf8b NaN — the unpack/pack format pairing is structurally
    # unsupported). bfloat16 inputs work fine without fp32 DEST.
    if not compute_kernel_config.fp32_dest_acc_en and query.dtype != ttnn.bfloat16:
        raise NotImplementedError(f"sdpa: fp32_dest_acc_en=False requires bfloat16 inputs (got {query.dtype})")

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

    program_descriptor = create_program_descriptor(
        query,
        key,
        value,
        attention_mask,
        output_tensor,
        scale=float(scale),
        compute_kernel_config=compute_kernel_config,
    )

    io_tensors = [query, key, value]
    if attention_mask is not None:
        io_tensors.append(attention_mask)
    io_tensors.append(output_tensor)  # output last
    return ttnn.generic_op(io_tensors, program_descriptor)
