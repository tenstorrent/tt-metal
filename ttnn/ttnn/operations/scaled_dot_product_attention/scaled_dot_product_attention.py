# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""Flash-Attention scaled_dot_product_attention — registry-model op file.

Computes ``softmax(Q·Kᵀ·scale + mask)·V`` with the Flash-Attention algorithm
(tiled online softmax, O(S) memory): the full S_q × S_kv score matrix is never
materialized; each (batch, head, query-tile-row) work unit streams the KV
sequence block-by-block and maintains running statistics (max m, sum l) and a
running output accumulator O.

This module is the source of truth for runtime support: it declares the four
registry declarations (INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate) inline,
exactly as eval/op_template.py prescribes. The golden suite
(eval/golden_tests/scaled_dot_product_attention/test_golden.py) imports
EXCLUSIONS / INPUT_TAGGERS / SUPPORTED from here; feature_spec.py owns TARGET +
INVALID.
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
# Each tagger maps (inputs_tuple, axes_dict) -> categorical label.
# inputs_tuple is (Q_shape, K_shape, V_shape). All shapes are (B, H, S, D).


def tag_alignment(inputs, axes):
    """Q's last two dims (S_q, D) vs the 32×32 tile grid.

    Both %32==0 -> tile_aligned; D not aligned -> w_non_aligned; else
    (D aligned, S_q not) -> h_non_aligned.
    """
    q = inputs[0]
    s_q, d = q[-2], q[-1]
    if s_q % 32 == 0 and d % 32 == 0:
        return "tile_aligned"
    if d % 32 != 0:
        return "w_non_aligned"
    return "h_non_aligned"


def tag_attention_kind(inputs, axes):
    """self when Q.S_q == K.S_kv, else cross."""
    return "self" if inputs[0][-2] == inputs[1][-2] else "cross"


def tag_kv_heads(inputs, axes):
    """mha when H_q == H_kv, mqa when H_kv == 1, gqa otherwise.

    Assumes H_q % H_kv == 0 (INPUTS never breaks that invariant).
    """
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
# 2. SUPPORTED — Phase-0 baseline
# ---------------------------------------------------------------------------

SUPPORTED = {
    # R1: float32 + bfloat8_b added. Input-side CBs follow the tensor dtype; the
    # matmul/reduce/eltwise *intermediate* CBs stay bf16 with fp32 DEST accumulation
    # (Issue #13364 — fp32 CB storage hangs this LLK). fp32 inputs unpack through
    # srcA/srcB (→ TF32) for the matmuls, which is production SDPA behavior.
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "attention_kind": ["self", "cross"],
    "kv_heads_mode": ["mha"],
    "mask_mode": ["none", "causal"],
    "scale_mode": ["auto", "explicit"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _check_structural(Q, K, V, attention_mask):
    """Structural shape contract → ValueError/RuntimeError (not a support refusal)."""
    q_shape = tuple(Q.shape)
    k_shape = tuple(K.shape)
    v_shape = tuple(V.shape)

    if not (len(q_shape) == len(k_shape) == len(v_shape) == 4):
        raise ValueError(
            f"scaled_dot_product_attention: Q/K/V must be rank-4 (B,H,S,D); "
            f"got Q={q_shape}, K={k_shape}, V={v_shape}"
        )
    if q_shape[-1] != k_shape[-1] or v_shape[-1] != k_shape[-1]:
        raise ValueError(
            f"scaled_dot_product_attention: head_dim mismatch — "
            f"Q.D={q_shape[-1]}, K.D={k_shape[-1]}, V.D={v_shape[-1]}"
        )
    if k_shape[-2] != v_shape[-2]:
        raise ValueError(f"scaled_dot_product_attention: K/V S_kv mismatch — K.S={k_shape[-2]}, V.S={v_shape[-2]}")
    if k_shape[1] != v_shape[1]:
        raise ValueError(f"scaled_dot_product_attention: K/V head mismatch — K.H={k_shape[1]}, V.H={v_shape[1]}")
    if k_shape[0] != v_shape[0]:
        raise ValueError(f"scaled_dot_product_attention: K/V batch mismatch — K.B={k_shape[0]}, V.B={v_shape[0]}")
    if q_shape[0] != k_shape[0]:
        raise ValueError(f"scaled_dot_product_attention: batch mismatch — Q.B={q_shape[0]}, K.B={k_shape[0]}")
    if k_shape[1] == 0 or q_shape[1] % k_shape[1] != 0:
        raise ValueError(
            f"scaled_dot_product_attention: H_q ({q_shape[1]}) must be a multiple of " f"H_kv ({k_shape[1]})"
        )

    if attention_mask is not None:
        m_shape = tuple(attention_mask.shape)
        if len(m_shape) != 4:
            raise ValueError(f"scaled_dot_product_attention: attention_mask must be rank-4; got {m_shape}")
        if m_shape[-2] != q_shape[-2] or m_shape[-1] != k_shape[-2]:
            raise ValueError(
                f"scaled_dot_product_attention: mask seq dims {m_shape[-2:]} must match "
                f"(S_q={q_shape[-2]}, S_kv={k_shape[-2]})"
            )
        if m_shape[0] != q_shape[0]:
            raise ValueError(f"scaled_dot_product_attention: mask batch {m_shape[0]} != Q batch {q_shape[0]}")
        if m_shape[1] not in (1, q_shape[1]):
            raise ValueError(f"scaled_dot_product_attention: mask head dim {m_shape[1]} must be 1 or H={q_shape[1]}")


def validate(Q, K, V, *, attention_mask=None, scale=None):
    """Runtime gate. Structural contract first, then SUPPORTED, then EXCLUSIONS."""
    _check_structural(Q, K, V, attention_mask)

    inputs = (tuple(Q.shape), tuple(K.shape), tuple(V.shape))

    axes = {
        "dtype": Q.dtype,
        "layout": Q.layout,
        "mask_mode": "none" if attention_mask is None else "causal",
        "scale_mode": "auto" if scale is None else "explicit",
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

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
# Public entry point
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    Q: "ttnn.Tensor",
    K: "ttnn.Tensor",
    V: "ttnn.Tensor",
    *,
    attention_mask: "ttnn.Tensor" = None,
    scale: float = None,
    memory_config: "ttnn.MemoryConfig" = None,
    compute_kernel_config: "ttnn.ComputeKernelConfig" = None,
) -> "ttnn.Tensor":
    """Flash-Attention SDPA: ``softmax(Q·Kᵀ·scale + mask)·V``.

    Args:
        Q: query  tensor (B, H,    S_q,  D), {bf16, fp32, bf8b}, TILE.
        K: key    tensor (B, H_kv, S_kv, D), same dtype as Q, TILE.
        V: value  tensor (B, H_kv, S_kv, D), same dtype as Q, TILE.
        attention_mask: optional additive mask (B, 1|H, S_q, S_kv), same dtype, TILE.
        scale: optional float; defaults to 1/sqrt(D).
        memory_config: output memory config (default DRAM interleaved).
        compute_kernel_config: optional ``ttnn.ComputeKernelConfig`` controlling
            math_fidelity / fp32_dest_acc_en / math_approx_mode / dst_full_sync_en.
            When None, the Phase-0 defaults (HiFi2, fp32_dest_acc_en=True,
            math_approx_mode=False, dst_full_sync_en=False) are used — byte-identical
            to prior behavior.

    Returns:
        Output tensor (B, H, S_q, D), same dtype as Q, TILE.
    """
    validate(Q, K, V, attention_mask=attention_mask, scale=scale)

    device = Q.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Resolve compute_kernel_config: None → Phase-0 hard-coded defaults; a user config
    # overrides per-field. init_device_compute_kernel_config validates against arch and
    # returns the arch's config type (the Wormhole/Blackhole alias is the same struct).
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        compute_kernel_config,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    )

    D = int(Q.shape[-1])
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    output_shape = list(Q.shape)  # (B, H, S_q, D) — same as Q
    # Output dtype follows the input dtype (R1) — the golden contract asserts
    # output.dtype == input dtype, and the writer drains an out-dtype CB.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        Q.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        Q,
        K,
        V,
        output_tensor,
        attention_mask=attention_mask,
        scale=float(scale),
        compute_kernel_config=compute_kernel_config,
    )

    io_tensors = [Q, K, V]
    if attention_mask is not None:
        io_tensors.append(attention_mask)
    io_tensors.append(output_tensor)  # output MUST be last

    return ttnn.generic_op(io_tensors, program_descriptor)
