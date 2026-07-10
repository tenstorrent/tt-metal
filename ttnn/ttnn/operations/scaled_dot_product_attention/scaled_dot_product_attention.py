# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Flash-Attention scaled_dot_product_attention — entry point + registry contract.

Computes `softmax(Q·Kᵀ·scale [+ mask]) · V` with the Flash-Attention algorithm
(tile over the KV sequence, online softmax with running max/sum). See
`op_design.md` for the full design.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .scaled_dot_product_attention_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Compute-kernel config default (single source of truth; imported by axes.py)
# ---------------------------------------------------------------------------
def default_compute_kernel_config() -> "ttnn.ComputeConfigDescriptor":
    """Fresh descriptor per call: HiFi4 + fp32 DEST acc + exact math.

    Phase-0 correctness requires fp32_dest_acc_en (both matmuls accumulate the
    K contraction in DEST). Resolve a caller's `None` through this.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — shape-facet projectors (single source of truth)
# ---------------------------------------------------------------------------
def tag_alignment(inputs, axes):
    """Q's (S_q, D) tile-alignment. inputs[0] = Q shape."""
    shape = inputs[0]
    S_q, D = shape[-2], shape[-1]
    if D % 32 != 0:
        return "w_non_aligned"
    if S_q % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def tag_attention_kind(inputs, axes):
    """self when Q.S == K.S else cross. inputs[1] = K shape."""
    return "self" if inputs[0][-2] == inputs[1][-2] else "cross"


def tag_kv_heads(inputs, axes):
    """mha/gqa/mqa from H_q vs H_kv (dim 1)."""
    H_q, H_kv = inputs[0][1], inputs[1][1]
    if H_q == H_kv:
        return "mha"
    if H_kv == 1:
        return "mqa"
    return "gqa"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
    "attention_kind": tag_attention_kind,
    "kv_heads_mode": tag_kv_heads,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0
# ---------------------------------------------------------------------------
SUPPORTED = {
    "dtype": [ttnn.bfloat16],
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
# float32 + fp32_dest_acc_en=False is legal-but-lossy (maxed input, non-maxed
# accumulation). Armed for when the dtype refinement lands.
EXCLUSIONS = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "verified"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------
def _shape_contract(query, key, value, attn_mask):
    """Structural checks independent of the axis rectangle (raise ValueError)."""
    q, k, v = list(query.shape), list(key.shape), list(value.shape)
    if len(q) != 4 or len(k) != 4 or len(v) != 4:
        raise ValueError("scaled_dot_product_attention: Q/K/V must be rank-4 (B,H,S,D)")
    B_q, H_q, S_q, D_q = q
    B_k, H_kv, S_kv, D_k = k
    if list(k) != list(v):
        raise ValueError(f"scaled_dot_product_attention: K shape {k} != V shape {v}")
    if D_q != D_k:
        raise ValueError(f"scaled_dot_product_attention: head_dim mismatch Q.D={D_q} K.D={D_k}")
    if B_q != B_k:
        raise ValueError(f"scaled_dot_product_attention: batch mismatch Q.B={B_q} K.B={B_k}")
    if H_kv == 0 or H_q % H_kv != 0:
        raise ValueError(f"scaled_dot_product_attention: H_q ({H_q}) must be a multiple of H_kv ({H_kv})")
    if attn_mask is not None:
        m = list(attn_mask.shape)
        if len(m) != 4 or m[0] != B_q or m[1] not in (1, H_q) or m[2] != S_q or m[3] != S_kv:
            raise ValueError(
                f"scaled_dot_product_attention: mask shape {m} must be "
                f"(B={B_q}, 1|H_q={H_q}, S_q={S_q}, S_kv={S_kv})"
            )


def validate(query, key, value, *, attn_mask=None, is_causal=False, scale=None, compute_kernel_config=None):
    # Mutually-exclusive kwargs
    if is_causal and attn_mask is not None:
        raise ValueError("scaled_dot_product_attention: is_causal and attn_mask are mutually exclusive")

    _shape_contract(query, key, value, attn_mask)

    # kwarg-derived axes
    if is_causal:
        mask_mode = "causal"
    elif attn_mask is not None:
        mask_mode = "custom"
    else:
        mask_mode = "none"
    scale_mode = "explicit" if scale is not None else "auto"
    fp32_dest_acc_en = bool(getattr(compute_kernel_config or default_compute_kernel_config(), "fp32_dest_acc_en", True))

    axes = {
        "dtype": query.dtype,
        "layout": query.layout,
        "mask_mode": mask_mode,
        "scale_mode": scale_mode,
        "fp32_dest_acc_en": fp32_dest_acc_en,
    }
    tagger_inputs = (list(query.shape), list(key.shape), list(value.shape))
    for name, tagger in INPUT_TAGGERS.items():
        axes[name] = tagger(tagger_inputs, axes)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: {axis}={axes[axis]!r} not in SUPPORTED {allowed}"
            )

    # 2. EXCLUSIONS — cell-level
    for exc in EXCLUSIONS:
        if all(axes.get(k) == val for k, val in exc.items()):
            raise ExcludedCell(f"scaled_dot_product_attention: unsupported combination (refinement candidate): {exc}")

    return axes


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
    compute_kernel_config: "ttnn.ComputeConfigDescriptor" = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    validate(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
    )

    resolved_cfg = compute_kernel_config or default_compute_kernel_config()

    D = int(query.shape[-1])
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    device = query.device()
    out_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_shape = list(query.shape)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        query.dtype,
        query.layout,
        device,
        out_memory_config,
    )

    tensors, program_descriptor = create_program_descriptor(
        query,
        key,
        value,
        output_tensor,
        attn_mask=attn_mask,
        scale=float(scale),
        compute_kernel_config=resolved_cfg,
    )

    return ttnn.generic_op(tensors, program_descriptor)
