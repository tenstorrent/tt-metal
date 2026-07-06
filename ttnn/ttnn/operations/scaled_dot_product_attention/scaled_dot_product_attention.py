# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scaled Dot Product Attention (Flash Attention) — entry point.

Computes softmax(Q @ K^T * scale) @ V using the Flash Attention algorithm:
tiled, online softmax, O(S) memory. The full S_q × S_kv attention matrix
is never materialized.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from .scaled_dot_product_attention_program_descriptor import create_program_descriptor


# ---------------------------------------------------------------------------
# Registry declarations (registry model)
# ---------------------------------------------------------------------------

# 1. INPUT_TAGGERS
#    Shape-derived axis: whether the attention is self-attention or cross-attention.
#    S_q == S_kv → self, S_q != S_kv → cross.


def tag_attention_kind(inputs, axes):
    """Project (Q, K, V) shapes onto self vs cross attention."""
    q_shape, k_shape, _ = inputs
    return "self" if q_shape[-2] == k_shape[-2] else "cross"


INPUT_TAGGERS = {
    "attention_kind": tag_attention_kind,
}


# 2. SUPPORTED
#    Phase 0: bf16, TILE_LAYOUT, no mask or custom mask, auto or explicit scale.

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "mask_mode": ["none", "custom"],
    "scale_mode": ["auto", "explicit"],
    "attention_kind": ["self", "cross"],
}


# 3. EXCLUSIONS
#    Phase 0 has no exclusions inside the SUPPORTED rectangle.

EXCLUSIONS: list[dict] = []


# 4. validate()


def validate(query, key, value, *, attn_mask=None, is_causal=False, scale=None):
    """Runtime gate — raises NotImplementedError for unsupported inputs."""
    if is_causal:
        raise UnsupportedAxisValue(
            "scaled_dot_product_attention: is_causal=True is not supported in Phase 0. "
            "Use a custom additive attn_mask to emulate causal masking."
        )
    if is_causal and attn_mask is not None:
        raise ValueError("scaled_dot_product_attention: is_causal and attn_mask are mutually exclusive.")

    # Determine mask_mode
    if attn_mask is not None:
        mask_mode = "custom"
    else:
        mask_mode = "none"

    # Determine scale_mode
    if scale is not None:
        scale_mode = "explicit"
    else:
        scale_mode = "auto"

    # Determine attention_kind
    q_shape = tuple(query.shape)
    k_shape = tuple(key.shape)
    attention_kind = "self" if q_shape[-2] == k_shape[-2] else "cross"

    axes = {
        "dtype": query.dtype,
        "layout": query.layout,
        "mask_mode": mask_mode,
        "scale_mode": scale_mode,
        "attention_kind": attention_kind,
    }

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(
                f"scaled_dot_product_attention: {axis}={axes[axis]!r} not in SUPPORTED {allowed}"
            )

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"scaled_dot_product_attention: unsupported combination (refinement candidate): {exc}")


def default_compute_kernel_config():
    """Single source of truth for compute kernel precision settings."""
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
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Compute scaled dot-product attention using the Flash Attention algorithm.

    Args:
        query:  (B, H, S_q, D), bf16, TILE_LAYOUT
        key:    (B, H, S_kv, D), bf16, TILE_LAYOUT
        value:  (B, H, S_kv, D), bf16, TILE_LAYOUT
        attn_mask: Optional (B, 1, S_q, S_kv) or (B, H, S_q, S_kv) additive mask
        is_causal: Must be False in Phase 0 (use custom mask instead)
        scale: Optional scale factor (default: 1/sqrt(D))
        memory_config: Output memory config (default: DRAM interleaved)

    Returns:
        (B, H, S_q, D) output tensor, bf16, TILE_LAYOUT
    """
    validate(query, key, value, attn_mask=attn_mask, is_causal=is_causal, scale=scale)

    device = query.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output shape is same as Q
    output_shape = list(query.shape)

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        query.dtype,
        query.layout,
        device,
        output_memory_config,
    )

    # Compute scale if not provided
    D = int(query.shape[-1])
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    compute_kernel_config = default_compute_kernel_config()

    program_descriptor = create_program_descriptor(
        query,
        key,
        value,
        output_tensor,
        attn_mask=attn_mask,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
    )

    io_tensors = [query, key, value, output_tensor]
    if attn_mask is not None:
        io_tensors.insert(3, attn_mask)  # mask before output

    return ttnn.generic_op(io_tensors, program_descriptor)
