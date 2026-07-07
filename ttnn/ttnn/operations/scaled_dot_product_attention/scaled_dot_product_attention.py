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
#    Shape-derived axes projected from (Q, K, V) shapes.


def tag_attention_kind(inputs, axes):
    """Project (Q, K, V) shapes onto self vs cross attention.

    S_q == S_kv → self, S_q != S_kv → cross.
    """
    q_shape, k_shape, _ = inputs
    return "self" if q_shape[-2] == k_shape[-2] else "cross"


def tag_kv_heads(inputs, axes):
    """Project (Q, K, V) shapes onto MHA / GQA / MQA.

    H_q == H_kv → mha, H_kv == 1 → mqa, otherwise → gqa.
    Assumes H_q % H_kv == 0 (INPUTS never breaks that invariant).
    """
    q_shape, k_shape, _ = inputs
    h_q = q_shape[1]
    h_kv = k_shape[1]
    if h_q == h_kv:
        return "mha"
    elif h_kv == 1:
        return "mqa"
    else:
        return "gqa"


def tag_alignment(inputs, axes):
    """Project Q's last two dims onto tile alignment categories.

    D % 32 == 0 and S_q % 32 == 0 → tile_aligned.
    D % 32 != 0 → w_non_aligned (last-dim / head-dim not tile-aligned).
    D % 32 == 0 but S_q % 32 != 0 → h_non_aligned (seq-dim not tile-aligned).
    """
    q_shape = inputs[0]
    d = q_shape[-1]
    s_q = q_shape[-2]
    if d % 32 != 0:
        return "w_non_aligned"
    elif s_q % 32 != 0:
        return "h_non_aligned"
    else:
        return "tile_aligned"


INPUT_TAGGERS = {
    "attention_kind": tag_attention_kind,
    "kv_heads_mode": tag_kv_heads,
    "alignment": tag_alignment,
}


# 2. SUPPORTED
#    Phase 0: bf16, TILE_LAYOUT, tile_aligned only, no mask or custom mask,
#    auto or explicit scale, self/cross attention, MHA only (no GQA/MQA),
#    fp32_dest_acc_en=True only.

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "mask_mode": ["none", "custom", "causal"],
    "scale_mode": ["auto", "explicit"],
    "attention_kind": ["self", "cross"],
    "kv_heads_mode": ["mha", "gqa", "mqa"],
    "fp32_dest_acc_en": [True, False],
}


# 3. EXCLUSIONS
#    fp32 input + fp32_dest_acc_en=False: the FP32 input requires fp32 dest
#    accumulation to maintain precision through the matmul/softmax chain.
#    16-bit dest would truncate the FP32 mantissa too aggressively for
#    the online softmax accumulation to converge. Mirrors the softmax op's
#    fp32+False EXCLUSION.

EXCLUSIONS: list[dict] = [
    {"dtype": ttnn.float32, "fp32_dest_acc_en": False},
    # Causal masking requires S_q == S_kv (decoder self-attention only).
    # A rectangular S_q × S_kv upper-triangular mask is mathematically
    # well-defined but corresponds to no real workload.
    {"mask_mode": "causal", "attention_kind": "cross"},
]


# 4. validate()


def validate(query, key, value, *, attn_mask=None, is_causal=False, scale=None, compute_kernel_config=None):
    """Runtime gate — raises NotImplementedError for unsupported inputs."""
    # is_causal + attn_mask are mutually exclusive (same as Torch)
    if is_causal and attn_mask is not None:
        raise ValueError("scaled_dot_product_attention: is_causal and attn_mask are mutually exclusive.")

    # Determine mask_mode
    if is_causal:
        mask_mode = "causal"
    elif attn_mask is not None:
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

    # Determine kv_heads_mode
    h_q = q_shape[1]
    h_kv = k_shape[1]
    if h_q == h_kv:
        kv_heads_mode = "mha"
    elif h_kv == 1:
        kv_heads_mode = "mqa"
    else:
        kv_heads_mode = "gqa"

    # Determine alignment
    d = q_shape[-1]
    s_q = q_shape[-2]
    if d % 32 != 0:
        alignment = "w_non_aligned"
    elif s_q % 32 != 0:
        alignment = "h_non_aligned"
    else:
        alignment = "tile_aligned"

    # Determine fp32_dest_acc_en from compute_kernel_config
    if compute_kernel_config is not None:
        fp32_dest_acc_en = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))
    else:
        fp32_dest_acc_en = True  # Phase 0 default

    axes = {
        "dtype": query.dtype,
        "layout": query.layout,
        "alignment": alignment,
        "mask_mode": mask_mode,
        "scale_mode": scale_mode,
        "attention_kind": attention_kind,
        "kv_heads_mode": kv_heads_mode,
        "fp32_dest_acc_en": fp32_dest_acc_en,
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
    compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,
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
        compute_kernel_config: Compute kernel precision config

    Returns:
        (B, H, S_q, D) output tensor, bf16, TILE_LAYOUT
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

    # Output shape is same as Q
    output_shape = list(query.shape)

    # Compute scale if not provided
    D = int(query.shape[-1])
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()

    # Pre-flight L1 budget check: estimate per-core CB footprint and refuse
    # shapes that would OOM. This prevents the TT_THROW RuntimeError from
    # crashing the shared module-scoped device in the golden test suite.
    # OOM shapes are Refinement 6 territory (large head_dim L1 budget).
    D_t = D // 32
    tile_bytes = ttnn.tile_size(query.dtype)
    fp32_dest = getattr(compute_kernel_config, "fp32_dest_acc_en", True)
    interm_bytes = ttnn.tile_size(ttnn.float32 if fp32_dest else ttnn.bfloat16)
    scaler_bytes = ttnn.tile_size(ttnn.bfloat16)  # scaler is always bf16
    # Q/K/V/O CBs: 2 * D_t pages each (double-buffered, B_q=1)
    input_cbs = 4 * (2 * D_t * tile_bytes)
    # Intermediate CBs: scores(2), m(2), l(2), o(2*D_t), m_new(2), psum(2),
    # pv(2), pv_out(2*D_t) — all use intermediate format
    S_kv_t = int(key.shape[2]) // 32
    interm_cbs = (2 + 2 + 2 + 2 * D_t + 2 + 2 + 2 + 2 * D_t) * interm_bytes
    # Scaler CB: 2 * num_kv_blocks pages, always bf16
    scaler_cbs = 2 * S_kv_t * scaler_bytes
    estimated_l1 = input_cbs + interm_cbs + scaler_cbs
    L1_BUDGET = 1_400_000  # Wormhole L1 is ~1.5MB; leave room for kernel/code
    if estimated_l1 > L1_BUDGET:
        raise NotImplementedError(
            f"scaled_dot_product_attention: L1 budget exceeded "
            f"({estimated_l1} > {L1_BUDGET} bytes) for D={D}, dtype={query.dtype}. "
            f"Large head_dim + float32 dtype requires Refinement 6 (L1 budget)."
        )

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        query.dtype,
        query.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        query,
        key,
        value,
        output_tensor,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        compute_kernel_config=compute_kernel_config,
    )

    io_tensors = [query, key, value, output_tensor]
    if attn_mask is not None:
        io_tensors.insert(3, attn_mask)  # mask before output

    return ttnn.generic_op(io_tensors, program_descriptor)
