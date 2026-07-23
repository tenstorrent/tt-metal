# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Public fused flash-attention operation."""

import math

import ttnn

from .config import (
    DEFAULT_FLASH_ATTENTION_PROGRAM_CONFIG,
    FlashAttentionProgramConfig,
    resolve_block_tiles,
    resolve_output_subblock,
)
from .program_descriptor import TILE, create_program_descriptor


def _golden_function(query, key, value, *, scale=None, **_):
    import torch

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    return torch.softmax(torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale, dim=-1).matmul(
        value.float()
    )


def _validate_dram_interleaved(name, tensor):
    memory_config = tensor.memory_config()
    if memory_config.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError(f"flash_attention: {name} must be in DRAM, got {memory_config.buffer_type}")
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(
            f"flash_attention: {name} must use INTERLEAVED memory layout, got {memory_config.memory_layout}"
        )


def validate_flash_attention_inputs(query, key, value, program_config):
    """Validate the deliberately narrow first implementation."""
    program_config.validate_basic()

    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.dtype != ttnn.bfloat16:
            raise ValueError(f"flash_attention: {name} must have bfloat16 dtype, got {tensor.dtype}")
        if tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"flash_attention: {name} must use TILE_LAYOUT")
        if len(tensor.shape) != 4:
            raise ValueError(f"flash_attention: {name} must have rank 4 [B,H,S,D], got shape {list(tensor.shape)}")
        _validate_dram_interleaved(name, tensor)

    q_shape, k_shape, v_shape = list(query.shape), list(key.shape), list(value.shape)
    if k_shape != v_shape:
        raise ValueError(f"flash_attention: key and value shapes must match, got K={k_shape}, V={v_shape}")
    if q_shape[:2] != k_shape[:2]:
        raise ValueError(
            f"flash_attention: query/key batch and head dimensions must match, got Q={q_shape[:2]}, K={k_shape[:2]}"
        )
    if q_shape[-1] != k_shape[-1]:
        raise ValueError(
            f"flash_attention: query/key head dimensions must match, got Q D={q_shape[-1]}, K D={k_shape[-1]}"
        )
    for name, shape in (("query", q_shape), ("key", k_shape), ("value", v_shape)):
        if any(dim < 1 for dim in shape):
            raise ValueError(f"flash_attention: {name} dimensions must be positive, got {shape}")

    q_seq, kv_seq, head_dim = q_shape[-2], k_shape[-2], q_shape[-1]
    if q_seq % TILE or kv_seq % TILE or head_dim % TILE:
        raise ValueError(
            "flash_attention: query sequence, key/value sequence, and head dimension must all be multiples of 32; "
            f"got Q S={q_seq}, KV S={kv_seq}, D={head_dim}"
        )

    q_seq_t, kv_seq_t, d_t = q_seq // TILE, kv_seq // TILE, head_dim // TILE
    qb = resolve_block_tiles(q_seq_t, program_config.query_block_tiles, 4)
    kb = resolve_block_tiles(kv_seq_t, program_config.key_block_tiles, 16)
    if q_seq_t % qb:
        raise ValueError(f"flash_attention: Q sequence has {q_seq_t} tiles, not divisible by query_block_tiles={qb}")
    if kv_seq_t % kb:
        raise ValueError(f"flash_attention: KV sequence has {kv_seq_t} tiles, not divisible by key_block_tiles={kb}")

    qk_h, qk_w = resolve_output_subblock(
        qb,
        kb,
        program_config.qk_output_subblock,
        program_config.dest_tile_capacity,
        prefer_wide=False,
    )
    pv_h, pv_w = resolve_output_subblock(qb, d_t, program_config.pv_output_subblock, program_config.dest_tile_capacity)
    if qb % qk_h or kb % qk_w:
        raise ValueError(f"flash_attention: qk_output_subblock={(qk_h, qk_w)} must divide QK block {(qb, kb)} tiles")
    if qb % pv_h or d_t % pv_w:
        raise ValueError(
            f"flash_attention: pv_output_subblock={(pv_h, pv_w)} must divide PV output block {(qb, d_t)} tiles"
        )


@ttnn.register_python_operation(name="ttnn.flash_attention", golden_function=_golden_function)
def flash_attention(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    *,
    scale: float | None = None,
    program_config: FlashAttentionProgramConfig | None = None,
) -> ttnn.Tensor:
    """Compute non-causal ``softmax(scale * Q @ Kᵀ) @ V`` as one device program.

    Inputs and output are rank-4 ``[batch, heads, sequence, head_dim]`` BF16 tile
    tensors in interleaved DRAM. Sequence and head dimensions must be tile
    aligned. The operation never materializes the score matrix in DRAM: each
    core keeps BF16 online-softmax state and rescale factors in L1 while
    streaming K/V blocks; reciprocal intermediates remain FP32. Horizontal per-head groups
    multicast each K/V block by default. The program configuration independently
    controls DEST accumulation and the probability/rescale exponent
    implementations.

    This first version intentionally has no causal mask, additive mask, dropout,
    GQA, or non-BF16 format path.

    Args:
        scale: Score scale. Defaults to ``1 / sqrt(head_dim)``.
        program_config: Performance knobs. Defaults to
            :class:`FlashAttentionProgramConfig`.
    """
    config = program_config or DEFAULT_FLASH_ATTENTION_PROGRAM_CONFIG
    if not isinstance(config, FlashAttentionProgramConfig):
        raise TypeError(
            "flash_attention: program_config must be FlashAttentionProgramConfig or None, "
            f"got {type(config).__name__}"
        )
    validate_flash_attention_inputs(query, key, value, config)

    resolved_scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else float(scale)
    if not math.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError(f"flash_attention: scale must be finite and positive, got {scale!r}")

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(query.shape)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        query.device(),
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = create_program_descriptor(query, key, value, output, scale=resolved_scale, program_config=config)
    return ttnn.generic_op([query, key, value, output], descriptor)
