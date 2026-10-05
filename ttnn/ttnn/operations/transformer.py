# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn
from ttnn.operations.transformer_golden import (
    chunk_gated_delta_rule_golden,
    chunked_flash_mla_prefill_golden,
    chunked_scaled_dot_product_attention_golden,
    exp_ring_joint_scaled_dot_product_attention_golden,
    flash_mla_prefill_golden,
    flash_multi_latent_attention_decode_golden,
    gated_delta_attn_seq_golden,
    joint_scaled_dot_product_attention_golden,
    paged_flash_multi_latent_attention_decode_golden,
    paged_scaled_dot_product_attention_decode_golden,
    ring_distributed_scaled_dot_product_attention_golden,
    ring_joint_scaled_dot_product_attention_golden,
    ring_mla_golden,
    scaled_dot_product_attention_decode_golden,
    scaled_dot_product_attention_golden,
    sparse_sdpa_golden,
    sparse_sdpa_msa_golden,
)

SDPAProgramConfig = ttnn._ttnn.operations.transformer.SDPAProgramConfig
PagedCacheGeometryOverride = ttnn._ttnn.operations.transformer.PagedCacheGeometryOverride
SparseKVFormat = ttnn._ttnn.operations.transformer.SparseKVFormat
ChunkGdnMonoProgramConfig = ttnn._ttnn.operations.transformer.ChunkGdnMonoProgramConfig
ChunkGdnPhasedProgramConfig = ttnn._ttnn.operations.transformer.ChunkGdnPhasedProgramConfig
ChunkGdnFusedProgramConfig = ttnn._ttnn.operations.transformer.ChunkGdnFusedProgramConfig


def _golden_function(
    input_tensor: ttnn.Tensor,
    kv_input_tensor: Optional[ttnn.Tensor] = None,
    *,
    num_heads,
    num_kv_heads=None,
    transpose_key=True,
    **_,
):
    import torch

    if kv_input_tensor is not None:
        input_tensor = torch.cat([input_tensor, kv_input_tensor], dim=-1)

    if num_kv_heads is None:
        num_kv_heads = num_heads

    batch_size, sequence_size, hidden_size = input_tensor.shape
    # Subtract head sizes for key and value
    head_size = hidden_size // (num_heads + 2 * num_kv_heads)

    q_hidden = num_heads * head_size
    kv_hidden = 2 * num_kv_heads * head_size

    query_flat = input_tensor[..., :q_hidden]
    kv_flat = input_tensor[..., q_hidden : q_hidden + kv_hidden]

    # Reshape Q, K, V
    query = query_flat.reshape(batch_size, sequence_size, num_heads, head_size)
    kv = kv_flat.reshape(batch_size, sequence_size, 2 * num_kv_heads, head_size)

    key = kv[..., :num_kv_heads, :]
    value = kv[..., num_kv_heads:, :]

    # Permute to (batch, num_heads, seq_len, head_size)
    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    if transpose_key:
        key = key.permute(0, 1, 3, 2).contiguous()

    return query, key, value


ttnn.attach_golden_function(
    ttnn.transformer.split_query_key_value_and_split_heads,
    golden_function=_golden_function,
)

ttnn.attach_golden_function(
    ttnn.experimental.split_query_key_value_and_split_heads,
    golden_function=_golden_function,
)


def _golden_function(input_tensor: ttnn.Tensor, *, head_size: int, attention_mask, **_):
    import torch

    if head_size is not None:
        scaler = 1 / (head_size**0.5)
    else:
        scaler = 1.0

    input_tensor = input_tensor * scaler

    if attention_mask is not None:
        input_tensor += attention_mask

    return torch.softmax(input_tensor, -1)


ttnn.attach_golden_function(
    ttnn.transformer.attention_softmax,
    golden_function=_golden_function,
)


ttnn.attach_golden_function(
    ttnn.transformer.attention_softmax_,
    golden_function=_golden_function,
)


def _golden_function(input_tensor: ttnn.Tensor, **_):
    import torch

    batch_size, num_heads, sequence_size, head_size = input_tensor.shape

    output_tensor = torch.permute(input_tensor, (0, 2, 1, 3)).contiguous().clone()
    output_tensor = (
        torch.reshape(output_tensor, (batch_size, sequence_size, num_heads * head_size)).contiguous().clone()
    )
    return output_tensor


ttnn.attach_golden_function(ttnn.transformer.concatenate_heads, golden_function=_golden_function)

ttnn.attach_golden_function(ttnn.experimental.concatenate_heads, golden_function=_golden_function)


def _golden_function(x, cos_cached, sin_cached, token_idx, **_):
    import torch

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx=0):
        cos = cos_cached[:, :, token_idx : token_idx + 1, ...]
        sin = sin_cached[:, :, token_idx : token_idx + 1, ...]
        x_embed = (x * cos) + (rotate_half(x) * sin)
        return x_embed

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx)
    return pt_out


ttnn.attach_golden_function(ttnn.experimental.rotary_embedding, golden_function=_golden_function)

ttnn.attach_golden_function(
    ttnn.transformer.chunk_gated_delta_rule,
    golden_function=chunk_gated_delta_rule_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.chunked_flash_mla_prefill,
    golden_function=chunked_flash_mla_prefill_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.chunked_scaled_dot_product_attention,
    golden_function=chunked_scaled_dot_product_attention_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.exp_ring_joint_scaled_dot_product_attention,
    golden_function=exp_ring_joint_scaled_dot_product_attention_golden,
    output_tensor_kwarg_names=(
        "persistent_output_buffer_k",
        "persistent_output_buffer_v",
    ),
)
ttnn.attach_golden_function(
    ttnn.transformer.flash_mla_prefill,
    golden_function=flash_mla_prefill_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.flash_multi_latent_attention_decode,
    golden_function=flash_multi_latent_attention_decode_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.gated_delta_attn_seq,
    golden_function=gated_delta_attn_seq_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.joint_scaled_dot_product_attention,
    golden_function=joint_scaled_dot_product_attention_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.paged_flash_multi_latent_attention_decode,
    golden_function=paged_flash_multi_latent_attention_decode_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.paged_scaled_dot_product_attention_decode,
    golden_function=paged_scaled_dot_product_attention_decode_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.ring_distributed_scaled_dot_product_attention,
    golden_function=ring_distributed_scaled_dot_product_attention_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.ring_joint_scaled_dot_product_attention,
    golden_function=ring_joint_scaled_dot_product_attention_golden,
    output_tensor_kwarg_names=(
        "persistent_output_buffer_k",
        "persistent_output_buffer_v",
        "persistent_output_buffer_joint_k",
        "persistent_output_buffer_joint_v",
    ),
)
ttnn.attach_golden_function(
    ttnn.transformer.ring_mla,
    golden_function=ring_mla_golden,
    output_tensor_kwarg_names=("persistent_output_buffer_kv",),
)
ttnn.attach_golden_function(
    ttnn.transformer.scaled_dot_product_attention,
    golden_function=scaled_dot_product_attention_golden,
)
ttnn.attach_golden_function(
    ttnn.transformer.scaled_dot_product_attention_decode,
    golden_function=scaled_dot_product_attention_decode_golden,
)


def _preprocess_sparse_sdpa_golden_inputs(function_args, function_kwargs):
    function_args = list(function_args)
    function_kwargs = dict(function_kwargs)
    if function_kwargs.get("block_cyclic_sp_axis") is not None:
        query = function_args[0] if function_args else function_kwargs["q"]
        mesh_device = query.device()
        if mesh_device is None:
            raise ValueError("Block-cyclic sparse SDPA comparison requires the query's mesh device")
        function_kwargs["_ttnn_sparse_sdpa_mesh_shape"] = tuple(int(dimension) for dimension in mesh_device.shape)

    if function_kwargs.get("kv_format") == SparseKVFormat.SCALED_FP8:
        if len(function_args) > 1:
            packed_kv = ttnn.decorators.to_torch_for_comparison(function_args[1], preserve_fp8_bytes=True)
            function_args[1] = packed_kv
        elif "kv" in function_kwargs:
            packed_kv = ttnn.decorators.to_torch_for_comparison(function_kwargs["kv"], preserve_fp8_bytes=True)
            function_kwargs["kv"] = packed_kv
        else:
            raise ValueError("Scaled-FP8 sparse SDPA comparison requires a KV tensor")
        # Global preprocessing converts the original FP8 argument independently. The wrapper's metadata merge
        # copies this operation-scoped override into the global golden call so packed mixed-format rows stay bytes.
        function_kwargs["_ttnn_sparse_sdpa_packed_kv"] = packed_kv
    return ttnn.decorators.default_preprocess_golden_function_inputs(function_args, function_kwargs)


ttnn.attach_golden_function(
    ttnn.transformer.sparse_sdpa,
    golden_function=sparse_sdpa_golden,
    preprocess_golden_function_inputs=_preprocess_sparse_sdpa_golden_inputs,
)
ttnn.attach_golden_function(
    ttnn.transformer.sparse_sdpa_msa,
    golden_function=sparse_sdpa_msa_golden,
)


__all__ = []
