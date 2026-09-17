# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn
from ttnn.operations.sdpa_reference import (
    golden_chunked_flash_mla_prefill,
    golden_chunked_scaled_dot_product_attention,
    golden_flash_mla_prefill,
    golden_flash_multi_latent_attention_decode,
    golden_joint_scaled_dot_product_attention,
    golden_paged_flash_multi_latent_attention_decode,
    golden_paged_scaled_dot_product_attention_decode,
    golden_scaled_dot_product_attention,
    golden_sparse_sdpa,
    sdpa_decode_reference,
)

SDPAProgramConfig = ttnn._ttnn.operations.transformer.SDPAProgramConfig
PagedCacheGeometryOverride = ttnn._ttnn.operations.transformer.PagedCacheGeometryOverride
SparseKVFormat = ttnn._ttnn.operations.transformer.SparseKVFormat


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
    ttnn.transformer.scaled_dot_product_attention, golden_function=golden_scaled_dot_product_attention
)
ttnn.attach_golden_function(ttnn.transformer.scaled_dot_product_attention_decode, golden_function=sdpa_decode_reference)


ttnn.attach_golden_function(
    ttnn.transformer.paged_scaled_dot_product_attention_decode,
    golden_function=golden_paged_scaled_dot_product_attention_decode,
)


ttnn.attach_golden_function(
    ttnn.transformer.chunked_scaled_dot_product_attention,
    golden_function=golden_chunked_scaled_dot_product_attention,
)


ttnn.attach_golden_function(
    ttnn.transformer.joint_scaled_dot_product_attention,
    golden_function=golden_joint_scaled_dot_product_attention,
)


# Ring attention outputs are distributed/reordered and cannot use the dense single-device golden.


ttnn.attach_golden_function(ttnn.transformer.sparse_sdpa, golden_function=golden_sparse_sdpa)
# sparse_sdpa_msa has a distinct block-sparse K/V contract and no local golden yet.


ttnn.attach_golden_function(ttnn.transformer.flash_mla_prefill, golden_function=golden_flash_mla_prefill)
# ring_mla returns distributed output and statistics, so the single-device prefill golden is not applicable.


ttnn.attach_golden_function(
    ttnn.transformer.flash_multi_latent_attention_decode,
    golden_function=golden_flash_multi_latent_attention_decode,
)


ttnn.attach_golden_function(
    ttnn.transformer.paged_flash_multi_latent_attention_decode,
    golden_function=golden_paged_flash_multi_latent_attention_decode,
)


ttnn.attach_golden_function(
    ttnn.transformer.chunked_flash_mla_prefill,
    golden_function=golden_chunked_flash_mla_prefill,
)


def _golden_function_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    *,
    scale=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    use_qk_l2norm=False,
    output_head_major=False,
    **_,
):
    import torch

    # Token-by-token recurrent gated delta rule (reference port). Inputs are [B, T, H, D].
    # Note: the op passes (g, beta) while the recurrence below uses (beta, g).
    def l2_norm(x):
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)

    if use_qk_l2norm:
        q = l2_norm(q)
        k = l2_norm(k)

    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]
    B, H, T, K = k.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    q = q * scale

    o = torch.zeros(B, H, T, V, device=v.device, dtype=torch.float32)
    h = torch.zeros(B, H, K, V, device=v.device, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.to(torch.float32)

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        b_beta = beta[:, :, i]
        # Decay state, read, compute delta, write, then query.
        h = h * g[:, :, i].exp()[..., None, None]
        b_v = b_v - (h * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)

    final_state = h if output_final_state else None
    o = o.transpose(1, 2).contiguous()
    if output_head_major:
        o = o.transpose(1, 2).reshape(B * H, T, V)
    return o, final_state


ttnn.attach_golden_function(ttnn.transformer.chunk_gated_delta_rule, golden_function=_golden_function_gated_delta_rule)
# gated_delta_attn_seq consumes precomputed chunk-state tensors, not raw q/k/v/g/beta inputs.


__all__ = []
