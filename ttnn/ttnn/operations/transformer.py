# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn

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


def _expand_gqa(q, k, v):
    """Repeat KV heads to match Q heads for grouped-query attention."""
    if k.shape[1] != q.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return k, v


def _make_sdpa_mask(q_len, k_len, *, is_causal, sliding_window_size=None, query_offset=0):
    import torch

    query_positions = torch.arange(q_len).view(-1, 1) + query_offset
    key_positions = torch.arange(k_len).view(1, -1)
    allowed = torch.ones((q_len, k_len), dtype=torch.bool)
    if is_causal:
        allowed &= key_positions <= query_positions
    if sliding_window_size:
        window_size = int(sliding_window_size)
        if is_causal:
            allowed &= key_positions > query_positions - window_size
        else:
            half_window = window_size // 2
            allowed &= key_positions >= query_positions - half_window
            allowed &= key_positions <= query_positions + half_window
    return torch.where(allowed, 0.0, float("-inf"))


def _sdpa_reference(
    q, k, v, *, attn_mask=None, is_causal=False, scale=None, sliding_window_size=None, attention_sink=None
):
    """Shared torch SDPA reference handling GQA, causal/sliding-window masks, scale, and attention sink."""
    import torch

    k, v = _expand_gqa(q, k, v)

    # Torch SDPA has no sliding-window flag, so materialize that mask explicitly.
    if sliding_window_size and attn_mask is None:
        q_len, k_len = q.shape[-2], k.shape[-2]
        attn_mask = _make_sdpa_mask(
            q_len,
            k_len,
            is_causal=is_causal,
            sliding_window_size=sliding_window_size,
            query_offset=k_len - q_len if is_causal else 0,
        )
        is_causal = False
        sliding_window_size = None

    if attention_sink is None:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
        )

    # Attention sink: an extra per-head logit that contributes to the softmax denominator only.
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if is_causal:
        causal_mask = _make_sdpa_mask(
            q.shape[-2],
            k.shape[-2],
            is_causal=True,
            query_offset=k.shape[-2] - q.shape[-2],
        )
        attn_mask = causal_mask if attn_mask is None else attn_mask + causal_mask
    if attn_mask is not None:
        logits = logits + attn_mask
    sink = attention_sink.float().reshape(1, logits.shape[1], 1, 1) * scale
    sink = sink.expand(logits.shape[0], logits.shape[1], logits.shape[-2], 1)
    full_logits = torch.cat([logits, sink], dim=-1)
    probs = torch.softmax(full_logits, dim=-1)[..., :-1]
    return torch.matmul(probs, v.float()).to(q.dtype)


def _make_windowed_sdpa_mask(q_len, k_len, cu_window_seqlens, query_offset):
    import torch

    boundaries = [int(value) for value in cu_window_seqlens.reshape(-1)]
    mask = torch.full((q_len, k_len), float("-inf"))
    for window_start, window_end in zip(boundaries, boundaries[1:]):
        local_start = max(window_start - query_offset, 0)
        local_end = min(window_end - query_offset, q_len)
        if local_start < local_end:
            mask[local_start:local_end, window_start:window_end] = 0.0
    return mask


def _golden_function_sdpa(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    *,
    attn_mask=None,
    is_causal=True,
    scale=None,
    sliding_window_size=None,
    attention_sink=None,
    cu_window_seqlens=None,
    windowed_q_token_offset=0,
    windowed_q_token_offset_tensor=None,
    **_,
):
    if cu_window_seqlens is not None:
        if windowed_q_token_offset_tensor is not None:
            windowed_q_token_offset = int(windowed_q_token_offset_tensor.reshape(-1)[0])
        attn_mask = _make_windowed_sdpa_mask(
            input_tensor_q.shape[-2],
            input_tensor_k.shape[-2],
            cu_window_seqlens,
            int(windowed_q_token_offset),
        )
        is_causal = False

    return _sdpa_reference(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        sliding_window_size=sliding_window_size,
        attention_sink=attention_sink,
    )


ttnn.attach_golden_function(ttnn.transformer.scaled_dot_product_attention, golden_function=_golden_function_sdpa)


def _resolve_cur_pos(cur_pos, cur_pos_tensor, batch_size, cache_len):
    if cur_pos_tensor is not None:
        return [int(x) for x in cur_pos_tensor.flatten().tolist()[:batch_size]]
    if cur_pos:
        return [int(x) for x in cur_pos]
    return [cache_len - 1] * batch_size


def _sdpa_decode_reference(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    *,
    is_causal=True,
    attn_mask=None,
    cur_pos=None,
    cur_pos_tensor=None,
    attention_sink=None,
    scale=None,
    sliding_window_size=None,
):
    import torch

    # Decode Q is [1, batch, heads, dim], while torch SDPA expects [batch, heads, 1, dim].
    query = input_tensor_q.permute(1, 2, 0, 3)
    batch_size = query.shape[0]
    k_len = input_tensor_k.shape[-2]
    positions = _resolve_cur_pos(cur_pos, cur_pos_tensor, batch_size, k_len)
    position_mask = torch.zeros((batch_size, 1, 1, k_len))
    for batch_index, position in enumerate(positions):
        if is_causal:
            position_mask[batch_index, :, :, position + 1 :] = float("-inf")
        if sliding_window_size:
            window_start = max(0, position - int(sliding_window_size) + 1)
            position_mask[batch_index, :, :, :window_start] = float("-inf")
    if attn_mask is not None:
        position_mask = position_mask + attn_mask

    output = _sdpa_reference(
        query,
        input_tensor_k,
        input_tensor_v,
        attn_mask=position_mask,
        is_causal=False,
        scale=scale,
        attention_sink=attention_sink,
    )
    return output.permute(2, 0, 1, 3)


def _golden_function_sdpa_decode(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    *,
    is_causal=True,
    attn_mask=None,
    cur_pos=None,
    cur_pos_tensor=None,
    attention_sink=None,
    scale=None,
    sliding_window_size=None,
    **_,
):
    return _sdpa_decode_reference(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        is_causal=is_causal,
        attn_mask=attn_mask,
        cur_pos=cur_pos,
        cur_pos_tensor=cur_pos_tensor,
        attention_sink=attention_sink,
        scale=scale,
        sliding_window_size=sliding_window_size,
    )


ttnn.attach_golden_function(
    ttnn.transformer.scaled_dot_product_attention_decode, golden_function=_golden_function_sdpa_decode
)


def _gather_paged_kv(cache, page_table):
    """Convert [blocks, kv_heads, block_size, dim] pages to [batch, kv_heads, sequence, dim]."""
    pages = cache[page_table.long()]
    batch_size, num_pages, num_heads, block_size, head_dim = pages.shape
    return pages.permute(0, 2, 1, 3, 4).reshape(batch_size, num_heads, num_pages * block_size, head_dim)


def _golden_function_paged_sdpa_decode(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    page_table_tensor,
    *,
    is_causal=True,
    attn_mask=None,
    cur_pos_tensor=None,
    attention_sink=None,
    scale=None,
    sliding_window_size=None,
    paged_cache_geometry=None,
    cache_position_modulo=None,
    **_,
):
    if paged_cache_geometry is not None or cache_position_modulo is not None:
        raise NotImplementedError("paged SDPA golden does not support cache geometry overrides or circular caches")

    return _sdpa_decode_reference(
        input_tensor_q,
        _gather_paged_kv(input_tensor_k, page_table_tensor),
        _gather_paged_kv(input_tensor_v, page_table_tensor),
        is_causal=is_causal,
        attn_mask=attn_mask,
        cur_pos_tensor=cur_pos_tensor,
        attention_sink=attention_sink,
        scale=scale,
        sliding_window_size=sliding_window_size,
    )


ttnn.attach_golden_function(
    ttnn.transformer.paged_scaled_dot_product_attention_decode, golden_function=_golden_function_paged_sdpa_decode
)


def _golden_function_chunked_sdpa(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    page_table_tensor,
    chunk_start_idx=None,
    *,
    chunk_start_idx_tensor=None,
    scale=None,
    paged_cache_geometry=None,
    **_,
):
    if paged_cache_geometry is not None:
        raise NotImplementedError("chunked SDPA golden does not support cache geometry overrides")
    if chunk_start_idx_tensor is not None:
        chunk_start_idx = int(chunk_start_idx_tensor.reshape(-1)[0])
    if chunk_start_idx is None:
        raise ValueError("chunk_start_idx or chunk_start_idx_tensor is required")

    key = _gather_paged_kv(input_tensor_k, page_table_tensor)
    value = _gather_paged_kv(input_tensor_v, page_table_tensor)
    prefix_end = int(chunk_start_idx) + input_tensor_q.shape[-2]
    key = key[..., :prefix_end, :]
    value = value[..., :prefix_end, :]
    causal_mask = _make_sdpa_mask(
        input_tensor_q.shape[-2],
        prefix_end,
        is_causal=True,
        query_offset=int(chunk_start_idx),
    )
    return _sdpa_reference(
        input_tensor_q,
        key,
        value,
        attn_mask=causal_mask,
        is_causal=False,
        scale=scale,
    )


ttnn.attach_golden_function(
    ttnn.transformer.chunked_scaled_dot_product_attention, golden_function=_golden_function_chunked_sdpa
)


def _golden_function_joint_sdpa(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    joint_tensor_q,
    joint_tensor_k,
    joint_tensor_v,
    *,
    joint_strategy="rear",
    scale=None,
    **_,
):
    import torch

    if joint_strategy != "rear":
        raise ValueError("joint_scaled_dot_product_attention only supports joint_strategy='rear'")

    # Joint attention concatenates the original and joint sequences along the sequence dim, runs SDPA once,
    # then splits the output back into the original and joint parts.
    q = torch.cat([input_tensor_q, joint_tensor_q], dim=-2)
    k = torch.cat([input_tensor_k, joint_tensor_k], dim=-2)
    v = torch.cat([input_tensor_v, joint_tensor_v], dim=-2)
    output = _sdpa_reference(q, k, v, is_causal=False, scale=scale)
    n = input_tensor_q.shape[-2]
    return output[..., :n, :], output[..., n:, :]


ttnn.attach_golden_function(
    ttnn.transformer.joint_scaled_dot_product_attention, golden_function=_golden_function_joint_sdpa
)


# Ring attention outputs are distributed/reordered and cannot use the dense single-device golden.


def _golden_function_sparse_sdpa(
    q,
    kv,
    indices,
    v_dim,
    *,
    kv_format=None,
    scale=None,
    cache_batch_idx=None,
    block_cyclic_sp_axis=None,
    block_cyclic_chunk_local=None,
    block_cyclic_cache_tp_sharded=False,
    attention_sink=None,
    **_,
):
    import torch

    if block_cyclic_sp_axis is not None or block_cyclic_chunk_local is not None or block_cyclic_cache_tp_sharded:
        raise NotImplementedError("sparse SDPA golden does not support block-cyclic cache layouts")

    batch_size, num_heads, sequence_length, key_dim = q.shape
    topk = indices.shape[-1]
    cache_batch_idx = 0 if cache_batch_idx is None else int(cache_batch_idx)
    cache = kv[cache_batch_idx, 0]
    if scale is None:
        scale = key_dim**-0.5

    selected_indices = indices.reshape(batch_size, sequence_length, topk)
    masked = selected_indices == 0xFFFFFFFF
    safe_indices = torch.where(masked, torch.zeros_like(selected_indices), selected_indices).long()
    expanded_cache = cache.unsqueeze(0).expand(batch_size, cache.shape[0], key_dim)
    selected_kv = torch.gather(
        expanded_cache.unsqueeze(1).expand(batch_size, sequence_length, cache.shape[0], key_dim),
        2,
        safe_indices.unsqueeze(-1).expand(batch_size, sequence_length, topk, key_dim),
    )
    logits = torch.einsum("bhsd,bsjd->bhsj", q.float(), selected_kv.float()) * scale
    logits = logits.masked_fill(masked.view(batch_size, 1, sequence_length, topk), float("-inf"))
    if attention_sink is not None:
        sink = attention_sink.float().reshape(1, num_heads, 1, 1)
        logits = torch.cat([logits, sink.expand(batch_size, num_heads, sequence_length, 1) * scale], dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    if attention_sink is not None:
        probabilities = probabilities[..., :-1]
    return torch.einsum("bhsj,bsjd->bhsd", probabilities, selected_kv[..., :v_dim]).to(q.dtype)


ttnn.attach_golden_function(ttnn.transformer.sparse_sdpa, golden_function=_golden_function_sparse_sdpa)
# sparse_sdpa_msa has a distinct block-sparse K/V contract and no local golden yet.


def _golden_function_flash_mla_prefill(
    input_tensor_q,
    input_tensor_k,
    head_dim_v=None,
    *,
    input_tensor_v=None,
    attn_mask=None,
    is_causal=True,
    scale=None,
    **_,
):
    import torch

    if input_tensor_v is not None:
        value = input_tensor_v
    elif torch.is_tensor(head_dim_v):
        value = head_dim_v
    else:
        if head_dim_v is None:
            raise ValueError("head_dim_v or input_tensor_v is required")
        value = input_tensor_k[..., : int(head_dim_v)]
    return _sdpa_reference(input_tensor_q, input_tensor_k, value, attn_mask=attn_mask, is_causal=is_causal, scale=scale)


ttnn.attach_golden_function(ttnn.transformer.flash_mla_prefill, golden_function=_golden_function_flash_mla_prefill)
# ring_mla returns distributed output and statistics, so the single-device prefill golden is not applicable.


def _golden_function_flash_mla_decode(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v=None,
    head_dim_v=None,
    *,
    is_causal=True,
    attn_mask=None,
    cur_pos=None,
    cur_pos_tensor=None,
    attention_sink=None,
    scale=None,
    sliding_window_size=None,
    **_,
):
    if head_dim_v is None:
        raise ValueError("head_dim_v is required")
    value = input_tensor_v if input_tensor_v is not None else input_tensor_k[..., :head_dim_v]
    return _sdpa_decode_reference(
        input_tensor_q,
        input_tensor_k,
        value,
        is_causal=is_causal,
        attn_mask=attn_mask,
        cur_pos=cur_pos,
        cur_pos_tensor=cur_pos_tensor,
        attention_sink=attention_sink,
        scale=scale,
        sliding_window_size=sliding_window_size,
    )


ttnn.attach_golden_function(
    ttnn.transformer.flash_multi_latent_attention_decode, golden_function=_golden_function_flash_mla_decode
)


def _golden_function_paged_flash_mla_decode(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v=None,
    head_dim_v=None,
    page_table_tensor=None,
    *,
    is_causal=True,
    attn_mask=None,
    cur_pos_tensor=None,
    attention_sink=None,
    scale=None,
    sliding_window_size=None,
    **_,
):
    if head_dim_v is None or page_table_tensor is None:
        raise ValueError("head_dim_v and page_table_tensor are required")
    key = _gather_paged_kv(input_tensor_k, page_table_tensor)
    value = _gather_paged_kv(input_tensor_v, page_table_tensor) if input_tensor_v is not None else key[..., :head_dim_v]
    return _sdpa_decode_reference(
        input_tensor_q,
        key,
        value,
        is_causal=is_causal,
        attn_mask=attn_mask,
        cur_pos_tensor=cur_pos_tensor,
        attention_sink=attention_sink,
        scale=scale,
        sliding_window_size=sliding_window_size,
    )


ttnn.attach_golden_function(
    ttnn.transformer.paged_flash_multi_latent_attention_decode, golden_function=_golden_function_paged_flash_mla_decode
)


def _golden_function_chunked_flash_mla_prefill(
    input_tensor_q, input_tensor_k, head_dim_v, page_table_tensor, chunk_start_idx, *, scale=None, **_
):
    key = _gather_paged_kv(input_tensor_k, page_table_tensor)
    prefix_end = int(chunk_start_idx) + input_tensor_q.shape[-2]
    key = key[..., :prefix_end, :]
    value = key[..., :head_dim_v]
    causal_mask = _make_sdpa_mask(
        input_tensor_q.shape[-2],
        prefix_end,
        is_causal=True,
        query_offset=int(chunk_start_idx),
    )
    return _sdpa_reference(
        input_tensor_q,
        key,
        value,
        attn_mask=causal_mask,
        is_causal=False,
        scale=scale,
    )


ttnn.attach_golden_function(
    ttnn.transformer.chunked_flash_mla_prefill, golden_function=_golden_function_chunked_flash_mla_prefill
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
