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


def _repeat_kv_heads(tensor, num_query_heads):
    if tensor.shape[1] == num_query_heads:
        return tensor
    if num_query_heads % tensor.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    return tensor.repeat_interleave(num_query_heads // tensor.shape[1], dim=1)


def _attention_reference(
    query,
    key,
    value,
    *,
    scale=None,
    attn_mask=None,
    is_causal=False,
    sliding_window_size=None,
    query_start=0,
    attention_sink=None,
    cu_window_seqlens=None,
):
    import torch

    query = query.float()
    key = _repeat_kv_heads(key.float(), query.shape[1])
    value = _repeat_kv_heads(value.float(), query.shape[1])
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    q_positions = query_start + torch.arange(query.shape[-2], device=query.device)
    k_positions = torch.arange(key.shape[-2], device=query.device)
    allowed = torch.ones((query.shape[-2], key.shape[-2]), dtype=torch.bool, device=query.device)
    if is_causal:
        allowed &= k_positions[None, :] <= q_positions[:, None]
    if sliding_window_size is not None:
        if is_causal:
            allowed &= k_positions[None, :] > q_positions[:, None] - sliding_window_size
        else:
            left_window = (sliding_window_size - 1) // 2
            right_window = sliding_window_size // 2
            allowed &= k_positions[None, :] >= q_positions[:, None] - left_window
            allowed &= k_positions[None, :] <= q_positions[:, None] + right_window

    if cu_window_seqlens is not None:
        boundaries = cu_window_seqlens.reshape(-1).to(torch.long)
        q_windows = torch.bucketize(q_positions, boundaries[1:], right=True)
        k_windows = torch.bucketize(k_positions, boundaries[1:], right=True)
        allowed &= q_windows[:, None] == k_windows[None, :]

    scores = scores.masked_fill(~allowed, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.float()

    if attention_sink is not None:
        sink = attention_sink.float().expand(*scores.shape[:-1], 1)
        probabilities = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
    else:
        probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value)


def _golden_function_scaled_dot_product_attention(
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
    **_,
):
    return _attention_reference(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        scale=scale,
        attn_mask=attn_mask,
        is_causal=is_causal,
        sliding_window_size=sliding_window_size,
        attention_sink=attention_sink,
        cu_window_seqlens=cu_window_seqlens,
    )


ttnn.attach_golden_function(
    ttnn.transformer.scaled_dot_product_attention,
    golden_function=_golden_function_scaled_dot_product_attention,
)


def _decode_attention_reference(
    query,
    key,
    value,
    positions,
    *,
    is_causal=True,
    scale=None,
    attn_mask=None,
    sliding_window_size=None,
    attention_sink=None,
    share_cache=False,
):
    import torch

    batch = query.shape[1]
    output = torch.zeros(1, batch, query.shape[2], value.shape[-1], dtype=torch.float32, device=query.device)
    if not is_causal:
        positions = [key.shape[-2] - 1] * batch
    elif positions is None or (hasattr(positions, "__len__") and len(positions) == 0):
        positions = [key.shape[-2] - 1] * batch
    if isinstance(positions, torch.Tensor):
        positions = positions.reshape(-1).tolist()

    for batch_index, position in enumerate(positions):
        position = int(position)
        if position < 0:
            continue
        cache_batch = 0 if share_cache else batch_index
        start = 0 if sliding_window_size is None else max(0, position + 1 - sliding_window_size)
        key_slice = key[cache_batch : cache_batch + 1, :, start : position + 1]
        value_slice = value[cache_batch : cache_batch + 1, :, start : position + 1]
        mask_slice = None
        if attn_mask is not None:
            mask_slice = attn_mask[batch_index : batch_index + 1, ..., start : position + 1]
            # Decode masks use device layout [batch, query, heads, keys], whereas the
            # Torch attention expression uses [batch, heads, query, keys].
            if mask_slice.ndim == 4 and mask_slice.shape[1] == 1:
                mask_slice = mask_slice.transpose(1, 2)
        sink_slice = attention_sink
        if attention_sink is not None and attention_sink.shape[0] == batch:
            sink_slice = attention_sink[batch_index : batch_index + 1]
        batch_output = _attention_reference(
            query[:, batch_index].unsqueeze(-2),
            key_slice,
            value_slice,
            scale=scale,
            attn_mask=mask_slice,
            attention_sink=sink_slice,
        )
        output[0, batch_index] = batch_output[0, :, 0]
    return output


def _golden_function_scaled_dot_product_attention_decode(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    *,
    is_causal=True,
    cur_pos=(),
    cur_pos_tensor=None,
    attn_mask=None,
    scale=None,
    sliding_window_size=None,
    attention_sink=None,
    share_cache=False,
    **_,
):
    positions = cur_pos_tensor if cur_pos_tensor is not None else cur_pos
    return _decode_attention_reference(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        positions,
        is_causal=is_causal,
        scale=scale,
        attn_mask=attn_mask,
        sliding_window_size=sliding_window_size,
        attention_sink=attention_sink,
        share_cache=bool(share_cache),
    )


ttnn.attach_golden_function(
    ttnn.transformer.scaled_dot_product_attention_decode,
    golden_function=_golden_function_scaled_dot_product_attention_decode,
)


def _reinterpret_paged_cache(cache, paged_cache_geometry):
    if paged_cache_geometry is None or not paged_cache_geometry.active():
        return cache
    block_size = int(paged_cache_geometry.block_size)
    num_kv_heads = int(paged_cache_geometry.num_kv_heads)
    if block_size <= 0 or num_kv_heads <= 0:
        raise ValueError("active paged cache geometry requires positive block_size and num_kv_heads")
    if block_size == cache.shape[2] and num_kv_heads == cache.shape[1]:
        return cache
    tile = 32
    num_blocks, allocated_kv_heads, allocated_block_size, allocated_head_dim = cache.shape
    if allocated_block_size % tile or allocated_head_dim % tile or block_size % tile:
        raise ValueError("paged cache geometry must be tile aligned")
    allocated_block_tiles = allocated_block_size // tile
    allocated_width_tiles = allocated_head_dim // tile
    view_block_tiles = block_size // tile
    allocated_tiles = allocated_kv_heads * allocated_block_tiles * allocated_width_tiles
    if allocated_tiles % (num_kv_heads * view_block_tiles):
        raise ValueError("paged cache geometry must preserve the per-block element count")
    view_width_tiles = allocated_tiles // (num_kv_heads * view_block_tiles)

    # Preserve the linear tile order of the physical cache while changing its logical
    # (KV heads, block rows, head width) view.
    cache = cache.reshape(num_blocks, allocated_kv_heads, allocated_block_tiles, tile, allocated_width_tiles, tile)
    cache = cache.permute(0, 1, 2, 4, 3, 5).contiguous()
    cache = cache.reshape(num_blocks, allocated_tiles, tile, tile)
    cache = cache.reshape(num_blocks, num_kv_heads, view_block_tiles, view_width_tiles, tile, tile)
    cache = cache.permute(0, 1, 2, 4, 3, 5).contiguous()
    return cache.reshape(num_blocks, num_kv_heads, block_size, view_width_tiles * tile)


def _unpage_cache(cache, page_table, *, paged_cache_geometry=None, cache_position_modulo=None):
    cache = _reinterpret_paged_cache(cache, paged_cache_geometry)
    block_size = cache.shape[2]
    batches = page_table.shape[0]
    sequence_length = page_table.shape[1] * block_size
    output = cache.new_empty(batches, cache.shape[1], sequence_length, cache.shape[-1])
    for batch_index in range(batches):
        for virtual_block in range(page_table.shape[1]):
            logical_block = virtual_block
            if cache_position_modulo is not None:
                logical_block %= int(cache_position_modulo) // block_size
            physical_block = int(page_table[batch_index, logical_block])
            start = virtual_block * block_size
            output[batch_index, :, start : start + block_size] = cache[physical_block]
    return output


def _golden_function_paged_scaled_dot_product_attention_decode(
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
    if cache_position_modulo is not None:
        import torch

        key_cache = _reinterpret_paged_cache(input_tensor_k, paged_cache_geometry)
        value_cache = _reinterpret_paged_cache(input_tensor_v, paged_cache_geometry)
        if cur_pos_tensor is None:
            raise ValueError("cache_position_modulo requires cur_pos_tensor")
        if not is_causal or attn_mask is not None:
            raise ValueError("cache_position_modulo golden supports causal decode without an explicit mask")
        positions = cur_pos_tensor.reshape(-1).tolist()
        modulo = int(cache_position_modulo)
        block_size = key_cache.shape[2]
        if modulo <= 0 or modulo % block_size:
            raise ValueError("cache_position_modulo must be a positive multiple of the cache block size")
        if sliding_window_size is not None and modulo < int(sliding_window_size):
            raise ValueError("cache_position_modulo must be at least sliding_window_size")
        output = torch.zeros(
            1,
            input_tensor_q.shape[1],
            input_tensor_q.shape[2],
            value_cache.shape[-1],
            dtype=torch.float32,
            device=input_tensor_q.device,
        )
        for batch_index, position in enumerate(positions):
            position = int(position)
            if position < 0:
                continue
            available = modulo if sliding_window_size is None else min(modulo, int(sliding_window_size))
            token_positions = range(max(0, position + 1 - available), position + 1)
            key_tokens = []
            value_tokens = []
            for token_position in token_positions:
                cache_position = token_position % modulo
                physical_block = int(page_table_tensor[batch_index, cache_position // block_size])
                block_offset = cache_position % block_size
                key_tokens.append(key_cache[physical_block, :, block_offset])
                value_tokens.append(value_cache[physical_block, :, block_offset])
            key = torch.stack(key_tokens, dim=-2).unsqueeze(0)
            value = torch.stack(value_tokens, dim=-2).unsqueeze(0)
            sink = attention_sink
            if sink is not None and sink.shape[0] == len(positions):
                sink = sink[batch_index : batch_index + 1]
            batch_output = _attention_reference(
                input_tensor_q[:, batch_index].unsqueeze(-2),
                key,
                value,
                scale=scale,
                attention_sink=sink,
            )
            output[0, batch_index] = batch_output[0, :, 0]
        return output

    key = _unpage_cache(
        input_tensor_k,
        page_table_tensor,
        paged_cache_geometry=paged_cache_geometry,
    )
    value = _unpage_cache(
        input_tensor_v,
        page_table_tensor,
        paged_cache_geometry=paged_cache_geometry,
    )
    return _decode_attention_reference(
        input_tensor_q,
        key,
        value,
        cur_pos_tensor,
        is_causal=is_causal,
        scale=scale,
        attn_mask=attn_mask,
        sliding_window_size=sliding_window_size,
        attention_sink=attention_sink,
    )


ttnn.attach_golden_function(
    ttnn.transformer.paged_scaled_dot_product_attention_decode,
    golden_function=_golden_function_paged_scaled_dot_product_attention_decode,
)


def _golden_function_chunked_scaled_dot_product_attention(
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
    if chunk_start_idx_tensor is not None:
        chunk_start_idx = int(chunk_start_idx_tensor.reshape(-1)[0])
    if chunk_start_idx is None:
        raise ValueError("chunk_start_idx or chunk_start_idx_tensor is required")
    key = _unpage_cache(input_tensor_k, page_table_tensor, paged_cache_geometry=paged_cache_geometry)
    value = _unpage_cache(input_tensor_v, page_table_tensor, paged_cache_geometry=paged_cache_geometry)
    end = int(chunk_start_idx) + input_tensor_q.shape[-2]
    return _attention_reference(
        input_tensor_q,
        key[..., :end, :],
        value[..., :end, :],
        scale=scale,
        is_causal=True,
        query_start=int(chunk_start_idx),
    )


ttnn.attach_golden_function(
    ttnn.transformer.chunked_scaled_dot_product_attention,
    golden_function=_golden_function_chunked_scaled_dot_product_attention,
)


def _golden_function_joint_scaled_dot_product_attention(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    joint_tensor_q,
    joint_tensor_k,
    joint_tensor_v,
    *,
    joint_strategy,
    scale=None,
    **_,
):
    import torch

    if joint_strategy != "rear":
        raise ValueError("joint_scaled_dot_product_attention only supports joint_strategy='rear'")
    query = torch.cat((input_tensor_q, joint_tensor_q), dim=-2)
    key = torch.cat((input_tensor_k, joint_tensor_k), dim=-2)
    value = torch.cat((input_tensor_v, joint_tensor_v), dim=-2)
    output = _attention_reference(query, key, value, scale=scale)
    split = input_tensor_q.shape[-2]
    return output[..., :split, :], output[..., split:, :]


ttnn.attach_golden_function(
    ttnn.transformer.joint_scaled_dot_product_attention,
    golden_function=_golden_function_joint_scaled_dot_product_attention,
)


def _golden_function_sparse_sdpa(
    q,
    kv,
    indices,
    v_dim,
    *,
    kv_format,
    scale=None,
    cache_batch_idx=None,
    block_cyclic_sp_axis=None,
    block_cyclic_chunk_local=None,
    **_,
):
    import torch

    if "BF16" not in str(kv_format):
        raise ValueError("sparse_sdpa golden supports only decoded BF16 KV tensors")
    if block_cyclic_sp_axis is not None or block_cyclic_chunk_local is not None:
        raise ValueError("sparse_sdpa block-cyclic cache placement is not representable by a CPU tensor")
    cache = kv[0 if cache_batch_idx is None else int(cache_batch_idx), 0].float()
    if v_dim > cache.shape[-1]:
        raise ValueError("sparse_sdpa v_dim cannot exceed the KV width")
    query = q.float()
    output = torch.zeros(*query.shape[:-1], v_dim, dtype=torch.float32, device=q.device)
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    token_indices = indices[0, 0].to(torch.long)
    for sequence_index in range(query.shape[-2]):
        selected = token_indices[sequence_index]
        selected = selected[(selected >= 0) & (selected < cache.shape[0])]
        keys = cache[selected]
        values = keys[:, :v_dim]
        scores = torch.matmul(query[0, :, sequence_index], keys.transpose(0, 1)) * scale
        output[0, :, sequence_index] = torch.matmul(torch.softmax(scores, dim=-1), values)
    return output


ttnn.attach_golden_function(ttnn.transformer.sparse_sdpa, golden_function=_golden_function_sparse_sdpa)


def _golden_function_sparse_sdpa_msa(
    q,
    k,
    v,
    indices,
    *,
    scale=None,
    block_size=128,
    cache_batch_idx=None,
    chunk_start_idx=None,
    cluster_axis=None,
    block_cyclic_sp_axis=None,
    block_cyclic_chunk_local=None,
    **_,
):
    import torch

    if cluster_axis is not None or block_cyclic_sp_axis is not None or block_cyclic_chunk_local is not None:
        raise ValueError("sparse_sdpa_msa mesh/block-cyclic layouts are not representable by a CPU tensor")
    cache_batch = 0 if cache_batch_idx is None else int(cache_batch_idx)
    query = q.float()
    key = k[cache_batch].float()
    value = v[cache_batch].float()
    if query.shape[1] % key.shape[0]:
        raise ValueError("sparse_sdpa_msa query head count must be divisible by KV head count")
    if key.shape[-2] % block_size:
        raise ValueError("sparse_sdpa_msa block_size must divide the KV sequence length")
    output = torch.zeros(*query.shape[:-1], value.shape[-1], dtype=torch.float32, device=q.device)
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    heads_per_kv = query.shape[1] // key.shape[0]

    for head in range(query.shape[1]):
        kv_head = head // heads_per_kv
        for sequence_index in range(query.shape[-2]):
            block_ids = indices[0, kv_head, sequence_index].to(torch.long)
            block_ids = block_ids[(block_ids >= 0) & (block_ids * block_size < key.shape[-2])]
            if block_ids.numel() == 0:
                continue
            token_ids = torch.cat(
                [torch.arange(block * block_size, (block + 1) * block_size, device=q.device) for block in block_ids]
            )
            if chunk_start_idx is not None:
                token_ids = token_ids[token_ids <= int(chunk_start_idx) + sequence_index]
            if token_ids.numel() == 0:
                continue
            keys = key[kv_head, token_ids]
            values = value[kv_head, token_ids]
            scores = torch.matmul(query[0, head, sequence_index], keys.transpose(0, 1)) * scale
            output[0, head, sequence_index] = torch.matmul(torch.softmax(scores, dim=-1), values)
    return output


ttnn.attach_golden_function(
    ttnn.transformer.sparse_sdpa_msa,
    golden_function=_golden_function_sparse_sdpa_msa,
)

# ring_distributed_scaled_dot_product_attention and
# ring_joint_scaled_dot_product_attention intentionally remain unattached. Their outputs
# are device-local sequence assignments (and, for ring-joint, persistent-buffer mutations
# plus streaming LSE state); comparison mode's single composed CPU tensor cannot represent
# those topology-dependent observables without discarding part of the contract.


__all__ = []
