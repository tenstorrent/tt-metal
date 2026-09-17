# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Torch reference implementations shared by TTNN SDPA goldens and tests."""


def expand_gqa(query, key, value):
    """Repeat KV heads to match query heads for grouped-query attention."""
    if key.shape[1] != query.shape[1]:
        repeat = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
    return key, value


def make_sdpa_mask(query_length, key_length, *, is_causal, sliding_window_size=None, query_offset=0):
    import torch

    query_positions = torch.arange(query_length).view(-1, 1) + query_offset
    key_positions = torch.arange(key_length).view(1, -1)
    allowed = torch.ones((query_length, key_length), dtype=torch.bool)
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


def sdpa_reference(
    query,
    key,
    value,
    *,
    attn_mask=None,
    is_causal=False,
    scale=None,
    sliding_window_size=None,
    attention_sink=None,
):
    """Compute dense SDPA with TTNN's GQA, window, mixed-dtype, and sink semantics."""
    import torch

    key, value = expand_gqa(query, key, value)

    # Torch SDPA has no sliding-window flag, so materialize that mask explicitly.
    if sliding_window_size and attn_mask is None:
        query_length, key_length = query.shape[-2], key.shape[-2]
        attn_mask = make_sdpa_mask(
            query_length,
            key_length,
            is_causal=is_causal,
            sliding_window_size=sliding_window_size,
            query_offset=key_length - query_length if is_causal else 0,
        )
        is_causal = False

    if attention_sink is None and query.dtype == key.dtype == value.dtype:
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, is_causal=is_causal, scale=scale
        )

    # Block-float K/V preprocesses to FP32 while Q stays BF16; explicit FP32 attention supports that mixed input.
    # This path also models an attention sink as an extra denominator-only logit.
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    if is_causal:
        causal_mask = make_sdpa_mask(
            query.shape[-2],
            key.shape[-2],
            is_causal=True,
            query_offset=key.shape[-2] - query.shape[-2],
        )
        attn_mask = causal_mask if attn_mask is None else attn_mask + causal_mask
    if attn_mask is not None:
        logits = logits + attn_mask
    if attention_sink is not None:
        sink = attention_sink.float().reshape(1, logits.shape[1], 1, 1) * scale
        sink = sink.expand(logits.shape[0], logits.shape[1], logits.shape[-2], 1)
        logits = torch.cat([logits, sink], dim=-1)
        probabilities = torch.softmax(logits, dim=-1)[..., :-1]
    else:
        probabilities = torch.softmax(logits, dim=-1)
    return torch.matmul(probabilities, value.float()).to(query.dtype)


def make_windowed_sdpa_mask(query_length, key_length, cu_window_seqlens, query_offset):
    import torch

    boundaries = [int(value) for value in cu_window_seqlens.reshape(-1)]
    mask = torch.full((query_length, key_length), float("-inf"))
    for window_start, window_end in zip(boundaries, boundaries[1:]):
        local_start = max(window_start - query_offset, 0)
        local_end = min(window_end - query_offset, query_length)
        if local_start < local_end:
            mask[local_start:local_end, window_start:window_end] = 0.0
    return mask


def golden_scaled_dot_product_attention(
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
        attn_mask = make_windowed_sdpa_mask(
            input_tensor_q.shape[-2],
            input_tensor_k.shape[-2],
            cu_window_seqlens,
            int(windowed_q_token_offset),
        )
        is_causal = False

    return sdpa_reference(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        sliding_window_size=sliding_window_size,
        attention_sink=attention_sink,
    )


def resolve_cur_pos(cur_pos, cur_pos_tensor, batch_size, cache_length):
    if cur_pos_tensor is not None:
        return [int(position) for position in cur_pos_tensor.flatten().tolist()[:batch_size]]
    if cur_pos:
        return [int(position) for position in cur_pos]
    return [cache_length - 1] * batch_size


def sdpa_decode_reference(
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
    """Compute decode SDPA for TTNN's [1, batch, heads, dim] query layout."""
    import torch

    # Decode Q is [1, batch, heads, dim], while torch SDPA expects [batch, heads, 1, dim].
    query = input_tensor_q.permute(1, 2, 0, 3)
    batch_size = query.shape[0]
    key_length = input_tensor_k.shape[-2]
    positions = resolve_cur_pos(cur_pos, cur_pos_tensor, batch_size, key_length)
    position_mask = torch.zeros((batch_size, 1, 1, key_length))
    for batch_index, position in enumerate(positions):
        if is_causal:
            position_mask[batch_index, :, :, position + 1 :] = float("-inf")
        if sliding_window_size:
            window_start = max(0, position - int(sliding_window_size) + 1)
            position_mask[batch_index, :, :, :window_start] = float("-inf")
    if attn_mask is not None:
        # TTNN decode masks are [batch_or_1, 1, heads, sequence], while torch SDPA expects
        # [batch_or_1, heads, 1, sequence].
        position_mask = position_mask + attn_mask.permute(0, 2, 1, 3)

    output = sdpa_reference(
        query,
        input_tensor_k,
        input_tensor_v,
        attn_mask=position_mask,
        is_causal=False,
        scale=scale,
        attention_sink=attention_sink,
    )
    return output.permute(2, 0, 1, 3)


def reinterpret_paged_cache(cache, view_num_heads, view_block_size, view_head_dim):
    """Reinterpret a TILE cache while preserving its linear per-block tile order."""
    num_blocks, alloc_num_heads, alloc_block_size, alloc_head_dim = cache.shape
    tile_size = 32
    dimensions = (alloc_block_size, alloc_head_dim, view_block_size, view_head_dim)
    if any(dimension % tile_size != 0 for dimension in dimensions):
        raise ValueError("paged cache geometry dimensions must be tile aligned")

    alloc_block_tiles = alloc_block_size // tile_size
    alloc_width_tiles = alloc_head_dim // tile_size
    view_block_tiles = view_block_size // tile_size
    view_width_tiles = view_head_dim // tile_size
    alloc_tiles = alloc_num_heads * alloc_block_tiles * alloc_width_tiles
    view_tiles = view_num_heads * view_block_tiles * view_width_tiles
    if alloc_tiles != view_tiles:
        raise ValueError("paged cache geometry must preserve the per-block tile count")

    # A plain reshape changes tile interpretation; flatten and rebuild the declared row-major tile grid instead.
    tiles = cache.view(num_blocks, alloc_num_heads, alloc_block_tiles, tile_size, alloc_width_tiles, tile_size)
    tiles = tiles.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(num_blocks, alloc_tiles, tile_size, tile_size)
    tiles = tiles.reshape(num_blocks, view_num_heads, view_block_tiles, view_width_tiles, tile_size, tile_size)
    return (
        tiles.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(num_blocks, view_num_heads, view_block_size, view_head_dim)
    )


def gather_paged_kv(cache, page_table, *, paged_cache_geometry=None, head_dim=None):
    """Convert paged [blocks, heads, block, dim] cache to dense [batch, heads, sequence, dim]."""
    if paged_cache_geometry is not None:
        # Shared HMA buffers can be allocated with another layer's shape; Q supplies this view's head dimension.
        cache = reinterpret_paged_cache(
            cache,
            paged_cache_geometry.num_kv_heads,
            paged_cache_geometry.block_size,
            head_dim,
        )
    pages = cache[page_table.long()]
    batch_size, num_pages, num_heads, block_size, head_dim = pages.shape
    return pages.permute(0, 2, 1, 3, 4).reshape(batch_size, num_heads, num_pages * block_size, head_dim)


def linearize_circular_cache(cache, cur_pos_tensor, capacity):
    import torch

    positions = [int(position) for position in cur_pos_tensor.reshape(-1)[: cache.shape[0]]]
    if len(positions) != cache.shape[0]:
        raise ValueError("cur_pos_tensor must contain one position per cache batch")
    if cache.shape[-2] < capacity:
        raise ValueError("page table does not cover cache_position_modulo")
    valid_lengths = [min(position + 1, capacity) for position in positions]
    sequence_length = max(1, max(valid_lengths))
    linear_cache = torch.zeros(
        cache.shape[0],
        cache.shape[1],
        sequence_length,
        cache.shape[-1],
        dtype=cache.dtype,
        device=cache.device,
    )
    for batch_index, (position, valid_length) in enumerate(zip(positions, valid_lengths)):
        if valid_length == 0:
            continue
        first_position = position - valid_length + 1
        physical_positions = torch.arange(first_position, position + 1, device=cache.device) % capacity
        linear_cache[batch_index, :, :valid_length, :] = cache[batch_index, :, physical_positions, :]
    return linear_cache, torch.tensor(
        [valid_length - 1 for valid_length in valid_lengths],
        dtype=cur_pos_tensor.dtype,
        device=cur_pos_tensor.device,
    )


def golden_paged_scaled_dot_product_attention_decode(
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
    head_dim = input_tensor_q.shape[-1]
    gathered_k = gather_paged_kv(
        input_tensor_k,
        page_table_tensor,
        paged_cache_geometry=paged_cache_geometry,
        head_dim=head_dim,
    )
    gathered_v = gather_paged_kv(
        input_tensor_v,
        page_table_tensor,
        paged_cache_geometry=paged_cache_geometry,
        head_dim=head_dim,
    )
    if cache_position_modulo is not None:
        # Rebuild each user's chronological window: only page lookup wraps, while decode positions remain absolute.
        gathered_k = gathered_k[..., :cache_position_modulo, :]
        gathered_v = gathered_v[..., :cache_position_modulo, :]
        if cur_pos_tensor is not None:
            gathered_k, relative_cur_pos = linearize_circular_cache(gathered_k, cur_pos_tensor, cache_position_modulo)
            gathered_v, _ = linearize_circular_cache(gathered_v, cur_pos_tensor, cache_position_modulo)
            cur_pos_tensor = relative_cur_pos

    return sdpa_decode_reference(
        input_tensor_q,
        gathered_k,
        gathered_v,
        is_causal=is_causal,
        attn_mask=attn_mask,
        cur_pos_tensor=cur_pos_tensor,
        attention_sink=attention_sink,
        scale=scale,
        sliding_window_size=sliding_window_size,
    )


def golden_chunked_scaled_dot_product_attention(
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
    """Compute chunked causal SDPA from paged K/V caches.

    ``chunk_start_idx`` is the first absolute Q position. Query row ``i`` can
    attend through key position ``chunk_start_idx + i``.
    """
    if paged_cache_geometry is not None:
        raise NotImplementedError("chunked SDPA golden does not support cache geometry overrides")
    if chunk_start_idx_tensor is not None:
        chunk_start_idx = int(chunk_start_idx_tensor.reshape(-1)[0])
    if chunk_start_idx is None:
        raise ValueError("chunk_start_idx or chunk_start_idx_tensor is required")

    key = gather_paged_kv(input_tensor_k, page_table_tensor)
    value = gather_paged_kv(input_tensor_v, page_table_tensor)
    prefix_end = int(chunk_start_idx) + input_tensor_q.shape[-2]
    key = key[..., :prefix_end, :]
    value = value[..., :prefix_end, :]
    causal_mask = make_sdpa_mask(
        input_tensor_q.shape[-2],
        prefix_end,
        is_causal=True,
        query_offset=int(chunk_start_idx),
    )
    return sdpa_reference(
        input_tensor_q,
        key,
        value,
        attn_mask=causal_mask,
        is_causal=False,
        scale=scale,
    )


def golden_joint_scaled_dot_product_attention(
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
    query = torch.cat([input_tensor_q, joint_tensor_q], dim=-2)
    key = torch.cat([input_tensor_k, joint_tensor_k], dim=-2)
    value = torch.cat([input_tensor_v, joint_tensor_v], dim=-2)
    output = sdpa_reference(query, key, value, is_causal=False, scale=scale)
    sequence_length = input_tensor_q.shape[-2]
    return output[..., :sequence_length, :], output[..., sequence_length:, :]


def sparse_mla_reference(query, kvpe, indices, scale, value_dim, attention_sink=None):
    """Torch reference for sparse MLA prefill.

    Absorbed MQA attends over the top-k latent rows selected by ``indices``;
    ``0xFFFFFFFF`` marks a masked slot. V is the leading ``value_dim`` columns
    of each key-width latent.

    Shapes:
        query: [batch, heads, sequence, key_dim]
        kvpe: [cache_length, key_dim]
        indices: [batch, 1, sequence, topk]
        output: [batch, heads, sequence, value_dim]
    """
    import torch

    batch_size, num_heads, sequence_length, key_dim = query.shape
    topk = indices.shape[-1]
    cache_length = kvpe.shape[0]
    selected_indices = indices.reshape(batch_size, sequence_length, topk)
    masked = selected_indices == 0xFFFFFFFF
    # Clamp sentinels in-bounds before gathering; the mask removes them from softmax.
    safe_indices = torch.where(masked, torch.zeros_like(selected_indices), selected_indices).long()
    cache = kvpe.unsqueeze(0).expand(batch_size, cache_length, key_dim)
    # Gather the selected latent rows once; all query heads share those rows.
    selected_kv = torch.gather(
        cache.unsqueeze(1).expand(batch_size, sequence_length, cache_length, key_dim),
        2,
        safe_indices.unsqueeze(-1).expand(batch_size, sequence_length, topk, key_dim),
    )
    logits = torch.einsum("bhsd,bsjd->bhsj", query.float(), selected_kv.float()) * scale
    logits = logits.masked_fill(masked.view(batch_size, 1, sequence_length, topk), float("-inf"))
    if attention_sink is not None:
        sink = attention_sink.float().reshape(1, num_heads, 1, 1)
        logits = torch.cat([logits, sink.expand(batch_size, num_heads, sequence_length, 1) * scale], dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    if attention_sink is not None:
        probabilities = probabilities[..., :-1]
    # The value view occupies the leading value_dim columns of each selected latent.
    return torch.einsum("bhsj,bsjd->bhsd", probabilities, selected_kv[..., :value_dim]).to(query.dtype)


def golden_sparse_sdpa(
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
    if block_cyclic_sp_axis is not None or block_cyclic_chunk_local is not None or block_cyclic_cache_tp_sharded:
        raise NotImplementedError("sparse SDPA golden does not support block-cyclic cache layouts")
    cache_batch_idx = 0 if cache_batch_idx is None else int(cache_batch_idx)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return sparse_mla_reference(q, kv[cache_batch_idx, 0], indices, scale, v_dim, attention_sink)


def golden_flash_mla_prefill(
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
    return sdpa_reference(
        input_tensor_q,
        input_tensor_k,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
    )


def golden_flash_multi_latent_attention_decode(
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
    return sdpa_decode_reference(
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


def golden_paged_flash_multi_latent_attention_decode(
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
    key = gather_paged_kv(input_tensor_k, page_table_tensor)
    value = gather_paged_kv(input_tensor_v, page_table_tensor) if input_tensor_v is not None else key[..., :head_dim_v]
    return golden_flash_multi_latent_attention_decode(
        input_tensor_q,
        key,
        value,
        head_dim_v,
        is_causal=is_causal,
        attn_mask=attn_mask,
        cur_pos_tensor=cur_pos_tensor,
        attention_sink=attention_sink,
        scale=scale,
        sliding_window_size=sliding_window_size,
    )


def golden_chunked_flash_mla_prefill(
    input_tensor_q,
    input_tensor_k,
    head_dim_v,
    page_table_tensor,
    chunk_start_idx,
    *,
    scale=None,
    **_,
):
    key = gather_paged_kv(input_tensor_k, page_table_tensor)
    prefix_end = int(chunk_start_idx) + input_tensor_q.shape[-2]
    key = key[..., :prefix_end, :]
    value = key[..., :head_dim_v]
    causal_mask = make_sdpa_mask(
        input_tensor_q.shape[-2],
        prefix_end,
        is_causal=True,
        query_offset=int(chunk_start_idx),
    )
    return sdpa_reference(
        input_tensor_q,
        key,
        value,
        attn_mask=causal_mask,
        is_causal=False,
        scale=scale,
    )
