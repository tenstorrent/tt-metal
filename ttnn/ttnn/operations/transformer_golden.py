# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared PyTorch golden functions for ``ttnn.transformer`` operations.

The mathematical references in this module are production-owned so tests,
sweeps, and operation registration all use the same implementation.
"""

import math
from typing import Optional


MASKED_INDEX = 0xFFFFFFFF  # sparse_sdpa masked-slot sentinel
SENTINEL = -1  # sparse_sdpa_msa masked-block sentinel
BLK_KV = 128  # MSA block size in tokens (= 4 tile rows)


def _scalar(value, default=None):
    import torch

    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item() if value.numel() else default
    return value


def _repeat_kv_heads(tensor, num_query_heads):
    num_kv_heads = tensor.shape[1]
    if num_kv_heads == num_query_heads:
        return tensor
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f"Q heads ({num_query_heads}) must be divisible by KV heads ({num_kv_heads})")
    return tensor.repeat_interleave(num_query_heads // num_kv_heads, dim=1)


def _paged_to_contiguous(cache, page_table, *, cache_position_modulo=None):
    import torch

    """Reconstruct logical [B, H, S, D] cache rows from [P, H, block, D] pages."""
    if cache.shape[0] == page_table.shape[0] and cache.ndim == 4 and cache.shape[2] > page_table.shape[-1]:
        return cache

    page_table = page_table.long().reshape(page_table.shape[0], -1)
    num_pages, num_heads, block_size, head_dim = cache.shape
    rows = []
    for page_row in page_table:
        physical_pages = page_row.clamp(0, num_pages - 1)
        row = cache[physical_pages].permute(1, 0, 2, 3).reshape(num_heads, -1, head_dim)
        if cache_position_modulo is not None:
            row = row[:, : int(cache_position_modulo)]
        rows.append(row)
    return torch.stack(rows)


def _window_mask(
    query_length,
    key_length,
    *,
    is_causal,
    sliding_window_size=None,
    query_start=None,
    device=None,
):
    import torch

    query_start = max(0, key_length - query_length) if query_start is None else int(query_start)
    query_positions = torch.arange(query_start, query_start + query_length, device=device).unsqueeze(1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)

    allowed = torch.ones((query_length, key_length), dtype=torch.bool, device=device)
    if is_causal:
        allowed &= key_positions <= query_positions
    if sliding_window_size is not None:
        window = int(sliding_window_size)
        if is_causal:
            allowed &= key_positions > query_positions - window
        else:
            half_window = window // 2
            allowed &= (key_positions >= query_positions - half_window) & (
                key_positions <= query_positions + half_window
            )
    return allowed


def _cu_window_mask(query_length, key_length, cu_window_seqlens, query_start, device):
    import torch

    boundaries = cu_window_seqlens.reshape(-1).long()
    query_positions = torch.arange(query_start, query_start + query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    allowed = torch.zeros((query_length, key_length), dtype=torch.bool, device=device)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        allowed |= ((query_positions >= start) & (query_positions < stop)).unsqueeze(1) & (
            (key_positions >= start) & (key_positions < stop)
        ).unsqueeze(0)
    return allowed


def _scaled_attention(
    query,
    key,
    value,
    *,
    attn_mask=None,
    is_causal=False,
    scale=None,
    sliding_window_size=None,
    attention_sink=None,
    query_start=None,
    cu_window_seqlens=None,
):
    """Dense grouped-query attention with TTNN masking and sink semantics."""
    import torch

    num_query_heads = query.shape[1]
    key = _repeat_kv_heads(key, num_query_heads)
    value = _repeat_kv_heads(value, num_query_heads)
    query_length = query.shape[-2]
    key_length = key.shape[-2]
    scale = query.shape[-1] ** -0.5 if scale is None else scale

    explicit_mask = attn_mask
    needs_position_mask = is_causal or sliding_window_size is not None or cu_window_seqlens is not None
    if needs_position_mask:
        if cu_window_seqlens is not None:
            position_mask = _cu_window_mask(
                query_length,
                key_length,
                cu_window_seqlens,
                int(_scalar(query_start, 0)),
                query.device,
            )
        else:
            position_mask = _window_mask(
                query_length,
                key_length,
                is_causal=is_causal,
                sliding_window_size=sliding_window_size,
                query_start=query_start,
                device=query.device,
            )
        position_mask = position_mask.reshape(1, 1, query_length, key_length)
        if explicit_mask is None:
            explicit_mask = position_mask
        else:
            explicit_mask = explicit_mask[..., :query_length, :key_length]
            if explicit_mask.dtype == torch.bool:
                explicit_mask = explicit_mask & position_mask
            else:
                explicit_mask = explicit_mask.masked_fill(~position_mask, float("-inf"))

    if attention_sink is None:
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=explicit_mask,
            scale=scale,
            is_causal=False,
        )

    scores = torch.matmul(query.float(), key.transpose(-2, -1).float()) * scale
    if explicit_mask is not None:
        explicit_mask = explicit_mask[..., :query_length, :key_length]
        if explicit_mask.dtype == torch.bool:
            scores = scores.masked_fill(~explicit_mask, float("-inf"))
        else:
            scores = scores + explicit_mask

    sink = attention_sink.float()
    if sink.ndim == 1:
        sink = sink.reshape(1, -1, 1, 1)
    elif sink.shape[-1] == num_query_heads and sink.shape[1] != num_query_heads:
        sink = sink.permute(0, 3, 1, 2)
    sink = sink.expand(scores.shape[0], num_query_heads, query_length, 1) * scale
    probabilities = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]
    return torch.matmul(probabilities, value.float()).to(query.dtype)


def scaled_dot_product_attention_golden(
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
    query_offset = _scalar(windowed_q_token_offset_tensor, windowed_q_token_offset)
    return _scaled_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        sliding_window_size=sliding_window_size,
        attention_sink=attention_sink,
        query_start=query_offset if cu_window_seqlens is not None else None,
        cu_window_seqlens=cu_window_seqlens,
    )


def _chunked_paged_attention(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    page_table_tensor,
    chunk_start_idx,
    *,
    scale=None,
):
    key = _paged_to_contiguous(input_tensor_k, page_table_tensor)
    value = _paged_to_contiguous(input_tensor_v, page_table_tensor)
    chunk_start_idx = int(_scalar(chunk_start_idx, 0))
    key_length = min(key.shape[-2], chunk_start_idx + input_tensor_q.shape[-2])
    return _scaled_attention(
        input_tensor_q,
        key[..., :key_length, :],
        value[..., :key_length, :],
        is_causal=True,
        scale=scale,
        query_start=chunk_start_idx,
    )


def chunked_scaled_dot_product_attention_golden(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    page_table_tensor,
    chunk_start_idx=None,
    *,
    chunk_start_idx_tensor=None,
    scale=None,
    **_,
):
    runtime_start = chunk_start_idx_tensor if chunk_start_idx_tensor is not None else chunk_start_idx
    return _chunked_paged_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        page_table_tensor,
        runtime_start,
        scale=scale,
    )


def joint_scaled_dot_product_attention_golden(
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
        raise ValueError(f"Only joint_strategy='rear' is supported, got {joint_strategy!r}")
    sequence_length = input_tensor_q.shape[-2]
    query = torch.cat([input_tensor_q, joint_tensor_q], dim=-2)
    key = torch.cat([input_tensor_k, joint_tensor_k], dim=-2)
    value = torch.cat([input_tensor_v, joint_tensor_v], dim=-2)
    output = _scaled_attention(query, key, value, scale=scale)
    return output[..., :sequence_length, :], output[..., sequence_length:, :]


def scaled_dot_product_attention_reference(Q, K, V, start_indices, padded_layer_len, scale, is_causal=True):
    """Decode reference migrated from ``mla_test_utils.py``."""
    import torch

    b, nh, _, _ = Q.shape
    _, nkv, _, _ = K.shape

    attn_mask = None
    if is_causal:
        attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
        for i in range(b):
            start_idx = start_indices[i]
            attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min
    else:
        raise AssertionError("Non-causal attention is not supported in this function.")

    Q_slice = Q[:, :nh, :, :]
    K_slice = K[:, :nkv, :padded_layer_len, :]
    K_slice = torch.cat([K_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_slice = V[:, :, :padded_layer_len, :]
    V_slice = torch.cat([V_slice[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    attn_mask_slice = attn_mask[:, :nh, :, :]
    return torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )


def scaled_dot_product_attention_reference_prefill(Q, K, V, scale, is_causal=True):
    """
    Memory-efficient full-sequence causal SDPA reference.
    Q: (B, nh, S, d_qk), K/V: (B, nkv, S, d)

    Chunks over heads (and, only for very long sequences, the Q sequence) so
    the [B, nh, S, S] attention matrix never materializes at full size.
    """
    import torch

    SEQ_CHUNK = 4096
    HEAD_CHUNK = 16

    B, nh, S, _ = Q.shape
    _, nkv, _, _ = V.shape
    Dv = V.shape[-1]
    head_rep = nh // nkv

    attn_out = torch.empty(B, nh, S, Dv, dtype=Q.dtype)
    for h_start in range(0, nh, HEAD_CHUNK):
        h_end = min(h_start + HEAD_CHUNK, nh)
        # Map each Q head in the chunk to its KV head (GQA broadcast) without
        # copying the full KV tensor.
        kv_idx = torch.arange(h_start, h_end) // head_rep
        k_heads = K[:, kv_idx]
        v_heads = V[:, kv_idx]
        q_heads = Q[:, h_start:h_end]
        for seq_start in range(0, S, SEQ_CHUNK):
            seq_end = min(seq_start + SEQ_CHUNK, S)
            q_chunk = q_heads[:, :, seq_start:seq_end]
            if is_causal and seq_start == 0 and seq_end == S:
                out = torch.nn.functional.scaled_dot_product_attention(
                    q_chunk, k_heads, v_heads, scale=scale, is_causal=True
                )
            elif is_causal:
                q_pos = torch.arange(seq_start, seq_end).unsqueeze(1)
                k_pos = torch.arange(seq_end).unsqueeze(0)
                mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
                out = torch.nn.functional.scaled_dot_product_attention(
                    q_chunk, k_heads[:, :, :seq_end], v_heads[:, :, :seq_end], attn_mask=mask, scale=scale
                )
            else:
                out = torch.nn.functional.scaled_dot_product_attention(q_chunk, k_heads, v_heads, scale=scale)
            attn_out[:, h_start:h_end, seq_start:seq_end] = out
    return attn_out


def _decode_layout(input_tensor_q, input_tensor_k):
    batch = input_tensor_k.shape[0]
    if input_tensor_q.shape[0] == 1 and input_tensor_q.shape[1] == batch:
        return input_tensor_q.permute(1, 2, 0, 3), True
    return input_tensor_q, False


def _decode_attention(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    *,
    cur_pos=None,
    cur_pos_tensor=None,
    attn_mask=None,
    is_causal=True,
    scale=None,
    sliding_window_size=None,
    attention_sink=None,
    share_cache=False,
):
    import torch

    query, transposed = _decode_layout(input_tensor_q, input_tensor_k)
    batch, num_heads, _, _ = query.shape
    positions = cur_pos_tensor if cur_pos_tensor is not None else cur_pos
    if positions is None or (not isinstance(positions, torch.Tensor) and len(positions) == 0):
        positions = [input_tensor_k.shape[-2] - 1] * batch
    if isinstance(positions, torch.Tensor):
        positions = positions.reshape(-1).tolist()

    outputs = []
    for batch_index in range(batch):
        position = int(positions[batch_index % len(positions)])
        if position < 0:
            outputs.append(torch.zeros_like(query[batch_index : batch_index + 1]))
            continue
        cache_batch = 0 if share_cache or input_tensor_k.shape[0] == 1 else batch_index
        key = input_tensor_k[cache_batch : cache_batch + 1, :, : position + 1]
        value = input_tensor_v[cache_batch : cache_batch + 1, :, : position + 1]
        mask = None if attn_mask is None else attn_mask[batch_index : batch_index + 1, ..., : position + 1]
        sink = attention_sink
        if sink is not None and sink.shape[0] == batch:
            sink = sink[batch_index : batch_index + 1]
        outputs.append(
            _scaled_attention(
                query[batch_index : batch_index + 1],
                key,
                value,
                attn_mask=mask,
                is_causal=is_causal,
                scale=scale,
                sliding_window_size=sliding_window_size,
                attention_sink=sink,
                query_start=position,
            )
        )
    output = torch.cat(outputs, dim=0)
    return output.permute(2, 0, 1, 3) if transposed else output


def scaled_dot_product_attention_decode_golden(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    *,
    is_causal=True,
    attn_mask=None,
    cur_pos=(),
    cur_pos_tensor=None,
    attention_sink=None,
    scale=None,
    sliding_window_size=None,
    share_cache=None,
    **_,
):
    return _decode_attention(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        cur_pos=cur_pos,
        cur_pos_tensor=cur_pos_tensor,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        sliding_window_size=sliding_window_size,
        attention_sink=attention_sink,
        share_cache=bool(share_cache),
    )


def paged_scaled_dot_product_attention_decode_golden(
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
    cache_position_modulo=None,
    **_,
):
    key = _paged_to_contiguous(input_tensor_k, page_table_tensor, cache_position_modulo=cache_position_modulo)
    value = _paged_to_contiguous(input_tensor_v, page_table_tensor, cache_position_modulo=cache_position_modulo)
    return _decode_attention(
        input_tensor_q,
        key,
        value,
        cur_pos_tensor=cur_pos_tensor,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        sliding_window_size=sliding_window_size,
        attention_sink=attention_sink,
    )


def flash_mla_prefill_golden(
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
    elif isinstance(head_dim_v, torch.Tensor):
        value = head_dim_v
    else:
        value = input_tensor_k[..., : int(head_dim_v)]
    if attn_mask is None:
        return scaled_dot_product_attention_reference_prefill(
            input_tensor_q,
            input_tensor_k,
            value,
            input_tensor_q.shape[-1] ** -0.5 if scale is None else scale,
            is_causal,
        )
    return _scaled_attention(
        input_tensor_q,
        input_tensor_k,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
    )


def chunked_flash_mla_prefill_golden(
    input_tensor_q,
    input_tensor_k,
    head_dim_v,
    page_table_tensor,
    chunk_start_idx,
    *,
    scale=None,
    **_,
):
    value = input_tensor_k[..., : int(head_dim_v)]
    return _chunked_paged_attention(
        input_tensor_q,
        input_tensor_k,
        value,
        page_table_tensor,
        chunk_start_idx,
        scale=scale,
    )


def flash_multi_latent_attention_decode_golden(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v=None,
    head_dim_v=None,
    **kwargs,
):
    value = input_tensor_k[..., : int(head_dim_v)] if input_tensor_v is None else input_tensor_v
    return scaled_dot_product_attention_decode_golden(
        input_tensor_q,
        input_tensor_k,
        value,
        **kwargs,
    )


def paged_flash_multi_latent_attention_decode_golden(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v=None,
    head_dim_v=None,
    page_table_tensor=None,
    **kwargs,
):
    value = input_tensor_k[..., : int(head_dim_v)] if input_tensor_v is None else input_tensor_v
    return paged_scaled_dot_product_attention_decode_golden(
        input_tensor_q,
        input_tensor_k,
        value,
        page_table_tensor,
        **kwargs,
    )


def sparse_mla(q, kvpe, indices, scale, v_dim, attention_sink=None):
    """Torch reference for sparse MLA prefill.

    ``q [1,H,S,K_DIM]``, ``kvpe [T,K_DIM]``,
    ``indices [1,1,S,k]`` -> ``out [1,H,S,v_dim]``.
    """
    import torch

    B, H, S, Dk = q.shape
    k = indices.shape[-1]
    T = kvpe.shape[0]
    idx = indices.reshape(B, S, k)
    masked = idx == MASKED_INDEX
    idx_safe = torch.where(masked, torch.zeros_like(idx), idx).to(torch.int64)
    kv = kvpe.unsqueeze(0).expand(B, T, Dk)
    sel = torch.gather(
        kv.unsqueeze(1).expand(B, S, T, Dk),
        2,
        idx_safe.view(B, S, k, 1).expand(B, S, k, Dk),
    )
    scores = torch.einsum("bhsd,bsjd->bhsj", q, sel) * scale
    scores = scores.masked_fill(masked.view(B, 1, S, k), float("-inf"))
    if attention_sink is not None:
        sink_scores = attention_sink.float().expand(B, H, S, 1) * scale
        scores = torch.cat([scores, sink_scores], dim=-1)
    probs = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
    if attention_sink is not None:
        probs = probs[..., :-1]
    return torch.einsum("bhsj,bsjd->bhsd", probs, sel[..., :v_dim])


def sparse_sdpa_golden(
    q,
    kv,
    indices,
    v_dim,
    *,
    scale=None,
    cache_batch_idx=None,
    attention_sink=None,
    **_,
):
    import torch

    scale = q.shape[-1] ** -0.5 if scale is None else scale
    cache_batch_idx = int(_scalar(cache_batch_idx, 0))
    kvpe = kv[cache_batch_idx, 0]
    if attention_sink is not None and attention_sink.shape[-1] == q.shape[1]:
        attention_sink = attention_sink.permute(0, 3, 1, 2)
    return sparse_mla(q, kvpe, indices.to(torch.int64), scale, v_dim, attention_sink)


def sparse_attention_ref_msa(q, k, v, indices, scale, *, blk_kv=BLK_KV, causal=False, chunk_start_idx=0):
    """MSA block-sparse reference: attend selected blocks, softmax, then PV."""
    import torch

    B, H, S, _ = q.shape
    n_kv, T = k.shape[1], k.shape[2]
    topk = indices.shape[-1]
    G = H // n_kv
    nblk = T // blk_kv
    assert T % blk_kv == 0 and H % n_kv == 0

    qf, kf, vf = q.float(), k.float(), v.float()
    v_dim = vf.shape[-1]

    # block_mask[b, g, s, blk] — set True only for valid selected blocks.
    block_mask = torch.zeros(B, n_kv, S, nblk, dtype=torch.bool)
    valid = (indices >= 0) & (indices != MASKED_INDEX)
    idx_safe = torch.where(valid, indices, torch.zeros_like(indices)).long()
    flat_mask = block_mask.view(-1, nblk)
    flat_valid = valid.reshape(-1, topk)
    flat_idx = idx_safe.reshape(-1, topk)
    row = torch.arange(flat_mask.shape[0]).unsqueeze(1)
    flat_mask[row.expand_as(flat_idx)[flat_valid], flat_idx[flat_valid]] = True

    token_mask = block_mask.repeat_interleave(blk_kv, dim=3).repeat_interleave(G, dim=1)
    kf = kf.repeat_interleave(G, dim=1)
    vf = vf.repeat_interleave(G, dim=1)
    scores = torch.einsum("bhsd,bhtd->bhst", qf * scale, kf)
    scores = scores.masked_fill(~token_mask, float("-inf"))
    if causal:
        q_pos = (torch.arange(S) + chunk_start_idx).view(1, 1, S, 1)
        kv_pos = torch.arange(T).view(1, 1, 1, T)
        scores = scores.masked_fill(kv_pos > q_pos, float("-inf"))

    row_has_value = (scores > float("-inf")).any(dim=-1, keepdim=True)
    scores = torch.where(row_has_value, scores, torch.zeros_like(scores))
    attn = torch.where(row_has_value, scores.softmax(dim=-1, dtype=torch.float32), torch.zeros_like(scores))
    return torch.einsum("bhst,bhtd->bhsd", attn, vf[..., :v_dim])


def sparse_sdpa_msa_golden(
    q,
    k,
    v,
    indices,
    *,
    scale=None,
    block_size=BLK_KV,
    cache_batch_idx=None,
    chunk_start_idx=None,
    **_,
):
    import torch

    scale = q.shape[-1] ** -0.5 if scale is None else scale
    cache_batch_idx = int(_scalar(cache_batch_idx, 0))
    return sparse_attention_ref_msa(
        q,
        k[cache_batch_idx : cache_batch_idx + 1],
        v[cache_batch_idx : cache_batch_idx + 1],
        indices.to(torch.int64),
        scale,
        blk_kv=block_size,
        causal=chunk_start_idx is not None,
        chunk_start_idx=int(_scalar(chunk_start_idx, 0)),
    )


def torch_sdpa_reference(q, k, v, is_causal=False, attention_sink=None):
    """
    Memory-efficient PyTorch reference for ring joint attention.

    Chunks over heads and the combined Q sequence so the [B, H, Sq, Sk]
    attention matrix never materializes at full size.
    """
    import torch

    SEQ_CHUNK = 4096
    HEAD_CHUNK = 16

    B, H, total_seq, _ = q.shape
    Dv = v.shape[-1]

    def take_heads(t, h_start, h_end):
        if t.shape[1] == H:
            return t[:, h_start:h_end]
        assert H % t.shape[1] == 0, f"Q heads must be divisible by KV heads, got H={H}, KV={t.shape[1]}"
        heads_per_kv = H // t.shape[1]
        kv_indices = torch.arange(h_start, h_end, device=t.device) // heads_per_kv
        return t[:, kv_indices]

    if attention_sink is not None:
        assert is_causal, "attention sink reference is defined for causal attention only"
        # Unlike PyTorch SDPA, the virtual sink key has no V row. Compute the
        # sink-aware softmax explicitly in compact blocks.
        SEQ_CHUNK = 512
        HEAD_CHUNK = 4
        scale = q.shape[-1] ** -0.5
        attn_out = torch.empty(B, H, total_seq, Dv, dtype=q.dtype)
        for h_start in range(0, H, HEAD_CHUNK):
            h_end = min(h_start + HEAD_CHUNK, H)
            q_heads = q[:, h_start:h_end]
            k_heads = take_heads(k, h_start, h_end)
            v_heads = take_heads(v, h_start, h_end)
            sink_scores = attention_sink[:, h_start:h_end].float() * scale
            for seq_start in range(0, total_seq, SEQ_CHUNK):
                seq_end = min(seq_start + SEQ_CHUNK, total_seq)
                scores = (
                    q_heads[:, :, seq_start:seq_end].float() @ k_heads[:, :, :seq_end].transpose(-2, -1).float()
                ) * scale
                q_pos = torch.arange(seq_start, seq_end, device=q.device).unsqueeze(1)
                k_pos = torch.arange(seq_end, device=q.device).unsqueeze(0)
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
                expanded_sink = sink_scores.expand(B, h_end - h_start, seq_end - seq_start, 1)
                weights = torch.softmax(torch.cat([scores, expanded_sink], dim=-1), dim=-1)[..., :-1]
                attn_out[:, h_start:h_end, seq_start:seq_end] = (weights @ v_heads[:, :, :seq_end].float()).to(q.dtype)
    elif total_seq <= SEQ_CHUNK and H <= HEAD_CHUNK:
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            take_heads(k, 0, H),
            take_heads(v, 0, H),
            is_causal=is_causal,
        )
    else:
        attn_out = torch.empty(B, H, total_seq, Dv, dtype=q.dtype)
        for h_start in range(0, H, HEAD_CHUNK):
            h_end = min(h_start + HEAD_CHUNK, H)
            q_heads = q[:, h_start:h_end]
            k_heads = take_heads(k, h_start, h_end)
            v_heads = take_heads(v, h_start, h_end)
            for seq_start in range(0, total_seq, SEQ_CHUNK):
                seq_end = min(seq_start + SEQ_CHUNK, total_seq)
                q_chunk = q_heads[:, :, seq_start:seq_end]
                if is_causal:
                    q_pos = torch.arange(seq_start, seq_end).unsqueeze(1)
                    k_pos = torch.arange(seq_end).unsqueeze(0)
                    mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
                    out = torch.nn.functional.scaled_dot_product_attention(
                        q_chunk,
                        k_heads[:, :, :seq_end],
                        v_heads[:, :, :seq_end],
                        attn_mask=mask,
                    )
                else:
                    out = torch.nn.functional.scaled_dot_product_attention(q_chunk, k_heads, v_heads)
                attn_out[:, h_start:h_end, seq_start:seq_end] = out
    return attn_out


def torch_sdpa(q, k, v, joint_q, joint_k, joint_v, num_devices):
    """Ring-merge reference migrated from ``test_ring_joint_attention.py``."""
    import torch

    scale = k.size(-1) ** -0.5
    seq_len = k.size(2)
    slice_seq_len = seq_len // num_devices
    out = None
    lse = None
    lse_list = []
    Q = torch.cat([q, joint_q], dim=2)
    for ring_id in range(num_devices):
        k_slice = k[:, :, ring_id * slice_seq_len : (ring_id + 1) * slice_seq_len, :]
        v_slice = v[:, :, ring_id * slice_seq_len : (ring_id + 1) * slice_seq_len, :]
        if ring_id == num_devices - 1:
            k_slice = torch.cat([k_slice, joint_k], dim=2)
            v_slice = torch.cat([v_slice, joint_v], dim=2)
        attn_weights = torch.matmul(Q, k_slice.transpose(-2, -1)) * scale
        cur_max, _ = torch.max(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - cur_max)
        cur_sum = torch.sum(attn_weights, dim=-1, keepdim=True)
        cur_out = torch.matmul(attn_weights, v_slice) / cur_sum
        cur_lse = cur_max + torch.log(cur_sum)
        if ring_id == 0:
            out = cur_out
            lse = cur_lse
        else:
            sig = torch.nn.functional.sigmoid(cur_lse - lse)
            out = out - sig * (out - cur_out)
            lse = lse - torch.nn.functional.logsigmoid(lse - cur_lse)
        lse_list.append(lse)
    return out, lse_list


def _ring_joint_golden(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    joint_tensor_q=None,
    joint_tensor_k=None,
    joint_tensor_v=None,
    *,
    joint_strategy="rear",
    logical_n=None,
    logical_l=0,
    is_causal=False,
    scale=None,
    attention_sink=None,
    sliding_window_size=None,
    kv_cache_batch_idx=None,
    **_,
):
    import torch

    if joint_strategy != "rear":
        raise ValueError(f"Only joint_strategy='rear' is supported, got {joint_strategy!r}")
    cache_batch_idx = int(_scalar(kv_cache_batch_idx, 0))
    if input_tensor_k.shape[0] != input_tensor_q.shape[0]:
        input_tensor_k = input_tensor_k[cache_batch_idx : cache_batch_idx + 1]
        input_tensor_v = input_tensor_v[cache_batch_idx : cache_batch_idx + 1]

    logical_n = int(_scalar(logical_n, input_tensor_k.shape[-2]))
    input_tensor_k = input_tensor_k[..., :logical_n, :]
    input_tensor_v = input_tensor_v[..., :logical_n, :]
    input_length = input_tensor_q.shape[-2]
    has_joint = joint_tensor_q is not None and joint_tensor_q.shape[-2] > 0
    if has_joint:
        logical_l = int(_scalar(logical_l, joint_tensor_q.shape[-2])) or joint_tensor_q.shape[-2]
        joint_tensor_q = joint_tensor_q[..., :logical_l, :]
        joint_tensor_k = joint_tensor_k[..., :logical_l, :]
        joint_tensor_v = joint_tensor_v[..., :logical_l, :]
        query = torch.cat([input_tensor_q, joint_tensor_q], dim=-2)
        key = torch.cat([input_tensor_k, joint_tensor_k], dim=-2)
        value = torch.cat([input_tensor_v, joint_tensor_v], dim=-2)
    else:
        query, key, value = input_tensor_q, input_tensor_k, input_tensor_v

    output = _scaled_attention(
        query,
        key,
        value,
        is_causal=is_causal,
        scale=scale,
        attention_sink=attention_sink,
        sliding_window_size=sliding_window_size,
    )
    scores = torch.matmul(query.float(), _repeat_kv_heads(key, query.shape[1]).transpose(-2, -1).float())
    scores *= query.shape[-1] ** -0.5 if scale is None else scale
    if is_causal:
        mask = _window_mask(
            query.shape[-2],
            key.shape[-2],
            is_causal=True,
            sliding_window_size=sliding_window_size,
            device=query.device,
        )
        scores = scores.masked_fill(~mask.reshape(1, 1, *mask.shape), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    joint_output = output[..., input_length:, :] if has_joint else output[..., :0, :]
    return output[..., :input_length, :], joint_output, lse


def ring_joint_scaled_dot_product_attention_golden(*args, **kwargs):
    return _ring_joint_golden(*args, **kwargs)


def exp_ring_joint_scaled_dot_product_attention_golden(*args, **kwargs):
    return _ring_joint_golden(*args, **kwargs)


def ring_mla_golden(
    input_tensor_q,
    input_tensor_kv,
    *,
    head_dim_v,
    logical_n,
    scale=None,
    kv_cache_batch_idx=None,
    **_,
):
    import torch

    cache_batch_idx = int(_scalar(kv_cache_batch_idx, 0))
    if input_tensor_kv.shape[0] != input_tensor_q.shape[0]:
        input_tensor_kv = input_tensor_kv[cache_batch_idx : cache_batch_idx + 1]
    logical_n = int(_scalar(logical_n, input_tensor_kv.shape[-2]))
    key = input_tensor_kv[..., :logical_n, :]
    value = key[..., : int(head_dim_v)]
    output = _scaled_attention(input_tensor_q, key, value, is_causal=True, scale=scale)
    scores = torch.matmul(
        input_tensor_q.float(), _repeat_kv_heads(key, input_tensor_q.shape[1]).transpose(-2, -1).float()
    )
    scores *= input_tensor_q.shape[-1] ** -0.5 if scale is None else scale
    mask = _window_mask(
        input_tensor_q.shape[-2],
        key.shape[-2],
        is_causal=True,
        device=input_tensor_q.device,
    )
    lse = torch.logsumexp(scores.masked_fill(~mask.reshape(1, 1, *mask.shape), float("-inf")), dim=-1, keepdim=True)
    return output, torch.cat([lse, lse], dim=-2)


def ring_distributed_scaled_dot_product_attention_golden(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    ring_size,
    ring_id=None,
    *,
    scale=None,
    page_table=None,
    chunk_start_idx=None,
    **_,
):
    """Return the early/late query chunks assigned to one ring rank."""
    import torch

    if page_table is not None:
        input_tensor_k = _paged_to_contiguous(input_tensor_k, page_table)
        input_tensor_v = _paged_to_contiguous(input_tensor_v, page_table)
    query_start = int(_scalar(chunk_start_idx, 0))
    key_length = min(input_tensor_k.shape[-2], query_start + input_tensor_q.shape[-2])
    output = _scaled_attention(
        input_tensor_q,
        input_tensor_k[..., :key_length, :],
        input_tensor_v[..., :key_length, :],
        is_causal=True,
        scale=scale,
        query_start=query_start,
    )
    ring_id = int(_scalar(ring_id, 0))
    chunk_size = output.shape[-2] // (2 * int(ring_size))
    first_chunk = output[..., ring_id * chunk_size : (ring_id + 1) * chunk_size, :]
    second_chunk_id = 2 * int(ring_size) - 1 - ring_id
    second_chunk = output[..., second_chunk_id * chunk_size : (second_chunk_id + 1) * chunk_size, :]
    return torch.cat([first_chunk, second_chunk], dim=-2)


def l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization along a given dimension."""
    import torch

    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_gated_delta_rule(
    q,
    k,
    v,
    beta,
    g,
    scale: Optional[float] = None,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
):
    """Token-by-token recurrent gated delta rule."""
    import torch

    if use_qk_l2norm:
        q = l2_norm(q, dim=-1)
        k = l2_norm(k, dim=-1)
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]
    B, H, T, K = k.shape
    V = v.shape[-1]
    scale = K**-0.5 if scale is None else scale
    q = q * scale
    output = torch.zeros(B, H, T, V, device=v.device, dtype=v.dtype)
    state = torch.zeros(B, H, K, V, device=v.device, dtype=v.dtype)
    if initial_state is not None:
        state = initial_state.to(torch.float32)
    for index in range(T):
        state = state.clone() * g[:, :, index].exp()[..., None, None]
        delta = v[:, :, index].clone() - (state.clone() * k[:, :, index, :, None]).sum(-2)
        delta = delta * beta[:, :, index, None]
        state = state.clone() + k[:, :, index].unsqueeze(-1) * delta.unsqueeze(-2)
        output[:, :, index] = torch.einsum("bhd,bhdm->bhm", q[:, :, index], state)
    return output.transpose(1, 2).contiguous(), state if output_final_state else None


def chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    chunk_size: int = 64,
    scale: Optional[float] = None,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
):
    """Chunked gated delta rule migrated from the FLA-derived model reference."""
    import torch

    if use_qk_l2norm:
        q = l2_norm(q, dim=-1)
        k = l2_norm(k, dim=-1)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]
    T = q.shape[-2]
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad_len))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
        beta = torch.nn.functional.pad(beta, (0, pad_len))
        g = torch.nn.functional.pad(g, (0, pad_len))

    B, H, L, K = q.shape
    V = v.shape[-1]
    q = q * scale
    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)

    def to_chunks(x):
        return x.reshape(B, H, -1, chunk_size, x.shape[-1])

    q_c, k_c, v_c = to_chunks(q), to_chunks(k), to_chunks(v)
    k_beta_c, v_beta_c = to_chunks(k_beta), to_chunks(v_beta)
    g_c = g.reshape(B, H, -1, chunk_size)
    decay = g_c.cumsum(dim=-1)
    decay_exp = decay.exp()[..., None]
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()
    attn = -((k_beta_c @ k_c.transpose(-1, -2)) * L_mask).masked_fill(mask_upper, 0)
    for index in range(1, chunk_size):
        attn[..., index, :index] = attn[..., index, :index].clone() + (
            attn[..., index, :index, None].clone() * attn[..., :index, :index].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    v_corrected = attn @ v_beta_c
    k_cumdecay = attn @ (k_beta_c * decay_exp)

    state = torch.zeros(B, H, K, V, device=q.device, dtype=q.dtype)
    if initial_state is not None:
        state = initial_state.to(torch.float32)
    num_chunks = L // chunk_size
    output = torch.zeros_like(v_corrected)
    mask_causal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for index in range(num_chunks):
        q_i, k_i = q_c[:, :, index], k_c[:, :, index]
        v_i = v_corrected[:, :, index]
        intra_attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, index]).masked_fill_(mask_causal, 0)
        v_new = v_i - k_cumdecay[:, :, index] @ state
        o_inter = (q_i * decay[:, :, index, :, None].exp()) @ state
        output[:, :, index] = o_inter + intra_attn @ v_new
        state = (
            state * decay[:, :, index, -1, None, None].exp()
            + (k_i * (decay[:, :, index, -1, None] - decay[:, :, index]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    output = output.reshape(B, H, -1, V)[:, :, :T].transpose(1, 2).contiguous()
    return output, state if output_final_state else None


def chunk_gated_delta_rule_golden(
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
    num_value_heads = v.shape[2]
    if q.shape[2] != num_value_heads:
        q = _repeat_kv_heads(q.transpose(1, 2), num_value_heads).transpose(1, 2)
        k = _repeat_kv_heads(k.transpose(1, 2), num_value_heads).transpose(1, 2)
    output, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=chunk_size,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
    )
    if output_head_major:
        output = output.permute(0, 2, 1, 3).reshape(-1, output.shape[1], output.shape[-1])
    return output, final_state


def gated_delta_attn_seq_golden(
    L_unit,
    v_beta_sc,
    k_bd_sc,
    intra_attn,
    q_decay,
    k_decay_t,
    dl_exp,
    L_inv,
    *,
    initial_state=None,
    **_,
):
    """Reference for the low-level blocked forward-substitution/state scan.

    ``L_inv`` is supplied to the device kernel as four diagonal block inverses.
    Mathematically the two forward-substitution passes are equivalent to
    ``solve_triangular(L_unit, rhs)`` and are expressed that way here.
    """
    import torch

    del L_inv
    batch_heads, num_chunks, _, key_dim = q_decay.shape
    value_dim = v_beta_sc.shape[-1]
    state = torch.zeros(batch_heads, key_dim, value_dim, dtype=q_decay.dtype, device=q_decay.device)
    if initial_state is not None:
        state = initial_state.float()
    outputs = []
    for chunk in range(num_chunks):
        lower = L_unit[:, chunk].float()
        v_cor = torch.linalg.solve_triangular(lower, v_beta_sc[:, chunk].float(), upper=False, unitriangular=False)
        k_cum = torch.linalg.solve_triangular(lower, k_bd_sc[:, chunk].float(), upper=False, unitriangular=False)
        v_new = v_cor - k_cum @ state
        output = q_decay[:, chunk].float() @ state + intra_attn[:, chunk].float() @ v_new
        state = state * dl_exp[:, chunk].float() + k_decay_t[:, chunk].float() @ v_new
        outputs.append(output)
    return torch.stack(outputs, dim=1), state


__all__ = [
    "BLK_KV",
    "MASKED_INDEX",
    "SENTINEL",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_golden",
    "chunked_flash_mla_prefill_golden",
    "chunked_scaled_dot_product_attention_golden",
    "exp_ring_joint_scaled_dot_product_attention_golden",
    "flash_mla_prefill_golden",
    "flash_multi_latent_attention_decode_golden",
    "gated_delta_attn_seq_golden",
    "joint_scaled_dot_product_attention_golden",
    "l2_norm",
    "paged_flash_multi_latent_attention_decode_golden",
    "paged_scaled_dot_product_attention_decode_golden",
    "recurrent_gated_delta_rule",
    "ring_distributed_scaled_dot_product_attention_golden",
    "ring_joint_scaled_dot_product_attention_golden",
    "ring_mla_golden",
    "scaled_dot_product_attention_decode_golden",
    "scaled_dot_product_attention_golden",
    "scaled_dot_product_attention_reference",
    "scaled_dot_product_attention_reference_prefill",
    "sparse_attention_ref_msa",
    "sparse_mla",
    "sparse_sdpa_golden",
    "sparse_sdpa_msa_golden",
    "torch_sdpa",
    "torch_sdpa_reference",
]
