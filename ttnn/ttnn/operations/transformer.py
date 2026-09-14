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


def _sdpa_reference(
    q, k, v, *, attn_mask=None, is_causal=False, scale=None, sliding_window_size=None, attention_sink=None
):
    """Shared torch SDPA reference handling GQA, causal/sliding-window masks, scale, and attention sink."""
    import torch

    k, v = _expand_gqa(q, k, v)

    # Sliding-window attention: build an explicit causal+window mask (torch SDPA has no window flag).
    if sliding_window_size and is_causal and attn_mask is None:
        q_len, k_len = q.shape[-2], k.shape[-2]
        offset = k_len - q_len
        qi = torch.arange(q_len).view(-1, 1) + offset
        kj = torch.arange(k_len).view(1, -1)
        allowed = (kj <= qi) & (kj > qi - int(sliding_window_size))
        attn_mask = torch.where(allowed, 0.0, float("-inf"))
        is_causal = False

    if attention_sink is None:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
        )

    # Attention sink: an extra per-head logit that contributes to the softmax denominator only.
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if attn_mask is not None:
        logits = logits + attn_mask
    sink = attention_sink.float().reshape(-1)[: logits.shape[1]] * scale
    sink = sink.reshape(1, -1, 1, 1).expand(logits.shape[0], logits.shape[1], logits.shape[-2], 1)
    full_logits = torch.cat([sink, logits], dim=-1)
    probs = torch.softmax(full_logits, dim=-1)[..., 1:]
    return torch.matmul(probs, v.float()).to(q.dtype)


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
    **_,
):
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
    import torch

    # Decode attends each query to cache positions [0, cur_pos]; mask out everything past it.
    k_len = input_tensor_k.shape[-2]
    positions = _resolve_cur_pos(cur_pos, cur_pos_tensor, input_tensor_q.shape[0], k_len)
    mask = torch.zeros((input_tensor_q.shape[0], 1, 1, k_len))
    for b, pos in enumerate(positions):
        mask[b, :, :, pos + 1 :] = float("-inf")
    return _sdpa_reference(
        input_tensor_q,
        input_tensor_k,
        input_tensor_v,
        attn_mask=mask,
        is_causal=False,
        scale=scale,
        sliding_window_size=sliding_window_size,
        attention_sink=attention_sink,
    )


ttnn.attach_golden_function(
    ttnn.transformer.scaled_dot_product_attention_decode, golden_function=_golden_function_sdpa_decode
)


def _gather_paged_kv(cache, page_table, positions):
    """Gather per-batch KV from a paged cache. cache: [num_blocks, block_size, nkv, d]; page_table: [B, max_blocks]."""
    import torch

    block_size = cache.shape[1]
    gathered_k = []
    for b in range(page_table.shape[0]):
        blocks = page_table[b].flatten().tolist()
        seq = torch.cat([cache[int(blk)] for blk in blocks], dim=0)  # [num_blocks*block_size, nkv, d]
        gathered_k.append(seq[: positions[b] + 1])
    return gathered_k


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
    **_,
):
    import torch

    # Gather each batch's KV pages, then run the decode reference per batch.
    k_len = input_tensor_k.shape[1] * page_table_tensor.shape[-1]
    positions = _resolve_cur_pos(None, cur_pos_tensor, input_tensor_q.shape[0], k_len)
    k_pages = _gather_paged_kv(input_tensor_k, page_table_tensor, positions)
    v_pages = _gather_paged_kv(input_tensor_v, page_table_tensor, positions)
    outputs = []
    for b in range(input_tensor_q.shape[0]):
        kb = k_pages[b].permute(1, 0, 2).unsqueeze(0)  # [1, nkv, len, d]
        vb = v_pages[b].permute(1, 0, 2).unsqueeze(0)
        qb = input_tensor_q[b : b + 1]
        outputs.append(_sdpa_reference(qb, kb, vb, is_causal=False, scale=scale, attention_sink=attention_sink))
    return torch.cat(outputs, dim=0)


ttnn.attach_golden_function(
    ttnn.transformer.paged_scaled_dot_product_attention_decode, golden_function=_golden_function_paged_sdpa_decode
)


def _golden_function_chunked_sdpa(
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    page_table_tensor,
    *_,
    scale=None,
    **__,
):
    # Chunked prefill computes full causal attention over the gathered prefix; chunking is a perf detail.
    return _golden_function_paged_sdpa_decode(
        input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, is_causal=True, scale=scale
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

    # Joint attention concatenates the original and joint sequences along the sequence dim, runs SDPA once,
    # then splits the output back into the original and joint parts.
    q = torch.cat([input_tensor_q, joint_tensor_q], dim=-2)
    k = torch.cat([input_tensor_k, joint_tensor_k], dim=-2)
    v = torch.cat([input_tensor_v, joint_tensor_v], dim=-2)
    output = _sdpa_reference(q, k, v, is_causal=False, scale=scale)
    n = input_tensor_q.shape[-2]
    return [output[..., :n, :], output[..., n:, :]]


ttnn.attach_golden_function(
    ttnn.transformer.joint_scaled_dot_product_attention, golden_function=_golden_function_joint_sdpa
)


# Ring/joint/distributed variants all compute the same full-attention reference; the ring/distributed
# decomposition is a load-balancing detail that does not change the math.
ttnn.attach_golden_function(
    ttnn.transformer.ring_joint_scaled_dot_product_attention, golden_function=_golden_function_joint_sdpa
)
ttnn.attach_golden_function(
    ttnn.transformer.exp_ring_joint_scaled_dot_product_attention, golden_function=_golden_function_joint_sdpa
)


def _golden_function_ring_distributed_sdpa(
    input_tensor_q, input_tensor_k, input_tensor_v, *, attn_mask=None, is_causal=True, scale=None, **_
):
    return _sdpa_reference(
        input_tensor_q, input_tensor_k, input_tensor_v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )


ttnn.attach_golden_function(
    ttnn.transformer.ring_distributed_scaled_dot_product_attention,
    golden_function=_golden_function_ring_distributed_sdpa,
)


def _golden_function_sparse_sdpa(q, kv, indices, v_dim, *, scale=None, attention_sink=None, **_):
    import torch

    # Sparse MLA: softmax(Q @ K^T * scale, masked to top-k selected latents) @ V, V = kv[..., :v_dim].
    # indices holds natural token positions per query; 0xFFFFFFFF marks masked sentinel entries.
    k = kv  # [1, 1, T, K_DIM]
    v = kv[..., :v_dim]
    if scale is None:
        scale = 1.0 / (kv.shape[-1] ** 0.5)
    s = q.shape[-2]
    topk = indices.shape[-1]
    # Gather the selected latents per query token.
    flat_idx = indices.reshape(s, topk).long()
    valid = flat_idx != 0xFFFFFFFF
    clamped = flat_idx.clamp(min=0, max=k.shape[-2] - 1)
    k_sel = k[0, 0][clamped]  # [S, TOPK, K_DIM]
    v_sel = v[0, 0][clamped][..., :v_dim]  # [S, TOPK, v_dim]
    logits = torch.einsum("hse,ste->hst", q[0].float(), k_sel.float()) * scale
    logits = logits.masked_fill(~valid.transpose(0, 1).unsqueeze(0), float("-inf"))
    if attention_sink is not None:
        sink = attention_sink.float().reshape(-1)[: logits.shape[0]] * scale
        logits = torch.cat([sink.reshape(-1, 1, 1).expand(logits.shape[0], s, 1), logits], dim=-1)
        probs = torch.softmax(logits, dim=-1)[..., 1:]
    else:
        probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("hst,ste->hse", probs, v_sel.float())
    return out.unsqueeze(0).to(q.dtype)


ttnn.attach_golden_function(ttnn.transformer.sparse_sdpa, golden_function=_golden_function_sparse_sdpa)
ttnn.attach_golden_function(ttnn.transformer.sparse_sdpa_msa, golden_function=_golden_function_sparse_sdpa)


def _resolve_mla_v(input_tensor_k, args, kwargs):
    """MLA derives V as a leading slice of the latent K cache, unless an explicit embedding-space V is given."""
    import torch

    explicit_v = kwargs.get("input_tensor_v")
    if explicit_v is not None:
        return explicit_v
    # An explicit positional V tensor takes priority; otherwise locate the head_dim_v integer.
    for arg in args:
        if torch.is_tensor(arg):
            return arg
    head_dim_v = kwargs.get("head_dim_v")
    if head_dim_v is None:
        head_dim_v = next((arg for arg in args if isinstance(arg, int)), None)
    return input_tensor_k[..., :head_dim_v]


def _golden_function_flash_mla_prefill(
    input_tensor_q, input_tensor_k, *args, attn_mask=None, is_causal=True, scale=None, **kwargs
):
    v = _resolve_mla_v(input_tensor_k, args, kwargs)
    return _sdpa_reference(input_tensor_q, input_tensor_k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale)


ttnn.attach_golden_function(ttnn.transformer.flash_mla_prefill, golden_function=_golden_function_flash_mla_prefill)
# ring_mla computes the same full prefill attention reference; the ring decomposition is a perf detail.
ttnn.attach_golden_function(ttnn.transformer.ring_mla, golden_function=_golden_function_flash_mla_prefill)


def _golden_function_flash_mla_decode(
    input_tensor_q,
    input_tensor_k,
    *args,
    is_causal=True,
    attn_mask=None,
    cur_pos=None,
    cur_pos_tensor=None,
    attention_sink=None,
    scale=None,
    **kwargs,
):
    import torch

    v = _resolve_mla_v(input_tensor_k, args, kwargs)
    k_len = input_tensor_k.shape[-2]
    positions = _resolve_cur_pos(cur_pos, cur_pos_tensor, input_tensor_q.shape[0], k_len)
    mask = torch.zeros((input_tensor_q.shape[0], 1, 1, k_len))
    for b, pos in enumerate(positions):
        mask[b, :, :, pos + 1 :] = float("-inf")
    return _sdpa_reference(
        input_tensor_q, input_tensor_k, v, attn_mask=mask, is_causal=False, scale=scale, attention_sink=attention_sink
    )


ttnn.attach_golden_function(
    ttnn.transformer.flash_multi_latent_attention_decode, golden_function=_golden_function_flash_mla_decode
)


def _golden_function_paged_flash_mla_decode(
    input_tensor_q,
    input_tensor_k,
    *args,
    cur_pos_tensor=None,
    attention_sink=None,
    scale=None,
    **kwargs,
):
    import torch

    # Signature: (q, k, input_tensor_v=None, head_dim_v, page_table_tensor, ...); v and head_dim_v may be
    # positional or keyword, so discriminate by type.
    tensors = [a for a in args if torch.is_tensor(a)]
    ints = [a for a in args if isinstance(a, int)]
    input_tensor_v = kwargs.get("input_tensor_v", tensors[0] if len(tensors) > 1 else None)
    head_dim_v = kwargs.get("head_dim_v", ints[0] if ints else None)
    page_table_tensor = kwargs.get("page_table_tensor", tensors[-1] if tensors else None)

    k_len = input_tensor_k.shape[1] * page_table_tensor.shape[-1]
    positions = _resolve_cur_pos(None, cur_pos_tensor, input_tensor_q.shape[0], k_len)
    k_pages = _gather_paged_kv(input_tensor_k, page_table_tensor, positions)
    if input_tensor_v is not None:
        v_pages = _gather_paged_kv(input_tensor_v, page_table_tensor, positions)
    else:
        v_pages = [kp[..., :head_dim_v] for kp in k_pages]
    outputs = []
    for b in range(input_tensor_q.shape[0]):
        kb = k_pages[b].permute(1, 0, 2).unsqueeze(0)
        vb = v_pages[b].permute(1, 0, 2).unsqueeze(0)
        outputs.append(
            _sdpa_reference(
                input_tensor_q[b : b + 1], kb, vb, is_causal=False, scale=scale, attention_sink=attention_sink
            )
        )
    return torch.cat(outputs, dim=0)


ttnn.attach_golden_function(
    ttnn.transformer.paged_flash_multi_latent_attention_decode, golden_function=_golden_function_paged_flash_mla_decode
)


def _golden_function_chunked_flash_mla_prefill(
    input_tensor_q, input_tensor_k, head_dim_v, page_table_tensor, *args, scale=None, **kwargs
):
    # Chunked MLA prefill computes full causal attention over the gathered prefix; chunking is a perf detail.
    v = input_tensor_k[..., :head_dim_v]
    return _golden_function_paged_sdpa_decode(
        input_tensor_q, input_tensor_k, v, page_table_tensor, is_causal=True, scale=scale
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
    use_qk_l2norm=False,
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
    if output_final_state:
        return [o, final_state]
    return o


ttnn.attach_golden_function(ttnn.transformer.chunk_gated_delta_rule, golden_function=_golden_function_gated_delta_rule)
ttnn.attach_golden_function(ttnn.transformer.gated_delta_attn_seq, golden_function=_golden_function_gated_delta_rule)


__all__ = []
