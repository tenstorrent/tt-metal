// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_recipe_dense.hpp"

#include "sdpa_recipe_blocking.hpp"

namespace ttnn::operations::transformer::sdpa::detail {
using namespace tt::tt_metal;
using ttnn::prim::SDPAInputs;
using ttnn::prim::SDPAParams;

DenseRecipeCall dense_recipe_call(const SDPAParams& attrs, const SDPAInputs& tensors) {
    const bool chunked = tensors.page_table.has_value();
    const bool windowed = tensors.cu_window_seqlens.has_value();
    // MLA without a V tensor reads V as K's first head_dim_v columns.
    const Tensor& v = tensors.v.value_or(tensors.k);
    DenseRecipeCall call{.segments = {{tensors.q, tensors.k, v}}};
    // Chunked prefill is causal over the paged cache, its Q at chunk_start_idx (scalar or device tensor); windowed
    // calls offset Q by windowed_q_token_offset (scalar or device tensor).
    call.key_range = RecipeKeyRange{
        .causal = attrs.is_causal || chunked,
        .sliding_window = attrs.sliding_window_size.value_or(0),
        .q_offset = chunked    ? static_cast<uint32_t>(attrs.chunk_start_idx.value_or(0))
                    : windowed ? attrs.windowed_q_token_offset
                               : 0u,
        .q_offset_tensor = chunked ? tensors.chunk_start_idx_tensor : tensors.windowed_q_token_offset_tensor,
        .segments = tensors.cu_window_seqlens,
        .page_table = tensors.page_table,
        .paged_geometry = attrs.paged_cache_geometry};
    call.options = RecipeDenseOptions{
        .head_dim_v = attrs.use_mla ? attrs.head_dim_v.value_or(0) : 0,
        .attention_sink = tensors.attention_sink,
        .output_concat_heads = attrs.output_concat_heads};
    return call;
}

PrecisionPolicy dense_recipe_policy(const SDPAParams& attrs, const SDPAInputs& tensors) {
    return resolve_recipe_policy(
        tensors.q,
        tensors.k,
        attrs.precision.value_or(SDPAPrecision::STANDARD),
        attrs.scale,
        std::nullopt,
        attrs.program_config);
}

namespace {
// The SDPA operation's argument rules for the recipe path (recipe_program checks shapes, types and layouts).
void validate_dense_recipe(const SDPAParams& attrs, const SDPAInputs& tensors, const DenseRecipeCall& call) {
    const bool windowed = tensors.cu_window_seqlens.has_value();
    const bool chunked = tensors.page_table.has_value();
    const uint32_t window = attrs.sliding_window_size.value_or(0);
    const uint32_t sq = tensors.q.logical_shape()[2], sk = tensors.k.logical_shape()[2];
    TT_FATAL(
        windowed || (attrs.windowed_q_token_offset == 0 && !tensors.windowed_q_token_offset_tensor),
        "windowed_q_token_offset requires cu_window_seqlens");
    TT_FATAL(
        !tensors.attn_mask || (!attrs.is_causal && window == 0 && !windowed && !chunked),
        "SDPA recipes take either attn_mask or is_causal / sliding_window_size / cu_window_seqlens / chunked prefill");
    // A block map only lets a reader skip fully masked blocks; the recipes apply the whole mask.
    TT_FATAL(!tensors.attn_mask_block_map || tensors.attn_mask, "attn_mask_block_map requires attn_mask");
    TT_FATAL(
        !attrs.output_concat_heads || !chunked, "output_concat_heads is not supported with chunked prefill");
    if (windowed) {
        TT_FATAL(!chunked, "Windowed SDPA does not support chunked prefill");
        TT_FATAL(window == 0, "Windowed SDPA does not support sliding_window_size");
        TT_FATAL(!tensors.attention_sink, "Windowed SDPA does not support attention_sink");
        TT_FATAL(sq <= sk, "windowed Q shard has {} rows, more than the K sequence length {}", sq, sk);
        TT_FATAL(
            attrs.windowed_q_token_offset % 32 == 0,
            "windowed_q_token_offset must be a multiple of 32, got {}",
            attrs.windowed_q_token_offset);
        TT_FATAL(
            tensors.windowed_q_token_offset_tensor || attrs.windowed_q_token_offset <= sk - sq,
            "windowed Q shard [{}, {} + {}) does not fit in the K sequence length {}",
            attrs.windowed_q_token_offset,
            attrs.windowed_q_token_offset,
            sq,
            sk);
    } else if (chunked) {
        if (attrs.chunk_start_idx) {
            const uint32_t k_rows = recipe_k_rows(tensors.k, call.key_range);
            TT_FATAL(*attrs.chunk_start_idx >= 0, "chunk_start_idx must be non-negative");
            TT_FATAL(
                k_rows >= sq + *attrs.chunk_start_idx,
                "K's sequence length must be >= Q's sequence length + chunk_start_idx. Got K: {}, Q: {}, "
                "chunk_start_idx: {}",
                k_rows,
                sq,
                *attrs.chunk_start_idx);
        }
    } else if (attrs.is_causal || window > 0) {
        TT_FATAL(
            sq == sk,
            "Causal or sliding-window SDPA requires Q and K to have the same sequence length. Got Q: {}, K: {}",
            sq,
            sk);
    }
}
}  // namespace

void resolve_dense_recipe(SDPAParams& attrs, const SDPAInputs& tensors) {
    const auto call = dense_recipe_call(attrs, tensors);
    validate_dense_recipe(attrs, tensors, call);
    const auto policy = dense_recipe_policy(attrs, tensors);
    if (tensors.attn_mask) {
        validate_recipe_mask(tensors.q, tensors.k, *tensors.attn_mask, policy);
    }
    const auto& q = tensors.q;
    const auto& qs = q.padded_shape();
    const uint32_t out_head_dim = call.options.head_dim_v ? call.options.head_dim_v : qs[3];
    const Tensor& v = call.segments.front()[2];
    attrs.program_config = resolve_dense_recipe_blocking(
        policy,
        q,
        tensors.k,
        nullptr,
        nullptr,
        attrs.program_config,
        tensors.attn_mask ? &*tensors.attn_mask : nullptr,
        recipe_output_l1_bytes(
            q, uint64_t{qs[0]} * qs[1] * (qs[2] / 32) * (out_head_dim / 32), 2048, attrs.output_mem_config),
        &call.key_range,
        &call.options,
        false,
        recipe_kv_storage(v.dtype()));
    attrs.recipe_l1_budget = recipe_l1_budget(
        call.segments,
        policy,
        attrs.program_config,
        tensors.attn_mask,
        attrs.output_mem_config,
        call.key_range,
        call.options);
}

TensorSpec dense_recipe_output_spec(const SDPAParams& attrs, const SDPAInputs& tensors) {
    const auto call = dense_recipe_call(attrs, tensors);
    return recipe_output_spec(
        tensors.q, call.options.head_dim_v, 0, call.options.output_concat_heads, attrs.output_mem_config);
}

ProgramDescriptor dense_recipe_descriptor(const SDPAParams& attrs, const SDPAInputs& tensors, Tensor& output) {
    const auto call = dense_recipe_call(attrs, tensors);
    return recipe_program(
        call.segments,
        dense_recipe_policy(attrs, tensors),
        attrs.program_config,
        tensors.attn_mask,
        attrs.scale,
        attrs.output_mem_config,
        call.key_range,
        call.options,
        {output},
        attrs.recipe_l1_budget);
}

}  // namespace ttnn::operations::transformer::sdpa::detail
