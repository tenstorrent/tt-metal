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
    validate_recipe_grid(attrs.program_config);
    {
        const uint32_t nqh = tensors.q.logical_shape()[1];
        // A paged cache viewed with another layer's geometry (paged_cache_geometry) has its own KV head count.
        const uint32_t nkv = chunked && attrs.paged_cache_geometry.num_kv_heads != 0
                                 ? attrs.paged_cache_geometry.num_kv_heads
                                 : tensors.k.logical_shape()[1];
        TT_FATAL(
            nkv > 0 && nqh >= nkv && nqh % nkv == 0,
            "Q num_heads must be >= K num_heads and divisible by K num_heads, and K num_heads must be greater than 0. "
            "Got Q: {}, K: {}",
            nqh,
            nkv);
    }
    TT_FATAL(
        windowed || (attrs.windowed_q_token_offset == 0 && !tensors.windowed_q_token_offset_tensor),
        "windowed_q_token_offset requires cu_window_seqlens");
    TT_FATAL(
        !tensors.attn_mask || (!attrs.is_causal && window == 0 && !windowed && !chunked),
        "SDPA recipes take either attn_mask or is_causal / sliding_window_size / cu_window_seqlens / chunked prefill");
    // A block map only lets a reader skip fully masked blocks; the recipes apply the whole mask (main's argument
    // rules for the map are kept, so callers see the same errors).
    if (tensors.attn_mask_block_map) {
        const auto& map = *tensors.attn_mask_block_map;
        TT_FATAL(
            !windowed && !chunked && window == 0,
            "attn_mask_block_map cannot be combined with cu_window_seqlens, sliding_window_size or chunked prefill");
        TT_FATAL(tensors.attn_mask.has_value(), "attn_mask_block_map needs attn_mask");
        TT_FATAL(attrs.program_config.has_value(), "attn_mask_block_map needs a program_config with the chunk sizes");
        TT_FATAL(
            map.storage_type() == StorageType::DEVICE && map.device() == tensors.q.device(),
            "attn_mask_block_map must be on the same device as Q");
        TT_FATAL(
            map.layout() == Layout::ROW_MAJOR && (map.dtype() == DataType::INT32 || map.dtype() == DataType::UINT32),
            "attn_mask_block_map must be a ROW_MAJOR int32 or uint32 tensor");
        TT_FATAL(map.memory_config().buffer_type() == BufferType::DRAM, "attn_mask_block_map must be in DRAM");
        const auto& mask_shape = tensors.attn_mask->logical_shape();
        const auto& map_shape = map.logical_shape();
        const uint32_t qc = attrs.program_config->q_chunk_size;
        const uint32_t kc = attrs.program_config->k_chunk_size;
        TT_FATAL(qc > 0 && kc > 0, "q_chunk_size ({}) and k_chunk_size ({}) must be greater than 0", qc, kc);
        const uint32_t nq = (sq + qc - 1) / qc;
        const uint32_t nk = (sk + kc - 1) / kc;
        TT_FATAL(
            map_shape.rank() == 4 && map_shape[0] == mask_shape[0] && map_shape[1] == mask_shape[1] &&
                map_shape[2] == nq && map_shape[3] == nk,
            "attn_mask_block_map must be [{}, {}, {}, {}], got {}",
            mask_shape[0],
            mask_shape[1],
            nq,
            nk,
            map_shape);
    }
    TT_FATAL(
        !attrs.output_concat_heads || !chunked, "output_concat_heads is not supported with chunked prefill");
    if (attrs.use_mla && tensors.v) {
        // Multi-latent attention with its own V: each Q head reads V head h / (H / Hv).
        const uint32_t nqh = tensors.q.logical_shape()[1], nvh = tensors.v->logical_shape()[1];
        TT_FATAL(
            nvh > 0 && nqh >= nvh && nqh % nvh == 0,
            "Q num_heads must be >= V num_heads and divisible by V num_heads, and V num_heads must be greater than 0. "
            "Got Q: {}, V: {}",
            nqh,
            nvh);
    } else if (tensors.v) {
        TT_FATAL(
            tensors.v->logical_shape()[1] == tensors.k.logical_shape()[1],
            "K and V num_heads must match. Got K: {}, V: {}",
            tensors.k.logical_shape()[1],
            tensors.v->logical_shape()[1]);
    }
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

namespace {
RecipeResolution resolve_dense_recipe_uncached(const SDPAParams& attrs, const SDPAInputs& tensors) {
    const auto call = dense_recipe_call(attrs, tensors);
    validate_dense_recipe(attrs, tensors, call);
    TT_FATAL(
        attrs.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "SDPA recipes require an interleaved (DRAM or L1) output memory config");
    const auto policy = dense_recipe_policy(attrs, tensors);
    if (tensors.attn_mask) {
        validate_recipe_mask(tensors.q, tensors.k, *tensors.attn_mask, policy);
    }
    const auto& q = tensors.q;
    const auto& qs = q.padded_shape();
    const uint32_t out_head_dim = call.options.head_dim_v ? call.options.head_dim_v : qs[3];
    const Tensor& v = call.segments.front()[2];
    RecipeResolution resolution;
    resolution.program_config = attrs.program_config;
    if (resolution.program_config &&
        (resolution.program_config->q_chunk_size > 1024 || resolution.program_config->q_chunk_size % 32 != 0 ||
         resolution.program_config->k_chunk_size % 32 != 0)) {
        // Chunk sizes the recipe kernels cannot run (tuned for the legacy kernels) are left to the op.
        resolution.program_config->q_chunk_size = 0;
        resolution.program_config->k_chunk_size = 0;
    }
    auto choose = [&] {
        resolution.program_config = resolve_dense_recipe_blocking(
            policy,
            q,
            tensors.k,
            nullptr,
            nullptr,
            resolution.program_config,
            tensors.attn_mask ? &*tensors.attn_mask : nullptr,
            recipe_output_l1_bytes(
                q, uint64_t{qs[0]} * qs[1] * (qs[2] / 32) * (out_head_dim / 32), 2048, attrs.output_mem_config),
            &call.key_range,
            &call.options,
            false,
            recipe_kv_storage(v.dtype()));
    };
    choose();
    if (resolution.program_config && !recipe_chunks_fit(
                                         call.segments,
                                         policy,
                                         resolution.program_config,
                                         tensors.attn_mask,
                                         attrs.output_mem_config,
                                         call.key_range,
                                         call.options)) {
        // The caller's chunks (tuned for the legacy kernels) do not fit the free L1 in any recipe layout: the op
        // chooses them on the caller's grid.
        resolution.program_config->q_chunk_size = 0;
        resolution.program_config->k_chunk_size = 0;
        choose();
    }
    resolution.l1_budget = recipe_l1_budget(
        call.segments,
        policy,
        resolution.program_config,
        tensors.attn_mask,
        attrs.output_mem_config,
        call.key_range,
        call.options);
    return resolution;
}
}  // namespace

void resolve_dense_recipe(SDPAParams& attrs, const SDPAInputs& tensors) {
    TT_FATAL(
        attrs.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "SDPA recipes require an interleaved (DRAM or L1) output memory config");
    const auto resolution = memoized_recipe_resolution(recipe_resolution_key(tensors.q, attrs, tensors), [&] {
        return resolve_dense_recipe_uncached(attrs, tensors);
    });
    attrs.program_config = resolution.program_config;
    attrs.recipe_l1_budget = resolution.l1_budget;
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
