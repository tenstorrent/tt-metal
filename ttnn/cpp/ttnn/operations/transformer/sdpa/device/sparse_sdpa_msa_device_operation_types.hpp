// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "block_cyclic_layout.hpp"  // ttnn::prim::BlockCyclicLayout (shared)
#include <bit>
#include <optional>
#include <tuple>

namespace ttnn::prim {

// Primitive for MSA block-sparse prefill. K/V are separate tiled caches; masking comes only from block ids in
// `indices` plus -1 sentinels. For GQA, each KV group owns H/n_kv query heads.
struct SparseSDPAMsaParams {
    float scale = 1.0f;         // compile-time; included in the program hash
    uint32_t block_size = 128;  // tokens per selected KV block
    DeviceComputeKernelConfig compute_kernel_config;
    // Selects one [B,n_kv,T,*] cache slot. The value is patched as runtime K/V tile offsets and is not hashed.
    std::optional<uint32_t> cache_batch_idx = std::nullopt;
    // Set -> the K/V cache is block-cyclic across SP; the gather kernels apply the invP block remap. Hashed.
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt;
    // Global position of query row 0.
    // Set -> enforce a token-level causal mask on the diagonal block (the query's own block, whose later tokens are
    // future). Unset -> no token-level causality; the op attends the full selected blocks.
    std::optional<uint32_t> chunk_start_idx = std::nullopt;
    // SP mesh axis used to derive the per-device chunk_start (chunk_start_idx + rank*S); host-side only.
    std::optional<uint32_t> cluster_axis = std::nullopt;

    bool has_indexed_kv_cache() const { return cache_batch_idx.has_value(); }
    bool causal_enabled() const { return chunk_start_idx.has_value(); }
    bool has_block_cyclic() const { return block_cyclic.has_value(); }

    // Mirrors the removed compute_program_hash: cache_batch_idx / chunk_start_idx / cluster_axis values are
    // runtime-patched or host-only (excluded), so only their derived booleans are hashed. block_cyclic shapes
    // the gather kernels (invP remap), so its presence and sp/chunk_local are hashed.
    static constexpr auto attribute_names = std::forward_as_tuple(
        "scale_bits",
        "block_size",
        "compute_kernel_config",
        "has_indexed_kv_cache",
        "causal_enabled",
        "has_block_cyclic",
        "block_cyclic_sp",
        "block_cyclic_chunk_local");
    auto attribute_values() const {
        return std::make_tuple(
            std::bit_cast<uint32_t>(scale),
            block_size,
            std::cref(compute_kernel_config),
            has_indexed_kv_cache(),
            causal_enabled(),
            has_block_cyclic(),
            block_cyclic.has_value() ? block_cyclic->sp : 0u,
            block_cyclic.has_value() ? block_cyclic->chunk_local : 0u);
    }
};

struct SparseSDPAMsaInputs {
    Tensor q;        // [1,H,S,d] bf16|fp8_e4m3 ROW_MAJOR
    Tensor k;        // [B,n_kv,T,d] TILE bf16|bfp8_b
    Tensor v;        // [B,n_kv,T,v_dim] TILE bf16|bfp8_b
    Tensor indices;  // [1,n_kv,S,TOPK] uint32 block ids; 0xFFFFFFFF is the sentinel

    // K/V logical shapes are hashed unconditionally: under a block-cyclic cache the shard stride gap is baked
    // into the gather kernels as a compile-time argument derived from T, so a different cache size must be a
    // distinct program even for an interleaved cache. (Params can't observe block_cyclic from here, so we hash
    // K/V T always — stricter than the old interleaved-only sentinel, but never a wrong cache hit.)
    static constexpr auto attribute_names = std::forward_as_tuple(
        "q_logical_shape",
        "q_dtype",
        "k_dtype",
        "k_memory_config",
        "k_logical_shape",
        "v_dtype",
        "v_memory_config",
        "v_logical_shape",
        "v_dim",
        "indices_logical_shape",
        "indices_dtype");
    auto attribute_values() const {
        return std::make_tuple(
            q.logical_shape(),
            q.dtype(),
            k.dtype(),
            std::cref(k.memory_config()),
            k.logical_shape(),
            v.dtype(),
            std::cref(v.memory_config()),
            v.logical_shape(),
            v.logical_shape()[3],
            indices.logical_shape(),
            indices.dtype());
    }
};

}  // namespace ttnn::prim
