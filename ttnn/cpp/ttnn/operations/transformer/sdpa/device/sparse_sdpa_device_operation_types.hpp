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

// Sparse MLA prefill (DeepSeek DSA).
struct SparseSDPAParams {
    float scale = 1.0f;  // compile-time (folded into the program hash)
    uint32_t v_dim;      // width of V (= leading v_dim cols of the K_DIM-wide KV cache); the output width
    uint32_t k_chunk_size = 128;
    DeviceComputeKernelConfig compute_kernel_config;
    // Indexed KV cache: when set, kv is a [B,1,T,K_DIM] shared cache and this selects the batch slot to
    // attend to (the gather page ids are offset by cache_batch_idx * T). It is a DYNAMIC runtime arg
    // (excluded from the program hash, re-applied every dispatch), so changing it does NOT recompile.
    std::optional<uint32_t> cache_batch_idx = std::nullopt;
    bool has_indexed_kv_cache() const { return cache_batch_idx.has_value(); }
    // The remap configuration is compile-time; T is part of the program hash for this path.
    std::optional<BlockCyclicLayout> block_cyclic = std::nullopt;
    bool has_block_cyclic() const { return block_cyclic.has_value(); }

    SparseSDPAParams(
        float scale,
        uint32_t v_dim,
        uint32_t k_chunk_size,
        DeviceComputeKernelConfig compute_kernel_config,
        std::optional<uint32_t> cache_batch_idx = std::nullopt,
        std::optional<BlockCyclicLayout> block_cyclic = std::nullopt) :
        scale(scale),
        v_dim(v_dim),
        k_chunk_size(k_chunk_size),
        compute_kernel_config(compute_kernel_config),
        cache_batch_idx(cache_batch_idx),
        block_cyclic(block_cyclic) {}

    static constexpr auto attribute_names = std::forward_as_tuple(
        "scale_bits",
        "v_dim",
        "k_chunk_size",
        "compute_kernel_config",
        "has_indexed_kv_cache",
        "has_block_cyclic",
        "block_cyclic_sp",
        "block_cyclic_chunk_local");
    auto attribute_values() const {
        return std::make_tuple(
            std::bit_cast<uint32_t>(scale),
            v_dim,
            k_chunk_size,
            std::cref(compute_kernel_config),
            has_indexed_kv_cache(),
            has_block_cyclic(),
            block_cyclic.has_value() ? block_cyclic->sp : 0u,
            block_cyclic.has_value() ? block_cyclic->chunk_local : 0u);
    }
};

struct SparseSDPAInputs {
    Tensor q;   // [1, H, S, K_DIM] bf16/fp8_e4m3 ROW_MAJOR  (K_DIM = head dim, e.g. 576)
    Tensor kv;  // [1, 1, T, K_DIM] bf16/fp8_e4m3 ROW_MAJOR  (or [B,1,T,K_DIM] when indexed; may be ND-sharded DRAM)
    Tensor indices;  // [1, 1, S, TOPK] uint32 ROW_MAJOR  (0xFFFFFFFF = masked)
    // Mirrors attrs.has_block_cyclic() for the kv shape hash sentinel (set at launch).
    bool block_cyclic_for_hash = false;

    SparseSDPAInputs(Tensor q_in, Tensor kv_in, Tensor indices_in, bool block_cyclic_for_hash_in = false) :
        q(std::move(q_in)),
        kv(std::move(kv_in)),
        indices(std::move(indices_in)),
        block_cyclic_for_hash(block_cyclic_for_hash_in) {}

    static constexpr auto attribute_names = std::forward_as_tuple(
        "q_logical_shape",
        "q_dtype",
        "kv_dtype",
        "kv_memory_config",
        "kv_logical_shape_if_sharded_or_block_cyclic",
        "indices_logical_shape",
        "indices_dtype");
    auto attribute_values() const {
        return std::make_tuple(
            q.logical_shape(),
            q.dtype(),
            kv.dtype(),
            std::cref(kv.memory_config()),
            (kv.memory_config().is_sharded() || block_cyclic_for_hash) ? kv.logical_shape() : tt::tt_metal::Shape{},
            indices.logical_shape(),
            indices.dtype());
    }
};

}  // namespace ttnn::prim
