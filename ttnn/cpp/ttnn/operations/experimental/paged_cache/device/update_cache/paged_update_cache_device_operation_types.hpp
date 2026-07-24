// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct PagedUpdateCacheParams {
    const std::vector<uint32_t> update_idxs;
    const uint32_t batch_offset;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
    const bool share_cache;
    const std::optional<std::set<ttnn::MeshCoordinate>> mesh_coords;
    // Optional per-call block_size, overriding cache.padded_shape[2]. Lets a single
    // physical buffer be addressed with different (block_size, head_dim) views as long
    // as num_kv_heads * block_size * head_dim is preserved (checked in
    // validate_on_program_cache_miss). Used by vLLM's hybrid kv-cache-groups path.
    const std::optional<uint32_t> block_size_override;
    // Optional per-call num_kv_heads, overriding cache.padded_shape[1]. Companion to
    // block_size_override for HMA cross-group sharing where sliding and full layers
    // have asymmetric kv-head counts (e.g. Gemma4-26B-A4B sliding kv=8 / full kv=2).
    // The decode-time input is height-sharded with the kv-heads dim padded to
    // TILE_HEIGHT, so the kernel can't infer the logical kv-head count from the input
    // tensor — the caller must pass it explicitly. The per-block element-count
    // invariant input_num_heads * effective_block_size * input_head_dim ==
    // cache_num_heads * cache_block_size * cache_head_dim is enforced in
    // validate_on_program_cache_miss.
    const std::optional<uint32_t> num_kv_heads_override;
    // Optional circular-buffer capacity (in tokens) for the cache view. When set, the
    // kernel computes ``update_idx %= cache_position_modulo`` before resolving the
    // page_table entry, so a bounded sliding-window cache of capacity N can be
    // indexed by absolute positions ≥ N without falling off the page_table or
    // collapsing to physical block 0. Required when paired with vLLM's
    // SlidingWindowSpec, which sizes the per-layer page_table to
    // sliding_window/block_size entries and zero-pads the rest. Must be a multiple of
    // (effective) block_size — otherwise a wrapped position would split across blocks
    // and the kernel can't address it. Paged-mode only (validated in
    // validate_on_program_cache_miss).
    const std::optional<uint32_t> cache_position_modulo;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "compute_kernel_config",
        "share_cache",
        "mesh_coords",
        "block_size_override",
        "num_kv_heads_override",
        "cache_position_modulo");
    auto attribute_values() const {
        return std::make_tuple(
            std::cref(compute_kernel_config),
            share_cache,
            mesh_coords,
            block_size_override,
            num_kv_heads_override,
            cache_position_modulo);
    }
};

struct PagedUpdateCacheInputs {
    Tensor cache_tensor;
    Tensor input_tensor;
    std::optional<Tensor> update_idxs_tensor;
    std::optional<Tensor> page_table;
};

}  // namespace ttnn::experimental::prim
