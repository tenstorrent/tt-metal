// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::experimental::prim {

struct PagedFillCacheParams {
    const uint32_t batch_idx_fallback;
    const std::optional<std::set<ttnn::MeshCoordinate>> mesh_coords;
    const bool noop = false;  // When true, kernels early exit
    // Optional per-call block_size override; see PagedUpdateCacheParams::block_size_override.
    const std::optional<uint32_t> block_size_override = std::nullopt;
    // Optional circular-buffer capacity (in tokens) for the cache view. When set, the
    // kernel computes ``seq_tile_id %= (cache_position_modulo / TILE_HEIGHT)`` before
    // resolving the page_table entry, so a bounded sliding-window cache can be filled
    // from a prefill longer than the capacity — only the last cache_position_modulo
    // tokens survive in the cache after the writes, exactly what attention needs.
    // Must be a multiple of the effective block_size. See
    // PagedUpdateCacheParams::cache_position_modulo for the companion decode-side kwarg.
    const std::optional<uint32_t> cache_position_modulo = std::nullopt;

    static constexpr auto attribute_names =
        std::forward_as_tuple("mesh_coords", "block_size_override", "cache_position_modulo");
    auto attribute_values() const {
        return std::forward_as_tuple(mesh_coords, block_size_override, cache_position_modulo);
    }
};

struct PagedFillCacheInputs {
    Tensor cache_tensor;  // also output tensor
    Tensor input_tensor;
    Tensor page_table;
    std::optional<Tensor> batch_idx_tensor_opt;
    // Optional per-request valid fill length (block-aligned, in tokens) as a 1-element
    // int device tensor. Read by the writer kernel at runtime, so it survives trace
    // capture/replay: for a bounded (cache_position_modulo) fill of a padded prompt,
    // the writer restricts the surviving ring window to the last capacity_t *real*
    // tiles (ending at valid_seq_len) instead of the padded end, matching the
    // host-side fill cap but per-request under a captured prefill trace.
    std::optional<Tensor> valid_seq_len_tensor_opt;
};

}  // namespace ttnn::experimental::prim
