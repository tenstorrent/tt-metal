// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>

#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/distributed/types.hpp"

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

// The block-cyclic chip axis of a KV cache, shared by the write (update_padded_kv_cache) and the read
// (read_padded_kv_cache) so both address the same rows. With a named cluster_axis the axis is sp (that mesh axis)
// times tp (tp_axis, KV dedup, or 1); with cluster_axis = nullopt it is the complete mesh in the cache's declared
// row-major topology order.

// Mesh extent along a shard axis (0 = rows, 1 = cols).
inline uint32_t mesh_axis_extent(const ttnn::distributed::MeshDeviceView& mesh_view, uint32_t axis) {
    return (axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
}

// TP dedup factor: the extent of the TP axis, or 1 when the cache is TP-replicated (tp_axis unset).
inline uint32_t tp_dedup_factor(
    const ttnn::distributed::MeshDeviceView& mesh_view, const std::optional<uint32_t>& tp_axis) {
    return tp_axis.has_value() ? mesh_axis_extent(mesh_view, *tp_axis) : 1;
}

inline uint32_t kv_cache_linear_factor(
    const Tensor& cache, const std::optional<uint32_t>& cluster_axis, const std::optional<uint32_t>& tp_axis) {
    return ::ttnn::ccl::get_topological_dimension(cache, cluster_axis) *
           tp_dedup_factor(cache.device()->get_view(), tp_axis);
}

// This chip's position on that axis: sp_coord * tp + tp_coord.
inline uint32_t kv_cache_linear_coord(
    const Tensor& cache,
    const ttnn::MeshCoordinate& coord,
    const std::optional<uint32_t>& cluster_axis,
    const std::optional<uint32_t>& tp_axis) {
    uint32_t sp_coord;
    if (cluster_axis.has_value()) {
        sp_coord = ::ttnn::ccl::get_linearized_index_from_physical_coord(cache, coord, cluster_axis);
    } else {
        // Complete-mesh cache ownership follows the tensor's declared canonical row-major topology. An
        // allocate-on-device tensor's storage coordinate vector may use a different internal order;
        // using that order would rotate the wrong cache shard after a nonzero kv_actual_global.
        //
        // This value is ALREADY the linearized sp*tp position the block-cyclic math wants:
        // row-major over the full mesh is exactly sp_coord*tp + tp_coord. tp_axis is rejected in this
        // mode, so tp_factor collapses to 1 and linear_coord reduces to this index unchanged.
        const auto& topology_coords = cache.tensor_topology().mesh_coords();
        const auto it = std::find(topology_coords.begin(), topology_coords.end(), coord);
        TT_FATAL(it != topology_coords.end(), "cache mesh coordinate {} is absent from its tensor topology", coord);
        sp_coord = static_cast<uint32_t>(std::distance(topology_coords.begin(), it));
    }
    const uint32_t tp_coord =
        tp_axis.has_value() ? ::ttnn::ccl::get_linearized_index_from_physical_coord(cache, coord, tp_axis) : 0;
    return sp_coord * tp_dedup_factor(cache.device()->get_view(), tp_axis) + tp_coord;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache
