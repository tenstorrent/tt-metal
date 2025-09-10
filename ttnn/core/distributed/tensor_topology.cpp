// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/tensor_topology.hpp"

namespace tt::tt_metal {

tt::tt_metal::distributed::MeshCoordinate TensorTopology::get_neighbor(
    const tt::tt_metal::distributed::MeshCoordinate& coord, int32_t offset, int32_t dim) const {
    const auto neighbor_coord = coord.get_neighbor(
        distribution_shape_, offset, dim, tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP);
    TT_FATAL(
        neighbor_coord.has_value(),
        "Failed to get neighbor for coordinate {} at dim {} with offset {}",
        coord,
        dim,
        offset);

    return neighbor_coord.value();
}

tt::tt_metal::distributed::MeshCoordinate TensorTopology::get_next_neighbor(
    const tt::tt_metal::distributed::MeshCoordinate& coord, int32_t dim) const {
    const auto next_coord =
        coord.get_neighbor(distribution_shape_, 1, dim, tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP);
    TT_FATAL(next_coord.has_value(), "Failed to get next neighbor for coordinate {} at dim {}", coord, dim);

    return next_coord.value();
}

tt::tt_metal::distributed::MeshCoordinate TensorTopology::get_prev_neighbor(
    const tt::tt_metal::distributed::MeshCoordinate& coord, int32_t dim) const {
    const auto prev_coord =
        coord.get_neighbor(distribution_shape_, -1, dim, tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP);
    TT_FATAL(prev_coord.has_value(), "Failed to get previous neighbor for coordinate {} at dim {}", coord, dim);

    return prev_coord.value();
}

tt::tt_metal::distributed::MeshCoordinate TensorTopology::get_device_coord(
    const tt::tt_metal::distributed::MeshCoordinate& coord) const {
    // Convert mesh coordinate to flattened index (row-major order)
    std::size_t flattened_index = 0;

    // Calculate row-major flattened index
    for (std::size_t i = 0; i < distribution_shape_.dims(); ++i) {
        flattened_index += coord[i] * distribution_shape_.get_stride(i);
    }

    // Return the mesh coordinate at the flattened position
    return mesh_coords_[flattened_index];
}

}  // namespace tt::tt_metal
