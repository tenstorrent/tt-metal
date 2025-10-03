// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/tensor_topology.hpp"

namespace tt::tt_metal {

TensorTopology TensorTopology::create_fully_replicated_tensor_topology(
    const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    tt::tt_metal::distributed::MeshShape distribution_shape =
        tt::tt_metal::distributed::MeshShape(mesh_shape.mesh_size());
    tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements = {
        tt::tt_metal::distributed::MeshMapperConfig::Replicate{}};
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords;
    mesh_coords.reserve(mesh_shape.mesh_size());
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        mesh_coords.push_back(coord);
    }

    return tt::tt_metal::TensorTopology(distribution_shape, placements, mesh_coords);
}

tt::tt_metal::distributed::MeshCoordinate TensorTopology::get_neighbor(
    const tt::tt_metal::distributed::MeshCoordinate& tensor_coord, int32_t offset, int32_t dim) const {
    const auto neighbor_coord = tensor_coord.get_neighbor(
        distribution_shape_, offset, dim, tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP);
    TT_FATAL(
        neighbor_coord.has_value(),
        "Failed to get neighbor for coordinate {} at dim {} with offset {}",
        tensor_coord,
        dim,
        offset);

    return neighbor_coord.value();
}

tt::tt_metal::distributed::MeshCoordinate TensorTopology::get_next_neighbor(
    const tt::tt_metal::distributed::MeshCoordinate& tensor_coord, int32_t dim) const {
    const auto next_coord = tensor_coord.get_neighbor(
        distribution_shape_, 1, dim, tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP);
    TT_FATAL(next_coord.has_value(), "Failed to get next neighbor for coordinate {} at dim {}", tensor_coord, dim);

    return next_coord.value();
}

tt::tt_metal::distributed::MeshCoordinate TensorTopology::get_prev_neighbor(
    const tt::tt_metal::distributed::MeshCoordinate& tensor_coord, int32_t dim) const {
    const auto prev_coord = tensor_coord.get_neighbor(
        distribution_shape_, -1, dim, tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP);
    TT_FATAL(prev_coord.has_value(), "Failed to get previous neighbor for coordinate {} at dim {}", tensor_coord, dim);

    return prev_coord.value();
}

tt::tt_metal::distributed::MeshCoordinate TensorTopology::get_device_coord(
    const tt::tt_metal::distributed::MeshCoordinate& tensor_coord) const {
    // Convert mesh coordinate to flattened index (row-major order)
    std::size_t flattened_index = 0;

    // Calculate row-major flattened index
    for (std::size_t i = 0; i < distribution_shape_.dims(); ++i) {
        flattened_index += tensor_coord[i] * distribution_shape_.get_stride(i);
    }

    // Return the mesh coordinate at the flattened position
    TT_FATAL(
        flattened_index < mesh_coords_.size(),
        "Flattened index {} is out of bounds of mesh_coordinates {} in tensor topology with distribution shape {}",
        flattened_index,
        mesh_coords_.size(),
        distribution_shape_);
    return mesh_coords_[flattened_index];
}

std::optional<tt::tt_metal::distributed::MeshCoordinate> TensorTopology::get_tensor_coord(
    const tt::tt_metal::distributed::MeshCoordinate& device_coord) const {
    // Search through all stored mesh coordinates to find a match
    // Assume that the mesh coordinates are unique (ie. tensor to device coords mapping is one-to-one)
    for (std::size_t tensor_coord_idx = 0; tensor_coord_idx < mesh_coords_.size(); ++tensor_coord_idx) {
        if (mesh_coords_[tensor_coord_idx] == device_coord) {
            // Convert the position index to tensor coordinate
            // This assumes that mesh coordinates are stored in row-major order

            // Create tensor coordinate container based on rank of distribution shape
            tt::stl::SmallVector<uint32_t> tensor_coord_values(distribution_shape_.dims());

            // Convert flattened index to tensor coordinate assuming row-major order
            // This is the inverse logic of get_device_coord
            for (int dim = static_cast<int>(distribution_shape_.dims()) - 1; dim >= 0; --dim) {
                tensor_coord_values[dim] = tensor_coord_idx % distribution_shape_[dim];
                tensor_coord_idx /= distribution_shape_[dim];
            }

            return tt::tt_metal::distributed::MeshCoordinate(tensor_coord_values);
        }
    }

    // No match found
    return std::nullopt;
}

bool operator==(const TensorTopology& lhs, const TensorTopology& rhs) {
    return lhs.distribution_shape() == rhs.distribution_shape() && lhs.placements() == rhs.placements() &&
           lhs.mesh_coords() == rhs.mesh_coords();
}

bool operator!=(const TensorTopology& lhs, const TensorTopology& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& os, const TensorTopology& tensor_topology) {
    os << "TensorTopology(";
    os << "distribution_shape=" << tensor_topology.distribution_shape();
    os << ", placements=" << tensor_topology.placements();
    os << ", mesh_coords=";
    for (size_t i = 0; i < tensor_topology.mesh_coords().size(); i++) {
        const auto& coord = tensor_topology.mesh_coords()[i];
        if (i > 0) {
            os << ", ";
        }
        os << coord;
    }
    os << ")";
    return os;
}

}  // namespace tt::tt_metal
