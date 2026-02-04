// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "distribution_mode.hpp"

namespace ttnn::distributed {

DistributionMode compute_distribution_mode(
    const std::optional<tt::tt_metal::distributed::MeshShape>& mesh_shape_override,
    const tt::tt_metal::distributed::MeshShape& device_shape) {
    if (!mesh_shape_override.has_value()) {
        // Note that when no shape is supplied, row-major order is equivalent to submesh.
        return DistributionMode::SUBMESH;
    }
    if (mesh_shape_override->dims() != device_shape.dims()) {
        // Shapes have different dimensions, so a reshape will be required.
        return DistributionMode::ROW_MAJOR;
    }
    // Check if `shape` fits within the mesh device. If it does, we can use submesh distribution. Otherwise,
    // a reshape will be required, and shards will be distributed in row-major order over the mesh device.
    for (size_t i = 0; i < mesh_shape_override->dims(); ++i) {
        if ((*mesh_shape_override)[i] > device_shape[i]) {
            return DistributionMode::ROW_MAJOR;
        }
    }
    return DistributionMode::SUBMESH;
}
std::vector<tt::tt_metal::distributed::MeshCoordinate> compute_distribution_to_mesh_mapping(
    const tt::tt_metal::distributed::MeshShape& distribution_shape,
    const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    DistributionMode mode = compute_distribution_mode(std::make_optional(distribution_shape), mesh_shape);
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords;

    if (mode == DistributionMode::SUBMESH) {
        // For SUBMESH mode, coordinates map directly (distribution coords match mesh coords)
        for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(distribution_shape)) {
            mesh_coords.emplace_back(coord);
        }
    } else {
        // For ROW_MAJOR mode, map distribution coordinates to mesh coordinates in row-major order
        auto mesh_range = tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape);
        auto mesh_iter = mesh_range.begin();
        for ([[maybe_unused]] const auto& dist_coord :
             tt::tt_metal::distributed::MeshCoordinateRange(distribution_shape)) {
            if (mesh_iter != mesh_range.end()) {
                mesh_coords.emplace_back(*mesh_iter);
                ++mesh_iter;
            } else {
                break;
            }
        }
    }

    return mesh_coords;
}

}  // namespace ttnn::distributed
