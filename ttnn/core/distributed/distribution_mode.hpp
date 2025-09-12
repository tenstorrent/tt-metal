// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn {
namespace distributed {

// Specifies how a tensor sharded over a specific shape will be distributed to a mesh device
enum class DistributionMode {
    // Tensor shards will be distributed in row-major order over a mesh device.
    ROW_MAJOR,

    // Shards will be mapped to a mesh device as is, preserving coordinates.
    // This requires a submesh to fit within the mesh device.
    SUBMESH,
};

// Computes the distribution mode based on mesh shape configuration.
DistributionMode compute_distribution_mode(
    const std::optional<tt::tt_metal::distributed::MeshShape>& mesh_shape_override,
    const tt::tt_metal::distributed::MeshShape& device_shape);

// Computes the ordered mesh coordinate mapping for the given distribution and mesh shapes.
std::vector<tt::tt_metal::distributed::MeshCoordinate> compute_distribution_to_mesh_mapping(
    const tt::tt_metal::distributed::MeshShape& distribution_shape,
    const tt::tt_metal::distributed::MeshShape& mesh_shape);

}  // namespace distributed
}  // namespace ttnn
