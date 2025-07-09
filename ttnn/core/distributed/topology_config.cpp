// SPDX-FileCopyrightText: © 2025 Tenstorrent AI UL LLC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/topology_config.hpp"

namespace tt::tt_metal {

tt::tt_metal::distributed::MeshCoordinate TopologyConfig::get_neighbor(
    const tt::tt_metal::distributed::MeshCoordinate& coord) const {
    // TODO: Implement topology-aware neighbor finding logic
    // For now, return the same coordinate as a placeholder

    return coord;
}

tt::tt_metal::distributed::MeshCoordinate TopologyConfig::get_device_coord(
    const tt::tt_metal::distributed::MeshCoordinate& coord) const {
    // Convert mesh coordinate to flattened index (row-major order)
    std::size_t flattened_index = 0;
    std::size_t stride = 1;

    // Calculate row-major flattened index
    for (int i = mesh_shape.dims() - 1; i >= 0; --i) {
        flattened_index += coord[i] * stride;
        stride *= mesh_shape[i];
    }

    // Return the device coordinate at the flattened position
    return device_coords[flattened_index];
}

}  // namespace tt::tt_metal
