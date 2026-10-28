// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill {

// Whether a kernel on every chip of `coords` can read that chip's mesh coordinate on device, from the chip id
// fabric's control plane writes into each Tensix core's routing info (kernels/mesh_coord_from_fabric.hpp): fabric
// is up, the mesh is 2D, and every chip's fabric chip id is its row-major index over this mesh, all in one fabric
// mesh. A program that would differ per chip only in values derived from the chip's coordinate can then be built
// once for every chip, which is what dispatching it costs on the host: one program instead of one per chip.
inline bool mesh_coord_from_fabric_available(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, const ttnn::MeshCoordinateRangeSet& coords) {
    if (tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::DISABLED) {
        return false;
    }
    const auto& shape = mesh_device.shape();
    if (shape.dims() != 2) {
        return false;
    }
    std::optional<tt::tt_fabric::MeshId> mesh_id;
    for (const auto& coord : coords.coords()) {
        const auto node = mesh_device.get_fabric_node_id(coord);
        if (mesh_id.has_value() && *mesh_id != node.mesh_id) {
            return false;
        }
        mesh_id = node.mesh_id;
        if (node.chip_id != coord[0] * shape[1] + coord[1]) {
            return false;
        }
    }
    return mesh_id.has_value();
}

}  // namespace ttnn::operations::experimental::deepseek_prefill
