// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_geometry.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

using tt::tt_metal::CoreCoord;
namespace mx = tt::tt_metal::experimental;

tt::tt_metal::distributed::MeshCoordinate flat_reference_coord(tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const auto ids = mesh_device.get_device_ids();
    TT_FATAL(!ids.empty(), "flat_routed_expert: the mesh device has no local chip");
    return mesh_device.get_view().find_device(ids.front());
}

CoreCoord flat_worker_noc0(tt::tt_metal::distributed::MeshDevice& mesh_device, const CoreCoord& logical) {
    return mx::GetWorkerNoc0Core(mesh_device, flat_reference_coord(mesh_device), logical);
}

CoreCoord flat_worker_noc0(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const CoreCoord& logical,
    const tt::tt_metal::distributed::MeshCoordinate& coord) {
    return mx::GetWorkerNoc0Core(mesh_device, coord, logical);
}

FlatDramGeometry FlatDramGeometry::of(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::optional<tt::tt_metal::distributed::MeshCoordinate>& coord) {
    FlatDramGeometry g;
    g.coord = coord.value_or(flat_reference_coord(mesh_device));
    g.banks = mesh_device.dram_grid_size().x;
    for (uint32_t b = 0; b < g.banks; ++b) {
        g.tiles.push_back(mx::GetDramBankTiles(mesh_device, g.coord, b));
    }
    return g;
}

CoreCoord FlatDramGeometry::endpoint(uint32_t bank, uint32_t noc) const {
    for (const auto& t : tiles.at(bank)) {
        if (t.endpoint_noc_mask & (1u << noc)) {
            return t.noc0_core;
        }
    }
    TT_THROW("flat_routed_expert: bank {} has no NOC{} endpoint DRAM tile", bank, noc);
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
