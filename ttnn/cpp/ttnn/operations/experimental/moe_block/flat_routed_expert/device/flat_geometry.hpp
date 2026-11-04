// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The chip geometry the flat expert's gate/up placement reads (host only: also on a mock cluster): each DRAM bank's
// NoC endpoint tiles and the worker cores, as NOC0 grid positions (tt-metalium/experimental/dram_geometry.hpp).
// Harvesting moves both, so the placement reads them per chip.

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/dram_geometry.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

// The DRAM tiles of every bank of one chip (default: the reference chip). Weight slot r lives in bank r % banks; a
// gate/up core reading on NoC n gets its responses from the bank's NoC n endpoint tile.
struct FlatDramGeometry {
    uint32_t banks = 0;
    tt::tt_metal::distributed::MeshCoordinate coord{0, 0};                       // the chip it was read from
    std::vector<std::vector<tt::tt_metal::experimental::DramBankTile>> tiles;  // per bank

    static FlatDramGeometry of(
        tt::tt_metal::distributed::MeshDevice& mesh_device,
        const std::optional<tt::tt_metal::distributed::MeshCoordinate>& coord = std::nullopt);

    // the NOC0 grid position of bank `bank`'s NoC `noc` endpoint tile
    tt::tt_metal::CoreCoord endpoint(uint32_t bank, uint32_t noc) const;
};

// The NOC0 grid coordinates of a worker core (what its kernels read as their own NoC position), on the reference chip
// or on chip `coord` (harvesting moves the physical columns)
tt::tt_metal::CoreCoord flat_worker_noc0(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const tt::tt_metal::CoreCoord& logical);
tt::tt_metal::CoreCoord flat_worker_noc0(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::CoreCoord& logical,
    const tt::tt_metal::distributed::MeshCoordinate& coord);

// The first chip this rank drives (the plan and the geometry are read from it).
tt::tt_metal::distributed::MeshCoordinate flat_reference_coord(tt::tt_metal::distributed::MeshDevice& mesh_device);

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
