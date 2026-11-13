// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::operations::experimental::deepseek_prefill {

// One program for a whole mesh whose kernels need a per-chip value.
//
// Program args are shared by every chip a program covers, so a per-chip value would force one program per chip. With
// fabric up, each core holds its chip's fabric chip id (routing_l1_info_t::my_device_id), so the op can pass a table
// of per-chip values indexed by fabric chip id instead; kernels/chip_position.hpp picks the entry. `value_of(coord)`
// is the value for the chip at `coord`, used for both the whole-mesh table and a per-chip program's one-entry table.

// Fabric init writes the routing tables for 1D and 2D configs only. Reads the default context, like other ttnn ops.
inline bool chip_lookup_available() {
    const auto config = tt::tt_fabric::GetFabricConfig();
    return tt::tt_fabric::is_1d_fabric_config(config) || tt::tt_fabric::is_2d_fabric_config(config);
}

// nullopt (one program per chip) without routing tables or when the chips span several fabric meshes.
template <typename Coords, typename ValueOf>  // Coords: any range of tt::tt_metal::distributed::MeshCoordinate
std::optional<std::vector<uint32_t>> chip_lookup_table(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, const Coords& coords, const ValueOf& value_of) {
    if (std::empty(coords) || !chip_lookup_available()) {
        return std::nullopt;
    }
    std::vector<uint32_t> table;
    std::optional<tt::tt_fabric::MeshId> mesh_id;
    for (const auto& coord : coords) {
        const auto node = mesh_device.get_fabric_node_id(coord);
        if (mesh_id.has_value() && *mesh_id != node.mesh_id) {
            return std::nullopt;
        }
        mesh_id = node.mesh_id;
        const uint32_t chip = node.chip_id;
        if (chip >= tt::tt_fabric::MAX_MESH_SIZE) {  // the routing table's my_device_id range
            return std::nullopt;
        }
        if (table.size() <= chip) {
            table.resize(chip + 1, 0);
        }
        table[chip] = value_of(coord);
    }
    return table;
}

// For the program hash: a whole-mesh table bakes in the coordinate -> fabric chip mapping, which the cache key (by
// coordinate) does not cover, e.g. after a reshape.
template <typename Coords>  // any range of tt::tt_metal::distributed::MeshCoordinate
uint64_t chip_mapping_hash(const tt::tt_metal::distributed::MeshDevice& mesh_device, const Coords& coords) {
    if (!chip_lookup_available()) {
        return 0;
    }
    uint64_t hash = 0xcbf29ce484222325ull;  // FNV-1a
    for (const auto& coord : coords) {
        const auto node = mesh_device.get_fabric_node_id(coord);
        for (const uint64_t value : {static_cast<uint64_t>(*node.mesh_id), static_cast<uint64_t>(node.chip_id)}) {
            hash = (hash ^ value) * 0x100000001b3ull;
        }
    }
    return hash;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill
