// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "ttnn/types.hpp"
#include <tt-metalium/tt_metal.hpp>

namespace ttnn {

namespace cluster {

tt::tt_metal::ClusterType get_cluster_type();
std::string serialize_cluster_descriptor();

// Resolve a FabricNodeId (mesh_id, chip_id) to the chip's hardware-stable 64-bit ASIC unique id.
// This is the physical, host-global-unique chip identity (the same value fabric sockets route by and
// the migration worker keys per-chip state on), NOT the process-local logical device id which
// collides across the meshes on a host.
std::uint64_t get_chip_unique_id_from_fabric_node_id(std::uint32_t mesh_id, std::uint32_t chip_id);

// Translate a core coordinate between coordinate systems ("LOGICAL", "NOC0", "NOC1",
// "TRANSLATED") for ONE chip. device_id is explicit and required: the mapping is built
// from that chip's harvesting configuration, so a coordinate translated with the wrong
// chip's mapping is silently wrong. Takes a device id rather than hanging off a
// MeshDevice for exactly that reason -- the caller usually knows which chip its data
// came from (e.g. the profiler CSV's PCIe slot), not which mesh happens to be open.
std::pair<std::uint32_t, std::uint32_t> translate_core_coord(
    int device_id, std::uint32_t x, std::uint32_t y, const std::string& from_system, const std::string& to_system);

}  // namespace cluster

using namespace cluster;

}  // namespace ttnn
