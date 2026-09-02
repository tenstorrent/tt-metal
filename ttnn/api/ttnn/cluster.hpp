// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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

}  // namespace cluster

using namespace cluster;

}  // namespace ttnn
