// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cluster.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/internal/cluster.hpp>

namespace ttnn {

namespace cluster {

tt::tt_metal::ClusterType get_cluster_type() { return tt::tt_metal::GetClusterType(); }

std::string serialize_cluster_descriptor() { return tt::tt_metal::SerializeClusterDescriptor(); }

std::uint64_t get_chip_unique_id_from_fabric_node_id(std::uint32_t mesh_id, std::uint32_t chip_id) {
    return *tt::tt_metal::internal::get_chip_unique_id_from_fabric_node_id(mesh_id, chip_id);
}

}  // namespace cluster

using namespace cluster;

}  // namespace ttnn
