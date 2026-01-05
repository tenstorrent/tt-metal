// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn {

namespace cluster {

tt::tt_metal::ClusterType get_cluster_type();
std::string serialize_cluster_descriptor();
tt::tt_metal::distributed::MeshShape get_mesh_shape();

}  // namespace cluster

using namespace cluster;

}  // namespace ttnn
