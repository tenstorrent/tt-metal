// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cluster.hpp"
#include <tt-metalium/tt_metal.hpp>

namespace ttnn {

namespace cluster {

tt::tt_metal::ClusterType get_cluster_type() { return tt::tt_metal::GetClusterType(); }

std::string serialize_cluster_descriptor() { return tt::tt_metal::SerializeClusterDescriptor(); }

}  // namespace cluster

using namespace cluster;

}  // namespace ttnn
