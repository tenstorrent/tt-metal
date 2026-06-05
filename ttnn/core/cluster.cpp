// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cluster.hpp"
#include <tt-metalium/tt_metal.hpp>

namespace ttnn {

namespace cluster {

tt::tt_metal::ClusterType get_cluster_type() { return tt::tt_metal::GetClusterType(); }

std::string serialize_cluster_descriptor() { return tt::tt_metal::SerializeClusterDescriptor(); }

bool get_enable_2_erisc_mode() { return tt::tt_metal::GetEnable2EriscMode(); }

uint64_t get_build_key() { return tt::tt_metal::GetBuildKey(); }

}  // namespace cluster

using namespace cluster;

}  // namespace ttnn
