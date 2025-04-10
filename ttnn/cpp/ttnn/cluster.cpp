// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cluster.hpp"
#include <umd/device/cluster.h>
#include <filesystem>

namespace ttnn {

namespace cluster {

std::string serialize_cluster_descriptor() {
    std::filesystem::path path = tt::umd::Cluster::serialize_to_file();
    return path.string();
};

}  // namespace cluster

using namespace cluster;

}  // namespace ttnn
