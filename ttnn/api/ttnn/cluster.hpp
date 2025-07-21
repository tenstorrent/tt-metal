// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"
#include <tt-metalium/tt_metal.hpp>

namespace ttnn {

namespace cluster {

tt::tt_metal::ClusterType get_cluster_type();
bool is_galaxy_cluster();
std::size_t number_of_user_devices();
std::string serialize_cluster_descriptor();

}  // namespace cluster

using namespace cluster;

}  // namespace ttnn
