// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ttnn {

namespace cluster {

std::string serialize_cluster_descriptor();

}  // namespace cluster

using namespace cluster;

}  // namespace ttnn
