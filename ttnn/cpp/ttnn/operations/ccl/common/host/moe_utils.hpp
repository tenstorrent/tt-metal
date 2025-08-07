// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::operations::ccl::common {

std::pair<std::vector<ttnn::MeshCoordinate>, std::array<bool, 4>> get_neighbors(
    const ttnn::distributed::MeshDeviceView& mesh_view,
    const ttnn::distributed::MeshCoordinate& mesh_coordinate,
    tt::tt_fabric::Topology topology,
    std::optional<uint32_t> axis);

// Utilities to code-gen variadic length containers for kernels
template <typename T>
std::string stringify(const T& vec) {
    std::string result = "{";
    for (const auto& elem : vec) {
        result += std::to_string(elem) + ", ";
    }
    result += "}";
    return result;
}

uint32_t get_linearized_index(const ttnn::MeshCoordinate& mesh_coordinate, const ttnn::MeshDeviceView& mesh_view);

}  // namespace ttnn::operations::ccl::common
