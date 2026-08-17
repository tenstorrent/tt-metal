// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>

#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
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

// How a MeshDevice view axis is actually wired. A view axis is not necessarily a fabric axis:
// a line view of a 2x4 board is the perimeter cycle, so its consecutive coords turn corners.
struct AxisGeometry {
    bool is_straight = true;        // every wired edge on the axis leaves in the same direction
    bool wrap_edge_exists = false;  // the closing edge of every line on the axis is wired
};

AxisGeometry get_axis_geometry(const tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t axis);

size_t get_num_links(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, std::optional<size_t> cluster_axis = std::nullopt);

}  // namespace ttnn::operations::ccl::common
