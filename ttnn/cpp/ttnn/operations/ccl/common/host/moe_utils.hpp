// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
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

// Same as `get_neighbors`, but restricts the search to a sub-range of the mesh.
// Used by MoE dispatch subgroups: the "world" each program sees is a single subgroup
// rather than the full mesh, so neighbors across a subgroup boundary must be pruned.
// `mesh_coordinate` must lie inside `range`.
std::pair<std::vector<ttnn::MeshCoordinate>, std::array<bool, 4>> get_neighbors_in_range(
    const ttnn::distributed::MeshCoordinateRange& range,
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

size_t get_num_links(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, std::optional<size_t> cluster_axis = std::nullopt);

// Partition a single contiguous MeshCoordinateRangeSet along `axis` into `num_subgroups`
// equal contiguous sub-ranges. Used by MoE dispatch subgroups so that each program factory
// can allocate per-subgroup semaphores and subgroup-scoped fabric state. Asserts the set
// contains exactly one range (true for tensors produced by the standard mesh mappers).
std::vector<ttnn::distributed::MeshCoordinateRange> split_into_subgroups(
    const ttnn::distributed::MeshCoordinateRangeSet& tensor_coords, uint32_t axis, uint32_t num_subgroups);

}  // namespace ttnn::operations::ccl::common
