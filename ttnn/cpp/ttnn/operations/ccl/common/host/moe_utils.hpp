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

size_t get_num_links(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, std::optional<size_t> cluster_axis = std::nullopt);

void append_fabric_mux_connection_ct_args(
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    uint32_t num_workers_per_direction,
    std::vector<uint32_t>& writer_ct_args);

void append_fabric_mux_connection_rt_args(
    const bool mux_connection_valid,
    const CoreCoord& mux_virtual_core,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    const CoreCoord& worker_logical_core,
    const uint32_t worker_per_direction_id,
    const bool is_termination_master,
    const CoreCoord termination_master_virtual_core,
    tt::tt_metal::Program& program,
    std::vector<uint32_t>& worker_rt_args);

}  // namespace ttnn::operations::ccl::common
