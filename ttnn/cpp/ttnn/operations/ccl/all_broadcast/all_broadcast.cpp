// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/all_broadcast/all_broadcast.hpp"
#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_device_operation.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <cstdint>
#include <optional>
#include <vector>
#include <deque>

namespace ttnn::operations::ccl {

std::vector<ttnn::Tensor> ExecuteAllBroadcast::invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<uint32_t> num_links,
    std::optional<ttnn::ccl::Topology> topology) {
    // Default values for num_links and topology
    if (cluster_axis == std::nullopt) {
        auto mesh_shape = input_tensor.device()->get_view().shape();
        // Check if flat mesh (1x...M...x1) where M = total mesh volume
        // if it is not flat, then we need to call all-broadcast on dim=-1 to dim=0
        // first all broadcast will be on the last dimension, producing a vector of tensors
        // we then recursively call all broadcast on each of the tensors in the vector
        if (!mesh_shape.is_line_topology()) {
            std::deque<ttnn::Tensor> tensors;
            tensors.push_back(input_tensor);
            for (uint32_t axis = 0; axis < mesh_shape.dims(); ++axis) {
                uint32_t num_tensors = tensors.size();
                for (uint32_t i = 0; i < num_tensors; ++i) {
                    auto tensor = std::move(tensors.front());
                    tensors.pop_front();
                    auto curr_tensors =
                        ttnn::all_broadcast(tensor, axis, subdevice_id, memory_config, num_links, topology);
                    tensors.insert(tensors.end(), curr_tensors.begin(), curr_tensors.end());
                }
            }
            return std::vector<ttnn::Tensor>(tensors.begin(), tensors.end());
        }
    }
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device is required for all_broadcast operation");
    tt::tt_fabric::Topology topology_ = topology.value_or(
        ::ttnn::ccl::get_usable_topology(input_tensor, tt::tt_fabric::get_fabric_topology(), cluster_axis));
    topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);
    uint32_t num_links_ = num_links.value_or(common::get_num_links(*mesh_device, cluster_axis));
    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::all_broadcast(input_tensor, cluster_axis, subdevice_id, memory_config_, num_links_, topology_);
}

}  // namespace ttnn::operations::ccl
