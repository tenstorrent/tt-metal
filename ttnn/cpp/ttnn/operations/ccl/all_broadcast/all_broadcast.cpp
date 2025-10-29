// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_broadcast.hpp"
#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_op.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/fabric.hpp>
namespace ttnn::operations::ccl {

std::vector<ttnn::Tensor> ExecuteAllBroadcast::invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<uint32_t> num_links,
    std::optional<ttnn::ccl::Topology> topology) {
    // Default values for num_links and topology
    auto mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Mesh device is required for all_broadcast operation");
    tt::tt_fabric::Topology topology_ = topology.value_or(
        ::ttnn::ccl::get_usable_topology(input_tensor, tt::tt_fabric::get_fabric_topology(), cluster_axis));
    uint32_t num_links_ = num_links.value_or(common::get_num_links(*mesh_device, cluster_axis));

    return ttnn::operations::ccl::all_broadcast(
        input_tensor, cluster_axis, subdevice_id, memory_config, num_links_, topology_);
}

}  // namespace ttnn::operations::ccl
