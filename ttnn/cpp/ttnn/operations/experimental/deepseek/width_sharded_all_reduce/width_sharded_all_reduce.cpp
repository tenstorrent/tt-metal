// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "width_sharded_all_reduce.hpp"
#include "device/width_sharded_all_reduce_device_operation.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::experimental::deepseek {

ttnn::Tensor width_sharded_all_reduce(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology) {
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "width_sharded_all_reduce: MeshDevice is required");
    tt::tt_fabric::Topology topology_ = topology.value_or(
        ::ttnn::ccl::get_usable_topology(input_tensor, tt::tt_fabric::get_fabric_topology(), cluster_axis));
    topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);
    std::optional<size_t> axis = cluster_axis.has_value() ? std::optional<size_t>(*cluster_axis) : std::nullopt;
    const uint32_t num_links_ =
        num_links.value_or(static_cast<uint32_t>(ttnn::operations::ccl::common::get_num_links(*mesh_device, axis)));
    return ttnn::prim::width_sharded_all_reduce(input_tensor, cluster_axis, subdevice_id, num_links_, topology_);
}

}  // namespace ttnn::experimental::deepseek
