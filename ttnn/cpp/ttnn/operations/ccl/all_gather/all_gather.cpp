// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <algorithm>
#include <tt-metalium/constants.hpp>

#include "all_gather.hpp"
#include "device/all_gather_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/composite_common.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology) {
    // If cluster_axis is None, but mesh shape is not 1xM or Mx1, then we call all-gather on cluster_axis=1, then
    // all-gather on cluster_axis=0
    if (cluster_axis == std::nullopt) {
        auto mesh_shape = input_tensor.device()->get_view().shape();
        // Check if flat mesh (1x...M...x1) where M = total mesh volume
        // if it is not flat, then we need to call all-gather on dim=-1 to dim=0
        if (!mesh_shape.is_line_topology()) {
            Tensor tensor = input_tensor;
            for (int i = mesh_shape.dims() - 1; i >= 0; --i) {
                tensor = ttnn::all_gather(
                    tensor, dim, i, subdevice_id, memory_config, optional_output_tensor, num_links, topology);
            }
            return tensor;
        }
    }

    auto mesh_device = input_tensor.device();
    uint32_t normalized_dim = input_tensor.logical_shape().get_normalized_index(dim);
    tt::tt_fabric::Topology topology_ = topology.value_or(
        ::ttnn::ccl::get_usable_topology(input_tensor, tt::tt_fabric::get_fabric_topology(), cluster_axis));
    topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);
    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());
    uint32_t num_links_ = num_links.value_or(common::get_num_links(*mesh_device, cluster_axis));

    if (composite_common::use_composite_all_gather(input_tensor, dim, memory_config)) {
        return composite_common::composite_all_gather(
            input_tensor, dim, num_links_, memory_config_, subdevice_id, cluster_axis);
    }

    return ttnn::prim::all_gather(
        input_tensor,
        normalized_dim,
        cluster_axis,
        subdevice_id,
        memory_config_,
        optional_output_tensor,
        num_links_,
        topology_);
}

}  // namespace ttnn::operations::ccl
