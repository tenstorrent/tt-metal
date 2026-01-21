// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <algorithm>
#include "all_gather.hpp"
#include "device/all_gather_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
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
    std::optional<tt::tt_fabric::Topology> topology,
    std::optional<uint32_t> chunks_per_sync,
    std::optional<uint32_t> num_workers_per_link,
    std::optional<uint32_t> num_buffers_per_channel,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    // If cluster_axis is None, but mesh shape is not 1xM or Mx1, then we call all-gather on cluster_axis=1, then
    // all-gather on cluster_axis=0
    if (cluster_axis == std::nullopt) {
        auto mesh_shape = input_tensor.device()->get_view().shape();
        // Check if flat mesh (1x...M...x1) where M = total mesh volume
        // if it is not flat, then we need to call all-gather on dim=-1 to dim=0
        if (!mesh_shape.is_line_topology()) {
            Tensor tensor = input_tensor;
            // Iterate through mesh dimensions in reverse order using reverse iterator
            auto mesh_view = mesh_shape.view();
            for (auto it = mesh_view.rbegin(); it != mesh_view.rend(); ++it) {
                auto axis = std::distance(mesh_view.begin(), it.base()) - 1;
                tensor = ttnn::all_gather(
                    tensor,
                    dim,
                    axis,
                    subdevice_id,
                    memory_config,
                    optional_output_tensor,
                    num_links,
                    topology,
                    chunks_per_sync,
                    num_workers_per_link,
                    num_buffers_per_channel,
                    sub_core_grid);
            }
            return tensor;
        }
    }

    auto* mesh_device = input_tensor.device();
    uint32_t normalized_dim = input_tensor.logical_shape().get_normalized_index(dim);
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);
    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());
    uint32_t num_links_ = num_links.value_or(common::get_num_links(*mesh_device, cluster_axis));
    if (composite_common::use_composite_all_gather(input_tensor, dim, memory_config)) {
        TT_FATAL(!sub_core_grid.has_value(), "Composite OP does not currently support sub core grid");
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
        topology_,
        chunks_per_sync,
        num_workers_per_link,
        num_buffers_per_channel,
        sub_core_grid);
}

}  // namespace ttnn::operations::ccl
