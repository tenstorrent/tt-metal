// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

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
    auto mesh_device = input_tensor.device();
    uint32_t normalized_dim = input_tensor.logical_shape().get_normalized_index(dim);
    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
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
