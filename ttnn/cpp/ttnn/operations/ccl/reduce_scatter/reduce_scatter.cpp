// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <tt-metalium/constants.hpp>

#include "reduce_scatter.hpp"
#include "device/reduce_scatter_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/fabric.hpp>

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteReduceScatter::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> axis,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    std::optional<tt::tt_fabric::Topology> topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) {
    auto mesh_device = input_tensor.device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    tt::tt_fabric::Topology topology_ = topology.value_or(tt::tt_fabric::get_fabric_topology());
    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::reduce_scatter(
        input_tensor, dim, axis, optional_output_tensor, topology_, memory_config_, subdevice_core_range_set);
}

}  // namespace ttnn::operations::ccl
