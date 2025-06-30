// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <tt-metalium/constants.hpp>

#include "all_to_all_dispatch.hpp"
#include "device/all_to_all_dispatch_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::ccl {

std::array<ttnn::Tensor, 2> ExecuteAllToAllDispatch::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    const std::optional<std::array<ttnn::Tensor, 2>>& optional_output_tensors,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<GlobalSemaphore>& global_semaphore) {
    auto mesh_device = input_tensor.mesh_device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    TT_FATAL(
        global_semaphore.has_value(),
        "Global semaphore is required for all_to_all_dispatch due to limitations in trace");

    return ttnn::prim::all_to_all_dispatch(
        input_tensor,
        expert_indices_tensor,
        expert_mapping_tensor,
        axis,
        optional_output_tensors,
        num_links,
        topology,
        memory_config.value_or(input_tensor.memory_config()),
        sd_id,
        global_semaphore);
}

}  // namespace ttnn::operations::ccl
