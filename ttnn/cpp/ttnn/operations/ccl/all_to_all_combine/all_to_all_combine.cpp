// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <tt-metalium/constants.hpp>

#include "all_to_all_combine.hpp"
#include "device/all_to_all_combine_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteAllToAllCombine::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const uint32_t num_links,
    const tt::tt_fabric::Topology topology,
    const GlobalSemaphore& global_semaphore
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id) {
    auto mesh_device = input_tensor.mesh_device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto sub_device_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);
        mesh_device, sub_device_cores, 0, tt::tt_metal::BufferType::L1));

    return ttnn::prim::all_to_all_combine(
        input_tensor,
        expert_indices_tensor,
        expert_mapping_tensor,
        num_links,
        topology,
        memory_config.value_or(input_tensor.memory_config()),
        sd_id,
        global_semaphore);
}

}  // namespace ttnn::operations::ccl
