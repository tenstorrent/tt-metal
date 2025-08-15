// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <tt-metalium/constants.hpp>

#include "all_to_all_combine.hpp"
#include "device/all_to_all_combine_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric.hpp>

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteAllToAllCombine::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    const std::optional<GlobalSemaphore>& global_semaphore,
    const bool locally_reduced,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<uint32_t>& axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const std::optional<GlobalSemaphore>& init_semaphore) {
    auto mesh_device = input_tensor.mesh_device();
    auto sd_id = subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    uint32_t num_links_ = num_links.value_or(1);
    tt::tt_fabric::Topology topology_ = topology.value_or(tt::tt_fabric::get_fabric_topology());
    auto memory_config_ = memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::all_to_all_combine(
        input_tensor,
        expert_mapping_tensor,
        expert_metadata_tensor,
        num_links_,
        topology_,
        memory_config_,
        global_semaphore,
        axis,
        subdevice_id,
        optional_output_tensor,
        locally_reduced,
        init_semaphore);
}

}  // namespace ttnn::operations::ccl
