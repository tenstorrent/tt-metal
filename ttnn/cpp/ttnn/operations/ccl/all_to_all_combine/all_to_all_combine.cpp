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

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteAllToAllCombine::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    const GlobalSemaphore& global_semaphore,
    const uint32_t num_links,
    const tt::tt_fabric::Topology topology,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<uint32_t>& axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    return ttnn::prim::all_to_all_combine(
        input_tensor,
        expert_mapping_tensor,
        expert_metadata_tensor,
        num_links,
        topology,
        memory_config.value_or(input_tensor.memory_config()),
        global_semaphore,
        axis,
        subdevice_id,
        optional_output_tensor);
}

}  // namespace ttnn::operations::ccl
