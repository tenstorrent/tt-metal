// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_command_processor_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_gather_command_processor_async/device/all_gather_command_processor_async_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllGatherCommandProcessorAsync::invoke(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    const GlobalSemaphore& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    return ttnn::operations::experimental::ccl::all_gather_command_processor_async(
        input_tensor,
        dim,
        multi_device_global_semaphore,
        persistent_output_buffer,
        num_links,
        memory_config,
        topology,
        cluster_axis,
        sub_device_id);
}

std::vector<ttnn::Tensor> ExecuteAllGatherCommandProcessorAsync::invoke(
    const std::vector<Tensor>& input_tensors,
    int32_t dim,
    const std::vector<global_semaphore::MultiDeviceGlobalSemaphore>& multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<uint32_t> cluster_axis,
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id) {
    return ttnn::operations::experimental::ccl::all_gather_command_processor_async(
        input_tensors,
        dim,
        multi_device_global_semaphore,
        persistent_output_buffer,
        num_links,
        memory_config,
        topology,
        cluster_axis,
        sub_device_id);
}

}  // namespace ttnn::operations::experimental::ccl
