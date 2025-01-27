// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "consolidate_cache.hpp"
#include "ttnn/operations/experimental/speculative_execution/swap_tensor/device/swap_tensor_op.hpp"

#include "device/consolidate_cache_op.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/run_operation.hpp"
#include "cpp/ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::speculative_execution {

ttnn::Tensor ExecuteConsolidateCache::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& priority_tensor,
    const ttnn::Tensor& other_priority_tensor,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    uint32_t num_links) {
    // ccl related
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    ttnn::ccl::Topology ccl_topology = ttnn::ccl::Topology::Linear;
    tt::log_info("devices: {}", devices);
    tt::log_info("num_devices: {}", num_devices);

    TT_FATAL(num_devices == 2, "Must have 2 devices for consolidate cache, got: {}", num_devices);

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    operation::launch_op(
        [devices, multi_device_global_semaphore, ccl_topology, num_links, queue_id](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            // ccl related
            auto this_device = input_tensors.at(0).device();
            uint32_t num_devices = devices.size();
            uint32_t device_index = 0;
            std::optional<IDevice*> forward_device = std::nullopt;
            std::optional<IDevice*> backward_device = std::nullopt;
            std::optional<GlobalSemaphore> semaphore = std::nullopt;
            uint32_t semaphore_index = 0;
            auto semaphores = multi_device_global_semaphore.global_semaphores;
            for (uint32_t i = 0; i < num_devices; ++i) {
                if (devices.at(i) == this_device) {
                    device_index = i;
                    semaphore = semaphores.at(i);
                    if (i != 0) {
                        backward_device = devices.at(i - 1);
                    }
                    if (i != num_devices - 1) {
                        forward_device = devices.at(i + 1);
                    }
                }
            }
            tt::log_info("num_devices: {}", num_devices);
            tt::log_info("device_index: {}", device_index);
            tt::log_info("backward_device: {}", backward_device);
            tt::log_info("forward_device: {}", forward_device);
            tt::log_info("semaphore: {}", semaphore->address());

            auto input_tensor = input_tensors.at(0);
            auto priority_tensor = input_tensors.at(1);
            auto other_priority_tensor = input_tensors.at(2);

            const auto& swapped_tensors = operation::run(
                SwapTensor{
                    .num_links = num_links,
                    .num_devices = num_devices,
                    .device_index = device_index,
                    .topology = ccl_topology,
                    .semaphore = semaphore,
                    .forward_device = forward_device,
                    .backward_device = backward_device},
                {input_tensor},
                {},
                {},
                queue_id);

            auto other_tensor = swapped_tensors.at(0);
            auto consolidated_cache_tensors = operation::run(
                ConsolidateCache{},
                {input_tensor, other_tensor, priority_tensor, other_priority_tensor},
                {},
                {},
                queue_id);
            return consolidated_cache_tensors;
        },
        {input_tensor, priority_tensor, other_priority_tensor},
        output_tensors,
        {});

    return output_tensors.at(0);
}

ttnn::Tensor ExecuteConsolidateCache::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& priority_tensor,
    const ttnn::Tensor& other_priority_tensor,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    uint32_t num_links) {
    return invoke(
        DefaultQueueId, input_tensor, priority_tensor, other_priority_tensor, multi_device_global_semaphore, num_links);
}

}  // namespace ttnn::operations::experimental::speculative_execution
