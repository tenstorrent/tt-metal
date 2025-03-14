// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <tt-metalium/constants.hpp>

#include "llama_reduce_scatter.hpp"
#include "device/llama_reduce_scatter_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
namespace ttnn::operations::experimental::ccl {
namespace detail {}  // namespace detail

ttnn::Tensor ExecuteLlamaReduceScatter::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const global_semaphore::MultiDeviceGlobalSemaphore& cross_device_semaphore,
    const SubDeviceId& subdevice_id,
    const uint32_t cluster_axis,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    bool enable_persistent_fabric = true;
    TT_FATAL(
        std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr,
        "all_gather_async op is only supported for Fast Dispatch");
    auto devices = input_tensor.get_workers();
    uint32_t ring_devices = devices.size();
    TT_FATAL(ring_devices > 1, "all_gather_async op will only work for ring_devices > 1, but has {}", ring_devices);
    ttnn::ccl::Topology ccl_topology = ttnn::ccl::Topology::Linear;

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input_tensor}))};

    std::cout << fmt::format(
                     "DEBUG: creating line_fabric with num devices: {}, num links: {}", devices.size(), num_links)
              << std::endl;
    std::cout << "DEBUG: line_fabric is created" << std::endl;
    std::vector<GlobalSemaphore> semaphores = cross_device_semaphore.global_semaphores;
    std::cout << "Calling llama_reduce_scatter" << std::endl;
    operation::launch_op(
        [dim, semaphores, subdevice_id, cluster_axis, ring_devices, memory_config, devices, num_links](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto input_tensor = input_tensors.at(0);
            std::cout << "input_tensor device id: " << input_tensor.device()->id() << std::endl;
            uint32_t ring_index = 0;  // Initialize device index
            std::optional<IDevice*> forward_device = std::nullopt;
            std::optional<IDevice*> backward_device = std::nullopt;
            std::optional<GlobalSemaphore> semaphore = std::nullopt;
            for (uint32_t i = 0; i < ring_devices; ++i) {
                if (devices.at(i) == input_tensor.device()) {
                    ring_index = i;
                    semaphore = semaphores.at(i);
                    if (i != 0) {
                        std::cout << "ring_index: " << ring_index << " backward_device: " << i - 1 << std::endl;
                        backward_device = devices.at(i - 1);
                    }
                    if (i != ring_devices - 1) {
                        std::cout << "ring_index: " << ring_index << " forward_device: " << i + 1 << std::endl;
                        forward_device = devices.at(i + 1);
                    }
                }
            }
            return {ttnn::prim::llama_reduce_scatter(
                input_tensor,
                dim,
                semaphore.value(),
                subdevice_id,
                ring_index,
                cluster_axis,
                forward_device,
                backward_device,
                ring_devices,
                num_links,
                memory_config)};
        },
        {input_tensor},
        output_tensors);
    auto output = output_tensors.at(0);
    // auto output = ttnn::prim::llama_reduce_scatter(input_tensor, dim, cross_device_semaphore, subdevice_id,
    // cluster_axis, forward_device, backward_device, ring_devices, memory_config);
    std::cout << output.memory_config() << std::endl;
    std::cout << output.get_logical_shape() << std::endl;
    return output;
}

}  // namespace ttnn::operations::experimental::ccl
