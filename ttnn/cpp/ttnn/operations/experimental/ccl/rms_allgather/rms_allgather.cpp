// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "rms_allgather.hpp"

namespace ttnn {
namespace operations::fused::normalization {

ttnn::Tensor ExecuteFusedRMSNorm::invoke(
    const ttnn::Tensor& input_tensor,
    const global_semaphore::MultiDeviceGlobalSemaphore& multi_device_global_semaphore,
    const ttnn::operations::normalization::LayerNormProgramConfig& program_config,
    const ttnn::ccl::Topology topology,
    const std::optional<const DataType> dtype,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& stats,
    bool is_pre,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const uint32_t num_links) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE
                    ? input_tensor.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    std::vector<GlobalSemaphore> semaphores = multi_device_global_semaphore.global_semaphores;
    std::optional<GlobalSemaphore> semaphore = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    auto devices = input_tensor.get_workers();
    uint32_t num_devices = devices.size();
    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    for (uint32_t i = 0; i < num_devices; ++i) {
        if (devices.at(i) == input_tensor.device()) {
            device_index = i;
            semaphore = semaphores.at(i);  // Get raw pointer
            if (i != 0) {
                backward_device = devices.at(i - 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices.at(num_devices - 1);
            }
            if (i != num_devices - 1) {
                forward_device = devices.at(i + 1);
            } else if (topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices.at(0);
            }
        }
    }
    return operation::run(
               RMSAllGather{
                   .eps = epsilon,
                   .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                   .program_config = program_config,
                   .compute_kernel_config = kernel_config_val,
                   .dtype = dtype,
                   .topology = topology,
                   .is_pre = is_pre,
                   .forward_device = forward_device,
                   .backward_device = backward_device,
                   .num_links = num_links,
                   .ring_size = num_devices,
                   .ring_index = device_index,
                   .semaphore = semaphore,
                   .sub_device_id = subdevice_id},
               {input_tensor},
               {residual_input_tensor, weight, stats})
        .at(0);
}

}  // namespace operations::fused::normalization

}  // namespace ttnn
