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
    const std::optional<const ttnn::Tensor>& bias,
    bool is_pre) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE
                    ? input_tensor.device()->arch()
                    : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    // Todo deal with .semaphore when fusing all gather
    return operation::run(
               RMSAllGather{
                   .eps = epsilon,
                   .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                   .program_config = program_config,
                   .compute_kernel_config = kernel_config_val,
                   .dtype = dtype,
                   .topology = topology,
                   .is_pre = is_pre},
               {input_tensor},
               {residual_input_tensor, weight, bias})
        .at(0);
}

}  // namespace operations::fused::normalization

}  // namespace ttnn
