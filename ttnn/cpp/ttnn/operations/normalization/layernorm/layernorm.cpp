// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm.hpp"

#include "device/layernorm_op.hpp"

namespace ttnn::operations::normalization {

ttnn::Tensor ExecuteLayerNorm::invoke(
    const ttnn::Tensor& input_tensor,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    return operation::run(
                LayerNorm{
                    .norm_type = LayerNormType::LAYERNORM,
                    .distributed_type = LayerNormDistributedType::NOT_DISTRIBUTED,
                    .eps = epsilon,
                    .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                    .program_config = program_config.value_or(LayerNormDefaultProgramConfig{}),
                    .compute_kernel_config = kernel_config_val},
                {input_tensor},
                {residual_input_tensor, weight, bias, std::nullopt}).at(0);
}

}  // namespace ttnn::operations::normalization
