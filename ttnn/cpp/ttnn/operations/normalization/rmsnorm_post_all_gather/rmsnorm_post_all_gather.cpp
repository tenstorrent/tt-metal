// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_post_all_gather.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_op.hpp"

namespace ttnn::operations::normalization {

ttnn::Tensor ExecuteRMSNormPostAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const LayerNormProgramConfig>& program_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const ttnn::Tensor>& stats) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch() : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    return operation::run(
                LayerNorm{
                    .norm_type = LayerNormType::RMSNORM,
                    .distributed_type = LayerNormStageType::POST_ALL_GATHER,
                    .eps = epsilon,
                    .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                    .program_config = program_config.value_or(LayerNormDefaultProgramConfig{}),
                    .compute_kernel_config = kernel_config_val},
                {input_tensor},
                {residual_input_tensor, weight, bias, stats}).at(0);
}

}  // namespace ttnn::operations::normalization
