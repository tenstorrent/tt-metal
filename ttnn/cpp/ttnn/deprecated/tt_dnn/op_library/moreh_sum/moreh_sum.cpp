// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum.hpp"

#include "moreh_sum_op.hpp"

namespace ttnn::operations::moreh {

ttnn::Tensor ExecuteRMSNorm::invoke(
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
                    .norm_type = LayerNormType::RMSNORM,
                    .eps = epsilon,
                    .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
                    .program_config = program_config.value_or(LayerNormDefaultProgramConfig{}),
                    .compute_kernel_config = kernel_config_val},
                {input_tensor},
                {residual_input_tensor, weight, bias}).at(0);
}


static ttnn::Tensor MorehSumOperation::invoke(
    const Tensor &input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keep_batch_dim,
    const std::optional<const Tensor> output,
    const MemoryConfig &output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    return invoke(DefaultQueueId, input, dim, keep_batch_dim, output, output_mem_config, compute_kernel_config);
}

static ttnn::Tensor MorehSumOperation::invoke(
    uint8_t queue_id,
    const Tensor &input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keep_batch_dim,
    const std::optional<const Tensor> output,
    const MemoryConfig &output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {

    return tt::operations::primary::moreh_sum(input, dim, keep_batch_dim, output, output_mem_config, compute_kernel_config, queue_id);
}

}  // namespace ttnn::operations::moreh
