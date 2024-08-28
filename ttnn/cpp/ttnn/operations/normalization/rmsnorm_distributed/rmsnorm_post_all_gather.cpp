// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_post_all_gather.hpp"

#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_post_all_gather_op.hpp"

namespace ttnn::operations::normalization {

ttnn::Tensor ExecuteRMSNormPostAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& stats,
    float epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? input_tensor.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    return operation::run(
                LayerNormPostAllGather{
                    .norm_type = LayerNormDistributedType::RMSNORM,
                    .eps = epsilon,
                    .memory_config = memory_config.value_or(input_tensor.memory_config()),
                    .compute_kernel_config = kernel_config_val},
                {input_tensor, stats},
                {weight, bias}).at(0);
}

}  // namespace ttnn::operations::normalization
