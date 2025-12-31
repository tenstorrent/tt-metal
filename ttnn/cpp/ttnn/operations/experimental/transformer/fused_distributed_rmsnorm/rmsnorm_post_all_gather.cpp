// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_post_all_gather.hpp"
#include "device/fused_rmsnorm_post_all_gather_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental {

Tensor wan_fused_rmsnorm_post_allgather(
    const Tensor& input_tensor,
    const Tensor& stats,
    float epsilon,
    uint32_t num_heads,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& transformation_mat,
    const std::optional<const Tensor>& rope_cos,
    const std::optional<const Tensor>& rope_sin,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const DataType>& dtype) {
    auto arch = input_tensor.device()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, false, true, false);

    using OperationType =
        operations::experimental::transformer::fused_rmsnorm_post_all_gather::FusedRMSNormPostAllGatherDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .eps = epsilon,
        .num_heads = num_heads,
        .memory_config = memory_config.value_or(input_tensor.memory_config()),
        .compute_kernel_config = kernel_config_val,
        .dtype = dtype,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor,
        .stats_tensor = stats,
        .weight = weight.has_value() ? std::optional<Tensor>(weight.value()) : std::nullopt,
        .transformation_mat =
            transformation_mat.has_value() ? std::optional<Tensor>(transformation_mat.value()) : std::nullopt,
        .rope_cos = rope_cos.has_value() ? std::optional<Tensor>(rope_cos.value()) : std::nullopt,
        .rope_sin = rope_sin.has_value() ? std::optional<Tensor>(rope_sin.value()) : std::nullopt,
    };

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
