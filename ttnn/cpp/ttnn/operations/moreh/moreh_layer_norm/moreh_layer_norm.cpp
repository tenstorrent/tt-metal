// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm.hpp"
#include "device/moreh_layer_norm_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> moreh_layer_norm(
    const Tensor& input,
    uint32_t normalized_dims,
    float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& output,
    const std::optional<const Tensor>& mean,
    const std::optional<const Tensor>& rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = operations::moreh::moreh_layer_norm::MorehLayerNormOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        normalized_dims,
        eps,
        memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{input, gamma, beta, output, mean, rstd};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn
