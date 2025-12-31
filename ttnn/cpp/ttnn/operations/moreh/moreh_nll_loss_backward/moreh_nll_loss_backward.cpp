// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_backward.hpp"
#include "device/moreh_nll_loss_backward_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

Tensor moreh_nll_loss_backward(
    const Tensor& target_tensor,
    const Tensor& output_grad_tensor,
    bool reduction_mean,
    const std::optional<Tensor>& weight_tensor,
    const std::optional<Tensor>& input_grad_tensor,
    const std::optional<Tensor>& divisor_tensor,
    int32_t ignore_index,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    using OperationType = operations::moreh::moreh_nll_loss_backward::MorehNllLossBackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        reduction_mean,
        ignore_index < 0 ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(ignore_index),
        memory_config.value_or(target_tensor.memory_config()),
        init_device_compute_kernel_config(target_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{
        target_tensor, output_grad_tensor, weight_tensor, divisor_tensor, input_grad_tensor};

    return device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn
