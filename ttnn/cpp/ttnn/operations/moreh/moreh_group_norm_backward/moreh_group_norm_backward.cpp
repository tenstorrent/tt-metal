// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward.hpp"

#include "device/gamma_beta_grad/moreh_group_norm_backward_gamma_beta_grad_device_operation.hpp"
#include "device/input_grad/moreh_group_norm_backward_input_grad_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {
std::vector<std::optional<Tensor>> MorehGroupNormBackward::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const uint32_t num_groups,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> input_grad,
    const std::optional<const Tensor> gamma_grad,
    const std::optional<const Tensor> beta_grad,
    const std::optional<MemoryConfig>& input_grad_memory_config,
    const std::optional<MemoryConfig>& gamma_grad_memory_config,
    const std::optional<MemoryConfig>& beta_grad_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    std::vector<std::optional<Tensor>> outputs;
    if (are_required_outputs[0]) {
        outputs.push_back(ttnn::prim::moreh_group_norm_backward_input_grad(output_grad,
                                                                           input,
                                                                           mean,
                                                                           rstd,
                                                                           num_groups,
                                                                           gamma,
                                                                           input_grad,
                                                                           input_grad_memory_config,
                                                                           compute_kernel_config));
    } else {
        outputs.push_back(std::nullopt);
    }
    if (are_required_outputs[1] || are_required_outputs[2]) {
        std::vector<bool> are_required_outputs_gamma_beta_grad = {are_required_outputs[1], are_required_outputs[2]};
        const auto& dgamma_dbeta =
            ttnn::prim::moreh_group_norm_backward_gamma_beta_grad(output_grad,
                                                                  input,
                                                                  mean,
                                                                  rstd,
                                                                  num_groups,
                                                                  are_required_outputs_gamma_beta_grad,
                                                                  gamma_grad,
                                                                  beta_grad,
                                                                  gamma_grad_memory_config,
                                                                  beta_grad_memory_config,
                                                                  compute_kernel_config);
        outputs.push_back(std::move(dgamma_dbeta[0]));
        outputs.push_back(std::move(dgamma_dbeta[1]));

    } else {
        outputs.push_back(std::nullopt);
        outputs.push_back(std::nullopt);
    }
    return std::move(outputs);
}
std::vector<Tensor> MorehGroupNormBackward::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_inputs) {
    return {
        Tensor(operation::get_workers_for_op_output(input_tensors, optional_inputs)),
        Tensor(operation::get_workers_for_op_output(input_tensors, optional_inputs)),
        Tensor(operation::get_workers_for_op_output(input_tensors, optional_inputs)),
    };
}

std::vector<bool> MorehGroupNormBackward::create_async_return_flag(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const uint32_t num_groups,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> input_grad,
    const std::optional<const Tensor> gamma_grad,
    const std::optional<const Tensor> beta_grad,
    const std::optional<MemoryConfig>& input_grad_memory_config,
    const std::optional<MemoryConfig>& gamma_grad_memory_config,
    const std::optional<MemoryConfig>& beta_grad_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return are_required_outputs;
}
}  // namespace ttnn::operations::moreh::moreh_group_norm_backward
