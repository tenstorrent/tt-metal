// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_layer_norm_backward.hpp"

#include "device/moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp"
#include "device/moreh_layer_norm_backward_input_grad_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_layer_norm_backward {
std::vector<std::optional<Tensor>> moreh_layer_norm_backward_gamma_beta_grad(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor>& gamma_grad,
    const std::optional<const Tensor>& beta_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    auto device = input.device();
    auto compute_kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<std::optional<Tensor>> outputs(2);
    if (!gamma_grad.has_value() && !beta_grad.has_value()) {
        return outputs;
    }

    const auto& ret = ttnn::prim::moreh_layer_norm_backward_gamma_beta_grad(
        output_grad,
        input,
        mean,
        rstd,
        normalized_dims,
        gamma_grad,
        beta_grad,
        memory_config,
        compute_kernel_config_val);
    return ret;
}

std::vector<std::optional<Tensor>> MorehLayerNormBackward::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& input_grad,
    const std::optional<const Tensor>& gamma_grad,
    const std::optional<const Tensor>& beta_grad,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    std::vector<std::optional<Tensor>> outputs;
    outputs.reserve(3);

    if (input_grad.has_value()) {
        outputs.push_back(ttnn::prim::moreh_layer_norm_backward_input_grad(
            output_grad, input, mean, rstd, normalized_dims, input_grad, gamma, memory_config, compute_kernel_config));
    } else {
        outputs.push_back(std::nullopt);
    }

    const auto& gamma_beta_grad = moreh_layer_norm_backward_gamma_beta_grad(
        output_grad, input, mean, rstd, normalized_dims, gamma_grad, beta_grad, memory_config, compute_kernel_config);
    outputs.push_back(gamma_beta_grad[0]);
    outputs.push_back(gamma_beta_grad[1]);

    return outputs;
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward
