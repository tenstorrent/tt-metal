// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <functional>
#include <optional>
#include <variant>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehLayerNormBackwardInputGrad {
    uint32_t normalized_dims;
    MemoryConfig memory_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
};

struct MorehLayerNormBackwardGammaBetaGrad {
    uint32_t normalized_dims;
    MemoryConfig memory_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
};

operation::ProgramWithCallbacks moreh_layernorm_backward_input_grad_impl(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const Tensor &input_grad,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<const Tensor> gamma = std::nullopt);

operation::ProgramWithCallbacks moreh_layernorm_backward_gamma_beta_grad_impl(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const ttnn::DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<const Tensor> gamma_grad = std::nullopt,
    const std::optional<const Tensor> beta_grad = std::nullopt);

Tensor moreh_layernorm_backward_input_grad(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor> input_grad = std::nullopt,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<MemoryConfig> &memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

std::vector<std::optional<Tensor>> moreh_layernorm_backward_gamma_beta_grad(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor> gamma_grad = std::nullopt,
    const std::optional<const Tensor> beta_grad = std::nullopt,
    const std::optional<MemoryConfig> &memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

std::vector<std::optional<Tensor>> moreh_layernorm_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<const Tensor> input_grad = std::nullopt,
    const std::optional<const Tensor> gamma_grad = std::nullopt,
    const std::optional<const Tensor> beta_grad = std::nullopt,
    const std::optional<MemoryConfig> &memory_config = std::nullopt,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt
    );

}  // namespace primary

}  // namespace operations

}  // namespace tt
