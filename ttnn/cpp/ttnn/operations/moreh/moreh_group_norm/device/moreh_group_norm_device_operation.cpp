// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_group_norm {
void MorehGroupNormOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    const auto& output = tensor_args.output;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    auto num_groups = operation_attributes.num_groups;

    check_tensor(input, "moreh_group_norm", "input");

    check_tensor(output, "moreh_group_norm", "output");
    check_tensor(mean, "moreh_group_norm", "mean");
    check_tensor(rstd, "moreh_group_norm", "rstd");

    check_tensor(gamma, "moreh_group_norm", "gamma");
    check_tensor(beta, "moreh_group_norm", "beta");

    // input (N, C, H, W)
    auto C = input.padded_shape()[1];
    TT_FATAL(C % num_groups == 0, "input_shape[1] must be divisible by num_groups.");
    // output (N, C, H, W)
    if (output.has_value()) {
        C = output.value().padded_shape()[1];
        TT_FATAL(C % num_groups == 0, "output_shape[1] must be divisible by num_groups.");
    }
    // gamma (1, 1, 1, C)
    if (gamma.has_value()) {
        C = gamma.value().logical_shape()[-1];
        TT_FATAL(C % num_groups == 0, "gamma_shape[-1] must be divisible by num_groups.");
    }
    // beta (1, 1, 1, C)
    if (beta.has_value()) {
        C = beta.value().logical_shape()[-1];
        TT_FATAL(C % num_groups == 0, "beta_shape[-1] must be divisible by num_groups.");
    }

    // mean (1, 1, N, num_groups)
    if (mean.has_value()) {
        TT_FATAL(mean.value().logical_shape()[-1] == num_groups, "mean_shape[-1] must match num_groups.");
    }
    // rstd (1, 1, N, num_groups)
    if (rstd.has_value()) {
        TT_FATAL(rstd.value().logical_shape()[-1] == num_groups, "rstd_shape[-1] must match num_groups.");
    }
}

MorehGroupNormOperation::program_factory_t MorehGroupNormOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MorehGroupNormFactory();
}

void MorehGroupNormOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

void MorehGroupNormOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

MorehGroupNormOperation::spec_return_value_t MorehGroupNormOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    auto dtype = tensor_args.input.dtype();
    Layout layout{Layout::TILE};
    // mean, rstd (1, 1, N, num_groups)
    const auto output_shape = tensor_args.input.logical_shape();
    const auto N = output_shape[0];
    const auto num_groups = operation_attributes.num_groups;
    Shape mean_rstd_shape({1, 1, N, num_groups});

    std::vector<std::optional<TensorSpec>> result;
    result.reserve(3);

    // output
    if (tensor_args.output.has_value()) {
        result.push_back(tensor_args.output->tensor_spec());
    } else {
        result.push_back(
            TensorSpec(output_shape, TensorLayout(dtype, PageConfig(layout), operation_attributes.memory_config)));
    }

    // mean
    if (tensor_args.mean.has_value()) {
        result.push_back(tensor_args.mean->tensor_spec());
    } else if (operation_attributes.are_required_outputs[1]) {
        result.push_back(
            TensorSpec(mean_rstd_shape, TensorLayout(dtype, PageConfig(layout), operation_attributes.memory_config)));
    } else {
        result.push_back(std::nullopt);
    }

    // rstd
    if (tensor_args.rstd.has_value()) {
        result.push_back(tensor_args.rstd->tensor_spec());
    } else if (operation_attributes.are_required_outputs[2]) {
        result.push_back(
            TensorSpec(mean_rstd_shape, TensorLayout(dtype, PageConfig(layout), operation_attributes.memory_config)));
    } else {
        result.push_back(std::nullopt);
    }
    return result;
}

MorehGroupNormOperation::tensor_return_value_t MorehGroupNormOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.input.device();

    std::vector<std::optional<Tensor>> result;
    result.reserve(3);

    // output
    if (tensor_args.output.has_value()) {
        result.push_back(tensor_args.output.value());
    } else {
        result.push_back(create_device_tensor(*output_specs[0], device));
    }

    // mean
    if (tensor_args.mean.has_value()) {
        result.push_back(tensor_args.mean.value());
    } else if (output_specs[1].has_value()) {
        result.push_back(create_device_tensor(*output_specs[1], device));
    } else {
        result.push_back(std::nullopt);
    }

    // rstd
    if (tensor_args.rstd.has_value()) {
        result.push_back(tensor_args.rstd.value());
    } else if (output_specs[2].has_value()) {
        result.push_back(create_device_tensor(*output_specs[2], device));
    } else {
        result.push_back(std::nullopt);
    }
    return result;
}
}  // namespace ttnn::operations::moreh::moreh_group_norm

namespace ttnn::prim {
ttnn::operations::moreh::moreh_group_norm::MorehGroupNormOperation::tensor_return_value_t moreh_group_norm(
    const Tensor& input,
    const uint32_t num_groups,
    const float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor>& output,
    const std::optional<const Tensor>& mean,
    const std::optional<const Tensor>& rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<MemoryConfig>& mean_memory_config,
    const std::optional<MemoryConfig>& rstd_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::moreh::moreh_group_norm::MorehGroupNormOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        num_groups,
        eps,
        are_required_outputs,
        memory_config.value_or(input.memory_config()),
        mean_memory_config.value_or(input.memory_config()),
        rstd_memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    auto tensor_args = OperationType::tensor_args_t{input, gamma, beta, output, mean, rstd};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
