// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_group_norm {
void MorehGroupNormOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    auto& output = tensor_args.output;
    auto& mean = tensor_args.mean;
    auto& rstd = tensor_args.rstd;

    auto& gamma = tensor_args.gamma;
    auto& beta = tensor_args.beta;

    auto num_groups = operation_attributes.num_groups;

    using namespace tt::operations::primary;

    check_tensor(input, "Moreh group norm", "input");

    check_tensor(output, "Moreh group norm", "output");
    check_tensor(mean, "Moreh group norm", "mean");
    check_tensor(rstd, "Moreh group norm", "rstd");

    check_tensor(gamma, "Moreh group norm", "gamma");
    check_tensor(beta, "Moreh group norm", "beta");

    // input (N, C, *)
    auto C = input.get_logical_shape()[1];
    TT_FATAL(C % num_groups == 0, "Moreh group norm: input_shape[1] must be divisible by num_groups.");
    // output (N, C, *)
    if (output.has_value()) {
        C = output.value().get_logical_shape()[1];
        TT_FATAL(C % num_groups == 0, "Moreh group norm: output_shape[1] must be divisible by num_groups.");
    }
    // gamma (1, C)
    if (gamma.has_value()) {
        C = gamma.value().get_logical_shape()[-1];
        TT_FATAL(C % num_groups == 0, "Moreh group norm: gamma_shape[-1] must be divisible by num_groups.");
    }
    // beta (1, C)
    if (beta.has_value()) {
        C = beta.value().get_logical_shape()[-1];
        TT_FATAL(C % num_groups == 0, "Moreh group norm: beta_shape[-1] must be divisible by num_groups.");
    }

    // mean (N, num_groups)
    if (mean.has_value()) {
        TT_FATAL(
            mean.value().get_logical_shape()[-1] == num_groups,
            "Moreh group norm: mean_shape[-1] must match num_groups.");
    }
    // rstd (N, num_groups)
    if (rstd.has_value()) {
        TT_FATAL(
            rstd.value().get_logical_shape()[-1] == num_groups,
            "Moreh group norm: rstd_shape[-1] must match num_groups.");
    }
}

MorehGroupNormOperation::program_factory_t MorehGroupNormOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    auto rank = input.get_logical_shape().rank();
    if (rank == 2) {
        return MorehGroupNorm2DFactory();
    }
    if (rank == 3) {
        return MorehGroupNorm3DFactory();
    }

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

MorehGroupNormOperation::shape_return_value_t MorehGroupNormOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    // output shape
    const auto output_shape = tensor_args.input.get_logical_shape();

    // mean, rstd (N, num_groups)
    const auto N = output_shape[0];
    const auto num_groups = operation_attributes.num_groups;

    SimpleShape mean_rstd_shape({N, num_groups});

    return {output_shape, mean_rstd_shape, mean_rstd_shape};
}

MorehGroupNormOperation::tensor_return_value_t MorehGroupNormOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    auto dtype = tensor_args.input.get_dtype();
    Layout layout{Layout::TILE};
    auto device = tensor_args.input.device();

    std::vector<std::optional<Tensor>> result;
    result.reserve(3);

    // output
    if (tensor_args.output.has_value()) {
        result.push_back(tensor_args.output.value());
    } else {
        result.push_back(
            create_device_tensor(output_shapes[0].value(), dtype, layout, device, operation_attributes.memory_config));
    }

    // mean
    if (tensor_args.mean.has_value()) {
        result.push_back(tensor_args.mean.value());
    } else if (operation_attributes.are_required_outputs[1]) {
        result.push_back(create_device_tensor(
            output_shapes[1].value(), dtype, layout, device, operation_attributes.mean_memory_config));
    } else {
        result.push_back(std::nullopt);
    }

    // rstd
    if (tensor_args.rstd.has_value()) {
        result.push_back(tensor_args.rstd.value());
    } else if (operation_attributes.are_required_outputs[2]) {
        result.push_back(create_device_tensor(
            output_shapes[2].value(), dtype, layout, device, operation_attributes.rstd_memory_config));
    } else {
        result.push_back(std::nullopt);
    }
    return std::move(result);
}

std::tuple<MorehGroupNormOperation::operation_attributes_t, MorehGroupNormOperation::tensor_args_t>
MorehGroupNormOperation::invoke(
    const Tensor& input,
    const uint32_t num_groups,
    const float eps,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor> output,
    const std::optional<const Tensor> mean,
    const std::optional<const Tensor> rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<MemoryConfig>& mean_memory_config,
    const std::optional<MemoryConfig>& rstd_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    operation_attributes_t operation_attributes{
        num_groups,
        eps,
        are_required_outputs,
        memory_config.value_or(input.memory_config()),
        mean_memory_config.value_or(input.memory_config()),
        rstd_memory_config.value_or(input.memory_config()),
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)};
    tensor_args_t tensor_args{input, gamma, beta, output, mean, rstd};
    return {operation_attributes, tensor_args};
}
}  // namespace ttnn::operations::moreh::moreh_group_norm
