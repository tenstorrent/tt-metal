// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_device_operation.hpp"

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
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

    check_tensor(input, "moreh_group_norm", "input");

    check_tensor(output, "moreh_group_norm", "output");
    check_tensor(mean, "moreh_group_norm", "mean");
    check_tensor(rstd, "moreh_group_norm", "rstd");

    check_tensor(gamma, "moreh_group_norm", "gamma");
    check_tensor(beta, "moreh_group_norm", "beta");

    // input (N, C, H, W)
    auto C = input.get_shape().value[1];
    TT_FATAL(C % num_groups == 0, "input_shape[1] must be divisible by num_groups.");
    // output (N, C, H, W)
    if (output.has_value()) {
        C = output.value().get_shape().value[1];
        TT_FATAL(C % num_groups == 0, "output_shape[1] must be divisible by num_groups.");
    }
    // gamma (1, 1, 1, C)
    if (gamma.has_value()) {
        C = gamma.value().get_shape().value.without_padding()[-1];
        TT_FATAL(C % num_groups == 0, "gamma_shape[-1] must be divisible by num_groups.");
    }
    // beta (1, 1, 1, C)
    if (beta.has_value()) {
        C = beta.value().get_shape().value.without_padding()[-1];
        TT_FATAL(C % num_groups == 0, "beta_shape[-1] must be divisible by num_groups.");
    }

    // mean (1, 1, N, num_groups)
    if (mean.has_value()) {
        TT_FATAL(
            mean.value().get_shape().value.without_padding()[-1] == num_groups,
            "mean_shape[-1] must match num_groups.");
    }
    // rstd (1, 1, N, num_groups)
    if (rstd.has_value()) {
        TT_FATAL(
            rstd.value().get_shape().value.without_padding()[-1] == num_groups,
            "rstd_shape[-1] must match num_groups.");
    }
}

MorehGroupNormOperation::program_factory_t MorehGroupNormOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
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
    // mean, rstd (1, 1, N, num_groups)
    const auto output_shape = tensor_args.input.get_shape();
    const auto N = output_shape.value[0];
    const auto num_groups = operation_attributes.num_groups;
    const std::vector<uint32_t> mean_rstd_origin_shape{
        1,
        1,
        TILE_HEIGHT * ((N + TILE_HEIGHT - 1) / TILE_HEIGHT),
        TILE_WIDTH * ((num_groups + TILE_WIDTH - 1) / TILE_WIDTH)};

    auto mean_rstd_padding = output_shape.value.padding();
    mean_rstd_padding[2] = Padding::PadDimension{0, TILE_HEIGHT - (N % TILE_HEIGHT)};
    mean_rstd_padding[3] = Padding::PadDimension{0, TILE_WIDTH - (num_groups % TILE_WIDTH)};

    Shape mean_rstd_shape = Shape(tt::tt_metal::LegacyShape(mean_rstd_origin_shape, mean_rstd_padding));
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
