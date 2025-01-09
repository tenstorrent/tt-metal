// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization {
void BatchNormOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& [input, batch_mean, batch_var, weight, bias, running_mean, running_var, output] = tensor_args;

    check_tensor(input, "batch_norm", "input");
    check_tensor(batch_mean, "batch_norm", "batch_mean");
    check_tensor(batch_var, "batch_norm", "batch_var");
    check_tensor(weight, "batch_norm", "weight");
    check_tensor(bias, "batch_norm", "bias");
    check_tensor(output, "batch_norm", "output");
    check_tensor(running_mean, "batch_norm", "running_mean");
    check_tensor(running_var, "batch_norm", "running_var");

    // input (N, C, H, W)
    auto C = input.get_logical_shape()[1];
    // output (N, C, H, W)
    if (output.has_value()) {
        auto check_C = output.value().get_logical_shape()[1];
        TT_FATAL(C == check_C, "output_shape[1] must be the same as input's channel size.");
    }

    // mean (1, C, 1, 1)
    TT_FATAL(batch_mean.get_logical_shape()[1] == C, "batch_mean_shape[1] must be the same as input's channel size.");
    // var (1, C, 1, 1)
    TT_FATAL(batch_var.get_logical_shape()[1] == C, "batch_var_shape[1] must be the same as input's channel size.");

    // weight (1, C, 1, 1)
    if (weight.has_value()) {
        TT_FATAL(
            weight.value().get_logical_shape()[1] == C, "weight_shape[1] must be the same as input's channel size.");
        TT_FATAL(
            weight.value().get_logical_shape()[1] == C, "weight_shape[1] must be the same as input's channel size.");
    }

    // bias (1, C, 1, 1)
    if (bias.has_value()) {
        TT_FATAL(bias.value().get_logical_shape()[1] == C, "bias_shape[1] must be the same as input's channel size.");
        TT_FATAL(bias.value().get_logical_shape()[1] == C, "bias_shape[1] must be the same as input's channel size.");
    }

    // running_mean (1, C, 1, 1)
    if (running_mean.has_value()) {
        TT_FATAL(
            running_mean.value().get_logical_shape()[1] == C,
            "running_mean_shape[1] must be the same as input's channel size.");
        TT_FATAL(
            running_mean.value().get_logical_shape()[1] == C,
            "running_mean_shape[1] must be the same as input's channel size.");
    }

    // running_var (1, C, 1, 1)
    if (running_var.has_value()) {
        TT_FATAL(
            running_var.value().get_logical_shape()[1] == C,
            "running_var_shape[1] must be the same as input's channel size.");
        TT_FATAL(
            running_var.value().get_logical_shape()[1] == C,
            "running_var_shape[1] must be the same as input's channel size.");
    }
}

BatchNormOperation::program_factory_t BatchNormOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return BatchNormFactory();
}

void BatchNormOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    const auto& [input, batch_mean, batch_var, weight, bias, running_mean, running_var, output] = tensor_args;

    TT_FATAL(input.get_layout() == Layout::TILE, "Input tensor must be must be tilized");
    TT_FATAL(
        input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Input tensor must be interleaved");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Output tensor to eltwise binary must be interleaved");

    TT_FATAL(batch_mean.get_layout() == Layout::TILE, "batch_mean tensor must be tilized");
    TT_FATAL(
        batch_mean.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "batch_mean tensor must be interleaved");

    TT_FATAL(batch_var.get_layout() == Layout::TILE, "batch_var tensor must be tilized");
    TT_FATAL(
        batch_var.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "batch_var tensor must be interleaved");

    if (weight.has_value()) {
        TT_FATAL(weight.value().get_layout() == Layout::TILE, "weight tensor must be tilized");
        TT_FATAL(
            weight.value().memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "weight tensor must be interleaved");
    }

    if (bias.has_value()) {
        TT_FATAL(bias.value().get_layout() == Layout::TILE, "bias tensor must be tilized");
        TT_FATAL(
            bias.value().memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "bias tensor must be interleaved");
    }

    if (running_mean.has_value()) {
        TT_FATAL(running_mean.value().get_layout() == Layout::TILE, "running_mean tensor must be tilized");
        TT_FATAL(
            running_mean.value().memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "running_mean tensor must be interleaved");
    }

    if (running_var.has_value()) {
        TT_FATAL(running_var.value().get_layout() == Layout::TILE, "running_var tensor must be tilized");
        TT_FATAL(
            running_var.value().memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
            "running_var tensor must be interleaved");
    }

    validate_tensors(operation_attributes, tensor_args);
};

void BatchNormOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

DataType BatchNormOperation::operation_attributes_t::get_dtype() const {
    return this->dtype.value_or(this->input_dtype);
}

BatchNormOperation::spec_return_value_t BatchNormOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto output_shape = tensor_args.input.get_logical_shape();
    return TensorSpec(
        output_shape,
        TensorLayout(operation_attributes.get_dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
}

BatchNormOperation::tensor_return_value_t BatchNormOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output;
    if (output_tensor.has_value()) {
        return output_tensor.value();
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

std::tuple<BatchNormOperation::operation_attributes_t, BatchNormOperation::tensor_args_t> BatchNormOperation::invoke(
    const Tensor& input,
    const Tensor& batch_mean,
    const Tensor& batch_var,
    const float eps,
    const float momentum,
    const bool training,
    std::optional<Tensor> weight,
    std::optional<Tensor> bias,
    std::optional<Tensor> running_mean,
    std::optional<Tensor> running_var,
    std::optional<Tensor> output,
    const std::optional<MemoryConfig>& memory_config) {
    operation_attributes_t operation_attributes{eps, momentum, training, memory_config.value_or(input.memory_config())};
    tensor_args_t tensor_args{
        input,
        batch_mean,
        batch_var,
        std::move(weight),
        std::move(bias),
        std::move(running_mean),
        std::move(running_var),
        std::move(output)};
    return {operation_attributes, tensor_args};
}
}  // namespace ttnn::operations::normalization
