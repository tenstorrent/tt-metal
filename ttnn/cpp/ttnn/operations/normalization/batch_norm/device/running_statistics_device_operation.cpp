// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "running_statistics_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization {
void RunningStatistics::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& [batch_mean, batch_var, running_mean, running_var] = tensor_args;

    check_tensor(batch_mean, "running_statistics", "batch_mean");
    check_tensor(batch_var, "running_statistics", "batch_var");
    check_tensor(running_mean, "running_statistics", "running_mean");
    check_tensor(running_var, "running_statistics", "running_var");

    // mean (1, C, 1, 1)
    auto C = batch_mean.get_logical_shape()[1];
    // var (1, C, 1, 1)
    TT_FATAL(batch_var.get_logical_shape()[1] == C, "batch_var_shape[1] must be the same as input's channel size.");

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

RunningStatistics::program_factory_t RunningStatistics::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return RunningStatisticsProgramFactory();
}

void RunningStatistics::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& [batch_mean, batch_var, running_mean, running_var] = tensor_args;

    TT_FATAL(batch_mean.get_layout() == Layout::TILE, "batch_mean tensor must be tilized");
    TT_FATAL(
        batch_mean.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "batch_mean tensor must be interleaved");

    TT_FATAL(batch_var.get_layout() == Layout::TILE, "batch_var tensor must be tilized");
    TT_FATAL(
        batch_var.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "batch_var tensor must be interleaved");

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

void RunningStatistics::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

DataType RunningStatistics::operation_attributes_t::get_dtype() const {
    return this->dtype.value_or(this->input_dtype);
}

RunningStatistics::spec_return_value_t RunningStatistics::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto output_shape = tensor_args.batch_mean.get_logical_shape();
    return TensorSpec(
        output_shape,
        TensorLayout(operation_attributes.get_dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
}

RunningStatistics::tensor_return_value_t RunningStatistics::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.batch_mean.device());
}

std::tuple<RunningStatistics::operation_attributes_t, RunningStatistics::tensor_args_t> RunningStatistics::invoke(
    const Tensor& batch_mean,
    const Tensor& batch_var,
    const float momentum,
    std::optional<Tensor> running_mean,
    std::optional<Tensor> running_var,
    const std::optional<MemoryConfig>& memory_config) {
    operation_attributes_t operation_attributes{momentum, memory_config.value_or(batch_mean.memory_config())};
    tensor_args_t tensor_args{batch_mean, batch_var, std::move(running_mean), std::move(running_var)};
    return {operation_attributes, tensor_args};
}
}  // namespace ttnn::operations::normalization
