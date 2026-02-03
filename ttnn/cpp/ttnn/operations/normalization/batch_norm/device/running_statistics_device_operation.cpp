// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "running_statistics_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "batch_norm_utils.hpp"

namespace ttnn::operations::normalization {

namespace {
inline void check_tensor_stat(const Tensor& tensor, std::string_view name, std::uint32_t input_c_dim) {
    TT_FATAL(tensor.layout() == Layout::TILE, "batch_norm only supports tiled layout. Got: {}", tensor.layout());
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::FLOAT32,
        "batch_norm only supports bfloat16, float32. Got: {}",
        tensor.dtype());
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE,
        "Operands to batch_norm need to be on device! Got: {}",
        tensor.storage_type());
    TT_FATAL(tensor.buffer() != nullptr, "Operands to batch_norm need to be allocated in buffers on device!");
    TT_FATAL(tensor.logical_shape().rank() == 4, "batch_norm supports tensors of rank 4");
    TT_FATAL(tensor.logical_shape()[1] == input_c_dim, "{}[1] must be the same as input's channel size.", name);
}
}  // namespace

void RunningStatistics::validate_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& [batch_mean, batch_var, running_mean, running_var] = tensor_args;

    // mean (1, C, 1, 1)
    auto C = batch_mean.logical_shape()[1];

    check_tensor_stat(batch_mean, "batch_mean_shape", C);
    check_tensor_stat(batch_var, "batch_var_shape", C);

    // running_mean (1, C, 1, 1)
    if (running_mean.has_value()) {
        check_tensor_stat(running_mean.value(), "running_mean_shape", C);
    }

    // running_var (1, C, 1, 1)
    if (running_var.has_value()) {
        check_tensor_stat(running_var.value(), "running_var_shape", C);
    }
}

RunningStatistics::program_factory_t RunningStatistics::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return RunningStatisticsProgramFactory();
}

void RunningStatistics::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& [batch_mean, batch_var, running_mean, running_var] = tensor_args;

    TT_FATAL(batch_mean.layout() == Layout::TILE, "batch_mean tensor must be tilized");
    TT_FATAL(
        batch_mean.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "batch_mean tensor must be interleaved");

    TT_FATAL(batch_var.layout() == Layout::TILE, "batch_var tensor must be tilized");
    TT_FATAL(
        batch_var.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "batch_var tensor must be interleaved");

    if (running_mean.has_value()) {
        TT_FATAL(running_mean.value().layout() == Layout::TILE, "running_mean tensor must be tilized");
        TT_FATAL(
            running_mean.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "running_mean tensor must be interleaved");
    }

    if (running_var.has_value()) {
        TT_FATAL(running_var.value().layout() == Layout::TILE, "running_var tensor must be tilized");
        TT_FATAL(
            running_var.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
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
    const auto output_shape = tensor_args.batch_mean.logical_shape();
    return TensorSpec(
        output_shape,
        TensorLayout(operation_attributes.get_dtype(), PageConfig(Layout::TILE), operation_attributes.memory_config));
}

RunningStatistics::tensor_return_value_t RunningStatistics::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.batch_mean.device());
}

}  // namespace ttnn::operations::normalization

namespace ttnn::prim {
ttnn::operations::normalization::RunningStatistics::tensor_return_value_t running_statistics(
    const Tensor& batch_mean,
    const Tensor& batch_var,
    float momentum,
    std::optional<Tensor> running_mean,
    std::optional<Tensor> running_var,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::normalization::RunningStatistics;
    OperationType::operation_attributes_t operation_attributes{
        momentum,
        memory_config.value_or(batch_mean.memory_config()),
        ttnn::operations::normalization::batch_norm::utils::resolve_compute_kernel_config(compute_kernel_config, batch_mean),
        batch_mean.dtype()};
    OperationType::tensor_args_t tensor_args{batch_mean, batch_var, std::move(running_mean), std::move(running_var)};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
