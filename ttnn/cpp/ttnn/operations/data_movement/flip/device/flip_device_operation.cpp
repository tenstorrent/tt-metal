// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/flip/device/flip_device_operation.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::data_movement {

FlipDeviceOperation::program_factory_t FlipDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto layout = input_tensor.layout();
    if (layout == Layout::TILE) {
        return MultiCoreTiled{};
    }
    return MultiCoreRowMajor{};
}

void FlipDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& dims = operation_attributes.dims;
    const auto input_rank = input_tensor.logical_shape().rank();

    TT_FATAL(input_rank <= 5, "Flip operation supports tensors with rank up to 5, got rank {}", input_rank);
    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");

    // Validate flip dimensions are within tensor rank
    for (auto dim : dims) {
        auto normalized_dim = input_tensor.logical_shape().get_normalized_index(dim);
        TT_FATAL(
            normalized_dim < input_rank, "Flip dimension {} is out of bounds for tensor with rank {}", dim, input_rank);
    }

    // Validate no duplicate dimensions
    std::set<int64_t> unique_dims;
    for (auto dim : dims) {
        auto normalized_dim = input_tensor.logical_shape().get_normalized_index(dim);
        TT_FATAL(unique_dims.insert(normalized_dim).second, "Duplicate dimension {} in flip dimensions", dim);
    }

    // TODO do we need this check?
    TT_FATAL(tensor_args.input_tensor.is_sharded() == false, "Flip operation does not support sharded input tensor");
}

void FlipDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // No additional validation needed on cache hit
}

FlipDeviceOperation::spec_return_value_t FlipDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout(input_tensor.dtype(), input_tensor.layout(), operation_attributes.output_mem_config));
}

FlipDeviceOperation::tensor_return_value_t FlipDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    // Create output tensor with same shape as input
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), input_tensor.device());
}

std::tuple<FlipDeviceOperation::operation_attributes_t, FlipDeviceOperation::tensor_args_t> FlipDeviceOperation::invoke(
    const Tensor& input_tensor,
    const SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return {
        operation_attributes_t{.dims = dims, .output_mem_config = memory_config.value_or(input_tensor.memory_config())},
        tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)}};
}

}  // namespace ttnn::operations::data_movement
