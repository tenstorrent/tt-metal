// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/permute/device/flip_device_operation.hpp"

namespace ttnn::operations::data_movement {

FlipDeviceOperation::program_factory_t FlipDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& dims = operation_attributes.dims;

    if (input_tensor.get_layout() == Layout::TILE) {
        return MultiCoreTiled{};
    }

    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        // Check if we're only flipping the last dimension (most common case)
        if (dims.size() == 1 && dims[0] == input_tensor.get_logical_shape().rank() - 1) {
            return MultiCoreRowMajor{};
        }
    }

    // For complex flip patterns, use generic implementation
    return MultiCoreGeneric{};
}

void FlipDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& dims = operation_attributes.dims;

    // Validate input tensor is on device
    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");

    // Validate tensor rank is within supported range (up to rank 5 as requested)
    const auto input_rank = input_tensor.get_logical_shape().rank();
    TT_FATAL(input_rank <= 5, "Flip operation supports tensors with rank up to 5, got rank {}", input_rank);

    // Validate flip dimensions are within tensor rank
    for (auto dim : dims) {
        auto normalized_dim = input_tensor.get_logical_shape().get_normalized_index(dim);
        TT_FATAL(
            normalized_dim < input_rank, "Flip dimension {} is out of bounds for tensor with rank {}", dim, input_rank);
    }

    // Validate no duplicate dimensions
    std::set<int64_t> unique_dims;
    for (auto dim : dims) {
        auto normalized_dim = input_tensor.get_logical_shape().get_normalized_index(dim);
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
        input_tensor.get_logical_shape(),
        TensorLayout(input_tensor.get_dtype(), input_tensor.get_layout(), operation_attributes.output_mem_config));
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
        operation_attributes_t{.dims = dims, .output_mem_config = memory_config.value_or(input_tensor.memor_config())},
        tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)}};
}

}  // namespace ttnn::operations::data_movement
