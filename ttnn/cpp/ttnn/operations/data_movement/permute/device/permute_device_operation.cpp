// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/cpp/ttnn/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"

namespace ttnn::operations::data_movement {

PermuteDeviceOperation::program_factory_t PermuteDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& dims = operation_attributes.dims;
    if (tensor_args.input_tensor.get_layout() == Layout::ROW_MAJOR) {
        // If the last dimension is not permuted, we can use the row-invariant kernel
        if (dims.back() == tensor_args.input_tensor.get_logical_shape().rank() - 1) {
            return MultiCoreRowInvariant{};
        }
        // Otherwise, we need to use the blocked generic, row moving kernel
        return MultiCoreBlockedGeneric{};
    } else {
        // If the input tensor is not row-major, we need to use the tiled kernels
        uint32_t rank = tensor_args.input_tensor.get_logical_shape().rank();
        if ((dims[rank - 1] == rank - 1 && dims[rank - 2] == rank - 2) ||
            (dims[rank - 1] == rank - 2 && dims[rank - 2] == rank - 1)) {
            return MultiCoreTileInvariant{};
        }
    }
    return MultiCoreBlockedGeneric{};
}

void PermuteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto& dims = attributes.dims;
    auto rank = tensor_args.input_tensor.get_logical_shape().rank();
    TT_FATAL(
        attributes.dims.size() == tensor_args.input_tensor.get_logical_shape().rank(),
        "Permute dimensions must match input tensor rank");
    TT_FATAL(tensor_args.input_tensor.is_sharded() == false, "Permute operation does not support sharded input tensor");
    TT_FATAL(
        tensor_args.input_tensor.get_layout() == Layout::ROW_MAJOR ||
            (tensor_args.input_tensor.get_layout() == Layout::TILE &&
             ((dims[rank - 1] == rank - 1 && dims[rank - 2] == rank - 2) ||
              (dims[rank - 1] == rank - 2 && dims[rank - 2] == rank - 1))),
        "Permute operation only supports row-major layout");
}

void PermuteDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

PermuteDeviceOperation::shape_return_value_t PermuteDeviceOperation::compute_output_shapes(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    SmallVector<uint32_t> shape;
    auto input_shape = tensor_args.input_tensor.get_logical_shape();
    shape.reserve(input_shape.rank());
    for (auto dim : attributes.dims) {
        shape.push_back(input_shape[dim]);
    }
    return ttnn::SimpleShape(shape);
}

PermuteDeviceOperation::tensor_return_value_t PermuteDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    const auto& input_tensor = tensor_args.input_tensor;
    return create_device_tensor(
        output_shape,
        input_tensor.dtype(),
        input_tensor.layout(),
        input_tensor.device(),
        operation_attributes.output_mem_config);
}

std::tuple<PermuteDeviceOperation::operation_attributes_t, PermuteDeviceOperation::tensor_args_t>
PermuteDeviceOperation::invoke(
    const Tensor& input_tensor,
    const SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return {
        operation_attributes_t{.dims = dims, .output_mem_config = memory_config.value_or(input_tensor.memory_config())},
        tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)}};
}

}  // namespace ttnn::operations::data_movement
