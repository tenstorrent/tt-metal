// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"

namespace ttnn::operations::data_movement {

PermuteDeviceOperation::program_factory_t PermuteDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto& dims = operation_attributes.dims;
    if (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR) {
        // If the last dimension is not permuted, we can use the row-invariant kernel
        if (dims.back() == tensor_args.input_tensor.logical_shape().rank() - 1) {
            return MultiCoreRowInvariant{};
        }
        // Otherwise, we need to use the blocked generic, row moving kernel
        return MultiCoreBlockedGeneric{};
    } else {
        // If the input tensor is not row-major, we need to use the tiled kernels
        uint32_t rank = tensor_args.input_tensor.logical_shape().rank();
        // When the tiled dimensions are not moved, we use this kernel
        if ((dims[rank - 1] == rank - 1 && dims[rank - 2] == rank - 2) ||
            (dims[rank - 1] == rank - 2 && dims[rank - 2] == rank - 1)) {
            return MultiCoreTileInvariant{};
        } else if (dims[rank - 1] == rank - 1 || dims[rank - 1] == rank - 2) {  // When only one of the tiled dimensions
                                                                                // is moved
            return MultiCoreTileRowInvariant{};
        } else {
            return MultiCoreTiledGeneric{};  // When both the tiled dimensions are moved
        }
    }
}

void PermuteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto& dims = attributes.dims;
    auto rank = tensor_args.input_tensor.logical_shape().rank();
    TT_FATAL(
        attributes.dims.size() == tensor_args.input_tensor.logical_shape().rank(),
        "Permute dimensions must match input tensor rank");
    TT_FATAL(tensor_args.input_tensor.is_sharded() == false, "Permute operation does not support sharded input tensor");
}

void PermuteDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

PermuteDeviceOperation::spec_return_value_t PermuteDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    SmallVector<uint32_t> shape;
    const auto& input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.logical_shape();
    shape.reserve(input_shape.rank());
    for (auto dim : attributes.dims) {
        shape.push_back(input_shape[dim]);
    }

    return TensorSpec(
        Shape(std::move(shape)),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), attributes.output_mem_config));
}

PermuteDeviceOperation::tensor_return_value_t PermuteDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

std::tuple<PermuteDeviceOperation::operation_attributes_t, PermuteDeviceOperation::tensor_args_t>
PermuteDeviceOperation::invoke(
    const Tensor& input_tensor,
    const SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    const std::optional<float>& pad_value) {
    return {
        operation_attributes_t{
            .dims = dims,
            .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
            .pad_value = pad_value},
        tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)}};
}

}  // namespace ttnn::operations::data_movement
