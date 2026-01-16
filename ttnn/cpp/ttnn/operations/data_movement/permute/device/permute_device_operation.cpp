// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "permute_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement {
PermuteDeviceOperation::program_factory_t PermuteDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& dims = operation_attributes.dims;
    if (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR) {
        // If the last dimension is not permuted, we can use the row-invariant kernel
        if (dims.back() == tensor_args.input_tensor.logical_shape().rank() - 1) {
            return MultiCoreRowInvariant{};
        }
        // Otherwise, we need to use the blocked generic, row moving kernel
        return MultiCoreBlockedGeneric{};
    }  // If the input tensor is not row-major, we need to use the tiled kernels
    uint32_t rank = tensor_args.input_tensor.logical_shape().rank();
    // When the tiled dimensions are not moved, we use this kernel
    if ((dims[rank - 1] == rank - 1 && dims[rank - 2] == rank - 2) ||
        (dims[rank - 1] == rank - 2 && dims[rank - 2] == rank - 1)) {
        return MultiCoreTileInvariant{};
    }
    if (dims[rank - 1] == rank - 1 || dims[rank - 1] == rank - 2) {  // When only one of the tiled dimensions
                                                                     // is moved
        return MultiCoreTileRowInvariant{};
    }
    return MultiCoreTiledGeneric{};  // When both the tiled dimensions are moved
}

void PermuteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        attributes.dims.size() == tensor_args.input_tensor.logical_shape().rank(),
        "Permute dimensions must match input tensor rank");
}

void PermuteDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {}

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

tt::tt_metal::operation::OpPerformanceModelGeneral<PermuteDeviceOperation::tensor_return_value_t>
PermuteDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*op_attr*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output, false, 0, true);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

PermuteDeviceOperation::tensor_return_value_t PermuteDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PermuteDeviceOperation::tensor_return_value_t permute(
    const Tensor& input_tensor,
    const SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    float pad_value) {
    using OperationType = ttnn::operations::data_movement::PermuteDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dims = dims,
            .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
            .pad_value = pad_value},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)});
}
}  // namespace ttnn::prim
