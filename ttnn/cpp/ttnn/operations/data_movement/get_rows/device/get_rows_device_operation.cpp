// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "get_rows_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

GetRowsDeviceOperation::program_factory_t GetRowsDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return GetRowsProgramFactory{};
}

void GetRowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& indices = tensor_args.indices_tensor;

    TT_FATAL(
        input.logical_shape().rank() == 2,
        "get_rows: input tensor must be rank 2 [H, W], got rank {} with shape {}",
        input.logical_shape().rank(),
        input.logical_shape());
    TT_FATAL(
        indices.logical_shape().rank() == 1,
        "get_rows: indices tensor must be rank 1 [N], got rank {} with shape {}",
        indices.logical_shape().rank(),
        indices.logical_shape());
    TT_FATAL(indices.logical_shape()[0] > 0, "get_rows: empty index vectors are not supported");
    TT_FATAL(input.logical_shape()[0] > 0, "get_rows: the input tensor must have at least one row");

    TT_FATAL(
        input.layout() == Layout::ROW_MAJOR, "get_rows: input tensor must be ROW_MAJOR, got {}", input.layout());
    TT_FATAL(
        indices.layout() == Layout::ROW_MAJOR, "get_rows: indices tensor must be ROW_MAJOR, got {}", indices.layout());

    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "get_rows: only interleaved input memory is supported in this milestone, got {}",
        input.memory_config().memory_layout());
    TT_FATAL(
        indices.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "get_rows: only interleaved indices memory is supported in this milestone, got {}",
        indices.memory_config().memory_layout());
    TT_FATAL(
        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "get_rows: only interleaved output memory is supported in this milestone, got {}",
        operation_attributes.output_mem_config.memory_layout());

    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32 ||
            input.dtype() == DataType::UINT32 || input.dtype() == DataType::INT32,
        "get_rows: input dtype must be BFLOAT16, FLOAT32, UINT32 or INT32, got {}",
        input.dtype());
    TT_FATAL(
        indices.dtype() == DataType::UINT32 || indices.dtype() == DataType::INT32,
        "get_rows: indices dtype must be UINT32 or INT32, got {}",
        indices.dtype());

    // The index values themselves are validated by the host wrapper before dispatch (it owns the
    // reject-negative/reject-out-of-range semantics); the device op never sees an out-of-range row.
    (void)indices;

    if (tensor_args.optional_output_tensor.has_value()) {
        const auto& output = tensor_args.optional_output_tensor.value();
        const ttnn::Shape expected_shape({indices.logical_shape()[0], input.logical_shape()[-1]});
        TT_FATAL(
            output.logical_shape() == expected_shape,
            "get_rows: preallocated output shape {} does not match the expected shape {}",
            output.logical_shape(),
            expected_shape);
        TT_FATAL(
            output.dtype() == input.dtype(),
            "get_rows: preallocated output dtype {} must match the input dtype {}",
            output.dtype(),
            input.dtype());
        TT_FATAL(
            output.layout() == Layout::ROW_MAJOR,
            "get_rows: preallocated output tensor must be ROW_MAJOR, got {}",
            output.layout());
    }
}

void GetRowsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

ttsl::hash::hash_t GetRowsDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<GetRowsDeviceOperation>, operation_attributes, tensor_args);
}

tt::tt_metal::TensorSpec GetRowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto& indices = tensor_args.indices_tensor;
    const ttnn::Shape output_shape({indices.logical_shape()[0], input.logical_shape()[-1]});
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    return tt::tt_metal::TensorSpec(
        output_shape,
        TensorLayout(input.dtype(), PageConfig(Layout::ROW_MAJOR), operation_attributes.output_mem_config));
}

Tensor GetRowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return *tensor_args.optional_output_tensor;
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

Tensor get_rows(
    const Tensor& input_tensor,
    const Tensor& indices_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& optional_output_tensor) {
    using OperationType = GetRowsDeviceOperation;
    auto memory_config = output_mem_config.value_or(input_tensor.memory_config());
    auto operation_attributes = OperationType::operation_attributes_t{.output_mem_config = memory_config};
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor = input_tensor,
        .indices_tensor = indices_tensor,
        .optional_output_tensor = optional_output_tensor,
    };
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
