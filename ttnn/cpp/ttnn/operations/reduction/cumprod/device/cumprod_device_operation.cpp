// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumprod_device_operation.hpp"
#include <magic_enum/magic_enum.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction {

CumprodDeviceOperation::program_factory_t CumprodDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MultiCoreCumprodProgramFactory{};
}

void CumprodDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor{tensor_args.input_tensor};
    auto& optional_out{tensor_args.optional_out};
    auto out_memory_config{optional_out.has_value() ? optional_out->memory_config() : attributes.output_memory_config};
    const auto& input_dtype{attributes.dtype};
    const auto& dim = attributes.dim;

    if (optional_out.has_value()) {
        const auto computed_output_shape{compute_output_specs(attributes, tensor_args).logical_shape()};
        const auto preallocated_output_shape = optional_out.value().logical_shape();
        TT_FATAL(
            computed_output_shape == preallocated_output_shape,
            "The shapes of the input and the preallocated tensors are not equal.\n"
            "Input tensor's shape: {}\n"
            "Preallocated tensor's shape: {}",
            computed_output_shape,
            preallocated_output_shape);
    }

    TT_FATAL(
        ((dim >= -static_cast<decltype(dim)>(input_tensor.padded_shape().rank())) &&
         (dim < static_cast<decltype(dim)>(input_tensor.padded_shape().rank()))),
        "The requested cumulation axis is {}, while the input thensor has rank {}.",
        dim,
        input_tensor.padded_shape().rank());

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "The ttnn.cumprod operation requires input to be on a Tenstorrent device. "
        "The input tensor is stored on {}.",
        magic_enum::enum_name(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "The ttnn.cumprod operation requires input to be allocated in buffers on the device. "
        "The buffer is null.");

    TT_FATAL(!input_tensor.is_sharded(), "The ttnn.cumprod operation does not support sharded input tensors.");

    TT_FATAL(
        input_tensor.layout() == Layout::TILE,
        "The provided input tensor has a non-tile layout: {}.",
        magic_enum::enum_name(input_tensor.layout()));

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "The ttnn.cumprod operation requires the memory layout of the input tensor to be "
        "interleaved. Instead, it is {}.",
        magic_enum::enum_name(input_tensor.memory_config().memory_layout()));
}

void CumprodDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

CumprodDeviceOperation::spec_return_value_t CumprodDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_out.has_value()) {
        return tensor_args.optional_out->tensor_spec();
    }

    auto output_layout{Layout::TILE};
    if (attributes.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input_tensor.layout();
    }

    const auto output_shape{tensor_args.input_tensor.logical_shape()};
    return TensorSpec{
        output_shape, TensorLayout{tensor_args.input_tensor.dtype(), output_layout, attributes.output_memory_config}};
}

CumprodDeviceOperation::tensor_return_value_t CumprodDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_out.has_value()) {
        // a copy of a Python object (referencing to the same tensor though) is returned here
        return *tensor_args.optional_out;
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

operation::Hash CumprodDeviceOperation::compute_program_hash(
    const operation_attributes_t& op_args, const tensor_args_t& tensor_args) {
    return operation::hash_operation<CumprodDeviceOperation>(
        select_program_factory(op_args, tensor_args).index(),
        op_args.dim,
        op_args.output_memory_config,
        tensor_args.input_tensor.logical_shape(),
        tensor_args.input_tensor.dtype(),
        tensor_args.input_tensor.memory_config(),
        tensor_args.input_tensor.layout());
}

CumprodDeviceOperation::invocation_result_t CumprodDeviceOperation::invoke(
    const Tensor& input_tensor,
    const int32_t& dim,
    std::optional<DataType>& dtype,
    std::optional<Tensor> optional_out,
    const MemoryConfig& memory_config,
    const QueueId& queue_id) {
    return {
        operation_attributes_t{dim, dtype.has_value() ? dtype.value() : DataType::INVALID, memory_config},
        tensor_args_t{input_tensor, std::move(optional_out)}};
}

}  // namespace ttnn::operations::reduction
