// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cumulation_device_operation.hpp"
#include <magic_enum/magic_enum.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::cumulation {

CumulationDeviceOperation::program_factory_t CumulationDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return CumulationProgramFactory{};
}

void CumulationDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor{tensor_args.input_tensor};
    auto& optional_out{tensor_args.opt_output};
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
        "ttnn cumulation operations require input to be on a Tenstorrent device. "
        "The input tensor is stored on {}.",
        magic_enum::enum_name(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "ttnn cumulation operations require to be allocated in buffers on the device. "
        "The buffer is null.");

    TT_FATAL(!input_tensor.is_sharded(), "ttnn cumulation operations do not support sharded input tensors.");

    TT_FATAL(
        input_tensor.layout() == Layout::TILE,
        "The provided input tensor has a non-tile layout: {}.",
        magic_enum::enum_name(input_tensor.layout()));

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "ttnn cumulation operations require the memory layout of the input tensor to be "
        "interleaved. Instead, it is {}.",
        magic_enum::enum_name(input_tensor.memory_config().memory_layout()));
}

void CumulationDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

CumulationDeviceOperation::spec_return_value_t CumulationDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.opt_output.has_value()) {
        return tensor_args.opt_output->tensor_spec();
    }

    auto output_layout{Layout::TILE};
    if (attributes.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input_tensor.layout();
    }

    const auto output_shape{tensor_args.input_tensor.logical_shape()};
    return TensorSpec{
        output_shape, TensorLayout{tensor_args.input_tensor.dtype(), output_layout, attributes.output_memory_config}};
}

CumulationDeviceOperation::tensor_return_value_t CumulationDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.opt_output.has_value()) {
        // a copy of a Python object (referencing to the same tensor though) is returned here
        return *tensor_args.opt_output;
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

operation::Hash CumulationDeviceOperation::compute_program_hash(
    const operation_attributes_t& op_args, const tensor_args_t& tensor_args) {
    return operation::hash_operation<CumulationDeviceOperation>(
        select_program_factory(op_args, tensor_args).index(),
        op_args.dim,
        op_args.output_memory_config,
        op_args.flip,
        op_args.dtype,
        op_args.op,
        tensor_args.input_tensor.logical_shape(),
        tensor_args.input_tensor.dtype(),
        tensor_args.input_tensor.memory_config(),
        tensor_args.opt_output.has_value() ? tensor_args.opt_output.value().logical_shape() : Shape{},
        tensor_args.opt_output.has_value() ? tensor_args.opt_output.value().memory_config() : MemoryConfig{},
        tensor_args.opt_output.has_value() ? tensor_args.opt_output.value().dtype() : DataType{});
}

CumulationDeviceOperation::invocation_result_t CumulationDeviceOperation::invoke(
    const Tensor& input_tensor,
    const int32_t& dim,
    std::optional<DataType>& dtype,
    std::optional<Tensor> optional_out,
    const std::optional<MemoryConfig>& memory_config,
    bool flip,
    CumulationOp op) {
    return {
        operation_attributes_t{
            dim,
            dtype.has_value() ? dtype.value() : DataType::INVALID,
            memory_config.has_value() ? *memory_config
                                      : (optional_out.has_value() ? optional_out->memory_config() : MemoryConfig{}),
            flip,
            op},
        tensor_args_t{input_tensor, std::move(optional_out)}};
}

}  // namespace ttnn::operations::reduction::cumulation
