// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cumprod_device_operation.hpp"
#include <magic_enum/magic_enum.hpp>

namespace ttnn::operations::experimental::reduction {

CumprodDeviceOperation::program_factory_t CumprodDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SingleCoreCumprodProgramFactory{};
}

void CumprodDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor{tensor_args.input_tensor};
    auto& optional_out{tensor_args.optional_out};
    auto out_memory_config{optional_out.has_value() ? optional_out->memory_config() : attributes.output_memory_config};
    auto out_dtype{DataType::INVALID};
    const auto& input_dtype{attributes.dtype};
    const auto& dim{attributes.dim};

    if (optional_out.has_value()) {
        // TODO(jbbieniekTT): in this case, automatic conversion should be performed as per Torch's policy
        // TT_FATAL(
        //     input_tensor.get_dtype() == optional_out->get_dtype(),
        //     "The dtype of the input tensor doesn't match the dtype of the preallocated tensor.\n"
        //     "Input tensor's dtype: {}\n"
        //     "Output tensor's dtype: {}",
        //     magic_enum::enum_name(input_tensor.get_dtype()),
        //     magic_enum::enum_name(optional_out->get_dtype()));

        const auto computed_output_shape{compute_output_specs(attributes, tensor_args).logical_shape()};
        const auto preallocated_output_shape = optional_out.value().get_logical_shape();
        TT_FATAL(
            computed_output_shape == preallocated_output_shape,
            "The shapes of the input and the preallocated tensors are not equal.\n"
            "Input tensor's shape: {}\n"
            "Preallocated tensor's shape: {}",
            computed_output_shape,
            preallocated_output_shape);

        out_dtype = optional_out->get_dtype();
    } else {
        // TODO(jbbieniekTT): in this case, automatic conversion should be performed as per Torch's policy
        // TT_FATAL(
        //     attributes.dtype == input_tensor.get_dtype(),
        //     "The input tensor's dtype doesn't match the dtype provided in the argument.\n"
        //     "Input tensor's dtype: {}\n"
        //     "Provided dtype: {}",
        //     magic_enum::enum_name(input_tensor.get_dtype()),
        //     magic_enum::enum_name(input_dtype));

        out_dtype = input_tensor.get_dtype();
    }

    TT_FATAL(
        ((dim >= -static_cast<decltype(dim)>(input_tensor.get_padded_shape().rank())) &&
         (dim < static_cast<decltype(dim)>(input_tensor.get_padded_shape().rank()))),
        "The requested cumulation axis is {}, while the input thensor has rank {}.",
        dim,
        input_tensor.get_padded_shape().rank());

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
        input_tensor.get_layout() == Layout::TILE,
        "The provided input tensor has a non-tile layout: {}.",
        magic_enum::enum_name(input_tensor.get_layout()));

    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "The ttnn.cumprod operation requires the memory layout of the input tensor to be "
        "interleaved. Instead, it is {}.",
        magic_enum::enum_name(input_tensor.memory_config().memory_layout));
}

void CumprodDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

CumprodDeviceOperation::spec_return_value_t CumprodDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_out.has_value()) {
        return tensor_args.optional_out->get_tensor_spec();
    }

    auto output_layout{Layout::TILE};
    if (attributes.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input_tensor.get_layout();
    }

    const auto output_shape{tensor_args.input_tensor.logical_shape()};
    return TensorSpec{
        output_shape,
        TensorLayout{tensor_args.input_tensor.get_dtype(), output_layout, attributes.output_memory_config}};
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

hash::hash_t CumprodDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return operation::hash_operation<CumprodDeviceOperation>(
        args,
        select_program_factory(args, tensor_args).index(),
        tensor_args.input_tensor.dtype(),
        tensor_args.input_tensor.get_padded_shape(),
        tensor_args.input_tensor.get_logical_shape(),
        std::get<DeviceStorage>(tensor_args.input_tensor.storage()).memory_config(),
        tensor_args.input_tensor.get_padded_shape().volume(),
        tensor_args.optional_out.has_value() ? tensor_args.optional_out->dtype() : DataType::INVALID,
        tensor_args.optional_out.has_value() ? tensor_args.optional_out->get_padded_shape() : Shape{});
}

}  // namespace ttnn::operations::experimental::reduction
