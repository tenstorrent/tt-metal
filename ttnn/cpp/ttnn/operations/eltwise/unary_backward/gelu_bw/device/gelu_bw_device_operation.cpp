// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gelu_bw_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "gelu_bw_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::unary_backward::gelu_bw {

void GeluBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& preallocated_input_grad = tensor_args.preallocated_input_grad;
    const auto& input_tensor = tensor_args.input;
    auto out_memory_config = args.output_memory_config;
    auto output_datatype = args.output_dtype;

    if (output_datatype == DataType::INVALID) {
        output_datatype = input_tensor.dtype();
    }

    if (preallocated_input_grad.has_value()) {
        out_memory_config = preallocated_input_grad->memory_config();
        output_datatype = preallocated_input_grad->dtype();
    }

    TT_FATAL(
        output_datatype == input_tensor.dtype(),
        "GELU_BW operation requires input and output data types to match. Input data type: {}, Output data type: {}",
        static_cast<int>(input_tensor.dtype()),
        static_cast<int>(output_datatype));

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "GELU_BW operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to GELU_BW need to be allocated in buffers on the device. Buffer is null.");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "GELU_BW operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        static_cast<int>(input_tensor.memory_config().memory_layout()),
        static_cast<int>(out_memory_config.memory_layout()));

    TT_FATAL(!input_tensor.is_sharded(), "GELU_BW operation does not support sharded input tensor.");

    TT_FATAL(
        input_tensor.layout() == Layout::TILE,
        "GELU_BW operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
        "tensor layout: {}",
        static_cast<int>(input_tensor.layout()));

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "GELU_BW operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
        "memory layout: `{}`",
        static_cast<int>(input_tensor.memory_config().memory_layout()));

    if (preallocated_input_grad.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto preallocated_output_shape = preallocated_input_grad.value().logical_shape();
        TT_FATAL(
            preallocated_output_shape == computed_output_shape,
            "When preallocated output tensor is used, GELU_BW operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            preallocated_output_shape);
    }
}

TensorSpec GeluBwDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_input_grad.has_value()) {
        return tensor_args.preallocated_input_grad->tensor_spec();
    }

    auto output_layout = Layout::TILE;
    if (args.output_memory_config.is_sharded()) {
        output_layout = tensor_args.input.layout();
    }

    DataType output_dtype = args.output_dtype;
    if (output_dtype == DataType::INVALID) {
        output_dtype = tensor_args.input.dtype();
    }

    const auto output_shape = tensor_args.input.logical_shape();
    return TensorSpec(output_shape, TensorLayout(output_dtype, output_layout, args.output_memory_config));
}

Tensor GeluBwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_input_grad.has_value()) {
        return *tensor_args.preallocated_input_grad;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t GeluBwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& grad_output = tensor_args.grad_output;
    const auto& input_shape = input_tensor.padded_shape();
    operation::Hash hash = operation::hash_operation<GeluBwDeviceOperation>(
        args,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        grad_output.dtype(),
        grad_output.memory_config(),
        input_shape.volume());

    return hash;
}

}  // namespace ttnn::operations::unary_backward::gelu_bw

namespace ttnn::operations::unary_backward::gelu_bw {

Tensor launch_gelu_bw(
    const Tensor& grad_output,
    const Tensor& input,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output) {
    auto operation_attributes = GeluBwDeviceOperation::operation_attributes_t{
        .output_dtype = output_dtype, .output_memory_config = output_memory_config};
    auto tensor_args = GeluBwDeviceOperation::tensor_args_t{
        .grad_output = grad_output, .input = input, .preallocated_input_grad = preallocated_output};

    return ttnn::device_operation::launch<GeluBwDeviceOperation>(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::unary_backward::gelu_bw
