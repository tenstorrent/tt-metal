// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gelu_bw_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "gelu_bw_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::unary_backward::gelu_bw {
namespace {

// GELU_BW supports only floating-point dtypes.
bool is_supported_dtype(DataType dtype) {
    return dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT8_B ||
           dtype == DataType::BFLOAT4_B;
}

void validate_tensor_contract(const Tensor& tensor, const std::string& name) {
    TT_FATAL(
        is_supported_dtype(tensor.dtype()),
        "GELU_BW operation requires a floating-point {} (BFLOAT16, FLOAT32, BFLOAT8_B or BFLOAT4_B). {} data "
        "type: {}",
        name,
        name,
        static_cast<int>(tensor.dtype()));

    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE,
        "GELU_BW operation requires {} to be on Device. {} storage type: {}",
        name,
        name,
        static_cast<int>(tensor.storage_type()));

    TT_FATAL(
        tensor.buffer() != nullptr,
        "GELU_BW operation requires {} to be allocated in a buffer on the device. Buffer is null.",
        name);

    TT_FATAL(!tensor.is_sharded(), "GELU_BW operation does not support sharded {} tensor.", name);

    TT_FATAL(
        tensor.layout() == Layout::TILE,
        "GELU_BW operation requires {} to be in Tile layout. {} layout: {}",
        name,
        name,
        static_cast<int>(tensor.layout()));

    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "GELU_BW operation requires {} to have Interleaved memory layout. {} memory layout: {}",
        name,
        name,
        static_cast<int>(tensor.memory_config().memory_layout()));
}

}  // namespace

void GeluBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& preallocated_input_grad = tensor_args.preallocated_input_grad;
    const auto& input_tensor = tensor_args.input;
    const auto& grad_output = tensor_args.grad_output;
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
        is_supported_dtype(input_tensor.dtype()),
        "GELU_BW operation requires a floating-point input (BFLOAT16, FLOAT32, BFLOAT8_B or BFLOAT4_B). Input data "
        "type: {}",
        static_cast<int>(input_tensor.dtype()));

    validate_tensor_contract(grad_output, "grad_output");

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

    TT_FATAL(
        grad_output.dtype() == input_tensor.dtype(),
        "GELU_BW operation requires grad_output and input data types to match. grad_output data type: {}, input data "
        "type: {}",
        static_cast<int>(grad_output.dtype()),
        static_cast<int>(input_tensor.dtype()));

    TT_FATAL(
        grad_output.logical_shape() == input_tensor.logical_shape(),
        "GELU_BW operation requires grad_output and input to have the same logical shape. grad_output logical shape: "
        "{}, input logical shape: {}",
        grad_output.logical_shape(),
        input_tensor.logical_shape());

    TT_FATAL(
        grad_output.device() == input_tensor.device(),
        "GELU_BW operation requires grad_output and input to be on the same device.");

    if (preallocated_input_grad.has_value()) {
        const auto& preallocated = preallocated_input_grad.value();
        validate_tensor_contract(preallocated, "preallocated input grad");

        TT_FATAL(
            preallocated.logical_shape() == input_tensor.logical_shape(),
            "When a preallocated output tensor is used, GELU_BW operation requires its shape to match the input shape. "
            "Input shape: {}, Preallocated output shape: {}",
            input_tensor.logical_shape(),
            preallocated.logical_shape());

        TT_FATAL(
            preallocated.device() == input_tensor.device(),
            "GELU_BW operation requires the preallocated input grad tensor to be on the same device as input.");
    }
}

tt::tt_metal::TensorSpec GeluBwDeviceOperation::compute_output_specs(
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
    return tt::tt_metal::TensorSpec(output_shape, TensorLayout(output_dtype, output_layout, args.output_memory_config));
}

Tensor GeluBwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_input_grad.has_value()) {
        return *tensor_args.preallocated_input_grad;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::operations::unary_backward::gelu_bw

namespace ttnn::prim {

Tensor gelu_bw(
    const Tensor& grad_output,
    const Tensor& input,
    const std::string& approximate,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::operations::unary_backward::gelu_bw::GeluBwDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .output_dtype = output_dtype, .output_memory_config = output_memory_config, .approximate = approximate};
    auto tensor_args = OperationType::tensor_args_t{
        .grad_output = grad_output, .input = input, .preallocated_input_grad = preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
