// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "intimg_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

IntImgDeviceOperation::program_factory_t IntImgDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return IntImgProgramFactory{};
}

void IntImgDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_shape = tensor_args.logical_shape();
    const auto& input_layout = tensor_args.layout();
    const auto& input_dtype = tensor_args.dtype();
    TT_FATAL(
        input_shape.rank() == 4,
        "intimg supports only 4D tensors, the input tensor has {} instead",
        input_shape.rank());
    TT_FATAL(input_shape[0] == 1, "intimg supports only one batch, found {} instead", input_shape[0]);
    TT_FATAL(input_layout == Layout::TILE, "only tile layout is supported, {} was provided instead", input_layout);
    TT_FATAL(
        input_dtype == DataType::BFLOAT16 || input_dtype == DataType::FLOAT32,
        "only bf16 and fp32 dtypes are supported, {} was provided instead",
        input_dtype);
}

void IntImgDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

IntImgDeviceOperation::spec_return_value_t IntImgDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args) {
    auto output_layout{Layout::TILE};
    const auto& input_tensor = tensor_args;
    const auto& output_shape{input_tensor.logical_shape()};
    return TensorSpec{output_shape, TensorLayout{input_tensor.dtype(), output_layout, input_tensor.memory_config()}};
}

IntImgDeviceOperation::tensor_return_value_t IntImgDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor intimg(const Tensor& input_tensor) {
    using OperationType = ttnn::experimental::prim::IntImgDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{};
    const auto& tensor_args = input_tensor;

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
