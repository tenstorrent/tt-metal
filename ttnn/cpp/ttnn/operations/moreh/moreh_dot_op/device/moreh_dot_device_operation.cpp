// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_dot_op {
MorehDotOperation::program_factory_t MorehDotOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // For now we litteraly don't care and return a single factory. Whatever
    return SingleCore{};
}

void MorehDotOperation::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    TT_FATAL(tt::operations::primary::is_1d_tensor(input_tensor_a), "Invalid input tensor dimensions.");
    TT_FATAL(tt::operations::primary::is_1d_tensor(input_tensor_b), "Invalid input tensor dimensions.");

    const auto& a_shape_wo_padding = input_tensor_a.get_shape();
    const auto& b_shape_wo_padding = input_tensor_b.get_shape();
    TT_FATAL(a_shape_wo_padding[3] == b_shape_wo_padding[3], "Shape without padding must be the same.");

    TT_FATAL(
        input_tensor_a.get_dtype() == DataType::BFLOAT16 || input_tensor_a.get_dtype() == DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
}

void MorehDotOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

void MorehDotOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

MorehDotOperation::shape_return_value_t MorehDotOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    const auto& input_tensor = tensor_args.input_tensor_a;
    auto output_shape = input_tensor.get_shape();
    auto padding = output_shape.padding();
    output_shape[3] = tt::constants::TILE_WIDTH;
    padding[3] = Padding::PadDimension{0, 31};
    return ttnn::Shape(output_shape, padding);
}


MorehDotOperation::tensor_return_value_t MorehDotOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    std::vector<Tensor> input_tensors = {tensor_args.input_tensor_a, tensor_args.input_tensor_b};
    const auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    const auto& input_tensor = tensor_args.input_tensor_a;
    return create_device_tensor(
        output_shape,
        input_tensor.tensor_attributes->dtype,
        input_tensor.tensor_attributes->layout,
        input_tensor.device(),
        operation_attributes.output_memory_config);
}

std::tuple<MorehDotOperation::operation_attributes_t, MorehDotOperation::tensor_args_t>
MorehDotOperation::invoke(
        const Tensor &input_tensor_a,
        const Tensor &input_tensor_b,
        const std::optional<DataType> dtype,
        const std::optional<MemoryConfig> &output_memory_config) {
    return {
        operation_attributes_t{dtype.value_or(input_tensor_a.dtype()), output_memory_config.value_or(input_tensor_a.memory_config())},
        tensor_args_t{input_tensor_a, input_tensor_b}
    };
}

}
