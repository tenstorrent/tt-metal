// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "tt_stl/type_name.hpp"
#include "ttnn/cpp/ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_dot_op {
MorehDotOperation::program_factory_t MorehDotOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // For now we litteraly don't care and return a single factory. Whatever
    return SingleCore{};
}

void MorehDotOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
}

void MorehDotOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
}

void MorehDotOperation::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    TT_ASSERT(tt::operations::primary::is_1d_tensor(input_tensor_a));
    TT_ASSERT(tt::operations::primary::is_1d_tensor(input_tensor_b));

    const auto& a_shape_wo_padding = input_tensor_a.get_legacy_shape().without_padding();
    const auto& b_shape_wo_padding = input_tensor_b.get_legacy_shape().without_padding();
    TT_ASSERT(a_shape_wo_padding[3] == b_shape_wo_padding[3]);

    TT_ASSERT(
        input_tensor_a.get_dtype() == DataType::BFLOAT16 || input_tensor_a.get_dtype() == DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_ASSERT(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_ASSERT(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");
    TT_ASSERT(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
}

MorehDotOperation::shape_return_value_t MorehDotOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    const auto& input_tensor = tensor_args.input_tensor_a;
    auto output_shape = input_tensor.get_shape().value;
    auto padding = output_shape.padding();
    output_shape[3] = TILE_WIDTH;
    padding[3] = Padding::PadDimension{0, 31};
    return ttnn::Shape{tt::tt_metal::Shape(output_shape, padding)};
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
        input_tensor.device());
}

std::tuple<MorehDotOperation::operation_attributes_t, MorehDotOperation::tensor_args_t>
MorehDotOperation::invoke(
        const Tensor &input_tensor_a,
        const Tensor &input_tensor_b,
        const DataType output_dtype,
        const std::optional<MemoryConfig> &output_mem_config) {
    return {
        operation_attributes_t{output_dtype, output_mem_config},
        tensor_args_t{input_tensor_a, input_tensor_b}
    };
}

}
