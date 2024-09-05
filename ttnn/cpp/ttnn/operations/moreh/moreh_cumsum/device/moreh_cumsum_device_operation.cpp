// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_cumsum_device_operation.hpp"

#include <cstdint>

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_cumsum {
void MorehCumsumDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_ASSERT((dim >= 0 && dim <= 3), "dim should be 0 - 3");
    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    auto input_shape = input.get_shape();
    const auto& output_shape = output.get_shape();
    auto input_shape_wo_padding = input.get_shape().value.without_padding();
    const auto& output_shape_wo_padding = output.get_shape().value.without_padding();

    for (int i = 0; i < input_shape.rank(); ++i) {
        TT_ASSERT(input_shape[i] == output_shape[i]);
        TT_ASSERT(input_shape_wo_padding[i] == output_shape_wo_padding[i]);
    }
}

MorehCumsumDeviceOperation::program_factory_t MorehCumsumDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehCumsumDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehCumsumDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehCumsumDeviceOperation::shape_return_value_t MorehCumsumDeviceOperation::compute_output_shapes(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto output_shape = input_tensor.get_shape().value;
    // auto padding = output_shape.padding();
    // output_shape[3] = TILE_WIDTH;
    // padding[3] = Padding::PadDimension{0, 31};
    return ttnn::Shape{tt::tt_metal::Shape(output_shape)};
    // Inplace
    // return Shape(NULL);
};

MorehCumsumDeviceOperation::tensor_return_value_t MorehCumsumDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    return create_device_tensor(
        output_shape,
        input_tensor.tensor_attributes->dtype,
        input_tensor.tensor_attributes->layout,
        input_tensor.device());
    // Inplace
    // return {};
}

std::tuple<MorehCumsumDeviceOperation::operation_attributes_t, MorehCumsumDeviceOperation::tensor_args_t>
MorehCumsumDeviceOperation::invoke(const Tensor& input, const Tensor& output, const int64_t dim, const bool flip) {
    return {
        operation_attributes_t{dim, flip},
        tensor_args_t{input, output}};
}
}  // namespace ttnn::operations::moreh::moreh_cumsum
