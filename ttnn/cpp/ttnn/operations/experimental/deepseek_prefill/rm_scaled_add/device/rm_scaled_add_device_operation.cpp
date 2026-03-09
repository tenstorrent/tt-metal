// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rm_scaled_add_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

void RmScaledAddDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& /* attrs */, const tensor_args_t& tensor_args) {
    const auto& input_a = tensor_args.input_a;
    const auto& input_b = tensor_args.input_b;

    TT_FATAL(input_a.dtype() == tt::tt_metal::DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
    TT_FATAL(input_b.dtype() == tt::tt_metal::DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");

    TT_FATAL(input_a.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");
    TT_FATAL(input_b.layout() == tt::tt_metal::Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for inputs!");

    const auto& a_shape = input_a.padded_shape();
    const auto& b_shape = input_b.padded_shape();

    TT_FATAL(a_shape == b_shape, "Input tensors must have the same shape!");

    // Validate the shape allows treating as tiles (7168 elements -> 7 tiles)
    uint32_t total_elements = a_shape.volume();
    TT_FATAL(total_elements % 1024 == 0, "Total elements must be divisible by 1024 (tile size)!");

    TT_FATAL(input_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Only interleaved memory is supported!");
    TT_FATAL(input_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Only interleaved memory is supported!");
}

RmScaledAddDeviceOperation::spec_return_value_t RmScaledAddDeviceOperation::compute_output_specs(
    const operation_attributes_t& /* attrs */, const tensor_args_t& tensor_args) {
    return tensor_args.input_a.tensor_spec();
}

tt::stl::hash::hash_t RmScaledAddDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input_a = tensor_args.input_a;
    const auto& input_shape = input_a.padded_shape();

    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<RmScaledAddDeviceOperation>(
        attrs.scale,
        input_a.dtype(),
        input_a.memory_config(),
        input_shape);

    return hash;
}

RmScaledAddDeviceOperation::tensor_return_value_t RmScaledAddDeviceOperation::create_output_tensors(
    const operation_attributes_t& /* attrs */, const tensor_args_t& tensor_args) {
    const auto& input_a = tensor_args.input_a;
    return create_device_tensor(
        input_a.tensor_spec(),
        input_a.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor rm_scaled_add(const Tensor& input_a, const Tensor& input_b, float scale) {
    using OperationType = ttnn::experimental::prim::RmScaledAddDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{.scale = scale};
    auto tensor_args = OperationType::tensor_args_t{.input_a = input_a, .input_b = input_b};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
