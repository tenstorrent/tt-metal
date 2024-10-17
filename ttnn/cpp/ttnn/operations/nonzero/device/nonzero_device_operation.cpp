// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nonzero_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::nonzero {
void NonzeroOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    auto input_shape = input.get_logical_shape();
    TT_FATAL(input_shape[0] == 1 and input_shape[1] == 1 and input_shape[2] == 1, "Input should be 1D");
    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR, "Currently only supporting row major layout");
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to Non-zero need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands to Non-zero need to be allocated in buffers on device!");
    TT_FATAL(
        input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Non-zero does not currently support sharding");
}

NonzeroOperation::program_factory_t NonzeroOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void NonzeroOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void NonzeroOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

NonzeroOperation::shape_return_value_t NonzeroOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    ttnn::SimpleShape num_non_zero_shape({1, 1, 1, 8});
    return {num_non_zero_shape, input.get_logical_shape()};
};

NonzeroOperation::tensor_return_value_t NonzeroOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_shapes = compute_output_shapes(operation_attributes, tensor_args);
    NonzeroOperation::tensor_return_value_t output_tensors;
    const auto device = tensor_args.input.device();
    output_tensors.push_back(create_device_tensor(
        output_shapes[0], DataType::UINT32, Layout::ROW_MAJOR, device, operation_attributes.memory_config));
    output_tensors.push_back(create_device_tensor(
        output_shapes[1], DataType::UINT32, Layout::ROW_MAJOR, device, operation_attributes.memory_config));
    return output_tensors;
}

std::tuple<NonzeroOperation::operation_attributes_t, NonzeroOperation::tensor_args_t> NonzeroOperation::invoke(
    const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    return {
        operation_attributes_t{
            memory_config.value_or(input.memory_config()),
        },
        tensor_args_t{
            input,
        },
    };
}
}  // namespace ttnn::operations::nonzero
