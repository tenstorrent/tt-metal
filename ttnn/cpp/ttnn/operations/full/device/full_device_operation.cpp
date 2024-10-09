// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_device_operation.hpp"

#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::full {
void FullOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // TODO: Validate input
}

FullOperation::program_factory_t FullOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void FullOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void FullOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

FullOperation::shape_return_value_t FullOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return operation_attributes.shape;
};

FullOperation::tensor_return_value_t FullOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_shapes(operation_attributes, tensor_args),
        operation_attributes.dtype,
        operation_attributes.layout,
        tensor_args.any.device(),
        operation_attributes.memory_config);
}

std::tuple<FullOperation::operation_attributes_t, FullOperation::tensor_args_t> FullOperation::invoke(
    const Shape& shape,
    const std::variant<int, float> fill_value,
    const Tensor& any,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    return {
        operation_attributes_t{
            shape,
            fill_value,
            dtype.value_or(any.get_dtype()),
            layout.value_or(any.get_layout()),
            memory_config.value_or(any.memory_config()),
        },
        tensor_args_t{
            any,
        },
    };
}
}  // namespace ttnn::operations::full
