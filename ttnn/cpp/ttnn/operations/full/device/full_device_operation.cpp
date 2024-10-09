// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::full {
void FullOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto any = tensor_args.any;
    TT_FATAL(any.storage_type() == StorageType::DEVICE, "Full: Any tensor must be on device");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Full: Not currently supporting sharding");
    TT_FATAL(operation_attributes.layout == Layout::TILE, "Full: Not currently supporting row major layout");
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

FullOperation::tensor_return_value_t FullOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        operation_attributes.shape,
        operation_attributes.dtype,
        operation_attributes.layout,
        tensor_args.any.device(),
        operation_attributes.memory_config);
}

std::tuple<FullOperation::operation_attributes_t, FullOperation::tensor_args_t> FullOperation::invoke(
    const SimpleShape& shape,
    const std::variant<float, int> fill_value,
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
