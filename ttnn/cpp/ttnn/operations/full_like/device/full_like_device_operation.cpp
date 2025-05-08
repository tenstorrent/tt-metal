// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_like_device_operation.hpp"

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::full_like {

FullLikeOperation::program_factory_t FullLikeOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void FullLikeOperation::validate(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    if (operation_attributes.dtype != input.get_dtype()) {
        TT_FATAL(
            input.get_layout() == Layout::TILE, "Full Like: Data type conversion is only supported with tile layout");
    }
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Full Like: Input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Full Like: Input must be allocated in buffer on device");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Full Like: Not currently supporting sharding");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Full Like: Not currently supporting sharding");
    TT_FATAL(operation_attributes.layout == Layout::TILE, "Full Like: Not currently supporting row major layout");
}

void FullLikeOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

void FullLikeOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}

FullLikeOperation::spec_return_value_t FullLikeOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return TensorSpec(
        tensor_args.input.get_logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
}

FullLikeOperation::tensor_return_value_t FullLikeOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

std::tuple<FullLikeOperation::operation_attributes_t, FullLikeOperation::tensor_args_t> FullLikeOperation::invoke(
    const Tensor& input,
    const std::variant<float, int> fill_value,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    return {
        operation_attributes_t{
            fill_value,
            dtype.value_or(input.dtype()),
            layout.value_or(input.layout()),
            memory_config.value_or(input.memory_config())},
        tensor_args_t{input}};
}

}  // namespace ttnn::operations::full_like
