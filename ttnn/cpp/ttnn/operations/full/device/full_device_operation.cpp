// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::full {
void FullOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto any = tensor_args.any;
    TT_FATAL(any.storage_type() == StorageType::DEVICE, "Full operation error: Any tensor must be on device");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Full operation error: Not currently supporting sharding");
    TT_FATAL(
        operation_attributes.layout == Layout::TILE, "Full operation error: Not currently supporting row major layout");

    const auto shape = operation_attributes.shape;

    TT_FATAL(
        shape.size() > 1,
        "Full operation error: Shape size must be greater than 1, but got shape size = {}",
        shape.size());

    for (int i = 0; i < shape.size(); i++) {
        TT_FATAL(
            shape[i] > 0,
            "Full operation error: Invalid shape at index {}. Each dimension of the shape must be greater than 0, but"
            "got shape[{}] = {}",
            i,
            i,
            shape[i]);
    }
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

FullOperation::spec_return_value_t FullOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    return TensorSpec(
        Shape(operation_attributes.shape),
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(operation_attributes.layout),
            operation_attributes.memory_config));
};

FullOperation::tensor_return_value_t FullOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.any.device());
}

std::tuple<FullOperation::operation_attributes_t, FullOperation::tensor_args_t> FullOperation::invoke(
    ttnn::SmallVector<uint32_t> shape,
    std::variant<float, int> fill_value,
    const Tensor& any,
    const std::optional<DataType>& dtype,
    const std::optional<Layout>& layout,
    const std::optional<MemoryConfig>& memory_config) {
    return {
        operation_attributes_t{
            std::move(shape),
            std::move(fill_value),
            dtype.value_or(any.dtype()),
            layout.value_or(any.layout()),
            memory_config.value_or(any.memory_config()),
        },
        tensor_args_t{
            any,
        },
    };
}
}  // namespace ttnn::operations::full
