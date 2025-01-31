// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expand_device_operation.hpp"

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::expand {
ExpandOperation::program_factory_t ExpandOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    switch (input.get_layout()) {
        case Layout::ROW_MAJOR: return ExpandRowMajorFactory{};
        default: TT_THROW("Expand: Unsupported input layout");
    }
}

void validate(
    const ExpandOperation::operation_attributes_t& operation_attributes,
    const ExpandOperation::tensor_args_t& tensor_args) {
    // We need to assert that the input and output are ROW_MAJOR. (unfortunately)

    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR, "Expand: Input tensor layout must be ROW_MAJOR");
    TT_FATAL(tensor_args.input.storage_type() == StorageType::DEVICE, "Expand: Input tensor need to be on device");
    TT_FATAL(tensor_args.input.buffer() != nullptr, "Expand: Input tensor need to be allocated in buffers on device");
    if (output.has_value()) {
        TT_FATAL(
            output->get_logical_shape() == operation_attributes.output_shape,
            "Expand: Output shape must match operation attributes");
        TT_FATAL(input.get_layout() == output->get_layout(), "Expand: Input and output must have same layout");
        TT_FATAL(input.get_dtype() == output->get_dtype(), "Expand: Input and output must have same dtype");
        TT_FATAL(input.device() == output->device(), "Expand: Input and output must be on the same device");
    }
}

void ExpandOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
};

void ExpandOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
};

ExpandOperation::spec_return_value_t ExpandOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->get_tensor_spec();
    }
    return TensorSpec(
        Shape{operation_attributes.output_shape},
        TensorLayout(
            tensor_args.input.get_dtype(),
            PageConfig(tensor_args.input.get_layout()),
            operation_attributes.memory_config));
};

ExpandOperation::tensor_return_value_t ExpandOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Let's just require it to be allocated ahead of time for now
    if (tensor_args.output.has_value()) {
        return {tensor_args.output.value()};
    }

    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

std::tuple<ExpandOperation::operation_attributes_t, ExpandOperation::tensor_args_t> ExpandOperation::invoke(
    const Tensor& input,
    const SmallVector<uint32_t>& output_shape,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config) {
    return {{output_shape, memory_config.value_or(input.memory_config())}, {input, output}};
}
}  // namespace ttnn::operations::expand
