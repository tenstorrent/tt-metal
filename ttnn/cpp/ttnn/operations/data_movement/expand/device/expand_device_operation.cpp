// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "expand_device_operation.hpp"

#include <cstdint>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::expand {
ExpandOperation::program_factory_t ExpandOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return Expand2DFactory{};
}

void validate(
    const ExpandOperation::operation_attributes_t& operation_attributes,
    const ExpandOperation::tensor_args_t& tensor_args) {
    // We need to assert that the input and output are ROW_MAJOR. (unfortunately)

    const auto& input = tensor_args.input;
    const auto& output = tensor_args.output;

    TT_FATAL(input.get_layout() == Layout::ROW_MAJOR, "Input tensor layout must be ROW_MAJOR");
    if (output.has_value()) {
        TT_FATAL(output.value().get_layout() == Layout::ROW_MAJOR, "Output tensor layout must be ROW_MAJOR");

        TT_FATAL(
            output->get_shape() == operation_attributes.output_shape, "Output shape must match operation attributes");
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

ExpandOperation::shape_return_value_t ExpandOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return Shape{operation_attributes.output_shape};
};

ExpandOperation::tensor_return_value_t ExpandOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Let's just require it to be allocated ahead of time for now
    if (tensor_args.output.has_value())
        return {tensor_args.output.value()};

    return create_device_tensor(
        compute_output_shapes(operation_attributes, tensor_args),
        tensor_args.input.get_dtype(),
        tensor_args.input.get_layout(),
        tensor_args.input.device(),
        operation_attributes.output_mem_config);
}

std::tuple<ExpandOperation::operation_attributes_t, ExpandOperation::tensor_args_t> ExpandOperation::invoke(
    const Tensor& input,
    const std::vector<uint32_t>& output_shape,

    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        {output_shape,
         output_mem_config.value_or(input.memory_config()),
         init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config)},
        {input, output}};
}
}  // namespace ttnn::operations::expand
