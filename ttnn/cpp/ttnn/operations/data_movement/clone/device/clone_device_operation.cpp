// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone_device_operation.hpp"

namespace ttnn::operations::data_movement::clone {
void CloneOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    if (operation_attributes.dtype != input.get_dtype())
        TT_FATAL(input.get_layout() == Layout::TILE, "Clone: data type conversion is only supported with tile layout");
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Clone: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Clone: input must be allocated in buffer on device");
    TT_FATAL(
        input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Clone: not currently support sharding");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Clone: not currently support sharding");
}

CloneOperation::program_factory_t CloneOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void CloneOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void CloneOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

CloneOperation::shape_return_value_t CloneOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input.get_logical_shape();
};

CloneOperation::tensor_return_value_t CloneOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return create_device_tensor(
        compute_output_shapes(operation_attributes, tensor_args),
        operation_attributes.dtype,
        input.get_layout(),
        input.device(),
        operation_attributes.memory_config);
}

std::tuple<CloneOperation::operation_attributes_t, CloneOperation::tensor_args_t> CloneOperation::invoke(
    const Tensor& input,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        operation_attributes_t{
            dtype.value_or(input.get_dtype()),
            memory_config.value_or(input.memory_config()),
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
        },
        tensor_args_t{input},
    };
}
}  // namespace ttnn::operations::data_movement::clone
