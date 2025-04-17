// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone_device_operation.hpp"

namespace ttnn::operations::data_movement::clone {
void CloneOperation::validate(const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    if (operation_attributes.dtype != input.get_dtype()) {
        TT_FATAL(input.get_layout() == Layout::TILE, "Clone: data type conversion is only supported with tile layout");
    }
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Clone: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Clone: input must be allocated in buffer on device");
    TT_FATAL(
        input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Clone: not currently support sharding");
    TT_FATAL(
        operation_attributes.memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Clone: not currently support sharding");
}

tt::tt_metal::ProgramDescriptor CloneOperation::create_program(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    return ProgramFactory::create(operation_attributes, tensor_args, output);
}

CloneOperation::spec_return_value_t CloneOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return TensorSpec(
        input.get_logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(input.get_layout()),
            operation_attributes.memory_config));
};

CloneOperation::tensor_return_value_t CloneOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.input.device());
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
