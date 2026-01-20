// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "clone_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::data_movement::clone {
void CloneOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    if (operation_attributes.dtype != input.dtype()) {
        TT_FATAL(input.layout() == Layout::TILE, "Clone: data type conversion is only supported with tile layout");
    }
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Clone: input must be on device");
    TT_FATAL(input.buffer() != nullptr, "Clone: input must be allocated in buffer on device");

    auto input_memory_layout = input.memory_config().memory_layout();
    auto output_memory_layout = operation_attributes.memory_config.memory_layout();
    bool input_sharded = input_memory_layout != TensorMemoryLayout::INTERLEAVED;
    bool output_sharded = output_memory_layout != TensorMemoryLayout::INTERLEAVED;

    if (input_sharded && output_sharded) {
        TT_FATAL(
            input_memory_layout == output_memory_layout,
            "Clone: input and output must have the same memory layout when both are sharded");

        auto input_shard_spec = input.buffer()->shard_spec();
        auto output_shard_spec = operation_attributes.memory_config.shard_spec();

        TT_FATAL(output_shard_spec.has_value(), "Clone: output memory config must have shard spec when sharded");

        TT_FATAL(
            input_shard_spec.tensor_shard_spec == output_shard_spec.value(),
            "Clone: input and output shard specs must be identical (same grid, shape, and orientation)");
    } else if (input_sharded || output_sharded) {
        TT_FATAL(
            false,
            "Clone: mixed sharded/interleaved layout not currently supported. Both input and output must have the same "
            "layout.");
    }
}

CloneOperation::program_factory_t CloneOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
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

CloneOperation::spec_return_value_t CloneOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            operation_attributes.dtype,
            tt::tt_metal::PageConfig(input.layout()),
            operation_attributes.memory_config,
            input.logical_shape(),
            input.padded_shape()));
};

CloneOperation::tensor_return_value_t CloneOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<CloneOperation::tensor_return_value_t>
CloneOperation::create_op_performance_model(
    const operation_attributes_t& /*op_attr*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::data_movement::clone

namespace ttnn::prim {
ttnn::Tensor clone(
    const Tensor& input,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType = ttnn::operations::data_movement::clone::CloneOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            dtype.value_or(input.dtype()),
            memory_config.value_or(input.memory_config()),
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4),
        },
        OperationType::tensor_args_t{input});
}
}  // namespace ttnn::prim
