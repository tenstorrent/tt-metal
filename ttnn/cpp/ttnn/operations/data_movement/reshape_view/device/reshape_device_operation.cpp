// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

ReshapeViewDeviceOperation::program_factory_t ReshapeViewDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    if (tensor_args.input.layout() == Layout::ROW_MAJOR) {
        return ReshapeViewRMProgramFactory{};
    }
    return ReshapeViewTiledProgramFactory{};
}

void ReshapeViewDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor_a = tensor_args.input;
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::UINT32 or
            input_tensor_a.dtype() == DataType::FLOAT32 or input_tensor_a.dtype() == DataType::INT32,
        "Can only work with bfloat16/float32 or int32/uint32 tensors");
    TT_FATAL(
        operation_attributes.output_mem_config.memory_layout() == input_tensor_a.memory_config().memory_layout(),
        "Output tensor must have the same memory layout as input tensor");
}

void ReshapeViewDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

ReshapeViewDeviceOperation::spec_return_value_t ReshapeViewDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input;
    auto mem_config = operation_attributes.output_mem_config;
    if (input_tensor_a.memory_config().is_sharded()) {
        auto shard_spec = input_tensor_a.shard_spec().value();
        shard_spec.shape[0] = operation_attributes.logical_output_shape[0];
        mem_config = mem_config.with_shard_spec(shard_spec);
    }
    return TensorSpec(
        operation_attributes.logical_output_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            input_tensor_a.dtype(),
            tt::tt_metal::PageConfig(input_tensor_a.layout()),
            mem_config,
            operation_attributes.logical_output_shape,
            operation_attributes.padded_output_shape));
}

tt::tt_metal::Tensor ReshapeViewDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t ReshapeViewDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "ReshapeViewDeviceOperation::compute_program_hash is called");

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    // don't hash on operation_attributes_t::recreate_mapping_tensor
    return tt::tt_metal::operation::hash_operation<ReshapeViewDeviceOperation>(
        operation_attributes.logical_output_shape,
        operation_attributes.output_mem_config,
        operation_attributes.sub_core_grid.has_value(),
        operation_attributes.sub_core_grid.has_value() ? operation_attributes.sub_core_grid.value()
                                                       : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        tensor_args,
        program_factory.index());
}

tt::tt_metal::Tensor reshape_view(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const ttnn::Shape& padded_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool recreate_mapping_tensor,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    return ttnn::device_operation::launch<ReshapeViewDeviceOperation>(
        ReshapeViewParams{
            logical_output_shape, padded_output_shape, output_mem_config, recreate_mapping_tensor, sub_core_grid},
        ReshapeViewInputs{input});
}

}  // namespace ttnn::prim
