// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_op.hpp"
#include "reshape_on_device/device/reshape_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ReshapeDeviceOperation::program_factory_t ReshapeDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    if (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR) {
        return ReshapeRMProgramFactory{};
    }
    return ReshapeTileProgramFactory{};
}

void ReshapeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor;
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::FLOAT32,
        "Input tensor dtype must be BFLOAT16 or FLOAT32 but got {}",
        input_tensor_a.dtype());

    TT_FATAL(
        input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR,
        "Only tile and row major reshape supported!");

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Use ttnn::reshape for reshaping sharded inputs");
    TT_FATAL(
        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Reshape does not currently support sharding. Use ttnn::reshape for reshaping sharded inputs");

    if (input_tensor_a.layout() == Layout::TILE) {
        TT_FATAL(
            input_tensor_a.physical_volume() % TILE_HW == 0,
            "Input tensor physical volume ({}) must be divisible by TILE_HW ({})",
            input_tensor_a.physical_volume(),
            TILE_HW);
    } else if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        uint32_t ROW_MAJOR_WIDTH = 8;
        TT_FATAL(
            input_tensor_a.padded_shape()[3] % ROW_MAJOR_WIDTH == 0 &&
                operation_attributes.padded_output_shape[3] % ROW_MAJOR_WIDTH == 0,
            "Operand/target width must be a multiple of 8");
    } else {
        TT_THROW("Unsupported layout for reshape");
    }
}

void ReshapeDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

ReshapeDeviceOperation::spec_return_value_t ReshapeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(
        operation_attributes.logical_output_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.tensor_spec().page_config(),
            operation_attributes.output_mem_config,
            operation_attributes.logical_output_shape,
            operation_attributes.padded_output_shape));
}

ReshapeDeviceOperation::tensor_return_value_t ReshapeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

tt::stl::hash::hash_t ReshapeDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return operation::hash_operation<ReshapeDeviceOperation>(
        operation_attributes.logical_output_shape,
        operation_attributes.padded_output_shape,
        operation_attributes.output_mem_config,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.layout(),
        input_tensor.padded_shape());
}

tt::tt_metal::Tensor reshape_on_device(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& logical_output_shape,
    const tt::tt_metal::Shape& padded_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    return ttnn::device_operation::launch<ReshapeDeviceOperation>(
        ReshapeOnDeviceParams{logical_output_shape, padded_output_shape, output_mem_config},
        ReshapeOnDeviceInputs{input_tensor});
}

}  // namespace ttnn::prim
