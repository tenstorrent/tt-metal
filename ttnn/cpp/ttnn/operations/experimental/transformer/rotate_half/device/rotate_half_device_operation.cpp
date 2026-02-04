// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotate_half_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

RotateHalfDeviceOperation::program_factory_t RotateHalfDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return RotateHalfProgramFactory{};
}

void RotateHalfDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void RotateHalfDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor = tensor_args;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to rotate half need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to rotate half need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.layout() == Layout::TILE), "Inputs to rotate half must be tilized");
    TT_FATAL(input_tensor.padded_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "RotateHalf does not currently support sharding");
    TT_FATAL(
        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "RotateHalf does not currently support sharding");
}

TensorSpec RotateHalfDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor = tensor_args;
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            PageConfig(Layout::TILE),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));
}

Tensor RotateHalfDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const TensorSpec spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor rotate_half(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::experimental::prim::RotateHalfDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{.output_mem_config = output_mem_config};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, input);
}

}  // namespace ttnn::prim
