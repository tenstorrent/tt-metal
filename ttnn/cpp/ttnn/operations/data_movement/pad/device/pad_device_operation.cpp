// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "pad_device_operation.hpp"
#include "pad_program_factory.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/full/device/full_device_operation.hpp"
#include "ttnn/operations/creation.hpp"

using namespace tt::tt_metal;
namespace ttnn::operations::data_movement::pad {

PadDeviceOperation::program_factory_t PadDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return program::PadProgramFactory{};
}

void PadDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void PadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input;
    auto logical_rank = input_tensor.logical_shape().rank();
    auto padded_rank = input_tensor.padded_shape().rank();
    TT_FATAL(logical_rank == padded_rank, "ttnn.pad: logical and padded shapes must have the same rank");
    TT_FATAL(input_tensor.logical_shape().rank() <= 4, "ttnn.pad: input tensor rank currently must be 4 or less");
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operand to pad needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operand to pad needs to be allocated in a buffer on device!");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::TILE || input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Error");
    if (input_tensor.layout() == Layout::TILE) {
        TT_FATAL(
            (operation_attributes.input_tensor_start[0] == 0 && operation_attributes.input_tensor_start[1] == 0 &&
             operation_attributes.input_tensor_start[2] == 0 && operation_attributes.input_tensor_start[3] == 0),
            "On device padding only supports padding at end of dims");
    }
    TT_FATAL(
        input_tensor.padded_shape()[0] + operation_attributes.input_tensor_start[0] <=
            operation_attributes.output_padded_shape[0],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.padded_shape()[1] + operation_attributes.input_tensor_start[1] <=
            operation_attributes.output_padded_shape[1],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.padded_shape()[2] + operation_attributes.input_tensor_start[2] <=
            operation_attributes.output_padded_shape[2],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.padded_shape()[3] + operation_attributes.input_tensor_start[3] <=
            operation_attributes.output_padded_shape[3],
        "Output size cannot fit input with offset");

    if (input_tensor.layout() == Layout::TILE) {
        TT_FATAL(
            (operation_attributes.output_padded_shape[2] % TILE_HEIGHT == 0),
            "Can only pad tilized tensor with full tiles");
        TT_FATAL(
            (operation_attributes.output_padded_shape[3] % TILE_WIDTH == 0),
            "Can only pad tilized tensor with full tiles");
        TT_FATAL(
            input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
                input_tensor.dtype() == DataType::INT32 || input_tensor.dtype() == DataType::UINT32 ||
                input_tensor.dtype() == DataType::UINT16,
            "Cannot pad tilized tensor with specified format");
    } else if (input_tensor.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
                input_tensor.dtype() == DataType::INT32 || input_tensor.dtype() == DataType::UINT32 ||
                input_tensor.dtype() == DataType::UINT16,
            "Cannot pad RM tensor with specified format");
    }

    if (input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "ttnn.pad: For sharded inputs, only height-sharding is supported.");
        TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "ttnn.pad: Only row-major sharded inputs are supported.");

        TT_FATAL(
            operation_attributes.output_mem_config.is_sharded(),
            "ttnn.pad: For sharded inputs, the output must be sharded.");
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "ttnn.pad: for sharded inputs, only height-sharding is supported for the output.");
    }
}

ttnn::TensorSpec PadDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return TensorSpec(
        operation_attributes.output_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config,
            operation_attributes.output_logical_shape,
            operation_attributes.output_padded_shape));
}

Tensor PadDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

std::tuple<PadDeviceOperation::operation_attributes_t, PadDeviceOperation::tensor_args_t> PadDeviceOperation::invoke(
    const Tensor& input,
    const ttnn::Shape& output_logical_shape,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    const float pad_value,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const bool use_multicore,
    const std::optional<Tensor>& preallocated_output) {
    return {
        operation_attributes_t{
            output_logical_shape, output_padded_shape, input_tensor_start, pad_value, output_mem_config, use_multicore},
        tensor_args_t{input, preallocated_output}};
}

}  // namespace ttnn::operations::data_movement::pad
