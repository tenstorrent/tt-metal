// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::cnn::to_chw {

ConvertToCHWDeviceOperation::program_factory_t ConvertToCHWDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return program::ConvertToCHWProgramFactory{};
}

void ConvertToCHWDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ConvertToCHWDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    const auto& input = tensor_args.input;
    const auto& shape = input.logical_shape();
    const auto& C = shape[-1];
    const auto& HW = shape[-2];

    TT_FATAL(shape.size() == 4, "Input shape must be rank 4 (was rank {})", shape.size());
    TT_FATAL(shape[0] == 1 && shape[1] == 1, "Expected input tensor to be shape [1, 1, HW, C]");
    TT_FATAL(C <= TILE_HEIGHT, "C must be less than or equal to 32 (was {})", C);
    TT_FATAL(HW % TILE_HEIGHT == 0, "HW must be divisible by tile size");

    TT_FATAL(input.is_sharded(), "Input tensor must be sharded");

    const auto& input_shard_spec = input.memory_config().shard_spec().value();
    TT_FATAL(
        input_shard_spec.shape[0] % TILE_HEIGHT == 0,
        "Shard height must be divisible by tile size");  // input shards can be padded so HW may not match shard height
    TT_FATAL(
        args.memory_config.is_sharded() &&
            args.memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
        "Output tensor must be width sharded");
}

TensorSpec ConvertToCHWDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& shape = tensor_args.input.logical_shape();
    const auto B = shape[0];
    const auto HW = shape[2];
    const auto C = shape[3];
    return TensorSpec(
        Shape({B, 1, C, HW}),
        tt::tt_metal::TensorLayout(
            args.dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), args.memory_config));
}

Tensor ConvertToCHWDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t ConvertToCHWDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<ConvertToCHWDeviceOperation>(
        args, program_factory.index(), input_tensor.dtype(), input_tensor.memory_config(), input_shape.volume());

    return hash;
}

}  // namespace ttnn::operations::experimental::cnn::to_chw

namespace ttnn::prim {

ttnn::operations::experimental::cnn::to_chw::ConvertToCHWDeviceOperation::tensor_return_value_t convert_to_chw(
    const Tensor& input, const std::optional<DataType>& dtype) {
    using OperationType = ttnn::operations::experimental::cnn::to_chw::ConvertToCHWDeviceOperation;

    TT_FATAL(input.is_sharded(), "Input tensor must be sharded to infer output memory config");

    const auto& input_memory_config = input.memory_config();
    TT_FATAL(
        input_memory_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        "Input tensor must be height sharded");

    const auto& input_shape = input.logical_shape();
    const auto C = input_shape[-1];

    const auto& input_shard_spec = input_memory_config.shard_spec().value();
    const auto input_shard_height = input_shard_spec.shape[0];
    const auto output_shard_width = input_shard_height;  // HW dimension per core stays the same

    const std::array<uint32_t, 2> output_shard_shape = {C, output_shard_width};
    auto output_shard_spec =
        tt::tt_metal::ShardSpec(input_shard_spec.grid, output_shard_shape, input_shard_spec.orientation);

    const auto output_memory_config = tt::tt_metal::MemoryConfig(
        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, input.memory_config().buffer_type(), output_shard_spec);

    auto operation_attributes = OperationType::operation_attributes_t{
        .memory_config = output_memory_config,
        .dtype = dtype.value_or(input.dtype()),
    };
    auto tensor_args = OperationType::tensor_args_t{.input = input};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
