// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "move_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::move {

MoveDeviceOperation::program_factory_t MoveDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    switch (operation_attributes.move_op_parallelization_strategy) {
        case MoveOpParallelizationStrategy::MULTI_CORE_SHARDED: return program::MoveShardedProgramFactory{};
        case MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP: return program::MoveOverlapProgramFactory{};
        case MoveOpParallelizationStrategy::MULTI_CORE: return program::MoveProgramFactory{};
        default: TT_FATAL(false, "Invalid move operation parallelization strategy");
    }
}

void MoveDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_tensor = tensor_args.output_tensor;

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "ttnn.move: input tensor must be on device. Got storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(input_tensor.buffer() != nullptr, "ttnn.move: input tensor must be allocated in buffer on device");

    TT_FATAL(
        output_tensor.storage_type() == StorageType::DEVICE,
        "ttnn.move: output tensor must be on device. Got storage type: {}",
        static_cast<int>(output_tensor.storage_type()));

    TT_FATAL(output_tensor.buffer() != nullptr, "ttnn.move: output tensor must be allocated in buffer on device");

    TT_FATAL(
        (input_tensor.dtype() == DataType::BFLOAT16 or input_tensor.dtype() == DataType::BFLOAT8_B or
         input_tensor.dtype() == DataType::FLOAT32 or input_tensor.dtype() == DataType::BFLOAT4_B or
         input_tensor.dtype() == DataType::UINT32 or input_tensor.dtype() == DataType::INT32),
        "ttnn.move: unsupported data type {}. Supported types: BFLOAT16, BFLOAT8_B, FLOAT32, BFLOAT4_B, UINT32, INT32",
        input_tensor.dtype());

    TT_FATAL(
        input_tensor.dtype() == output_tensor.dtype(),
        "ttnn.move: input and output tensors must have the same dtype. Input dtype: {}, Output dtype: {}",
        input_tensor.dtype(),
        output_tensor.dtype());

    if (operation_attributes.move_op_parallelization_strategy == MoveOpParallelizationStrategy::MULTI_CORE_SHARDED) {
        TT_FATAL(
            input_tensor.memory_config().is_sharded(),
            "ttnn.move: MULTI_CORE_SHARDED strategy requires input tensor to be sharded. Got memory layout: {}",
            input_tensor.memory_config().memory_layout());

        TT_FATAL(
            input_tensor.shard_spec().has_value(),
            "ttnn.move: MULTI_CORE_SHARDED strategy requires input tensor to have shard spec");

        TT_FATAL(
            operation_attributes.output_mem_config.is_sharded(),
            "ttnn.move: MULTI_CORE_SHARDED strategy requires output memory config to be sharded. Got memory layout: {}",
            operation_attributes.output_mem_config.memory_layout());

        TT_FATAL(
            output_tensor.shard_spec().has_value(),
            "ttnn.move: MULTI_CORE_SHARDED strategy requires output tensor to have shard spec");
    }

    if (operation_attributes.move_op_parallelization_strategy == MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP) {
        auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
        TT_FATAL(
            (compute_with_storage_grid_size.x > 1 and compute_with_storage_grid_size.y > 1),
            "ttnn.move: MULTI_CORE_OVERLAP strategy requires at least 2x2 compute grid. Got: {}x{}",
            compute_with_storage_grid_size.x,
            compute_with_storage_grid_size.y);
    }
}

void MoveDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const Tensor& output_tensor = tensor_args.output_tensor;

    // Validate tensors are still on device with valid buffers
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "ttnn.move: input tensor must be on device on cache hit. Got storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(input_tensor.buffer() != nullptr, "ttnn.move: input tensor buffer must be allocated on cache hit");

    TT_FATAL(
        output_tensor.storage_type() == StorageType::DEVICE,
        "ttnn.move: output tensor must be on device on cache hit. Got storage type: {}",
        static_cast<int>(output_tensor.storage_type()));

    TT_FATAL(output_tensor.buffer() != nullptr, "ttnn.move: output tensor buffer must be allocated on cache hit");
}

MoveDeviceOperation::spec_return_value_t MoveDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Output spec is same as output tensor spec
    return tensor_args.output_tensor.tensor_spec();
}

MoveDeviceOperation::tensor_return_value_t MoveDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Output tensor is already created and passed in tensor_args
    return tensor_args.output_tensor;
}

std::tuple<MoveDeviceOperation::operation_attributes_t, MoveDeviceOperation::tensor_args_t> MoveDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const MoveOpParallelizationStrategy& move_op_parallelization_strategy) {
    return {
        operation_attributes_t{
            .output_mem_config = output_mem_config,
            .move_op_parallelization_strategy = move_op_parallelization_strategy},
        tensor_args_t{.input_tensor = input_tensor, .output_tensor = output_tensor}};
}

}  // namespace ttnn::operations::data_movement::move
