// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

std::pair<bool, std::string> ShardedToInterleavedDeviceOperation::validate_inputs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return {false, "Operands to shard need to be on device!"};
    }
    if (input_tensor.buffer() == nullptr) {
        return {false, "Operands to shard need to be allocated in buffers on device!"};
    }
    if (!input_tensor.memory_config().is_sharded()) {
        return {false, "Input tensor must be sharded"};
    }
    if (input_tensor.memory_config().buffer_type() != BufferType::L1) {
        return {false, "Input tensor must be in L1"};
    }
    if (tensor_args.preallocated_output.has_value()) {
        const auto& output_tensor = tensor_args.preallocated_output.value();
        if (output_tensor.memory_config() != args.output_mem_config) {
            return {false, "Mismatched output memory config"};
        }
        if (output_tensor.dtype() != args.output_dtype) {
            return {false, "Mismatched output dtype"};
        }
        if (output_tensor.storage_type() != StorageType::DEVICE) {
            return {false, "Operands to shard need to be on device!"};
        }
        if (output_tensor.buffer() == nullptr) {
            return {false, "Operands to shard need to be allocated in buffers on device!"};
        }
        if (output_tensor.device() != input_tensor.device()) {
            return {false, "Operands to shard need to be on the same device!"};
        }
        if (output_tensor.layout() != input_tensor.layout()) {
            return {false, "Output tensor layout must match input tensor layout"};
        }
    }
    if (args.output_mem_config.memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        return {false, "Output memory config must be Interleaved"};
    }
    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        return {
            false,
            "sharded_to_interleaved does not support ND sharding. Please use ttnn.to_memory_config or ttnn.copy "
            "instead."};
    }
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        uint32_t l1_alignment = hal::get_l1_alignment();
        if ((*input_tensor.memory_config().shard_spec()).shape[1] * input_tensor.element_size() % l1_alignment != 0) {
            return {false, fmt::format("Shard page size must be aligned to {}B for L1 Tensor", l1_alignment)};
        }
    }
    if (input_tensor.dtype() != args.output_dtype) {
        if (input_tensor.layout() != Layout::TILE) {
            return {false, "If diff output type, tensor must be TILED"};
        }
    }
    if (input_tensor.layout() == Layout::TILE) {
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {
                false,
                fmt::format(
                    "sharded_to_interleaved requires standard 32x32 tiles, got {}x{}",
                    tile.get_height(),
                    tile.get_width())};
        }
        if (tensor_args.preallocated_output.has_value()) {
            auto out_tile = tensor_args.preallocated_output.value().tensor_spec().tile();
            if (out_tile != tile) {
                return {false, "Output tensor tile shape must match input tensor tile shape"};
            }
        }
    }
    return {true, ""};
}

void ShardedToInterleavedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto [valid, msg] = validate_inputs(args, tensor_args);
    TT_FATAL(valid, "{}", msg);
}

TensorSpec ShardedToInterleavedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            args.output_dtype,
            PageConfig(input_tensor.layout()),
            args.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));
}

Tensor ShardedToInterleavedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(spec, input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor>
ShardedToInterleavedDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) const {
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(tensor_args.input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {tensor_args.input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

Tensor sharded_to_interleaved(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output) {
    return ttnn::device_operation::launch<ShardedToInterleavedDeviceOperation>(
        ShardedToInterleavedParams{output_mem_config, output_dtype, 1, 0},
        ShardedToInterleavedInputs{input_tensor, preallocated_output});
}

}  // namespace ttnn::prim
