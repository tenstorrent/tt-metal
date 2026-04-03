// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_op.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <ttnn/operation.hpp>
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

std::pair<bool, std::string> InterleavedToShardedDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_mem_config = operation_attributes.output_mem_config;
    const auto& output_dtype = operation_attributes.output_dtype;

    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return {false, "Operands to shard need to be on device!"};
    }
    if (input_tensor.buffer() == nullptr) {
        return {false, "Operands to shard need to be allocated in buffers on device!"};
    }
    if (tensor_args.output_tensor.has_value()) {
        const auto& output_tensor = tensor_args.output_tensor.value();
        if (output_tensor.logical_shape() != input_tensor.logical_shape()) {
            return {false, "Mismatched output shape"};
        }
        if (output_tensor.memory_config() != output_mem_config) {
            return {false, "Mismatched output memory config"};
        }
        if (output_tensor.dtype() != output_dtype) {
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
    if (input_tensor.memory_config().memory_layout() != tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        return {false, "Input tensor memory layout must be INTERLEAVED"};
    }
    if (!output_mem_config.is_sharded()) {
        return {false, "Output memory config must be sharded"};
    }
    if (output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::ND_SHARDED) {
        return {false, "interleaved_to_sharded does not support ND sharding. Please use ttnn.to_memory_config or ttnn.copy instead."};
    }
    if (output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
        if (output_mem_config.buffer_type() != tt::tt_metal::BufferType::L1) {
            return {false, "We don't support DRAM block sharding"};
        }
    }
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        if ((*output_mem_config.shard_spec()).shape[1] * input_tensor.element_size() %
                tt::tt_metal::hal::get_l1_alignment() != 0) {
            return {false, "Shard page size must currently have L1 aligned page size"};
        }
    }
    if (input_tensor.dtype() != output_dtype) {
        if (input_tensor.layout() != Layout::TILE) {
            return {false, "Input tensor layout must be TILE when dtype conversion is needed"};
        }
    }
    if (input_tensor.layout() == Layout::TILE) {
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
        }
        if (tensor_args.output_tensor.has_value()) {
            auto out_tile = tensor_args.output_tensor.value().tensor_spec().tile();
            if (out_tile != tile) {
                return {false, "Output tensor tile shape must match input tensor tile shape"};
            }
        }
    }
    return {true, ""};
}

void InterleavedToShardedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto [valid, msg] = validate_inputs(operation_attributes, tensor_args);
    TT_FATAL(valid, "{}", msg);
}

InterleavedToShardedDeviceOperation::spec_return_value_t InterleavedToShardedDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value().tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));
}

InterleavedToShardedDeviceOperation::tensor_return_value_t InterleavedToShardedDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, input_tensor.device());
}

ttsl::hash::hash_t InterleavedToShardedDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return tt::tt_metal::operation::hash_operation<InterleavedToShardedDeviceOperation>(
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.keep_l1_aligned,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.layout(),
        input_tensor.padded_shape());
}

Tensor interleaved_to_sharded(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    bool keep_l1_aligned,
    const std::optional<Tensor>& preallocated_output) {
    return ttnn::device_operation::launch<InterleavedToShardedDeviceOperation>(
        InterleavedToShardedParams{output_mem_config, output_dtype, keep_l1_aligned},
        InterleavedToShardedInputs{input_tensor, preallocated_output});
}
}  // namespace ttnn::prim
