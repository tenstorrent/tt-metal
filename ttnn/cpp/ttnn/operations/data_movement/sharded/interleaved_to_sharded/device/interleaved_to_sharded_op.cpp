// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "interleaved_to_sharded_op.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <ttnn/operation.hpp>
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::data_movement {

InterleavedToShardedDeviceOperation::program_factory_t InterleavedToShardedDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return interleaved_to_sharded::InterleavedToShardedProgramFactory{};
}

void InterleavedToShardedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_mem_config = operation_attributes.output_mem_config;
    const auto& output_dtype = operation_attributes.output_dtype;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");

    if (tensor_args.output_tensor.has_value()) {
        const auto& output_tensor = tensor_args.output_tensor.value();
        TT_FATAL(output_tensor.logical_shape() == input_tensor.logical_shape(), "Mismatched output shape");
        TT_FATAL(output_tensor.memory_config() == output_mem_config, "Mismatched output memory config");
        TT_FATAL(output_tensor.dtype() == output_dtype, "Mismatched output dtype");
        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
        TT_FATAL(output_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
        TT_FATAL(output_tensor.device() == input_tensor.device(), "Operands to shard need to be on the same device!");
    }

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor memory layout must be INTERLEAVED but got {}",
        input_tensor.memory_config().memory_layout());
    TT_FATAL(output_mem_config.is_sharded(), "Output memory config must be sharded");
    if (output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(output_mem_config.buffer_type() == BufferType::L1, "We don't support DRAM block sharding");
    }
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            0 == (*output_mem_config.shard_spec()).shape[1] * input_tensor.element_size() %
                     tt::tt_metal::hal::get_l1_alignment(),
            "Shard page size must currently have L1 aligned page size");
    }
    if (input_tensor.dtype() != output_dtype) {
        TT_FATAL(
            input_tensor.layout() == Layout::TILE,
            "Input tensor layout must be TILE when dtype conversion is needed but got {}",
            input_tensor.layout());
    }
}

void InterleavedToShardedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

InterleavedToShardedDeviceOperation::spec_return_value_t InterleavedToShardedDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return tensor_args.output_tensor.value().tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    return tt::tt_metal::TensorSpec(
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

tt::stl::hash::hash_t InterleavedToShardedDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    auto program_factory = select_program_factory(operation_attributes, tensor_args);
    return tt::tt_metal::operation::hash_operation<InterleavedToShardedDeviceOperation>(
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.keep_l1_aligned,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.layout(),
        input_tensor.padded_shape());
}
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::InterleavedToShardedDeviceOperation::tensor_return_value_t interleaved_to_sharded(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    bool keep_l1_aligned,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::operations::data_movement::InterleavedToShardedDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{output_mem_config, output_dtype, keep_l1_aligned},
        OperationType::tensor_args_t{input_tensor, preallocated_output});
}
}  // namespace ttnn::prim
