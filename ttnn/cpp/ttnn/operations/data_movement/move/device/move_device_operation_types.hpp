// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>

namespace ttnn::prim {

enum class MoveOpParallelizationStrategy { MULTI_CORE, MULTI_CORE_OVERLAP, MULTI_CORE_SHARDED };

// Snapshot of input tensor attributes captured before deallocation.
// The move operation deallocates the input tensor early, so we capture all
// required attributes beforehand and flow this through the operation.
// A new Buffer object is created pointing to the original address.
struct MoveInputTensorSnapshot {
    // Tensor-level attributes
    tt::tt_metal::DataType dtype;
    tt::tt_metal::Layout layout;
    ttnn::Shape logical_shape;
    ttnn::Shape padded_shape;
    std::optional<tt::tt_metal::ShardSpec> shard_spec;
    tt::tt_metal::MemoryConfig memory_config;
    size_t element_size;

    // Buffer created at the original address - safe to use after tensor deallocation
    std::shared_ptr<tt::tt_metal::Buffer> buffer;

    // Pre-captured buffer properties for convenience
    uint32_t buffer_address;
    tt::tt_metal::DeviceAddr buffer_size;
    tt::tt_metal::DeviceAddr buffer_page_size;
    tt::tt_metal::DeviceAddr buffer_aligned_page_size;
    tt::tt_metal::DeviceAddr buffer_aligned_size_per_bank;
    uint32_t buffer_alignment;
    tt::tt_metal::BufferType buffer_type;

    // Physical volume needed for overlap program factory
    size_t physical_volume;

    // Factory method to create snapshot from a tensor
    static MoveInputTensorSnapshot create(const Tensor& tensor) {
        using namespace tt::tt_metal;
        auto* original_buf = tensor.buffer();

        // Capture sharding args from original buffer
        BufferShardingArgs sharding_args;
        if (original_buf->has_shard_spec()) {
            sharding_args = BufferShardingArgs(
                original_buf->buffer_distribution_spec(), original_buf->shard_spec(), original_buf->buffer_layout());
        } else {
            sharding_args = BufferShardingArgs(
                original_buf->buffer_distribution_spec(), std::nullopt, original_buf->buffer_layout());
        }

        // Create a new buffer at the same address
        // (sorry Audrey)
        auto new_buffer = Buffer::create(
            original_buf->device(),
            original_buf->address(),
            original_buf->size(),
            original_buf->page_size(),
            original_buf->buffer_type(),
            sharding_args,
            original_buf->bottom_up(),
            original_buf->sub_device_id());

        return MoveInputTensorSnapshot{
            .dtype = tensor.dtype(),
            .layout = tensor.layout(),
            .logical_shape = tensor.logical_shape(),
            .padded_shape = tensor.padded_shape(),
            .shard_spec = tensor.shard_spec(),
            .memory_config = tensor.memory_config(),
            .element_size = tensor.element_size(),
            .buffer = new_buffer,
            .buffer_address = original_buf->address(),
            .buffer_size = original_buf->size(),
            .buffer_page_size = original_buf->page_size(),
            .buffer_aligned_page_size = original_buf->aligned_page_size(),
            .buffer_aligned_size_per_bank = original_buf->aligned_size_per_bank(),
            .buffer_alignment = original_buf->alignment(),
            .buffer_type = original_buf->buffer_type(),
            .physical_volume = tensor.physical_volume(),
        };
    }
};

// Operation attributes - includes input snapshot since the input tensor is
// deallocated before the device operation runs. The snapshot is placed here
// rather than in tensor_args because the device operation framework expects
// tensor_args to contain only Tensor objects for visitation/hashing.
struct MoveOperationAttributes {
    tt::tt_metal::MemoryConfig output_mem_config;
    MoveOpParallelizationStrategy move_op_parallelization_strategy;
    bool backwards = false;
    MoveInputTensorSnapshot input_snapshot;
};

// Tensor arguments - only contains output tensor since input is deallocated
// and its attributes are captured in operation_attributes.input_snapshot
struct MoveTensorArgs {
    Tensor output_tensor;
};

}  // namespace ttnn::prim
