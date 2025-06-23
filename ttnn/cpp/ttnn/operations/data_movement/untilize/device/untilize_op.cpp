// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_op.hpp"

#include "ttnn/run_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "untilize_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void Untilize::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor_a = input_tensors.at(0);

    uint32_t tensor_width = input_tensor_a.padded_shape()[-1];
    uint32_t tensor_height = input_tensor_a.physical_volume() / tensor_width;

    bool input_is_sharded = input_tensor_a.is_sharded();
    bool output_is_sharded = this->output_mem_config.is_sharded();

    BufferType input_buffer_type = input_tensor_a.memory_config().buffer_type();
    BufferType output_buffer_type = this->output_mem_config.buffer_type();

    TensorMemoryLayout input_memory_layout = input_tensor_a.memory_config().memory_layout();
    TensorMemoryLayout output_memory_layout = this->output_mem_config.memory_layout();

    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to untilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    // Input must be in valid tile layout
    TT_FATAL(tensor_width % TILE_WIDTH == 0, "Width must be evenly divisible into tiles");
    TT_FATAL(tensor_height % TILE_HEIGHT == 0, "Height must be evenly divisible into tiles");

    // Special conditions for sub_core_grids special case
    if (this->sub_core_grids.has_value()) {
        TT_FATAL(
            input_memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Input memory layout must be interleaved when sub_core_grid argument provided");
        TT_FATAL(
            output_memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Output memory layout must be interleaved when sub_core_grid argument provided");
        TT_FATAL(
            this->use_multicore == true,
            "sub_core_grid implementation only supported when use_multicore flag argument is set to true");
    }

    // If input is sharded, then the shard shape must be in multiples of tiles
    if (input_is_sharded) {
        std::array<uint32_t, 2> input_shard_shape = input_tensor_a.shard_spec().value().shape;
        uint32_t input_shard_width = input_shard_shape[1];
        uint32_t input_shard_height = input_shard_shape[0];
        TT_FATAL(
            input_shard_width % TILE_WIDTH == 0,
            "Input shard width {} must be a multiple of tile width",
            input_shard_width);
        TT_FATAL(
            input_shard_height % TILE_HEIGHT == 0,
            "Input shard height {} must be a multiple of tile height",
            input_shard_height);
    }

    // We don't support input or output uneven sharding for the single core implementation
    if (!this->use_multicore) {
        // Check for input uneven sharding
        if (input_is_sharded) {
            std::array<uint32_t, 2> input_shard_shape = input_tensor_a.shard_spec().value().shape;
            uint32_t input_shard_width = input_shard_shape[1];
            uint32_t input_shard_height = input_shard_shape[0];
            TT_FATAL(
                tensor_width % input_shard_width == 0,
                "Uneven input shard width {} for tensor width {} not supported for single core implementation",
                input_shard_width,
                tensor_width);
            TT_FATAL(
                tensor_height % input_shard_height == 0,
                "Uneven input shard height {} for tensor height {} not supported for single core implementation",
                input_shard_height,
                tensor_height);
        }
        // Check for output uneven sharding
        if (output_is_sharded) {
            std::array<uint32_t, 2> output_shard_shape = this->output_mem_config.shard_spec().value().shape;
            uint32_t output_shard_width = output_shard_shape[1];
            uint32_t output_shard_height = output_shard_shape[0];
            TT_FATAL(
                tensor_width % output_shard_width == 0,
                "Uneven output shard width {} for tensor width {} not supported for single core implementation",
                output_shard_width,
                tensor_width);
            TT_FATAL(
                tensor_height % output_shard_height == 0,
                "Uneven output shard height {} for tensor height {} not supported for single core implementation",
                output_shard_height,
                tensor_height);
        }
    }

    // We always support uneven input sharding for the multicore implementation. Uneven output sharding is only
    // supported if the input and output memory layouts are identical (i.e. height->height, width->width, block->block)
    // and the input and output shard specs are identical. Otherwise uneven output sharding is not supported.
    if (output_is_sharded) {
        std::array<uint32_t, 2> output_shard_shape = this->output_mem_config.shard_spec().value().shape;
        uint32_t output_shard_width = output_shard_shape[1];
        uint32_t output_shard_height = output_shard_shape[0];

        bool output_is_uneven_sharded_width_wise = tensor_width % output_shard_width != 0;
        bool output_is_uneven_sharded_height_wise = tensor_height % output_shard_height != 0;
        if (output_is_uneven_sharded_width_wise || output_is_uneven_sharded_height_wise) {
            TT_FATAL(
                input_memory_layout == output_memory_layout,
                "Input and output memory layouts must be identical if output is uneven sharded");
            TT_FATAL(
                input_tensor_a.shard_spec().value() == this->output_mem_config.shard_spec().value(),
                "Input and output shard specs must be identical if output is uneven sharded");
        }
    }

    // Multicore implementation doesn't support input DRAM sharding
    if (this->use_multicore && input_is_sharded) {
        TT_FATAL(input_buffer_type == BufferType::L1, "Multicore implementation doesn't support DRAM sharding");
    }

    // We don't support output DRAM block sharding
    if (output_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_FATAL(output_buffer_type == BufferType::L1, "We don't support DRAM block sharding");
    }
}

std::vector<ttnn::TensorSpec> Untilize::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();

    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::ROW_MAJOR),
            this->output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}

operation::ProgramWithCallbacks Untilize::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    bool input_is_sharded = input_tensor_a.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    BufferType input_buffer_type = input_tensor_a.memory_config().buffer_type();
    BufferType output_buffer_type = output_tensor.memory_config().buffer_type();

    TensorMemoryLayout input_memory_layout = input_tensor_a.memory_config().memory_layout();
    TensorMemoryLayout output_memory_layout = output_tensor.memory_config().memory_layout();

    if (!this->use_multicore) {
        // Single core implementation
        return detail::untilize_single_core(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }
    if (this->sub_core_grids.has_value()) {
        // If sub_core_grids parameter is provided, use custom sub_core_grid implementation instead
        // of the standard multicore implementation or the block multicore implementation.
        // Note that this implementation does not support sharding, which is enforced in validate().
        return detail::untilize_multi_core_sub_core_grids(
            input_tensor_a,
            output_tensor,
            this->use_pack_untilize,
            this->fp32_dest_acc_en,
            this->sub_core_grids.value());
    }
    if (!this->enough_space_height && !input_is_sharded && !output_is_sharded) {
        // Optimized special case implementation, only supported when neither input or output is sharded
        return detail::untilize_multi_core_block(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }
    if (input_is_sharded && output_is_sharded && input_buffer_type == BufferType::L1 &&
        output_buffer_type == BufferType::L1 && input_memory_layout == output_memory_layout &&
        input_tensor_a.shard_spec() == output_tensor.shard_spec()) {
        // Optimized special case implementation for when both input and output are sharded, both are located in L1,
        // have identical memory layouts (i.e. height->height, width->width, block->block), and have identical shard
        // specs
        return detail::untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }

    // Default multi core implementation
    return detail::untilize_multi_core(input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
}

}  // namespace ttnn::operations::data_movement
