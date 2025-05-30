// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_op.hpp"

#include "ttnn/run_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "untilize_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

namespace untilize_helpers {
uint32_t get_num_cores(CoreCoord grid_size, uint32_t num_blocks) {
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;
    if (num_blocks <= num_cores) {
        num_cores = num_blocks;
    } else {
        uint32_t num_blocks_per_core = std::ceil((float)num_blocks / num_cores);
        num_cores = std::ceil((float)num_blocks / num_blocks_per_core);
    }
    return num_cores;
}
}  // namespace untilize_helpers

void Untilize::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to untilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_FATAL(input_tensor_a.padded_shape()[-1] % TILE_WIDTH == 0, "Width must be evenly divisible into tiles");
    TT_FATAL(
        (input_tensor_a.physical_volume() / input_tensor_a.padded_shape()[-1]) % TILE_HEIGHT == 0,
        "Height must be evenly divisible into tiles");

    bool input_is_sharded = input_tensor_a.is_sharded();
    bool output_is_sharded = this->output_mem_config.is_sharded();
    if (input_is_sharded && output_is_sharded) {
        // Sharded to sharded
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == this->output_mem_config.memory_layout(),
            "Input shard memory layout must match output shard memory layout");

        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                input_tensor_a.shard_spec().value().grid.ranges().size() == 1,
                "Block sharding cores must be on a single contiguous rectangle");
        }
    } else if (input_is_sharded && !output_is_sharded) {
        // Sharded to interleaved
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            TT_FATAL(
                input_tensor_a.shard_spec().value().grid.ranges().size() == 1,
                "Block sharding cores must be on a single contiguous rectangle");
        }
    } else if (!input_is_sharded && output_is_sharded) {
        // Interleaved to sharded
        TT_FATAL(
            this->output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "Only supporting converting interleaved to height sharded");

        uint32_t num_tiles = input_tensor_a.volume() / TILE_HW;
        uint32_t num_tiles_per_block = input_tensor_a.get_padded_shape()[-1] / TILE_WIDTH;
        uint32_t num_blocks = std::ceil((float)num_tiles / num_tiles_per_block);
        uint32_t num_cores =
            untilize_helpers::get_num_cores(input_tensor_a.device()->compute_with_storage_grid_size(), num_blocks);
        uint32_t fused_height = input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1];
        TT_FATAL(fused_height % num_cores == 0, "Error");
    } else {
        // Interleaved to interleaved
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Don't support single_bank as input memory layout");
        TT_FATAL(
            this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Don't support single_bank as output memory layout");
    }
}

std::vector<ttnn::TensorSpec> Untilize::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();

    if (!this->use_multicore) {
        return {TensorSpec(
            input_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                output_dtype,
                PageConfig(Layout::ROW_MAJOR),
                this->output_mem_config,
                input_tensor.logical_shape(),
                input_tensor.padded_shape()))};
    }

    if (this->output_mem_config.is_sharded()) {
        if (input_tensor.memory_config().is_sharded()) {
            auto mem_config = this->output_mem_config.with_shard_spec(input_tensor.memory_config().shard_spec());
            return {TensorSpec(
                input_tensor.logical_shape(),
                TensorLayout::fromPaddedShape(
                    output_dtype,
                    PageConfig(Layout::ROW_MAJOR),
                    mem_config,
                    input_tensor.logical_shape(),
                    input_tensor.padded_shape()))};
        }

    bool input_is_sharded = input_tensor_a.is_sharded();
    bool output_is_sharded = this->output_mem_config.is_sharded();
    if (input_is_sharded && output_is_sharded) {
        // Sharded to sharded
        auto output_mem_config = this->output_mem_config.with_shard_spec(input_tensor.shard_spec());
        return {TensorSpec(
            input_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                output_dtype,
                PageConfig(Layout::ROW_MAJOR),
                output_mem_config,
                input_tensor.get_logical_shape(),
                input_tensor.get_padded_shape()))};
    } else if (input_is_sharded && !output_is_sharded) {
        // Sharded to interleaved
        return {TensorSpec(
            input_tensor.get_logical_shape(),
            TensorLayout::fromPaddedShape(
                output_dtype,
                PageConfig(Layout::ROW_MAJOR),
                output_mem_config,
                input_tensor.get_logical_shape(),
                input_tensor.get_padded_shape()))};
    } else if (!input_is_sharded && output_is_sharded) {
        // Interleaved to sharded
        uint32_t num_tiles = input_tensor.volume() / TILE_HW;
        uint32_t num_tiles_per_block = input_tensor.get_padded_shape()[-1] / TILE_WIDTH;
        uint32_t num_blocks = std::ceil((float)num_tiles / num_tiles_per_block);
        uint32_t num_cores =
            untilize_helpers::get_num_cores(input_tensor.device()->compute_with_storage_grid_size(), num_blocks);
        uint32_t fused_height = input_tensor.volume() / input_tensor.get_padded_shape()[-1];

        auto shard_grid = tt::tt_metal::num_cores_to_corerangeset(
            num_cores, input_tensor.device()->compute_with_storage_grid_size(), true);
        std::array<uint32_t, 2> shard_shape = {fused_height / num_cores, input_tensor.get_padded_shape()[-1]};
        ShardSpec shard_spec(shard_grid, shard_shape, ShardOrientation::ROW_MAJOR);
        auto output_mem_config = this->output_mem_config.with_shard_spec(shard_spec);

        return {TensorSpec(
            input_tensor.get_logical_shape(),
            TensorLayout::fromPaddedShape(
                output_dtype,
                PageConfig(Layout::ROW_MAJOR),
                output_mem_config,
                input_tensor.get_logical_shape(),
                input_tensor.get_padded_shape()))};
    } else {
        // Interleaved to interleaved
        return {TensorSpec(
            input_tensor.get_logical_shape(),
            TensorLayout::fromPaddedShape(
                output_dtype,
                PageConfig(Layout::ROW_MAJOR),
                output_mem_config,
                input_tensor.get_logical_shape(),
                input_tensor.get_padded_shape()))};
    }

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

    if (!this->use_multicore) {
        return detail::untilize_single_core(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }
    if (this->sub_core_grids.has_value()) {
        // If sub_core_grids parameter is provided, use custom sub_core_grid implementation instead
        // of the standard multicore implementation or the block multicore implementation
        return detail::untilize_multi_core_sub_core_grids(
            input_tensor_a,
            output_tensor,
            this->use_pack_untilize,
            this->fp32_dest_acc_en,
            this->sub_core_grids.value());
    }
    if (!this->enough_space_height && !input_tensor_a.is_sharded() && !output_tensor.is_sharded()) {
        return detail::untilize_multi_core_block(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }

    return detail::untilize_multi_core(input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
}

}  // namespace ttnn::operations::data_movement
