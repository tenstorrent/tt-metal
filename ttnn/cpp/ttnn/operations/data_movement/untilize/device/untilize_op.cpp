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
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to untilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Can only untilize tile major data");

    // Input must be in valid tile layout
    TT_FATAL(input_tensor_a.get_padded_shape()[-1] % TILE_WIDTH == 0, "Width must be evenly divisible into tiles");
    TT_FATAL(
        (input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1]) % TILE_HEIGHT == 0,
        "Height must be evenly divisible into tiles");

    // Only support interleaved or sharded memory layout
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::SINGLE_BANK,
        "Input memory layout must be interleaved or sharded");
    TT_FATAL(
        this->output_mem_config.memory_layout() != TensorMemoryLayout::SINGLE_BANK,
        "Output memory layout must be interleaved or sharded");

    // Special conditions for sub_core_grids special case
    if (this->sub_core_grids.has_value()) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input memory layout must be interleaved when sub_core_grid argument provided");
        TT_FATAL(
            this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Output memory layout must be interleaved when sub_core_grid argument provided");
        TT_FATAL(
            this->use_multicore == true,
            "sub_core_grid implementation only supported when use_multicore flag argument is set to true");
    }

    // Shard shape restrictions
    if (input_tensor_a.is_sharded()) {
        std::array<uint32_t, 2> input_shard_shape = input_tensor_a.shard_spec().value().shape;
        TT_FATAL(
            input_tensor_a.get_padded_shape()[-1] % input_shard_shape[1] == 0,
            "Uneven input shard shape not supported");
        TT_FATAL(
            (input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1]) % input_shard_shape[0] == 0,
            "Uneven input shard shape not supported");
    }
    if (this->output_mem_config.is_sharded()) {
        std::array<uint32_t, 2> output_shard_shape = this->output_mem_config.shard_spec().value().shape;
        TT_FATAL(
            input_tensor_a.get_padded_shape()[-1] % output_shard_shape[1] == 0,
            "Uneven output shard shape not supported");
        TT_FATAL(
            (input_tensor_a.volume() / input_tensor_a.get_padded_shape()[-1]) % output_shard_shape[0] == 0,
            "Uneven output shard shape not supported");
    }
}

std::vector<ttnn::TensorSpec> Untilize::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor = input_tensors.at(0);
    DataType output_dtype =
        input_tensor.get_dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.get_dtype();

    return {TensorSpec(
        input_tensor.get_logical_shape(),
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::ROW_MAJOR),
            this->output_mem_config,
            input_tensor.get_logical_shape(),
            input_tensor.get_padded_shape()))};
}

operation::ProgramWithCallbacks Untilize::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    bool input_is_sharded = input_tensor_a.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

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
    if (input_is_sharded && output_is_sharded && input_tensor_a.shard_spec() == output_tensor.shard_spec()) {
        // Optimized special case implementation for when both input and output are sharded, have identical shard specs,
        // and have identical memory layouts
        return detail::untilize_multi_core_input_and_output_shard_spec_indentical(
            input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
    }

    // Default multi core implementation
    return detail::untilize_multi_core(input_tensor_a, output_tensor, this->use_pack_untilize, this->fp32_dest_acc_en);
}

}  // namespace ttnn::operations::data_movement
