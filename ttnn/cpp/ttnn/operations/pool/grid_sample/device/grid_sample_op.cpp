// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample_op.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::grid_sample {
using namespace tt;
using namespace tt::tt_metal;

void GridSample::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& grid_tensor = input_tensors.at(1);

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device!");
    TT_FATAL(grid_tensor.storage_type() == StorageType::DEVICE, "Grid tensor must be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in buffer on device!");
    TT_FATAL(grid_tensor.buffer() != nullptr, "Grid tensor must be allocated in buffer on device!");

    // Shape validation
    TT_FATAL(
        input_tensor.logical_shape().rank() == 4,
        "Input tensor must be 4D (N, H, W, C), but got shape {} with {} dimensions",
        input_tensor.logical_shape(),
        input_tensor.logical_shape().rank());
    TT_FATAL(
        grid_tensor.logical_shape().rank() == 4,
        "Grid tensor must be 4D (N, H_out, W_out, coords), but got shape {} with {} dimensions",
        grid_tensor.logical_shape(),
        grid_tensor.logical_shape().rank());

    uint32_t grid_last_dim = grid_tensor.logical_shape()[-1];
    if (use_precomputed_grid_) {
        TT_FATAL(
            grid_last_dim % PRECOMPUTED_GRID_ELEMENTS_PER_POINT == 0 &&
                grid_last_dim >= PRECOMPUTED_GRID_ELEMENTS_PER_POINT,
            "Precomputed grid tensor last dimension must be a multiple of {} (for h_nw, w_nw, weight_nw, weight_ne, "
            "weight_sw, weight_se), but got {} in shape {}",
            PRECOMPUTED_GRID_ELEMENTS_PER_POINT,
            grid_last_dim,
            grid_tensor.logical_shape());
    } else {
        TT_FATAL(
            grid_last_dim % STANDARD_GRID_ELEMENTS_PER_POINT == 0 && grid_last_dim >= STANDARD_GRID_ELEMENTS_PER_POINT,
            "Standard grid tensor last dimension must be a multiple of {} (for x, y coordinates), but got {} in shape "
            "{}",
            STANDARD_GRID_ELEMENTS_PER_POINT,
            grid_last_dim,
            grid_tensor.logical_shape());
    }

    TT_FATAL(
        input_tensor.logical_shape()[0] == grid_tensor.logical_shape()[0],
        "Batch size mismatch: input tensor shape {} has batch size {}, but grid tensor shape {} has batch size {}",
        input_tensor.logical_shape(),
        input_tensor.logical_shape()[0],
        grid_tensor.logical_shape(),
        grid_tensor.logical_shape()[0]);

    // batch_output_channels validation - must have batched input (K > 1)
    if (batch_output_channels_) {
        const uint32_t num_elements_per_grid_point =
            use_precomputed_grid_ ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT : STANDARD_GRID_ELEMENTS_PER_POINT;
        const uint32_t grid_batching_factor = grid_last_dim / num_elements_per_grid_point;
        TT_FATAL(
            grid_batching_factor > 1,
            "batch_output_channels=True requires grid batching factor K > 1. Use a batched grid with multiple "
            "coordinate sets per row of grid.");
    }

    // Data type validation
    TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "Input tensor must be BFLOAT16");
    if (use_precomputed_grid_) {
        TT_FATAL(grid_tensor.dtype() == DataType::BFLOAT16, "Precomputed grid tensor must be BFLOAT16");
    } else {
        TT_FATAL(
            grid_tensor.dtype() == DataType::BFLOAT16 || grid_tensor.dtype() == DataType::FLOAT32,
            "Grid tensor must be BFLOAT16 or FLOAT32");
    }

    // Layout validation
    TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "Input tensor must be ROW_MAJOR layout");
    TT_FATAL(grid_tensor.layout() == Layout::ROW_MAJOR, "Grid tensor must be ROW_MAJOR layout");

    // Parameter validation - currently only support fixed configuration
    TT_FATAL(mode_ == "bilinear", "Only bilinear interpolation mode is currently supported");
    TT_FATAL(padding_mode_ == "zeros", "Only zeros padding mode is currently supported");

    // Memory layout validation - support interleaved and height sharded
    auto input_memory_layout = input_tensor.memory_config().memory_layout();
    auto grid_memory_layout = grid_tensor.memory_config().memory_layout();
    auto output_memory_layout = output_mem_config_.memory_layout();

    // Input tensor can be interleaved or height sharded
    TT_FATAL(
        input_memory_layout == TensorMemoryLayout::INTERLEAVED ||
            input_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Input tensor must have INTERLEAVED or HEIGHT_SHARDED memory layout");

    // Grid tensor can be interleaved or height sharded
    TT_FATAL(
        grid_memory_layout == TensorMemoryLayout::INTERLEAVED ||
            grid_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Grid tensor must have INTERLEAVED or HEIGHT_SHARDED memory layout");

    // Output can be interleaved or height sharded
    TT_FATAL(
        output_memory_layout == TensorMemoryLayout::INTERLEAVED ||
            output_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
        "Output tensor must have INTERLEAVED or HEIGHT_SHARDED memory layout");

    TT_FATAL(
        input_tensor.padded_shape()[-1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor last dimension must be divisible by TILE_WIDTH ({}), but got {} in padded shape {}",
        tt::constants::TILE_WIDTH,
        input_tensor.padded_shape()[-1],
        input_tensor.padded_shape());
    const uint32_t max_tiles_per_reduction = 8;
    TT_FATAL(
        input_tensor.padded_shape()[-1] <= tt::constants::TILE_WIDTH * max_tiles_per_reduction,
        "Wide reduction not supported: input tensor width {} exceeds maximum {} (TILE_WIDTH {} * max_tiles {}), padded "
        "shape: {}",
        input_tensor.padded_shape()[-1],
        tt::constants::TILE_WIDTH * max_tiles_per_reduction,
        tt::constants::TILE_WIDTH,
        max_tiles_per_reduction,
        input_tensor.padded_shape());
}

std::vector<TensorSpec> GridSample::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& grid_tensor = input_tensors.at(1);

    const auto& input_shape = input_tensor.logical_shape();
    const auto& grid_shape = grid_tensor.logical_shape();

    // Extract dimensions
    uint32_t N = input_shape[0];
    uint32_t C = input_shape[-1];
    uint32_t H_out = grid_shape[1];
    uint32_t W_out = grid_shape[2];
    uint32_t grid_last_dim = grid_shape[-1];

    // Calculate the number of batched grid points per grid row

    const uint32_t num_of_elements_per_grid_point =
        use_precomputed_grid_ ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT : STANDARD_GRID_ELEMENTS_PER_POINT;
    uint32_t grid_batching_factor = grid_last_dim / num_of_elements_per_grid_point;

    // Define output shape based on batch_output_channels flag
    ttnn::Shape output_logical_shape;
    if (batch_output_channels_) {
        // batch_output_channels=True: extend channels (legacy behavior)
        // Output shape: (N, H_out, W_out, C * grid_batching_factor)
        uint32_t C_out = C * grid_batching_factor;
        output_logical_shape = ttnn::Shape({N, H_out, W_out, C_out});
    } else {
        // batch_output_channels=False: extend W dimension (default behavior)
        // Output shape: (N, H_out, W_out * grid_batching_factor, C)
        uint32_t W_out_extended = W_out * grid_batching_factor;
        output_logical_shape = ttnn::Shape({N, H_out, W_out_extended, C});
    }

    // Output has same data type as input
    const DataType output_data_type = input_tensor.dtype();

    // Output layout is ROW_MAJOR (same as input)
    const Layout output_layout = Layout::ROW_MAJOR;

    // Determine the memory config of the output
    MemoryConfig output_memory_config = output_mem_config_;

    if (grid_tensor.memory_config().is_sharded()) {
        // If the grid tensor is sharded, the output tensor will also be sharded
        // The shard shape for the output should be:
        // - Height: same as grid shard_spec.shape[0] if the channels get extended, otherwise grid shard_spec.shape[0] *
        // grid_batching_factor
        // - Width: num of channels in the input times the channel extend factor (padded shape of the input)
        // - Shard orientation: same as for the grid
        // - Core grid: same as for the grid
        // - Memory layout: HEIGHT_SHARDED

        const ShardSpec grid_shard_spec = grid_tensor.shard_spec().value();

        // Calculate output shard dimensions
        const uint32_t output_shard_height =
            grid_shard_spec.shape[0] * (batch_output_channels_ ? 1 : grid_batching_factor);  // Output height
        const uint32_t input_padded_channel_width = input_tensor.padded_shape()[-1];
        const uint32_t output_shard_width =
            input_padded_channel_width * (batch_output_channels_ ? grid_batching_factor : 1);  // Input channels * channel extend factor

        // Use the same core grid and orientation as the grid tensor
        const CoreRangeSet output_core_range_set = grid_shard_spec.grid;
        const ShardOrientation output_shard_orientation = grid_shard_spec.orientation;
        const TensorMemoryLayout output_memory_layout = TensorMemoryLayout::HEIGHT_SHARDED;
        const BufferType output_buffer_type = BufferType::L1;

        const ShardSpec output_shard_spec =
            ShardSpec(output_core_range_set, {output_shard_height, output_shard_width}, output_shard_orientation);

        output_memory_config = MemoryConfig(output_memory_layout, output_buffer_type, output_shard_spec);
    }

    const auto& grid_padded_shape = grid_tensor.padded_shape();
    const auto& input_padded_shape = input_tensor.padded_shape();

    // Batch and height dimensions: same as grid tensor's padded shape
    uint32_t N_padded = grid_padded_shape[0];
    uint32_t H_out_padded = grid_padded_shape[1];

    // Width dimension: expand by grid_batching_factor if batch_output_channels=false
    uint32_t W_out_padded;
    if (batch_output_channels_) {
        W_out_padded = grid_padded_shape[2];  // batch_output_channels=true: use grid's padded width
    } else {
        W_out_padded = grid_padded_shape[2] * grid_batching_factor;  // batch_output_channels=false: expand width
    }

    // Channel dimension: use input tensor's padded channels and apply channel extend factor
    uint32_t C_padded = input_padded_shape[-1] * (batch_output_channels_ ? grid_batching_factor : 1);

    ttnn::Shape output_padded_shape({N_padded, H_out_padded, W_out_padded, C_padded});

    return {TensorSpec(
        output_logical_shape,
        TensorLayout::fromPaddedShape(
            output_data_type,
            PageConfig(output_layout),
            output_memory_config,
            output_logical_shape,
            output_padded_shape))};
}

operation::ProgramWithCallbacks GridSample::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const Tensor& input_tensor = input_tensors.at(0);
    const Tensor& grid_tensor = input_tensors.at(1);
    Tensor& output_tensor = output_tensors.at(0);

    return grid_sample_program_factory(
        input_tensor, grid_tensor, output_tensor, mode_, padding_mode_, use_precomputed_grid_, batch_output_channels_);
}

}  // namespace ttnn::operations::grid_sample
