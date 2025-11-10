// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample_op.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

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
            grid_last_dim % (mode_ == "nearest" ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST
                                                : PRECOMPUTED_GRID_ELEMENTS_PER_POINT) ==
                    0 &&
                grid_last_dim >= (mode_ == "nearest" ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST
                                                     : PRECOMPUTED_GRID_ELEMENTS_PER_POINT),
            "Precomputed grid tensor last dimension must be a multiple of {} (for h_nw, w_nw, weight_nw, weight_ne, "
            "weight_sw, weight_se), but got {} in shape {}",
            (mode_ == "nearest" ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST : PRECOMPUTED_GRID_ELEMENTS_PER_POINT),
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
            use_precomputed_grid_
                ? mode_ == "nearest" ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST : PRECOMPUTED_GRID_ELEMENTS_PER_POINT
                : STANDARD_GRID_ELEMENTS_PER_POINT;
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

    TT_FATAL(padding_mode_ == "zeros", "Only zeros padding mode is currently supported");

    // Mode and precomputed grid compatibility validation
    TT_FATAL(
        !(mode_ == "nearest" && !use_precomputed_grid_),
        "use_precomputed_grid = false is not supported with mode = 'nearest'. Please use precomputed grid with nearest "
        "mode.");

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
        use_precomputed_grid_
            ? mode_ == "nearest" ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST : PRECOMPUTED_GRID_ELEMENTS_PER_POINT
            : STANDARD_GRID_ELEMENTS_PER_POINT;
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

    // Get grid padded shape - needed for both sharded and interleaved cases
    const auto& grid_padded_shape = grid_tensor.padded_shape();
    const auto& input_padded_shape = input_tensor.padded_shape();

    // For bilinear mode with interleaved input/grid, keep output interleaved to avoid core count issues
    // For nearest mode or sharded cases, use sharded output
    if ((mode_ == "bilinear" && !grid_tensor.memory_config().is_sharded() &&
         output_mem_config_.memory_layout() == TensorMemoryLayout::INTERLEAVED)) {
        // Bilinear mode with interleaved tensors - use interleaved output (original behavior)
        output_memory_config = output_mem_config_;
    } else {
        // Check if user provided a shard spec - if so, respect it
        if (output_mem_config_.shard_spec().has_value() && !grid_tensor.memory_config().is_sharded()) {
            // User provided shard spec - use it as is
            output_memory_config = output_mem_config_;
        } else {
            // User didn't provide shard spec - generate one automatically
            // Nearest mode or sharded cases - create sharded output configuration
            const uint32_t input_padded_channel_width = input_tensor.padded_shape()[-1];

            CoreRangeSet output_core_range_set;
            ShardOrientation output_shard_orientation;
            uint32_t grid_points_per_shard;

            if (grid_tensor.memory_config().is_sharded()) {
                // Case 1: Grid is sharded - use its sharding configuration
                const ShardSpec grid_shard_spec = grid_tensor.shard_spec().value();

                // Use the same core grid and orientation as the grid tensor
                output_core_range_set = grid_shard_spec.grid;
                output_shard_orientation = grid_shard_spec.orientation;
                grid_points_per_shard = grid_shard_spec.shape[0];
            } else {
                // Case 2: Grid is not sharded - create sharding based on grid dimensions
                const uint32_t total_grid_points = grid_padded_shape[1] * grid_padded_shape[2];  // H * W

                // Get device compute grid for sharding
                tt::tt_metal::IDevice* device = input_tensor.device();
                const auto compute_grid_size = device->compute_with_storage_grid_size();

                // Split grid points across available cores
                auto
                    [num_cores_used,
                     all_cores_range,
                     core_group_1_range,
                     core_group_2_range,
                     num_points_1,
                     num_points_2] = tt::tt_metal::split_work_to_cores(compute_grid_size, total_grid_points);

                output_core_range_set = all_cores_range;
                output_shard_orientation = ShardOrientation::ROW_MAJOR;
                grid_points_per_shard = num_points_1;  // Use primary group size
            }

            // Calculate output shard dimensions based on grid points per core and batching
            // For nearest interpolation, each grid point produces one output point per channel
            const uint32_t output_shard_height =
                batch_output_channels_ ? grid_points_per_shard : grid_points_per_shard * grid_batching_factor;

            // Output width is the number of input channels, extended by batching factor if needed
            const uint32_t output_shard_width =
                batch_output_channels_ ? input_padded_channel_width * grid_batching_factor : input_padded_channel_width;

            // Create sharding configuration
            const TensorMemoryLayout output_memory_layout = TensorMemoryLayout::HEIGHT_SHARDED;
            const BufferType output_buffer_type = BufferType::L1;

            const ShardSpec output_shard_spec =
                ShardSpec(output_core_range_set, {output_shard_height, output_shard_width}, output_shard_orientation);

            output_memory_config = MemoryConfig(output_memory_layout, output_buffer_type, output_shard_spec);
        }
    }

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

    if (mode_ == "bilinear") {
        return grid_sample_bilinear_program_factory(
            input_tensor,
            grid_tensor,
            output_tensor,
            padding_mode_,
            align_corners_,
            use_precomputed_grid_,
            batch_output_channels_);
    } else {
        return grid_sample_nearest_program_factory(
            input_tensor,
            grid_tensor,
            output_tensor,
            padding_mode_,
            align_corners_,
            use_precomputed_grid_,
            batch_output_channels_);
    }
}

}  // namespace ttnn::operations::grid_sample
