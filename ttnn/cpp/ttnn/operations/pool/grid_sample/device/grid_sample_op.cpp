// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample_op.hpp"

#include "ttnn/tensor/types.hpp"
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

    TT_FATAL(input_tensor.logical_shape().rank() == 4, "Input tensor must be 4D (N, H, W, C)");
    TT_FATAL(grid_tensor.logical_shape().rank() == 4, "Grid tensor must be 4D (N, H_out, W_out, multiple_of_2_or_6)");

    uint32_t grid_last_dim = grid_tensor.logical_shape()[-1];
    if (use_precomputed_grid_) {
        TT_FATAL(
            grid_last_dim % PRECOMPUTED_GRID_ELEMENTS_PER_POINT == 0 &&
                grid_last_dim >= PRECOMPUTED_GRID_ELEMENTS_PER_POINT,
            "Grid tensor last dimension must be a multiple of 6 (multiple sets of h_nw, w_nw, weight_nw, weight_ne, "
            "weight_sw, weight_se)");
    } else {
        TT_FATAL(
            grid_last_dim % STANDARD_GRID_ELEMENTS_PER_POINT == 0 && grid_last_dim >= STANDARD_GRID_ELEMENTS_PER_POINT,
            "Grid tensor last dimension must be a multiple of 2 (multiple sets of x, y relative coordinates)");
    }

    TT_FATAL(
        input_tensor.logical_shape()[0] == grid_tensor.logical_shape()[0],
        "Batch size mismatch between input and grid");

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

    // Memory layout validation - for now only support interleaved
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only interleaved memory layout is currently supported for input tensor");
    TT_FATAL(
        grid_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only interleaved memory layout is currently supported for grid tensor");
    TT_FATAL(
        output_mem_config_.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only interleaved memory layout is currently supported for output tensor");
    TT_FATAL(
        input_tensor.logical_shape()[-1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor last dimension must be divisible by TILE_WIDTH");
    const uint32_t max_tiles_per_reduction = 8;
    TT_FATAL(
        input_tensor.logical_shape()[-1] <= tt::constants::TILE_WIDTH * max_tiles_per_reduction,
        "Wide reduction is currently not supported");
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

    // Calculate the grid batching factor
    const uint32_t num_of_elements_per_grid_point =
        use_precomputed_grid_ ? PRECOMPUTED_GRID_ELEMENTS_PER_POINT : STANDARD_GRID_ELEMENTS_PER_POINT;
    uint32_t grid_batching_factor = grid_last_dim / num_of_elements_per_grid_point;

    // Define output shape based on batch_output_channels flag
    ttnn::Shape output_shape;
    if (batch_output_channels_) {
        // batch_output_channels=True: batch output channels
        // Output shape: (N, H_out, W_out, C * grid_batching_factor)
        uint32_t C_out = C * grid_batching_factor;
        output_shape = ttnn::Shape({N, H_out, W_out, C_out});
    } else {
        // batch_output_channels=False: extend W dimension (default behavior)
        // Output shape: (N, H_out, W_out * grid_batching_factor, C)
        uint32_t W_out_extended = W_out * grid_batching_factor;
        output_shape = ttnn::Shape({N, H_out, W_out_extended, C});
    }

    // Output has same data type as input
    const DataType output_data_type = input_tensor.dtype();

    // Output layout is ROW_MAJOR (same as input)
    const Layout output_layout = Layout::ROW_MAJOR;

    return {
        TensorSpec(output_shape, TensorLayout(output_data_type, PageConfig(output_layout), this->output_mem_config_))};
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
