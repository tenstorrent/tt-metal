// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_program_factory_common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/assert.hpp>
#include "ttnn/types.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization::softmax_backward {

// Estimate L1 memory usage for non-streaming kernel
// Returns true if tensor fits in L1 memory for non-streaming approach
bool should_use_non_streaming_kernel(uint32_t num_rows, uint32_t width_tiles, uint32_t tile_size) {
    // L1 memory available for circular buffers (conservative estimate)
    constexpr uint32_t L1_AVAILABLE_FOR_CBS = 1024 * 1024;  // ~1MB available

    const uint32_t total_tiles = num_rows * width_tiles;

    // Memory requirements for non-streaming kernel:
    // - Y input: total_tiles
    // - grad input: total_tiles
    // - output: total_tiles
    // - mul_cb: width_tiles (one row at a time)
    // - sum_reduce_cb: 1 tile
    // - ones_cb: 1 tile

    const uint32_t memory_needed = (total_tiles * 2) * tile_size +  // Y and grad inputs (2x buffer)
                                   (total_tiles * 2) * tile_size +  // Output (2x buffer)
                                   (width_tiles * 2) * tile_size +  // mul_cb (2x buffer)
                                   tile_size +                      // sum_reduce_cb
                                   tile_size;                       // ones_cb

    return memory_needed < L1_AVAILABLE_FOR_CBS;
}

// Helper function to get common tensor properties
void get_tensor_properties(
    const ttnn::Tensor& softmax_output,
    const operation_attributes_t& operation_attributes,
    uint32_t& num_rows,
    uint32_t& width_tiles,
    uint32_t& mask_w,
    DataFormat& input_data_format,
    DataFormat& output_data_format,
    DataFormat& intermed_data_format,
    uint32_t& input_tile_size,
    uint32_t& output_tile_size,
    uint32_t& intermed_tile_size,
    const ttnn::Tensor& tensor_return_value) {
    const ttnn::Shape& shape = softmax_output.padded_shape();
    const ttnn::Shape& logical_shape = softmax_output.logical_shape();
    const size_t rank = shape.rank();
    const uint32_t dim = operation_attributes.dim;

    TT_FATAL(
        dim == rank - 1 || dim == static_cast<uint32_t>(-1),
        "Currently only supporting softmax_backward on last dimension");

    const uint32_t height = shape[-2];
    const uint32_t width = shape[-1];
    const uint32_t height_tiles = height / constants::TILE_HEIGHT;
    width_tiles = width / constants::TILE_WIDTH;

    // Get logical width to determine padding mask
    const uint32_t logical_width = logical_shape[-1];
    mask_w = logical_width % constants::TILE_WIDTH;  // Position where padding starts in last tile

    // Calculate number of tiles to process
    const auto num_outer_dims = softmax_output.physical_volume() / height / width;
    num_rows = num_outer_dims * height_tiles;

    // Data formats
    input_data_format = datatype_to_dataformat_converter(softmax_output.dtype());
    output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    intermed_data_format = DataFormat::Float16_b;  // Use bfloat16 for intermediate calculations

    input_tile_size = tile_size(input_data_format);
    output_tile_size = tile_size(output_data_format);
    intermed_tile_size = tile_size(intermed_data_format);
}

// Helper function to create precise compute config
ComputeConfig precise(std::vector<uint32_t> compile_time_args, std::map<std::string, std::string> defines) {
    ComputeConfig config;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.math_fidelity = MathFidelity::HiFi4;
    config.compile_args = std::move(compile_time_args);
    config.defines = std::move(defines);
    return config;
}

}  // namespace ttnn::operations::normalization::softmax_backward
