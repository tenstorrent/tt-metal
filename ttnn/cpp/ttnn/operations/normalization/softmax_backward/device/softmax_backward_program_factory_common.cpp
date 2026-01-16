// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_program_factory_common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/assert.hpp>
#include "ttnn/types.hpp"

namespace ttnn::operations::normalization::softmax_backward {

// Estimate L1 memory usage for non-streaming kernel
// Returns: (use_non_streaming_kernel, estimated_memory_bytes)
std::pair<bool, uint32_t> should_use_non_streaming_kernel(uint32_t width_tiles, uint32_t tile_size) {
    // L1 memory available for circular buffers (conservative estimate)
    constexpr uint32_t L1_AVAILABLE_FOR_CBS = 1024 * 1024;  // ~1MB available

    // Memory requirements for non-streaming kernel (processes one row at a time):
    // - src0_cb (Y input): width_tiles * 2 (double-buffered)
    // - src1_cb (grad input): width_tiles * 2 (double-buffered)
    // - out_cb (output): width_tiles * 2 (double-buffered)
    // - intermed0_cb (mul_cb: y * grad): width_tiles * 2 (double-buffered)
    // - intermed1_cb (sum_reduce_cb): 1 tile
    // - ones_cb: 1 tile

    const uint32_t memory_needed = (width_tiles * 2 * tile_size) +  // src0_cb: Y input
                                   (width_tiles * 2 * tile_size) +  // src1_cb: grad input
                                   (width_tiles * 2 * tile_size) +  // out_cb: output
                                   (width_tiles * 2 * tile_size) +  // intermed0_cb: y * grad
                                   (1 * tile_size) +                // intermed1_cb: sum_reduce_cb
                                   (1 * tile_size);                 // ones_cb

    return {memory_needed < L1_AVAILABLE_FOR_CBS, memory_needed};
}

// Helper function to get common tensor properties
void get_tensor_properties(
    const ttnn::Tensor& softmax_output,
    const operation_attributes_t& operation_attributes,
    uint32_t& num_rows,
    uint32_t& width_tiles,
    uint32_t& mask_w,
    tt::DataFormat& input_data_format,
    tt::DataFormat& output_data_format,
    tt::DataFormat& intermed_data_format,
    uint32_t& input_tile_size,
    uint32_t& output_tile_size,
    uint32_t& intermed_tile_size,
    const ttnn::Tensor& tensor_return_value) {
    using namespace tt::tt_metal;
    const uint32_t dim = operation_attributes.dim;

    TT_FATAL(
        dim == softmax_output.logical_shape().rank() - 1 || dim == static_cast<uint32_t>(-1),
        "Currently only supporting softmax_backward on last dimension");

    const uint32_t height = softmax_output.logical_shape()[-2];
    const uint32_t width = softmax_output.logical_shape()[-1];
    const uint32_t height_tiles = height / tt::constants::TILE_HEIGHT;
    width_tiles = width / tt::constants::TILE_WIDTH;

    // Get logical width to determine padding mask
    const uint32_t logical_width = softmax_output.logical_shape()[-1];
    mask_w = logical_width % tt::constants::TILE_WIDTH;  // Position where padding starts in last tile

    // Calculate number of tiles to process
    const uint64_t num_outer_dims = softmax_output.physical_volume() / height / width;
    num_rows = num_outer_dims * height_tiles;

    // Data formats
    input_data_format = datatype_to_dataformat_converter(softmax_output.dtype());
    output_data_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    intermed_data_format = tt::DataFormat::Float16_b;  // Use bfloat16 for intermediate calculations

    input_tile_size = tile_size(input_data_format);
    output_tile_size = tile_size(output_data_format);
    intermed_tile_size = tile_size(intermed_data_format);
}

// Helper function to create precise compute config
tt::tt_metal::ComputeConfig precise(
    std::vector<uint32_t> compile_time_args, std::map<std::string, std::string> defines) {
    tt::tt_metal::ComputeConfig config;
    config.fp32_dest_acc_en = true;
    config.math_approx_mode = false;
    config.math_fidelity = MathFidelity::HiFi4;
    config.compile_args = std::move(compile_time_args);
    config.defines = std::move(defines);
    return config;
}

}  // namespace ttnn::operations::normalization::softmax_backward
