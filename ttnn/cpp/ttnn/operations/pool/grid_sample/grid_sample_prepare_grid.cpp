// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample_prepare_grid.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "ttnn/tensor/types.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

namespace ttnn::operations::grid_sample {

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Unified helper function for grid preprocessing (both nearest and bilinear modes)
template <typename InputType, typename OutputType>
tt::tt_metal::HostBuffer create_host_buffer_for_grid_preprocessing(
    const Tensor& input_tensor,
    const ttnn::Shape& output_shape,
    const std::string& mode,
    bool align_corners,
    const std::vector<uint32_t>& tensor_input_shape) {
    auto input_buffer = tt::tt_metal::host_buffer::get_as<InputType>(input_tensor);
    std::vector<OutputType> output_buffer(output_shape.volume());

    // Extract dimensions
    uint32_t input_h = tensor_input_shape[1];
    uint32_t input_w = tensor_input_shape[2];

    // Extract dimensions from grid tensor shape
    auto grid_shape = input_tensor.logical_shape();
    uint32_t grid_n = grid_shape[0];
    uint32_t grid_h = grid_shape[1];
    uint32_t grid_w = grid_shape[2];

    // Scale factors for coordinate transformation based on align_corners mode
    float height_scale, height_offset, width_scale, width_offset;

    if (align_corners) {
        // align_corners=True: maps [-1,1] to [0, size-1]
        height_scale = (input_h > 1) ? static_cast<float>(input_h - 1) * 0.5f : 0.0f;
        width_scale = (input_w > 1) ? static_cast<float>(input_w - 1) * 0.5f : 0.0f;
        height_offset = 0.0f;
        width_offset = 0.0f;
    } else {
        // align_corners=False: maps [-1,1] to [-0.5, size-0.5]
        height_scale = static_cast<float>(input_h) * 0.5f;
        width_scale = static_cast<float>(input_w) * 0.5f;
        height_offset = -0.5f;
        width_offset = -0.5f;
    }

    // Process each grid point
    for (uint32_t n = 0; n < grid_n; n++) {
        for (uint32_t h = 0; h < grid_h; h++) {
            for (uint32_t w = 0; w < grid_w; w++) {
                // Calculate input indices for grid coordinates
                uint32_t x_idx = (((n * grid_h + h) * grid_w + w) * 2) + 0;  // x coordinate
                uint32_t y_idx = (((n * grid_h + h) * grid_w + w) * 2) + 1;  // y coordinate

                // Extract normalized coordinates [-1, 1]
                float x_coord = static_cast<float>(input_buffer[x_idx]);
                float y_coord = static_cast<float>(input_buffer[y_idx]);

                // Transform to image coordinates - use consistent formula
                float h_coord_image = ((y_coord + 1.0f) * height_scale) + height_offset;
                float w_coord_image = ((x_coord + 1.0f) * width_scale) + width_offset;

                if (mode == "nearest") {
                    // For nearest neighbor: use round() for align_corners=True, floor(coord + 0.5) for
                    // align_corners=False
                    int32_t h_nearest, w_nearest;
                    if (align_corners) {
                        h_nearest = static_cast<int32_t>(std::round(h_coord_image));
                        w_nearest = static_cast<int32_t>(std::round(w_coord_image));
                    } else {
                        h_nearest = static_cast<int32_t>(std::floor(h_coord_image + 0.5f));
                        w_nearest = static_cast<int32_t>(std::floor(w_coord_image + 0.5f));
                    }

                    // Check if coordinates are valid - proper pixel bounds
                    bool h_valid = (h_nearest >= 0) && (h_nearest < static_cast<int32_t>(input_h));
                    bool w_valid = (w_nearest >= 0) && (w_nearest < static_cast<int32_t>(input_w));

                    // Calculate output indices for nearest mode (2 values per point)
                    uint32_t output_base = (((n * grid_h + h) * grid_w + w) * 2);

                    // Store optimized coordinates with validity information
                    if constexpr (std::is_same_v<OutputType, bfloat16>) {
                        if (h_valid && w_valid) {
                            // Store valid coordinates as int16 bit patterns
                            int16_t h_clamped = static_cast<int16_t>(std::clamp(h_nearest, -32768, 32767));
                            int16_t w_clamped = static_cast<int16_t>(std::clamp(w_nearest, -32768, 32767));

                            uint16_t h_bits = static_cast<uint16_t>(h_clamped);
                            uint16_t w_bits = static_cast<uint16_t>(w_clamped);

                            output_buffer[output_base + 0] = std::bit_cast<bfloat16>(h_bits);  // h coordinate
                            output_buffer[output_base + 1] = std::bit_cast<bfloat16>(w_bits);  // w coordinate
                        } else {
                            // Store sentinel value for invalid coordinates (use -1)
                            uint16_t invalid_sentinel = static_cast<uint16_t>(-1);
                            output_buffer[output_base + 0] = std::bit_cast<bfloat16>(invalid_sentinel);  // invalid h
                            output_buffer[output_base + 1] = std::bit_cast<bfloat16>(invalid_sentinel);  // invalid w
                        }
                    } else {
                        if (h_valid && w_valid) {
                            output_buffer[output_base + 0] = static_cast<OutputType>(h_nearest);  // h coordinate
                            output_buffer[output_base + 1] = static_cast<OutputType>(w_nearest);  // w coordinate
                        } else {
                            // Store sentinel value for invalid coordinates (use -1)
                            output_buffer[output_base + 0] = static_cast<OutputType>(-1);  // invalid h
                            output_buffer[output_base + 1] = static_cast<OutputType>(-1);  // invalid w
                        }
                    }
                } else {  // bilinear mode
                    // Get corner pixel coordinates (floor operation)
                    int32_t h0 = static_cast<int32_t>(std::floor(h_coord_image));
                    int32_t w0 = static_cast<int32_t>(std::floor(w_coord_image));
                    int32_t h1 = h0 + 1;
                    int32_t w1 = w0 + 1;

                    // Boundary checks
                    bool h0_valid = (h0 >= 0) && (h0 < static_cast<int32_t>(input_h));
                    bool h1_valid = (h1 >= 0) && (h1 < static_cast<int32_t>(input_h));
                    bool w0_valid = (w0 >= 0) && (w0 < static_cast<int32_t>(input_w));
                    bool w1_valid = (w1 >= 0) && (w1 < static_cast<int32_t>(input_w));

                    // Calculate interpolation weights
                    float h_frac = h_coord_image - static_cast<float>(h0);
                    float w_frac = w_coord_image - static_cast<float>(w0);
                    float h_frac_inv = 1.0f - h_frac;
                    float w_frac_inv = 1.0f - w_frac;

                    // Compute bilinear weights with boundary conditions
                    float weight_nw = (h0_valid && w0_valid) ? h_frac_inv * w_frac_inv : 0.0f;
                    float weight_ne = (h0_valid && w1_valid) ? h_frac_inv * w_frac : 0.0f;
                    float weight_sw = (h1_valid && w0_valid) ? h_frac * w_frac_inv : 0.0f;
                    float weight_se = (h1_valid && w1_valid) ? h_frac * w_frac : 0.0f;

                    // Clamp coordinates to 16-bit range for storing as bfloat16
                    int16_t h0_clamped = static_cast<int16_t>(std::clamp(h0, -32768, 32767));
                    int16_t w0_clamped = static_cast<int16_t>(std::clamp(w0, -32768, 32767));

                    // Calculate output indices for bilinear mode (6 values per point)
                    uint32_t base_idx = ((n * grid_h + h) * grid_w + w) * 6;

                    // Store results
                    if constexpr (std::is_same_v<OutputType, bfloat16>) {
                        // Reinterpret int16 bits as bfloat16 for coordinates
                        uint16_t h0_bits = static_cast<uint16_t>(h0_clamped);
                        uint16_t w0_bits = static_cast<uint16_t>(w0_clamped);
                        output_buffer[base_idx + 0] = std::bit_cast<bfloat16>(h0_bits);
                        output_buffer[base_idx + 1] = std::bit_cast<bfloat16>(w0_bits);
                        // Convert weights to bfloat16
                        output_buffer[base_idx + 2] = bfloat16(weight_nw);
                        output_buffer[base_idx + 3] = bfloat16(weight_ne);
                        output_buffer[base_idx + 4] = bfloat16(weight_sw);
                        output_buffer[base_idx + 5] = bfloat16(weight_se);
                    }
                }
            }
        }
    }

    return tt::tt_metal::HostBuffer(std::move(output_buffer));
}

// Template function to convert tensor based on input and output types
template <typename InputType, typename OutputType>
Tensor convert_grid_tensor(
    const Tensor& input_tensor,
    const std::string& mode,
    bool align_corners,
    const ttnn::Shape& output_shape,
    const std::vector<uint32_t>& tensor_input_shape,
    DataType output_dtype) {
    auto compute = [&](const tt::tt_metal::HostBuffer& /*input_host_buffer*/) {
        return create_host_buffer_for_grid_preprocessing<InputType, OutputType>(
            input_tensor, output_shape, mode, align_corners, tensor_input_shape);
    };

    const TensorSpec output_spec(
        output_shape,
        tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    TT_FATAL(is_cpu_tensor(input_tensor), "Prepare_grid_sample_grid only supports host tensors");

    return Tensor(input_tensor.host_storage().transform(compute), output_spec, input_tensor.tensor_topology());
}

}  // anonymous namespace

ttnn::Tensor prepare_grid_sample_grid(
    const ttnn::Tensor& grid,
    const std::vector<uint32_t>& input_shape,
    const std::string& mode,
    const std::string& padding_mode,
    bool align_corners,
    const std::optional<DataType>& output_dtype) {
    // Validate inputs
    TT_FATAL(is_cpu_tensor(grid), "Grid tensor must be on host");
    TT_FATAL(grid.layout() == Layout::ROW_MAJOR, "Grid tensor must be in row major layout");
    TT_FATAL(grid.logical_shape().rank() == 4, "Grid tensor must be 4D");
    TT_FATAL(grid.logical_shape()[-1] == 2, "Grid tensor last dimension must be 2 (x, y coordinates)");
    TT_FATAL(padding_mode == "zeros", "Currently only 'zeros' padding mode is supported");
    TT_FATAL(input_shape.size() == 4, "Input shape must have 4 dimensions [N, H, W, C]");
    TT_FATAL(
        output_dtype == DataType::BFLOAT16 || !output_dtype.has_value(),
        "Currently only BFLOAT16 is supported for the grid output dtype");

    // Determine output data type
    DataType out_dtype = output_dtype.value_or(DataType::BFLOAT16);

    // Validate input grid data type
    TT_FATAL(grid.dtype() == DataType::FLOAT32, "Currently only float32 input grid is supported");

    // Calculate output shape: (N, H_out, W_out, 6) for bilinear, (N, H_out, W_out, 2) for nearest
    auto grid_shape = grid.logical_shape();
    uint32_t elements_per_point = (mode == "nearest") ? 2 : 6;
    ttnn::Shape output_shape({grid_shape[0], grid_shape[1], grid_shape[2], elements_per_point});

    // Dispatch based on output data type
    switch (out_dtype) {
        case DataType::BFLOAT16:
            return convert_grid_tensor<float, bfloat16>(
                grid, mode, align_corners, output_shape, input_shape, out_dtype);
        default: TT_THROW("Unsupported output data type for prepare_grid_sample_grid: {}", out_dtype);
    }
}

}  // namespace ttnn::operations::grid_sample
