// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/bcast/bcast.hpp"

#include "ttnn/operations/data_movement/bcast/device/bcast_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::data_movement {

// Does a broadcast
Tensor BcastOperation::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    BcastOpMath bcast_op,
    BcastOpDim bcast_dim,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output_tensor) {
    using namespace tt::constants;

    auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());

    if (bcast_dim == BcastOpDim::W) {
        TT_FATAL(
            input_tensor_a.padded_shape()[-2] == input_tensor_b.padded_shape()[-2],
            "Input tensor A height ({}) must equal input tensor B height ({}) for width broadcast",
            input_tensor_a.padded_shape()[-2],
            input_tensor_b.padded_shape()[-2]);
        if (input_tensor_b.layout() == Layout::TILE) {
            TT_FATAL(
                input_tensor_b.padded_shape()[-1] == TILE_WIDTH,
                "Input tensor B width ({}) must equal TILE_WIDTH ({}) for tile layout",
                input_tensor_b.padded_shape()[-1],
                TILE_WIDTH);
        } else if (input_tensor_b.layout() == Layout::ROW_MAJOR) {
            TT_FATAL(
                input_tensor_b.padded_shape()[-1] == 1 || input_tensor_b.padded_shape()[-1] == TILE_WIDTH,
                "Input tensor B width ({}) must be 1 or TILE_WIDTH ({}) for row major layout",
                input_tensor_b.padded_shape()[-1],
                TILE_WIDTH);
        } else {
            TT_THROW("Unsupported layout");
        }
    } else if (bcast_dim == BcastOpDim::H) {
        TT_FATAL(
            input_tensor_a.padded_shape()[-1] == input_tensor_b.padded_shape()[-1],
            "Input tensor A width ({}) must equal input tensor B width ({}) for height broadcast",
            input_tensor_a.padded_shape()[-1],
            input_tensor_b.padded_shape()[-1]);
        if (input_tensor_b.layout() == Layout::TILE) {
            TT_FATAL(
                input_tensor_b.padded_shape()[-2] == TILE_HEIGHT,
                "Input tensor B height ({}) must equal TILE_HEIGHT ({}) for tile layout",
                input_tensor_b.padded_shape()[-2],
                TILE_HEIGHT);
        } else if (input_tensor_b.layout() == Layout::ROW_MAJOR) {
            TT_FATAL(
                input_tensor_b.padded_shape()[-2] == 1 || input_tensor_b.padded_shape()[-2] == TILE_HEIGHT,
                "Input tensor B height ({}) must be 1 or TILE_HEIGHT ({}) for row major layout",
                input_tensor_b.padded_shape()[-2],
                TILE_HEIGHT);
        } else {
            TT_THROW("Unsupported layout");
        }
    } else if (bcast_dim == BcastOpDim::HW) {
        if (input_tensor_b.layout() == Layout::TILE) {
            TT_FATAL(
                input_tensor_b.padded_shape()[-2] == TILE_HEIGHT && input_tensor_b.padded_shape()[-1] == TILE_WIDTH,
                "Error");
        } else if (input_tensor_b.layout() == Layout::ROW_MAJOR) {
            TT_FATAL(
                (input_tensor_b.padded_shape()[-2] == 1 && input_tensor_b.padded_shape()[-1] == 1) ||
                    (input_tensor_b.padded_shape()[-2] == TILE_HEIGHT &&
                     input_tensor_b.padded_shape()[-1] == TILE_WIDTH),
                "Error");
        }
    }

    // Format inputs: device operation requires TILE layout, but API accepts both TILE and ROW_MAJOR
    auto format_input = [](const Tensor& input) -> Tensor {
        if (input.layout() == Layout::TILE) {
            // Already in TILE layout - no formatting needed (already tile-aligned)
            return input;
        } else {
            // ROW_MAJOR → TILE conversion needed
            // Use compute_padded_shape to calculate tile-aligned shape
            Shape tile_aligned_shape = compute_padded_shape(input.padded_shape(), TILE_HEIGHT, TILE_WIDTH);

            PadValue pad_value_variant;
            if (input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32) {
                pad_value_variant = 0.0f;
            } else {
                pad_value_variant = (uint32_t)0;
            }
            // tilize_with_val_padding handles both padding and tilization
            return ttnn::tilize_with_val_padding(input, tile_aligned_shape, pad_value_variant, input.memory_config());
        }
    };

    Tensor formatted_a = format_input(input_tensor_a);
    Tensor formatted_b = format_input(input_tensor_b);

    return tt::tt_metal::operation::run(
               EltwiseBinaryBroadcast{bcast_op, bcast_dim, output_memory_config},
               {formatted_a, formatted_b},
               {},
               {output_tensor})
        .at(0);
}

}  // namespace ttnn::operations::data_movement
