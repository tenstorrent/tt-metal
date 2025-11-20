// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "op_slicing.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/slice_write/slice_write.hpp"
#include "ttnn/operations/experimental/padded_slice/padded_slice.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
namespace ttnn::operations::op_slicing {

void run_sliced_op(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& output_tensor,
    OpSliceAttr* op_slice_attr,
    Op2DSliceConfig dram_slice_config) {
    tt::tt_metal::Layout output_layout = output_tensor.layout();
    auto [batch_size, output_height, output_width, output_channels] = output_tensor.logical_shape().to_array_4D();
    auto [in_batch_, input_height, input_width, input_channels] = input_tensor.logical_shape().to_array_4D();

    uint32_t slice_rounding_value = 1;
    if (output_layout == tt::tt_metal::Layout::TILE &&
        dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_WIDTH) {
        // In DRAM Slicing with Tile Layout, the width must be a multiple of TILE_HEIGHT.
        slice_rounding_value = tt::constants::TILE_HEIGHT;
    }

    const uint32_t output_sliced_dim =
        dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;
    const uint32_t min_output_slice_size =
        tt::div_up(output_sliced_dim, slice_rounding_value) / dram_slice_config.num_slices;
    const uint32_t output_slice_rem =
        tt::div_up(output_sliced_dim, slice_rounding_value) % dram_slice_config.num_slices;

    uint32_t slice_index = 0;
    uint32_t output_slice_dim_start = 0;

    while ((output_slice_dim_start < output_sliced_dim) && (slice_index < dram_slice_config.num_slices)) {
        const uint32_t output_slice_size =
            slice_rounding_value * (min_output_slice_size + ((slice_index < output_slice_rem) ? 1 : 0));
        const uint32_t output_slice_dim_end = std::min(output_sliced_dim, output_slice_dim_start + output_slice_size);
        const uint32_t this_output_slice_dim = output_slice_dim_end - output_slice_dim_start;

        if (this_output_slice_dim == 0) {
            // No work to be done in this interation, so skip it.
            slice_index++;
            continue;
        }

        uint32_t output_slice_height_start, output_slice_height_end, input_slice_height_start, input_slice_height_end;
        uint32_t output_slice_width_start, output_slice_width_end, input_slice_width_start, input_slice_width_end;
        if (dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT) {
            output_slice_height_start = output_slice_dim_start;
            output_slice_height_end = output_slice_dim_end;
            output_slice_width_start = 0;
            output_slice_width_end = output_width;
            auto [input_slice_start, input_slice_end] = op_slice_attr->get_input_slice(
                {output_slice_height_start, output_slice_width_start},
                {output_slice_height_end, output_slice_width_end});
            std::tie(input_slice_height_start, input_slice_width_start) = input_slice_start;
            std::tie(input_slice_height_end, input_slice_width_end) = input_slice_end;

            input_slice_width_start = 0;
            input_slice_width_end = input_width;

            input_slice_height_start = std::max<int>(0, input_slice_height_start);
            input_slice_height_end = std::min<int>(input_height, input_slice_height_end);
            if (input_slice_height_start >= input_slice_height_end) {
                // No work to be done in this interation, so skip it.
                slice_index++;
                continue;
            }
        } else {
            output_slice_height_start = 0;
            output_slice_height_end = output_height;
            output_slice_width_start = output_slice_dim_start;
            output_slice_width_end = output_slice_dim_end;

            auto [input_slice_start, input_slice_end] = op_slice_attr->get_input_slice(
                {output_slice_height_start, output_slice_width_start},
                {output_slice_height_end, output_slice_width_end});
            std::tie(input_slice_height_start, input_slice_width_start) = input_slice_start;
            std::tie(input_slice_height_end, input_slice_width_end) = input_slice_end;

            input_slice_height_start = 0;
            input_slice_height_end = input_height;
            input_slice_width_start = std::max<int>(0, input_slice_width_start);
            input_slice_width_end = std::min<int>(input_width, input_slice_width_end);

            if (input_slice_width_start >= input_slice_width_end) {
                // No work to be done in this interation, so skip it.
                slice_index++;
                continue;
            }
        }

        log_trace(
            tt::LogOp,
            "Op {} DRAM Slicing: Slice {}: Output Slice Start: ({}, {}), End: ({}, {})",
            op_slice_attr->name(),
            slice_index,
            output_slice_height_start,
            output_slice_width_start,
            output_slice_height_end,
            output_slice_width_end);
        log_trace(
            tt::LogOp,
            "Op {} DRAM Slicing: Slice {}: Input Slice Start: ({}, {}), End: ({}, {})",
            op_slice_attr->name(),
            slice_index,
            input_slice_height_start,
            input_slice_width_start,
            input_slice_height_end,
            input_slice_width_end);

        const uint32_t output_slice_height = output_slice_height_end - output_slice_height_start;

        const uint32_t output_slice_width = output_slice_width_end - output_slice_width_start;

        log_debug(
            tt::LogOp,
            "Input Slice : {},{} ->  {},{}, Output Slice {} x {}",
            input_slice_height_start,
            input_slice_width_start,
            input_slice_height_end,
            input_slice_width_end,
            output_slice_height,
            output_slice_width);

        auto sliced_input_tensor_memory_config = op_slice_attr->get_input_memory_config(
            {output_slice_height_start, output_slice_width_start}, {output_slice_height_end, output_slice_width_end});

        const Tensor sliced_input_tensor = ttnn::experimental::padded_slice(
            input_tensor,
            ttnn::SmallVector<uint32_t>{0, input_slice_height_start, input_slice_width_start, 0},  // Start
            ttnn::SmallVector<uint32_t>{batch_size, input_slice_height_end, input_slice_width_end, input_channels},
            ttnn::SmallVector<uint32_t>{1, 1, 1, 1},  // Step
            sliced_input_tensor_memory_config);

        ttnn::Tensor sliced_output_tensor = op_slice_attr->run_L1_op(
            sliced_input_tensor,
            {output_slice_height_start, output_slice_width_start},
            {output_slice_height_end, output_slice_width_end});

        // slice_write supports all sharding layouts for tiled inputs. For row major, height & block sharding are
        // supported.
        if (sliced_output_tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED &&
            sliced_output_tensor.memory_config().memory_layout() != TensorMemoryLayout::BLOCK_SHARDED &&
            output_tensor.layout() == Layout::ROW_MAJOR) {
            sliced_output_tensor = ttnn::to_memory_config(
                sliced_output_tensor, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1});
        }
        if (sliced_output_tensor.layout() != Layout::ROW_MAJOR && output_layout == Layout::ROW_MAJOR) {
            sliced_output_tensor = ttnn::untilize(sliced_output_tensor);
        }
        if (sliced_output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED) {
            // slice_write expects the output tensor to be correctly shaped when its in interleaved memory layout.
            sliced_output_tensor = ttnn::reshape(
                sliced_output_tensor,
                ttnn::Shape({batch_size, output_slice_height, output_slice_width, output_channels}),
                ttnn::Shape(
                    {batch_size, output_slice_height, output_slice_width, sliced_output_tensor.padded_shape()[3]}));
        }
        ttnn::experimental::slice_write(
            sliced_output_tensor,
            output_tensor,
            ttnn::SmallVector<uint32_t>{0, output_slice_height_start, output_slice_width_start, 0},
            ttnn::SmallVector<uint32_t>{batch_size, output_slice_height_end, output_slice_width_end, output_channels},
            ttnn::SmallVector<uint32_t>{1, 1, 1, 1});
        output_slice_dim_start += output_slice_size;
        slice_index++;
    }
}
}  // namespace ttnn::operations::op_slicing
