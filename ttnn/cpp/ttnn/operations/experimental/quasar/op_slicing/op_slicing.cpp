// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "op_slicing.hpp"
#include <tuple>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/functions.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/experimental/quasar/slice_write/slice_write.hpp>
#include <ttnn/operations/experimental/quasar/padded_slice/padded_slice.hpp>

namespace ttnn::operations::experimental::quasar::op_slicing {

// Reuse the arch-neutral base types and determine_slice_config from the shared op.
using namespace ttnn::operations::op_slicing;

// Compute the rounding value for slice boundaries based on output layout and slice type.
// For tiled outputs, slices must align to tile boundaries (32 elements).
// (Copied file-static from the shared op_slicing.cpp: run_sliced_op depends on it, and the
// shared definition has internal linkage so it is not visible across translation units.)
static uint32_t compute_slice_rounding_value(
    tt::tt_metal::Layout output_layout, Op2DSliceConfig::SliceType slice_type) {
    if (output_layout == tt::tt_metal::Layout::TILE && slice_type == Op2DSliceConfig::SliceType::DRAM_WIDTH) {
        return tt::constants::TILE_HEIGHT;
    }
    return 1;
}

// Compute the maximum number of slices allowed for the given output dimension and layout.
// (Copied file-static from the shared op_slicing.cpp for the same reason as above.)
static uint32_t compute_max_num_slices(
    uint32_t output_sliced_dim, uint32_t slice_rounding_value, tt::tt_metal::Layout output_layout) {
    if (output_layout == tt::tt_metal::Layout::TILE) {
        return tt::div_up(output_sliced_dim, slice_rounding_value);
    }
    return output_sliced_dim;
}

void run_sliced_op(
    const ttnn::Tensor& input_tensor,
    std::vector<OpSliceAttr::RefTensor>& output_tensors,
    OpSliceAttr* op_slice_attr,
    const std::optional<Op2DSliceConfig> dram_slice_config_) {
    Op2DSliceConfig dram_slice_config;

    tt::tt_metal::Layout output_layout = output_tensors[0].get().layout();
    uint32_t num_output_tensors = output_tensors.size();
    auto [batch_size, output_height, output_width, output_channels] =
        output_tensors[0].get().logical_shape().to_array_4D();
    auto [in_batch_, input_height, input_width, input_channels] = input_tensor.logical_shape().to_array_4D();

    log_debug(
        tt::LogOp,
        "run_sliced_op called: output_layout={}, output_shape={}x{}, dram_slice_config_.has_value()={}",
        output_layout == tt::tt_metal::Layout::TILE ? "TILE" : "ROW_MAJOR",
        output_height,
        output_width,
        dram_slice_config_.has_value());

    if (dram_slice_config_.has_value() && dram_slice_config_.value().num_slices > 0) {
        dram_slice_config = dram_slice_config_.value();
        log_debug(tt::LogOp, "Using provided slice config: num_slices={}", dram_slice_config.num_slices);
    } else {
        log_debug(tt::LogOp, "Calling determine_slice_config to auto-determine configuration");
        dram_slice_config = determine_slice_config(
            op_slice_attr,
            input_tensor.logical_shape(),
            output_tensors[0].get().logical_shape(),
            dram_slice_config_,
            output_layout,
            input_tensor.device());
        log_debug(
            tt::LogOp, "Auto determined DRAM Slice Config as {} for {}", dram_slice_config, op_slice_attr->name());

        // If auto-determination resulted in num_slices==1, convert to L1_FULL to avoid DRAM slicing overhead
        // A single slice means the entire operation fits in L1, so we should use the L1 path instead
        if (dram_slice_config.num_slices == 1) {
            log_debug(tt::LogOp, "Auto-determined num_slices=1, converting to L1_FULL for {}", op_slice_attr->name());
            dram_slice_config.slice_type = Op2DSliceConfig::SliceType::L1_FULL;
        }
    }

    TT_FATAL(
        dram_slice_config.num_slices > 0,
        "DRAM slicing configuration failed for {} with output layout {}. Unable to find a valid slice configuration. "
        "This indicates that even with maximum slicing granularity, the operation requires more L1 memory than "
        "available. "
        "Output shape: {}x{}, Available L1: {} bytes",
        op_slice_attr->name(),
        output_layout == tt::tt_metal::Layout::TILE ? "TILE" : "ROW_MAJOR",
        output_height,
        output_width,
        input_tensor.device()->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_free_bytes);

    log_debug(tt::LogOp, "{} DRAM with Slice Config {}", op_slice_attr->name(), dram_slice_config);

    const uint32_t slice_rounding_value = compute_slice_rounding_value(output_layout, dram_slice_config.slice_type);

    // DRAM_HEIGHT = slice along image height, DRAM_WIDTH = slice along image width
    const uint32_t output_sliced_dim =
        dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;

    const uint32_t max_num_slices = compute_max_num_slices(output_sliced_dim, slice_rounding_value, output_layout);

    if (max_num_slices == 1) {
        log_debug(
            tt::LogOp,
            "Op with Output Dimensions {}x{}, {} and {} can't be sliced. The L1 version of the op will be directly "
            "called on the full input. ",
            output_height,
            output_width,
            output_layout,
            dram_slice_config.slice_type);
    }
    TT_FATAL(
        dram_slice_config.num_slices <= max_num_slices,
        "Number of slices ({}) exceeds the maximum allowed ({}) for the given output dimension and alignment.",
        dram_slice_config.num_slices,
        max_num_slices);

    if (dram_slice_config.num_slices == 1) {
        for (auto& this_output_tensor : output_tensors) {
            this_output_tensor.get().deallocate(true);
        }
        auto op_output_tensors = op_slice_attr->run_L1_op(input_tensor, {0, 0}, {output_height, output_width});
        for (uint32_t i = 0; i < num_output_tensors; i++) {
            output_tensors[i].get() = ttnn::to_memory_config(
                op_output_tensors[i],
                tt::tt_metal::MemoryConfig{
                    TensorMemoryLayout::INTERLEAVED,
                    BufferType::DRAM,
                });
        }

        return;
    }

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
            // No work to be done in this iteration, so skip it.
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
            std::tie(input_slice_height_start, std::ignore) = input_slice_start;
            std::tie(input_slice_height_end, std::ignore) = input_slice_end;

            input_slice_width_start = 0;
            input_slice_width_end = input_width;

            input_slice_height_start = std::max<int>(0, input_slice_height_start);
            input_slice_height_end = std::min<int>(input_height, input_slice_height_end);
            if (input_slice_height_start >= input_slice_height_end) {
                // No work to be done in this iteration, so skip it.
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
            std::tie(std::ignore, input_slice_width_start) = input_slice_start;
            std::tie(std::ignore, input_slice_width_end) = input_slice_end;

            input_slice_height_start = 0;
            input_slice_height_end = input_height;
            input_slice_width_start = std::max<int>(0, input_slice_width_start);
            input_slice_width_end = std::min<int>(input_width, input_slice_width_end);

            if (input_slice_width_start >= input_slice_width_end) {
                // No work to be done in this iteration, so skip it.
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

        // Quasar fork: unconditionally use the quasar Metal-2 padded_slice.
        const Tensor sliced_input_tensor = ttnn::operations::experimental::quasar::padded_slice(
            input_tensor,
            ttsl::SmallVector<uint32_t>{0, input_slice_height_start, input_slice_width_start, 0},  // Start
            ttsl::SmallVector<uint32_t>{batch_size, input_slice_height_end, input_slice_width_end, input_channels},
            ttsl::SmallVector<uint32_t>{1, 1, 1, 1},  // Step
            sliced_input_tensor_memory_config);

        auto sliced_output_tensors = op_slice_attr->run_L1_op(
            sliced_input_tensor,
            {output_slice_height_start, output_slice_width_start},
            {output_slice_height_end, output_slice_width_end});
        TT_FATAL(
            sliced_output_tensors.size() == num_output_tensors,
            "Number of output tensors from run_L1_op {} does not match the expected number of output tensors {}",
            sliced_output_tensors.size(),
            num_output_tensors);
        for (uint32_t output_tensor_index = 0; output_tensor_index < num_output_tensors; output_tensor_index++) {
            auto& sliced_output_tensor = sliced_output_tensors[output_tensor_index];
            auto& output_tensor = output_tensors[output_tensor_index].get();
            // slice_write supports all sharding layouts for tiled inputs. For row major, height & block sharding are
            // supported.
            if (sliced_output_tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED &&
                sliced_output_tensor.memory_config().memory_layout() != TensorMemoryLayout::BLOCK_SHARDED &&
                output_layout == Layout::ROW_MAJOR) {
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
            // [#48552] Width tile-pad unblock. A DRAM-sliced conv output can arrive flattened as
            // [1,1,oh*padded_ow,C] where padded_ow is the tile/face-aligned width (e.g. 112 -> 128), while
            // the target slice region uses the TRUE width (output_slice_width). slice_write asserts
            // actual_shape.volume() == input.logical_volume(), which then fails (e.g. 401408 vs 458752, an
            // 8/7 width pad). fix_conv_output_logical_nhw() only strips the trailing height/NHW over-count
            // and is fed the padded ow, so it cannot see the per-row width pad, and it never runs on this
            // assembled tensor. Recover the true 4D shape here: logical uses the true width, padded keeps the
            // tile-padded width, so slice_write does the correct strided copy of the valid columns. No-op
            // when the flattened logical NHW already equals the true region (the common, already-correct
            // case, including the per-slice writes above).
            {
                const auto& lg = sliced_output_tensor.logical_shape();
                const uint64_t region_nhw =
                    static_cast<uint64_t>(batch_size) * output_slice_height * output_slice_width;
                const uint64_t hw_rows = static_cast<uint64_t>(batch_size) * output_slice_height;
                if (lg.rank() == 4 && lg[0] == 1 && lg[1] == 1 && static_cast<uint64_t>(lg[2]) > region_nhw &&
                    hw_rows != 0 && (static_cast<uint64_t>(lg[2]) % hw_rows) == 0) {
                    const uint32_t padded_ow = static_cast<uint32_t>(static_cast<uint64_t>(lg[2]) / hw_rows);
                    sliced_output_tensor = ttnn::reshape(
                        sliced_output_tensor,
                        ttnn::Shape({batch_size, output_slice_height, output_slice_width, output_channels}),
                        ttnn::Shape(
                            {batch_size, output_slice_height, padded_ow, sliced_output_tensor.padded_shape()[3]}));
                }
            }
            // Quasar fork: unconditionally use the quasar Metal-2 slice_write.
            ttnn::operations::experimental::quasar::slice_write(
                sliced_output_tensor,
                output_tensor,
                ttsl::SmallVector<uint32_t>{0, output_slice_height_start, output_slice_width_start, 0},
                ttsl::SmallVector<uint32_t>{
                    batch_size, output_slice_height_end, output_slice_width_end, output_channels},
                ttsl::SmallVector<uint32_t>{1, 1, 1, 1});
        }
        output_slice_dim_start += output_slice_size;
        slice_index++;
    }
}

}  // namespace ttnn::operations::experimental::quasar::op_slicing
