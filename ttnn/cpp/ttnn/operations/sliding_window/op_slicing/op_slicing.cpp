// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "op_slicing.hpp"
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/functions.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/experimental/slice_write/slice_write.hpp>
#include <ttnn/operations/experimental/padded_slice/padded_slice.hpp>
namespace ttnn::operations::op_slicing {

static uint32_t compute_L1_usage_for_slice_config(
    const Shape& input_shape,
    const Shape& output_shape,
    tt::tt_metal::Layout output_layout,
    OpSliceAttr* op_slice_attr,
    const Op2DSliceConfig& dram_slice_config) {
    TT_FATAL(
        dram_slice_config.num_slices > 0, "Number of slices must be greater than 0 for DRAM L1 usage calculation.");
    auto [batch_size, output_height, output_width, output_channels] = output_shape.to_array_4D();
    auto [in_batch_, input_height, input_width, input_channels] = input_shape.to_array_4D();

    const uint32_t output_sliced_dim =
        dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;

    uint32_t slice_rounding_value = 1;

    if (output_layout == tt::tt_metal::Layout::TILE &&
        dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_WIDTH) {
        // Slice Write requires that the slice boundaries and shard boundaries are aligned to the tile boundaries.
        slice_rounding_value = tt::constants::TILE_HEIGHT;
    }

    const uint32_t min_output_slice_size =
        tt::div_up(output_sliced_dim, slice_rounding_value) / dram_slice_config.num_slices;
    const uint32_t output_slice_rem =
        tt::div_up(output_sliced_dim, slice_rounding_value) % dram_slice_config.num_slices;

    uint32_t max_memory_consumed = 0;
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

        max_memory_consumed = std::max(
            max_memory_consumed,
            op_slice_attr->get_L1_usage(
                {output_slice_height_start, output_slice_width_start},
                {output_slice_height_end, output_slice_width_end},
                dram_slice_config));
        output_slice_dim_start += output_slice_size;
        slice_index++;
    }
    return max_memory_consumed;
}

// Decide whether to slice along height or width based on input dimensions and output layout
// We ideally want to slice along the width dimension as it results in smaller halo size
// However, in case of very tall and narrow inputs, slicing along height is preferred to avoid
// very small slice sizes
// Additionally, for tiled outputs, there is a constraint that each slice's width must be a multiple of TILE_HEIGHT
// In this case, slicing along height is preferred to avoid this constraint.
static Op2DSliceConfig::SliceType best_guess_slice_type(
    uint32_t input_height, uint32_t input_width, Layout output_layout) {
    if (output_layout == Layout::ROW_MAJOR) {
        float threshold_ratio = 3.0;
        if (input_height > input_width * threshold_ratio) {
            return Op2DSliceConfig::SliceType::DRAM_HEIGHT;
        }
        return Op2DSliceConfig::SliceType::DRAM_WIDTH;
    }
    if (input_width < 200) {
        return Op2DSliceConfig::SliceType::DRAM_HEIGHT;
    }
    if (input_height > input_width) {
        return Op2DSliceConfig::SliceType::DRAM_HEIGHT;
    }
    return Op2DSliceConfig::SliceType::DRAM_WIDTH;
}

Op2DSliceConfig determine_slice_config(
    OpSliceAttr* op_slice_attr,
    const ttnn::Shape& input_shape,
    const ttnn::Shape& output_shape,
    const std::optional<Op2DSliceConfig> slice_config_,
    const tt::tt_metal::Layout output_layout,
    MeshDevice* device) {
    if (slice_config_.has_value() && slice_config_.value().num_slices > 0) {
        return slice_config_.value();
    }
    bool auto_slice_type = !slice_config_.has_value();
    auto L1_stats = device->allocator()->get_statistics(tt::tt_metal::BufferType::L1);
    Op2DSliceConfig return_slice_config;

    uint32_t output_height = output_shape[1];
    uint32_t output_width = output_shape[2];
    uint32_t current_num_slices = 1;
    log_debug(tt::LogOp, "DRAM Auto slice with {} free memory", L1_stats.total_free_bytes);
    const uint32_t output_sliced_dim =
        return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;

    if (auto_slice_type) {
        // Start with width slicing as it is more memory efficient.
        return_slice_config.slice_type = best_guess_slice_type(input_shape[1], input_shape[2], output_layout);
    } else {
        return_slice_config.slice_type = slice_config_.value().slice_type;
    }
    while (current_num_slices <= ((output_sliced_dim + 1) / 2)) {
        return_slice_config.num_slices = current_num_slices;
        uint32_t l1_usage = compute_L1_usage_for_slice_config(
            input_shape, output_shape, output_layout, op_slice_attr, return_slice_config);
        log_debug(
            tt::LogOp,
            "DRAM Auto slice for {} op with {} slices requires {} L1 memory",
            op_slice_attr->name(),
            current_num_slices,
            l1_usage);
        if (L1_stats.total_free_bytes >= l1_usage) {
            break;
        }
        current_num_slices++;
    }
    if (output_layout == tt::tt_metal::Layout::TILE &&
        return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_WIDTH) {
        // The rounding up of the slice size to tile height can result in the slice_config having a larger num_slices
        // than needed. So it is clamped.
        const uint32_t max_slices = tt::div_up(output_sliced_dim, tt::constants::TILE_HEIGHT);
        return_slice_config.num_slices = std::min(return_slice_config.num_slices, max_slices);
    }
    if (auto_slice_type && current_num_slices > ((output_sliced_dim - 1) / 2) &&
        output_layout == tt::tt_metal::Layout::TILE &&
        return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_WIDTH) {
        // For Tiled output with width slicing, we may not be able to find a suitable number of slices due to the
        // TILE_HEIGHT constraint.
        //  In this case, we switch to height slicing and try again.
        log_debug(
            tt::LogOp,
            "DRAM Auto slice could not find suitable number of slices with width slicing, switching to height "
            "slicing");
        return determine_slice_config(
            op_slice_attr,
            input_shape,
            output_shape,
            Op2DSliceConfig{.slice_type = Op2DSliceConfig::SliceType::DRAM_HEIGHT, .num_slices = 0},
            output_layout,
            device);
    }
    if (current_num_slices > output_sliced_dim) {
        log_warning(
            tt::LogOp,
            "DRAM Auto slice could not find suitable number of slices Slice config = {}",
            return_slice_config);
    }
    return return_slice_config;
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

    if (dram_slice_config_.has_value() && dram_slice_config_.value().num_slices > 0) {
        dram_slice_config = dram_slice_config_.value();
    } else {
        dram_slice_config = determine_slice_config(
            op_slice_attr,
            input_tensor.logical_shape(),
            output_tensors[0].get().logical_shape(),
            dram_slice_config_,
            output_layout,
            input_tensor.device());
        log_info(tt::LogOp, "Auto determined DRAM Slice Config as {} for {}", dram_slice_config, op_slice_attr->name());
    }

    log_debug(tt::LogOp, "{} DRAM with Slice Config {}", op_slice_attr->name(), dram_slice_config);
    TT_FATAL(dram_slice_config.num_slices > 0, " Number of slices should be greater than 0 for DRAM Slicing");

    uint32_t slice_rounding_value = 1;
    if (output_layout == tt::tt_metal::Layout::TILE &&
        dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_WIDTH) {
        // In DRAM Slicing with Tile Layout, the width must be a multiple of TILE_HEIGHT.
        slice_rounding_value = tt::constants::TILE_HEIGHT;
    }

    const uint32_t output_sliced_dim =
        dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;

    uint32_t max_num_slices = tt::div_up(output_sliced_dim, slice_rounding_value);
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
    dram_slice_config.num_slices = std::min(dram_slice_config.num_slices, max_num_slices);

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

    if (output_sliced_dim == 1) {
        dram_slice_config.num_slices = 1;
    } else {
        TT_ASSERT(
            dram_slice_config.num_slices < output_sliced_dim,
            " Number of slices {} should be less than the dimension {} being sliced in DRAM Slicing for {}",
            dram_slice_config.num_slices,
            output_sliced_dim,
            op_slice_attr->name());
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
            ttnn::experimental::slice_write(
                sliced_output_tensor,
                output_tensor,
                ttnn::SmallVector<uint32_t>{0, output_slice_height_start, output_slice_width_start, 0},
                ttnn::SmallVector<uint32_t>{
                    batch_size, output_slice_height_end, output_slice_width_end, output_channels},
                ttnn::SmallVector<uint32_t>{1, 1, 1, 1});
        }
        output_slice_dim_start += output_slice_size;
        slice_index++;
    }
}
}  // namespace ttnn::operations::op_slicing
