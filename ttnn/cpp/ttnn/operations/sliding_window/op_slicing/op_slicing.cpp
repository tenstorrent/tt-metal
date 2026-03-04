// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
#include <ttnn/operations/experimental/slice_write/slice_write.hpp>
#include <ttnn/operations/experimental/padded_slice/padded_slice.hpp>
namespace ttnn::operations::op_slicing {

// Compute the rounding value for slice boundaries based on output layout and slice type.
// For tiled outputs, slices must align to tile boundaries (32 elements).
static uint32_t compute_slice_rounding_value(
    tt::tt_metal::Layout output_layout, Op2DSliceConfig::SliceType slice_type) {
    if (output_layout == tt::tt_metal::Layout::TILE && slice_type == Op2DSliceConfig::SliceType::DRAM_WIDTH) {
        return tt::constants::TILE_HEIGHT;
    }
    return 1;
}

// Compute the maximum number of slices allowed for the given output dimension and layout.
static uint32_t compute_max_num_slices(
    uint32_t output_sliced_dim, uint32_t slice_rounding_value, tt::tt_metal::Layout output_layout) {
    if (output_layout == tt::tt_metal::Layout::TILE) {
        return tt::div_up(output_sliced_dim, slice_rounding_value);
    }
    return output_sliced_dim;
}

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

    // DRAM_HEIGHT = slice along image height, DRAM_WIDTH = slice along image width
    const uint32_t output_sliced_dim =
        dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;

    const uint32_t slice_rounding_value = compute_slice_rounding_value(output_layout, dram_slice_config.slice_type);

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
            std::tie(input_slice_height_start, std::ignore) = input_slice_start;
            std::tie(input_slice_height_end, std::ignore) = input_slice_end;

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
            std::tie(std::ignore, input_slice_width_start) = input_slice_start;
            std::tie(std::ignore, input_slice_width_end) = input_slice_end;

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

// Internal helper that tracks whether we've already attempted a fallback
static Op2DSliceConfig determine_slice_config_internal(
    OpSliceAttr* op_slice_attr,
    const ttnn::Shape& input_shape,
    const ttnn::Shape& output_shape,
    const std::optional<Op2DSliceConfig> slice_config_,
    const tt::tt_metal::Layout output_layout,
    MeshDevice* device,
    bool conv_bypass,
    bool is_retry_attempt) {
    if (slice_config_.has_value() && slice_config_.value().num_slices > 0) {
        return slice_config_.value();
    }
    bool auto_slice_type = !slice_config_.has_value();
    auto L1_stats = device->allocator()->get_statistics(tt::tt_metal::BufferType::L1);
    Op2DSliceConfig return_slice_config;

    uint32_t output_height = output_shape[1];
    uint32_t output_width = output_shape[2];
    uint32_t current_num_slices = 1;

    if (auto_slice_type) {
        // Start with width slicing as it is more memory efficient.
        return_slice_config.slice_type = best_guess_slice_type(input_shape[1], input_shape[2], output_layout);
    } else {
        return_slice_config.slice_type = slice_config_.value().slice_type;
    }

    log_warning(tt::LogOp, "DRAM Auto slice with {} free memory", L1_stats.total_free_bytes);
    log_warning(
        tt::LogOp,
        "Determining slice config: output_layout={}, output_height={}, output_width={}, auto_slice_type={}",
        output_layout == tt::tt_metal::Layout::TILE ? "TILE" : "ROW_MAJOR",
        output_height,
        output_width,
        auto_slice_type);

    const uint32_t slice_rounding_value = compute_slice_rounding_value(output_layout, return_slice_config.slice_type);

    // DRAM_HEIGHT = slice along image height, DRAM_WIDTH = slice along image width
    const uint32_t output_sliced_dim =
        return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;

    const uint32_t max_num_slices = compute_max_num_slices(output_sliced_dim, slice_rounding_value, output_layout);

    log_warning(
        tt::LogOp,
        "Max possible slices for {} layout and {}-slicing: {} (output_sliced_dim={})",
        output_layout == tt::tt_metal::Layout::TILE ? "TILE" : "ROW_MAJOR",
        return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? "height" : "width",
        max_num_slices,
        output_sliced_dim);

    bool found_valid_config = false;
    while (current_num_slices <= max_num_slices) {
        return_slice_config.num_slices = current_num_slices;
        uint32_t l1_usage = compute_L1_usage_for_slice_config(
            input_shape, output_shape, output_layout, op_slice_attr, return_slice_config);
        log_warning(
            tt::LogOp,
            "Trying num_slices={}: L1 usage={}, available={}",
            current_num_slices,
            l1_usage,
            L1_stats.total_free_bytes);
        if (L1_stats.total_free_bytes >= l1_usage) {
            found_valid_config = true;
            log_warning(tt::LogOp, "Found valid config with num_slices={}, L1 usage={}", current_num_slices, l1_usage);
            break;
        }
        current_num_slices++;
    }

    if (auto_slice_type && !is_retry_attempt && current_num_slices > max_num_slices) {
        // Could not find a suitable number of slices for the initial slice type.
        // Try the opposite slice dimension before giving up.
        log_warning(
            tt::LogOp,
            "Failed to find valid config with {}-slicing. Attempting fallback to {}-slicing.",
            return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? "height" : "width",
            return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? "width" : "height");

        if (return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_WIDTH) {
            // Switch from width slicing to height slicing and try again.
            return determine_slice_config_internal(
                op_slice_attr,
                input_shape,
                output_shape,
                Op2DSliceConfig{.slice_type = Op2DSliceConfig::SliceType::DRAM_HEIGHT, .num_slices = 0},
                output_layout,
                device,
                conv_bypass,
                true);  // Mark as retry attempt
        }
        // Switch from height slicing to width slicing and try again.
        return determine_slice_config_internal(
            op_slice_attr,
            input_shape,
            output_shape,
            Op2DSliceConfig{.slice_type = Op2DSliceConfig::SliceType::DRAM_WIDTH, .num_slices = 0},
            output_layout,
            device,
            conv_bypass,
            true);  // Mark as retry attempt
    }

    // If we haven't found a valid config, this is fatal
    TT_FATAL(
        found_valid_config || conv_bypass,
        "DRAM Auto slice could not find valid slice configuration. Tried up to {} slices for {}-slicing on output "
        "dimension {}. Available L1: {} bytes. Operation requires more memory than available even with maximum "
        "slicing.",
        current_num_slices - 1,
        return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? "height" : "width",
        output_sliced_dim,
        L1_stats.total_free_bytes);

    return return_slice_config;
}

// Public wrapper that starts the slice configuration search
Op2DSliceConfig determine_slice_config(
    OpSliceAttr* op_slice_attr,
    const ttnn::Shape& input_shape,
    const ttnn::Shape& output_shape,
    const std::optional<Op2DSliceConfig> slice_config_,
    const tt::tt_metal::Layout output_layout,
    MeshDevice* device,
    bool conv_bypass) {
    return determine_slice_config_internal(
        op_slice_attr, input_shape, output_shape, slice_config_, output_layout, device, conv_bypass, false);
}

void run_sliced_op(
    const ttnn::Tensor& input_tensor,
    std::vector<OpSliceAttr::RefTensor>& output_tensors,
    OpSliceAttr* op_slice_attr,
    const std::optional<Op2DSliceConfig> dram_slice_config_,
    bool conv_bypass) {
    Op2DSliceConfig dram_slice_config;

    tt::tt_metal::Layout output_layout = output_tensors[0].get().layout();
    uint32_t num_output_tensors = output_tensors.size();
    auto [batch_size, output_height, output_width, output_channels] =
        output_tensors[0].get().logical_shape().to_array_4D();
    auto [in_batch_, input_height, input_width, input_channels] = input_tensor.logical_shape().to_array_4D();

    log_warning(
        tt::LogOp,
        "run_sliced_op called: output_layout={}, output_shape={}x{}, dram_slice_config_.has_value()={}",
        output_layout == tt::tt_metal::Layout::TILE ? "TILE" : "ROW_MAJOR",
        output_height,
        output_width,
        dram_slice_config_.has_value());

    if (dram_slice_config_.has_value() && dram_slice_config_.value().num_slices > 0) {
        dram_slice_config = dram_slice_config_.value();
        log_warning(tt::LogOp, "Using provided slice config: num_slices={}", dram_slice_config.num_slices);
    } else {
        log_warning(tt::LogOp, "Calling determine_slice_config to auto-determine configuration");
        dram_slice_config = determine_slice_config(
            op_slice_attr,
            input_tensor.logical_shape(),
            output_tensors[0].get().logical_shape(),
            dram_slice_config_,
            output_layout,
            input_tensor.device());
        log_info(tt::LogOp, "Auto determined DRAM Slice Config as {} for {}", dram_slice_config, op_slice_attr->name());
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
        dram_slice_config.num_slices <= max_num_slices || conv_bypass,
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
            std::tie(input_slice_height_start, std::ignore) = input_slice_start;
            std::tie(input_slice_height_end, std::ignore) = input_slice_end;

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
            std::tie(std::ignore, input_slice_width_start) = input_slice_start;
            std::tie(std::ignore, input_slice_width_end) = input_slice_end;

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
