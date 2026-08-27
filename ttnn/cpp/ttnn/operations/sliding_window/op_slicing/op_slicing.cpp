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
#include <ttnn/operations/experimental/slice_write/slice_write.hpp>
#include <ttnn/operations/experimental/padded_slice/padded_slice.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>
#include <ttnn/operations/data_movement/concat/concat.hpp>
namespace ttnn::operations::op_slicing {

// Ops opt in to channel slicing by overriding these; the defaults keep every existing
// implementation spatial-only.
uint32_t OpSliceAttr::channel_slice_granularity() const { return 0; }

uint32_t OpSliceAttr::get_L1_usage_for_channel_slice(
    uint32_t /*channel_start*/, uint32_t /*channel_end*/, const op_slicing::Op2DSliceConfig& /*slice_config*/) const {
    TT_THROW("{} does not support channel slicing", name());
}

std::vector<ttnn::Tensor> OpSliceAttr::run_L1_op_channel_slice(
    const ttnn::Tensor& /*sliced_input_tensor*/, uint32_t /*channel_start*/, uint32_t /*channel_end*/) {
    TT_THROW("{} does not support channel slicing", name());
}

// Compute the rounding value for slice boundaries based on output layout and slice type.
// For tiled outputs, slices must align to tile boundaries (32 elements).
static uint32_t compute_slice_rounding_value(
    tt::tt_metal::Layout output_layout, Op2DSliceConfig::SliceType slice_type) {
    if (output_layout == tt::tt_metal::Layout::TILE && slice_type == Op2DSliceConfig::SliceType::DRAM_WIDTH) {
        return tt::constants::TILE_HEIGHT;
    }
    return 1;
}

static const char* slice_type_name(Op2DSliceConfig::SliceType slice_type) {
    switch (slice_type) {
        case Op2DSliceConfig::SliceType::DRAM_HEIGHT: return "height";
        case Op2DSliceConfig::SliceType::DRAM_WIDTH: return "width";
        case Op2DSliceConfig::SliceType::DRAM_CHANNEL: return "channel";
        case Op2DSliceConfig::SliceType::L1_FULL: return "L1_FULL";
    }
    return "unknown";
}

// Channel slices must respect the op's own alignment requirement rather than the output layout:
// the channel dimension is the innermost (stick) dimension, not the tilized height.
static uint32_t compute_channel_slice_rounding_value(const OpSliceAttr* op_slice_attr) {
    const uint32_t granularity = op_slice_attr->channel_slice_granularity();
    TT_FATAL(granularity > 0, "{} does not support channel slicing", op_slice_attr->name());
    return granularity;
}

// Walk the channel slices implied by num_slices, invoking `visit(start, end)` for each.
// Shared by the L1 usage estimate and the execution loop so the two cannot disagree.
template <typename VisitFn>
static void for_each_channel_slice(uint32_t channels, uint32_t rounding_value, uint32_t num_slices, VisitFn&& visit) {
    TT_FATAL(num_slices > 0, "Channel slicing requires at least one slice");
    const uint32_t rounded_units = tt::div_up(channels, rounding_value);
    const uint32_t min_units_per_slice = rounded_units / num_slices;
    const uint32_t units_remainder = rounded_units % num_slices;

    uint32_t channel_start = 0;
    for (uint32_t slice_index = 0; slice_index < num_slices && channel_start < channels; slice_index++) {
        const uint32_t slice_units = min_units_per_slice + ((slice_index < units_remainder) ? 1 : 0);
        const uint32_t channel_end = std::min(channels, channel_start + slice_units * rounding_value);
        if (channel_end > channel_start) {
            visit(channel_start, channel_end);
        }
        channel_start = channel_end;
    }
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

    if (dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_CHANNEL) {
        uint32_t max_memory_consumed = 0;
        for_each_channel_slice(
            output_channels,
            compute_channel_slice_rounding_value(op_slice_attr),
            dram_slice_config.num_slices,
            [&](uint32_t channel_start, uint32_t channel_end) {
                max_memory_consumed = std::max(
                    max_memory_consumed,
                    op_slice_attr->get_L1_usage_for_channel_slice(channel_start, channel_end, dram_slice_config));
            });
        return max_memory_consumed;
    }

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

            input_slice_width_start = std::max<int>(0, input_slice_width_start);
            input_slice_width_end = std::min<int>(input_width, input_slice_width_end);

            if (input_slice_width_start >= input_slice_width_end) {
                // No work to be done in this iteration, so skip it.
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

    log_debug(tt::LogOp, "DRAM Auto slice with {} free memory", L1_stats.total_free_bytes);
    log_debug(
        tt::LogOp,
        "Determining slice config: output_layout={}, output_height={}, output_width={}, auto_slice_type={}",
        output_layout == tt::tt_metal::Layout::TILE ? "TILE" : "ROW_MAJOR",
        output_height,
        output_width,
        auto_slice_type);

    const bool channel_slicing = return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_CHANNEL;

    const uint32_t slice_rounding_value =
        channel_slicing ? compute_channel_slice_rounding_value(op_slice_attr)
                        : compute_slice_rounding_value(output_layout, return_slice_config.slice_type);

    // DRAM_HEIGHT = slice along image height, DRAM_WIDTH = slice along image width,
    // DRAM_CHANNEL = slice along channels
    const uint32_t output_sliced_dim =
        channel_slicing ? output_shape[3]
                        : (return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? output_height
                                                                                                     : output_width);

    // Channel slices are bounded by the op's alignment, not by the output layout's tiling.
    const uint32_t max_num_slices =
        channel_slicing ? tt::div_up(output_sliced_dim, slice_rounding_value)
                        : compute_max_num_slices(output_sliced_dim, slice_rounding_value, output_layout);

    log_debug(
        tt::LogOp,
        "Max possible slices for {} layout and {}-slicing: {} (output_sliced_dim={})",
        output_layout == tt::tt_metal::Layout::TILE ? "TILE" : "ROW_MAJOR",
        slice_type_name(return_slice_config.slice_type),
        max_num_slices,
        output_sliced_dim);

    bool found_valid_config = false;
    while (current_num_slices <= max_num_slices) {
        return_slice_config.num_slices = current_num_slices;
        uint32_t l1_usage = compute_L1_usage_for_slice_config(
            input_shape, output_shape, output_layout, op_slice_attr, return_slice_config);
        log_debug(
            tt::LogOp,
            "Trying num_slices={}: L1 usage={}, available={}",
            current_num_slices,
            l1_usage,
            L1_stats.total_free_bytes);
        if (L1_stats.total_free_bytes >= l1_usage) {
            found_valid_config = true;
            log_debug(tt::LogOp, "Found valid config with num_slices={}, L1 usage={}", current_num_slices, l1_usage);
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
            slice_type_name(return_slice_config.slice_type),
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
            true);  // Mark as retry attempt
    }

    // Both spatial axes are exhausted. If the op has no cross-channel reduction, channels are still
    // a legal axis to split on -- and the only one left when the image is small but very deep
    // (e.g. a wide depthwise short convolution, whose output is 1 x 23 x 8192).
    if (!found_valid_config && !channel_slicing && op_slice_attr->channel_slice_granularity() > 0) {
        log_warning(
            tt::LogOp,
            "Failed to find valid config with {}-slicing. Attempting fallback to channel-slicing.",
            slice_type_name(return_slice_config.slice_type));
        return determine_slice_config_internal(
            op_slice_attr,
            input_shape,
            output_shape,
            Op2DSliceConfig{.slice_type = Op2DSliceConfig::SliceType::DRAM_CHANNEL, .num_slices = 0},
            output_layout,
            device,
            true);  // Mark as retry attempt
    }

    // If we haven't found a valid config, this is fatal
    TT_FATAL(
        found_valid_config,
        "DRAM Auto slice could not find valid slice configuration. Tried up to {} slices for {}-slicing on output "
        "dimension {}. Available L1: {} bytes. Operation requires more memory than available even with maximum "
        "slicing.",
        current_num_slices - 1,
        slice_type_name(return_slice_config.slice_type),
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
    MeshDevice* device) {
    return determine_slice_config_internal(
        op_slice_attr, input_shape, output_shape, slice_config_, output_layout, device, false);
}

// Execute the op one channel range at a time. Unlike the spatial loop there is no halo: slice N
// reads exactly the channels it writes, so a plain slice replaces padded_slice and the op reshards
// the activation itself.
static void run_channel_sliced_op(
    const ttnn::Tensor& input_tensor,
    std::vector<OpSliceAttr::RefTensor>& output_tensors,
    OpSliceAttr* op_slice_attr,
    uint32_t num_slices,
    tt::tt_metal::Layout output_layout) {
    auto [batch_size, output_height, output_width, output_channels] =
        output_tensors[0].get().logical_shape().to_array_4D();
    auto [in_batch_, input_height, input_width, input_channels] = input_tensor.logical_shape().to_array_4D();
    const uint32_t num_output_tensors = output_tensors.size();

    // No halo means the input and output channel ranges coincide, which only holds for ops whose
    // channels map one-to-one (pooling, depthwise convolution).
    TT_FATAL(
        input_channels == output_channels,
        "Channel slicing requires matching input and output channel counts for {}, got {} and {}",
        op_slice_attr->name(),
        input_channels,
        output_channels);

    // slice_write cannot take a nonzero offset in the last dimension for a tiled sharded input
    // (slice_write_tiled_sharded_input_program_factory: output_tensor_start[-1] == 0), so channel
    // slices are gathered and concatenated rather than written in place. The preallocated output is
    // replaced, as in the num_slices == 1 path.
    const MemoryConfig dram_interleaved{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    std::vector<std::vector<ttnn::Tensor>> gathered_slices(num_output_tensors);

    for_each_channel_slice(
        output_channels,
        compute_channel_slice_rounding_value(op_slice_attr),
        num_slices,
        [&](uint32_t channel_start, uint32_t channel_end) {
            log_trace(
                tt::LogOp,
                "Op {} DRAM channel slicing: channels [{}, {})",
                op_slice_attr->name(),
                channel_start,
                channel_end);

            const Tensor sliced_input_tensor = ttnn::slice(
                input_tensor,
                ttsl::SmallVector<uint32_t>{0, 0, 0, channel_start},
                ttsl::SmallVector<uint32_t>{batch_size, input_height, input_width, channel_end},
                ttsl::SmallVector<uint32_t>{1, 1, 1, 1});

            auto sliced_output_tensors =
                op_slice_attr->run_L1_op_channel_slice(sliced_input_tensor, channel_start, channel_end);
            TT_FATAL(
                sliced_output_tensors.size() == num_output_tensors,
                "Number of output tensors from run_L1_op_channel_slice {} does not match the expected number of "
                "output tensors {}",
                sliced_output_tensors.size(),
                num_output_tensors);

            const uint32_t channel_slice_size = channel_end - channel_start;
            for (uint32_t output_tensor_index = 0; output_tensor_index < num_output_tensors; output_tensor_index++) {
                auto& sliced_output_tensor = sliced_output_tensors[output_tensor_index];
                // Spill each slice to DRAM so concat sees uniform, unsharded inputs and L1 is freed
                // for the next slice.
                sliced_output_tensor = ttnn::to_memory_config(sliced_output_tensor, dram_interleaved);
                if (sliced_output_tensor.layout() != Layout::ROW_MAJOR && output_layout == Layout::ROW_MAJOR) {
                    sliced_output_tensor = ttnn::untilize(sliced_output_tensor);
                }
                // The op returns a flattened activation; restore [N, H, W, C] so the concatenated
                // result carries the 4D shape the caller allocated (it flattens this afterwards).
                sliced_output_tensor = ttnn::reshape(
                    sliced_output_tensor,
                    ttnn::Shape({batch_size, output_height, output_width, channel_slice_size}),
                    ttnn::Shape({batch_size, output_height, output_width, sliced_output_tensor.padded_shape()[3]}));
                gathered_slices[output_tensor_index].push_back(sliced_output_tensor);
            }
        });

    for (uint32_t output_tensor_index = 0; output_tensor_index < num_output_tensors; output_tensor_index++) {
        auto& output_tensor = output_tensors[output_tensor_index].get();
        // Free the preallocated output before concat allocates the real one.
        output_tensor.deallocate(true);
        output_tensor = ttnn::concat(gathered_slices[output_tensor_index], 3, dram_interleaved);
    }
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
        log_debug(tt::LogOp, "Auto determined DRAM Slice Config as {} for {}", dram_slice_config, op_slice_attr->name());

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

    // Channels are sliced by their own loop: the bounds below are spatial, and the channel axis has
    // no halo to compute.
    if (dram_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_CHANNEL) {
        run_channel_sliced_op(input_tensor, output_tensors, op_slice_attr, dram_slice_config.num_slices, output_layout);
        return;
    }

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

        const Tensor sliced_input_tensor = ttnn::experimental::padded_slice(
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
            ttnn::experimental::slice_write(
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
}  // namespace ttnn::operations::op_slicing
