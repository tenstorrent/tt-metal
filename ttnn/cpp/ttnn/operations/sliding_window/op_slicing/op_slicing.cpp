// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "op_slicing.hpp"
#include <map>
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

    const uint32_t slice_rounding_value = compute_slice_rounding_value(output_layout, return_slice_config.slice_type);

    // DRAM_HEIGHT = slice along image height, DRAM_WIDTH = slice along image width
    const uint32_t output_sliced_dim =
        return_slice_config.slice_type == Op2DSliceConfig::SliceType::DRAM_HEIGHT ? output_height : output_width;

    const uint32_t max_num_slices = compute_max_num_slices(output_sliced_dim, slice_rounding_value, output_layout);

    log_debug(
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

    // If we haven't found a valid config, this is fatal
    TT_FATAL(
        found_valid_config,
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
    MeshDevice* device) {
    return determine_slice_config_internal(
        op_slice_attr, input_shape, output_shape, slice_config_, output_layout, device, false);
}

// The starts of fixed slices of `slice` along an axis of `extent`: one every `slice`, the last shifted back to end at
// the extent, so all have the same size. An extent of at most one slice gets one slice at 0, which reaches past it.
static std::vector<uint32_t> fixed_slice_starts(uint32_t extent, uint32_t slice) {
    std::vector<uint32_t> starts;
    if (extent <= slice) {
        return {0};
    }
    for (uint32_t start = 0; start + slice < extent; start += slice) {
        starts.push_back(start);
    }
    starts.push_back(extent - slice);
    return starts;
}

static void run_fixed_sliced_op(
    const ttnn::Tensor& input_tensor,
    std::vector<OpSliceAttr::RefTensor>& output_tensors,
    OpSliceAttr* op_slice_attr,
    const Op2DSliceConfig& slice_config) {
    TT_FATAL(op_slice_attr->enable_fixed_slices(), "{} does not support fixed 2D DRAM slices", op_slice_attr->name());
    TT_FATAL(
        input_tensor.layout() == Layout::ROW_MAJOR && !input_tensor.is_sharded(),
        "Fixed 2D DRAM slices need a row-major interleaved input, got {} {}",
        input_tensor.layout(),
        input_tensor.memory_config());
    TT_FATAL(
        slice_config.slice_width % tt::constants::TILE_HEIGHT == 0,
        "The width of fixed 2D DRAM slices ({}) must be whole tiles",
        slice_config.slice_width);
    auto [batch_size, output_height, output_width, output_channels] =
        output_tensors[0].get().logical_shape().to_array_4D();
    const uint32_t input_channels = input_tensor.logical_shape()[-1];
    for (const auto& output_tensor : output_tensors) {
        TT_FATAL(output_tensor.get().layout() == Layout::TILE, "Fixed 2D DRAM slices need a tiled output");
    }

    // Along W the slices start on whole tiles of the tiled output, so they cover its width rounded up to whole tiles.
    auto h_starts = fixed_slice_starts(output_height, slice_config.slice_height);
    auto w_starts =
        fixed_slice_starts(tt::round_up(output_width, tt::constants::TILE_HEIGHT), slice_config.slice_width);

    // The input extent of the slice at an output start. A slice whose input window starts in the op's leading
    // padding reads fewer input rows or columns and pads that part itself; past the leading padding all slices read
    // the same extent, here taken at a start far from the edge.
    const auto input_extent = [&](uint32_t h_start, uint32_t w_start) {
        const auto [input_start, input_end] = op_slice_attr->get_input_slice(
            {h_start, w_start}, {h_start + slice_config.slice_height, w_start + slice_config.slice_width});
        return std::pair<uint32_t, uint32_t>{
            std::get<0>(input_end) - std::get<0>(input_start), std::get<1>(input_end) - std::get<1>(input_start)};
    };
    constexpr uint32_t far_start = 1U << 16;
    const auto [interior_h, interior_w] = input_extent(far_start, far_start);
    // Only slices at start 0 may lie in the leading padding, so there are at most four kinds (top-left, top, left,
    // interior) whatever the size. The last slice, shifted back to end at the edge, could land in the padding when
    // the padding is wider than the stride (a 5x5 kernel with padding 2, say): it is moved forward until it doesn't,
    // reaching further past the end, which is clipped. Along W it moves by whole tiles.
    if (h_starts.size() > 1) {
        while (input_extent(h_starts.back(), far_start).first < interior_h) {
            h_starts.back()++;
        }
    }
    if (w_starts.size() > 1) {
        while (input_extent(far_start, w_starts.back()).second < interior_w) {
            w_starts.back() += tt::constants::TILE_HEIGHT;
        }
    }
    for (uint32_t i = 1; i < h_starts.size(); i++) {
        TT_FATAL(
            input_extent(h_starts[i], far_start).first == interior_h,
            "{} fixed 2D DRAM slices: the slice height {} is too small for the op's top padding",
            op_slice_attr->name(),
            slice_config.slice_height);
    }
    for (uint32_t i = 1; i < w_starts.size(); i++) {
        TT_FATAL(
            input_extent(far_start, w_starts[i]).second == interior_w,
            "{} fixed 2D DRAM slices: the slice width {} is too small for the op's left padding",
            op_slice_attr->name(),
            slice_config.slice_width);
    }

    // The memory config of a slice depends on its input extent, which tells the slice's leading padding, so the
    // configs, which take a while to work out, are worked out once per extent. Fixed slices don't check
    // get_L1_usage: a slice too big for L1 fails when its L1 op allocates, not up front.
    std::map<std::pair<uint32_t, uint32_t>, MemoryConfig> input_memory_configs;
    for (const uint32_t h_start : h_starts) {
        for (const uint32_t w_start : w_starts) {
            const OpSliceAttr::IOShape output_start{h_start, w_start};
            const OpSliceAttr::IOShape output_end{
                h_start + slice_config.slice_height, w_start + slice_config.slice_width};
            const auto [input_start, input_end] = op_slice_attr->get_input_slice(output_start, output_end);
            log_trace(
                tt::LogOp,
                "{} fixed DRAM slice: output ({}, {}) to ({}, {}), input ({}, {}) to ({}, {})",
                op_slice_attr->name(),
                h_start,
                w_start,
                std::get<0>(output_end),
                std::get<1>(output_end),
                std::get<0>(input_start),
                std::get<1>(input_start),
                std::get<0>(input_end),
                std::get<1>(input_end));
            const std::pair<uint32_t, uint32_t> kind{
                std::get<0>(input_end) - std::get<0>(input_start), std::get<1>(input_end) - std::get<1>(input_start)};
            auto memory_config = input_memory_configs.find(kind);
            if (memory_config == input_memory_configs.end()) {
                memory_config =
                    input_memory_configs.emplace(kind, op_slice_attr->get_input_memory_config(output_start, output_end))
                        .first;
            }
            // The input slice may reach past the input, where padded_slice reads zeros; it and slice_write take
            // their offsets at run time, so every slice of every input size shares their programs.
            const Tensor sliced_input_tensor = ttnn::experimental::padded_slice(
                input_tensor,
                ttsl::SmallVector<uint32_t>{0, std::get<0>(input_start), std::get<1>(input_start), 0},
                ttsl::SmallVector<uint32_t>{batch_size, std::get<0>(input_end), std::get<1>(input_end), input_channels},
                ttsl::SmallVector<uint32_t>{1, 1, 1, 1},
                memory_config->second,
                std::nullopt,
                std::nullopt,
                /*runtime_shapes=*/true);
            auto sliced_output_tensors = op_slice_attr->run_L1_op(sliced_input_tensor, output_start, output_end);
            TT_FATAL(
                sliced_output_tensors.size() == output_tensors.size(),
                "Number of output tensors from run_L1_op {} does not match the expected number of output tensors {}",
                sliced_output_tensors.size(),
                output_tensors.size());
            for (uint32_t i = 0; i < output_tensors.size(); i++) {
                TT_FATAL(
                    sliced_output_tensors[i].is_sharded() && sliced_output_tensors[i].layout() == Layout::TILE,
                    "Fixed 2D DRAM slices need a sharded tiled output from {}",
                    op_slice_attr->name());
                // Written clipped to the output.
                ttnn::experimental::slice_write(
                    sliced_output_tensors[i],
                    output_tensors[i].get(),
                    ttsl::SmallVector<uint32_t>{0, h_start, w_start, 0},
                    ttsl::SmallVector<uint32_t>{
                        batch_size, std::get<0>(output_end), std::get<1>(output_end), output_channels},
                    ttsl::SmallVector<uint32_t>{1, 1, 1, 1},
                    /*runtime_shapes=*/true);
            }
        }
    }
}

void run_sliced_op(
    const ttnn::Tensor& input_tensor,
    std::vector<OpSliceAttr::RefTensor>& output_tensors,
    OpSliceAttr* op_slice_attr,
    const std::optional<Op2DSliceConfig> dram_slice_config_) {
    TT_FATAL(
        !dram_slice_config_.has_value() ||
            (dram_slice_config_->slice_height > 0) == (dram_slice_config_->slice_width > 0),
        "Fixed 2D DRAM slices need both slice_height and slice_width, got {} and {}",
        dram_slice_config_.has_value() ? dram_slice_config_->slice_height : 0,
        dram_slice_config_.has_value() ? dram_slice_config_->slice_width : 0);
    if (dram_slice_config_.has_value() && dram_slice_config_->is_fixed()) {
        run_fixed_sliced_op(input_tensor, output_tensors, op_slice_attr, dram_slice_config_.value());
        return;
    }
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
