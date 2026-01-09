// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "sliding_window.hpp"
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <tt-logger/tt-logger.hpp>
#include <vector>
#include <tt_stl/assert.hpp>
#include <shared_mutex>

using namespace tt::tt_metal;

namespace ttnn::operations::sliding_window {

std::size_t SlidingWindowConfig::get_hash() const { return std::hash<std::string>{}(to_string()); }

std::array<uint32_t, 4> get_pair_n4_padding(
    const std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>& padding) {
    std::array<uint32_t, 4> ret_padding{};
    std::visit(
        [&](auto&& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, std::pair<uint32_pair_t, uint32_pair_t>>) {
                ret_padding[0] = value.first.first;
                ret_padding[1] = value.first.second;
                ret_padding[2] = value.second.first;
                ret_padding[3] = value.second.second;
            } else if constexpr (std::is_same_v<T, std::array<uint32_t, 4>>) {
                ret_padding = value;
            } else if constexpr (std::is_same_v<T, std::array<uint32_t, 2>>) {
                ret_padding[0] = value[0];
                ret_padding[1] = value[0];
                ret_padding[2] = value[1];
                ret_padding[3] = value[1];
            } else if constexpr (std::is_same_v<T, uint32_pair_t>) {
                ret_padding[0] = value.first;
                ret_padding[1] = value.first;
                ret_padding[2] = value.second;
                ret_padding[3] = value.second;
            }
        },
        padding);
    log_debug(
        tt::LogOp, "Padding = ({}, {}), ({}, {})", ret_padding[0], ret_padding[1], ret_padding[2], ret_padding[3]);
    return ret_padding;
}
/**
 * Return the input shape (excluding depth)
 */
ttnn::Shape SlidingWindowConfig::get_input_shape() const {
    return ttnn::Shape({batch_size, std::get<0>(input_hw), std::get<1>(input_hw)});
}

bool SlidingWindowConfig::has_parallel_config() const {
    return num_cores_nhw > 0 && num_cores_c > 0 && !core_range_set.ranges().empty();
}
/**
 * Calculate the window op output shape, excludes the channel dimension since this config is independent of the depth.
 */
ttnn::Shape SlidingWindowConfig::get_output_shape() const {
    if (is_transpose) {
        TT_FATAL(!ceil_mode, "ceil_mode is not supported for transposed operation");

        // This is the inverse calculation of the shape used in the forward pass.
        // Given the same values of stride, padding, dilation, and kernel size, the output shape of conv_transpose2d is
        // the input shape of conv2d, and vice versa.
        uint32_t output_h = ((input_hw.first - 1) * stride_hw.first) - get_pad_h() +
                            (dilation_hw.first * (window_hw.first - 1)) + output_pad_hw.first + 1;
        uint32_t output_w = ((input_hw.second - 1) * stride_hw.second) - get_pad_w() +
                            (dilation_hw.second * (window_hw.second - 1)) + output_pad_hw.second + 1;
        log_trace(
            tt::LogOp,
            "SlidingWindowConfig::get_output_shape(): {} {} {} {}",
            batch_size,
            output_h,
            output_w,
            "is_transpose==True");
        return ttnn::Shape({batch_size, output_h, output_w, 0});
    }

    uint32_t output_h;
    uint32_t output_w;

    // Note Pytorch doesn't support dilation for average pool, but TTNN may in the future
    // thus output size calculation is the same for average and max pool
    output_h = ((input_hw.first + get_pad_h() + get_ceil_pad_h() - (dilation_hw.first * (window_hw.first - 1)) - 1) /
                stride_hw.first) +
               1;
    output_w = ((input_hw.second + get_pad_w() + get_ceil_pad_w() - (dilation_hw.second * (window_hw.second - 1)) - 1) /
                stride_hw.second) +
               1;

    if (is_bilinear) {
        TT_FATAL(!ceil_mode, "ceil_mode is not supported for bilinear operation");
        TT_FATAL(
            padding[0] == 1 && padding[1] == 1 && padding[2] == 1 && padding[3] == 1,
            "Bilinear operation requires padding to be (1, 1, 1, 1)");
        TT_FATAL(stride_hw.first == 1 && stride_hw.second == 1, "Bilinear operation requires stride to be (1, 1)");
        TT_FATAL(
            dilation_hw.first == 1 && dilation_hw.second == 1, "Bilinear operation requires dilation to be (1, 1)");

        // for bilinear upsampling, output dimensions are scaled by scale factors
        output_h = input_hw.first * scale_h;
        output_w = input_hw.second * scale_w;
    }
    log_trace(
        tt::LogOp,
        "SlidingWindowConfig::get_output_shape():: {} {} {} for input {}",
        batch_size,
        output_h,
        output_w,
        *this);
    return ttnn::Shape({batch_size, output_h, output_w, 0});
}

uint32_t SlidingWindowConfig::get_pad_top() const { return padding[0]; }
uint32_t SlidingWindowConfig::get_pad_bottom() const { return padding[1]; }
uint32_t SlidingWindowConfig::get_pad_left() const { return padding[2]; }
uint32_t SlidingWindowConfig::get_pad_right() const { return padding[3]; }
uint32_t SlidingWindowConfig::get_pad_h() const { return padding[0] + padding[1]; }
uint32_t SlidingWindowConfig::get_pad_w() const { return padding[2] + padding[3]; }

uint32_pair_t SlidingWindowConfig::get_ceil_pad_hw() const {
    if (!ceil_mode) {
        return {0, 0};
    }
    if (ceil_pad_hw.has_value()) {
        return ceil_pad_hw.value();
    }
    float output_h_float =
        (float)(input_hw.first + get_pad_h() - (dilation_hw.first * (window_hw.first - 1)) - 1) / stride_hw.first;
    float output_w_float =
        (float)(input_hw.second + get_pad_w() - (dilation_hw.second * (window_hw.second - 1)) - 1) / stride_hw.second;

    uint32_t output_h = std::ceil(output_h_float) + 1;
    uint32_t output_w = std::ceil(output_w_float) + 1;

    // adjust the output shape if the last kernel position is in the padding region
    if (((output_h - 1) * stride_hw.first) >= (input_hw.first + padding[0])) {
        output_h--;
    }
    if (((output_w - 1) * stride_hw.second) >= (input_hw.second + padding[2])) {
        output_w--;
    }

    // Calculate effective kernel size with dilation
    uint32_t effective_kernel_h = (dilation_hw.first * (window_hw.first - 1)) + 1;
    uint32_t effective_kernel_w = (dilation_hw.second * (window_hw.second - 1)) + 1;

    // extra_padding = ceil size - non ceil size
    int32_t padding_calc_h = (stride_hw.first * (output_h - 1)) + effective_kernel_h - input_hw.first - get_pad_h();
    uint32_t ceil_padding_h = (padding_calc_h > 0) ? static_cast<uint32_t>(padding_calc_h) : 0;

    int32_t padding_calc_w = (stride_hw.second * (output_w - 1)) + effective_kernel_w - input_hw.second - get_pad_w();
    uint32_t ceil_padding_w = (padding_calc_w > 0) ? static_cast<uint32_t>(padding_calc_w) : 0;
    return {ceil_padding_h, ceil_padding_w};
}
uint32_t SlidingWindowConfig::get_ceil_pad_h() const { return get_ceil_pad_hw().first; }

uint32_t SlidingWindowConfig::get_ceil_pad_w() const { return get_ceil_pad_hw().second; }

ttnn::Shape SlidingWindowConfig::get_transposed_full_input_shape() const {
    TT_FATAL(
        is_transpose == true,
        "SlidingWindowConfig::get_transposed_full_input_shape() is only valid for transposed operation");
    auto output_shape = get_output_shape();
    uint32_t full_input_height = output_shape[1] + (dilation_hw.first * (window_hw.first - 1));
    uint32_t full_input_width = output_shape[2] + (dilation_hw.second * (window_hw.second - 1));
    return ttnn::Shape({batch_size, full_input_height, full_input_width, 0});
}

std::array<uint32_pair_t, 2> SlidingWindowConfig::get_transposed_real_padding() const {
    TT_FATAL(
        is_transpose == true,
        "SlidingWindowConfig::get_transposed_full_input_shape() is only valid for transposed operation");

    auto full_input_shape = get_transposed_full_input_shape();
    // Size of input after adding interleaved 0s.
    uint32_t strided_input_height = ((input_hw.first - 1) * stride_hw.first) + 1;
    uint32_t strided_input_width = ((input_hw.second - 1) * stride_hw.second) + 1;

    uint32_t base_pad_height = dilation_hw.first * (window_hw.first - 1);
    uint32_t base_pad_width = dilation_hw.second * (window_hw.second - 1);
    uint32_t input_pad_top = base_pad_height - padding[0];
    uint32_t input_pad_bottom = base_pad_height - padding[1];

    uint32_t input_pad_left = base_pad_width - padding[2];
    uint32_t input_pad_right = base_pad_width - padding[3];

    TT_FATAL(
        full_input_shape[1] - strided_input_height == input_pad_top + input_pad_bottom + output_pad_hw.first,
        "Calculated input padding height does not match the expected value.");
    TT_FATAL(
        full_input_shape[2] - strided_input_width == input_pad_left + input_pad_right + output_pad_hw.second,
        "Calculated input padding width does not match the expected value.");

    return {std::pair{input_pad_top, input_pad_bottom}, std::pair{input_pad_left, input_pad_right}};
}

/**
 * Calculate output tensor shard height
 */
uint32_t SlidingWindowConfig::get_output_shard_y(bool snap_to_tile) const {
    TT_ASSERT(has_parallel_config(), "Parallel config is not set in SlidingWindowConfig");
    ttnn::Shape output_shape = get_output_shape();
    uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
    uint32_t output_nhw_padded =
        tt::round_up(output_nhw, num_cores_nhw * (snap_to_tile ? tt::constants::TILE_HEIGHT : 1));
    log_trace(
        tt::LogOp,
        "output_nhw: {} output_nhw_padded: {} num_cores_nhw: {}",
        output_nhw,
        output_nhw_padded,
        num_cores_nhw);
    return (output_nhw_padded / num_cores_nhw);
}

std::vector<bool> generate_pad_metadata(const SlidingWindowConfig& config) {
    if (config.is_transpose) {
        auto full_input_shape = config.get_transposed_full_input_shape();
        auto full_input_height = full_input_shape[1];
        auto full_input_width = full_input_shape[2];

        auto real_padding = config.get_transposed_real_padding();
        std::vector<bool> pad_metadata(config.batch_size * full_input_height * full_input_width, false);

        // Size of input after adding interleaved 0s.
        uint32_t strided_input_height = ((config.input_hw.first - 1) * config.stride_hw.first) + 1;
        uint32_t strided_input_width = ((config.input_hw.second - 1) * config.stride_hw.second) + 1;

        auto [input_pad_top, input_pad_bottom] = real_padding[0];
        auto [input_pad_left, input_pad_right] = real_padding[1];

        for (uint32_t b = 0; b < config.batch_size; ++b) {
            for (uint32_t h = 0; h < full_input_height; ++h) {
                for (uint32_t w = 0; w < full_input_width; ++w) {
                    if (h < input_pad_top || h >= strided_input_height + input_pad_top || w < input_pad_left ||
                        w >= strided_input_width + input_pad_left) {
                        pad_metadata[(b * full_input_height * full_input_width) + (h * full_input_width) + w] = true;
                    } else {
                        if (((h - input_pad_top) % config.stride_hw.first != 0 ||
                             ((w - input_pad_left) % config.stride_hw.second != 0))) {
                            pad_metadata[(b * full_input_height * full_input_width) + (h * full_input_width) + w] =
                                true;
                        }
                    }
                }
            }
        }
        return pad_metadata;
    }
    uint32_t ceil_padding_h = config.get_ceil_pad_h();
    uint32_t ceil_padding_w = config.get_ceil_pad_w();
    uint32_t padded_input_h = config.input_hw.first + config.get_pad_h() + ceil_padding_h;
    uint32_t padded_input_w = config.input_hw.second + config.get_pad_w() + ceil_padding_w;
    std::vector<bool> pad_metadata(config.batch_size * padded_input_h * padded_input_w, false);

    for (uint32_t b = 0; b < config.batch_size; ++b) {
        for (uint32_t h = 0; h < padded_input_h; ++h) {
            for (uint32_t w = 0; w < padded_input_w; ++w) {
                if (h < config.padding[0] || h >= (config.padding[0] + config.input_hw.first) ||
                    w < config.padding[2] || w >= (config.padding[2] + config.input_hw.second)) {
                    pad_metadata[(b * padded_input_h * padded_input_w) + (h * padded_input_w) + w] = true;
                }
            }
        }
    }
    return pad_metadata;
}
std::vector<uint32_t> generate_op_trace_metadata(const SlidingWindowConfig& config) {
    if (config.is_bilinear) {
        return generate_op_trace_metadata_bilinear(config);
    }

    ttnn::Shape output_shape = config.get_output_shape();
    uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
    std::vector<uint32_t> op_trace_metadata(output_nhw, 0);

    if (config.is_transpose) {
        auto full_input_shape = config.get_transposed_full_input_shape();
        uint32_t padded_input_h = full_input_shape[1];
        uint32_t padded_input_w = full_input_shape[2];
        uint32_t i = 0;
        for (uint32_t b = 0; b < output_shape[0]; ++b) {
            for (uint32_t h = 0; h < output_shape[1]; ++h) {
                for (uint32_t w = 0; w < output_shape[2]; ++w) {
                    // In Transpose as Conv2d, Stride is always 1
                    uint32_t input_index = (b * padded_input_h * padded_input_w) + (h * padded_input_w) + w;
                    op_trace_metadata[i++] = input_index;
                }
            }
        }
    } else {
        uint32_t ceil_padding_h = config.get_ceil_pad_h();
        uint32_t ceil_padding_w = config.get_ceil_pad_w();

        uint32_t padded_input_h = config.input_hw.first + config.get_pad_h() + ceil_padding_h;
        uint32_t padded_input_w = config.input_hw.second + config.get_pad_w() + ceil_padding_w;
        uint32_t i = 0;
        for (uint32_t b = 0; b < output_shape[0]; ++b) {
            for (uint32_t h = 0; h < output_shape[1]; ++h) {
                for (uint32_t w = 0; w < output_shape[2]; ++w) {
                    uint32_t input_index = (b * padded_input_h * padded_input_w) +
                                           (h * config.stride_hw.first * padded_input_w) +
                                           (w * config.stride_hw.second);
                    op_trace_metadata[i++] = input_index;
                }
            }
        }
    }
    return op_trace_metadata;
}
std::vector<uint32_t> generate_op_trace_metadata_bilinear(const SlidingWindowConfig& config) {
    const auto output_shape = config.get_output_shape();

    const uint32_t padded_input_h = config.input_hw.first + config.get_pad_h();
    const uint32_t padded_input_w = config.input_hw.second + config.get_pad_w();

    // Calculate scale factors
    const float scale_h_inv = 1.0f / static_cast<float>(config.scale_h);
    const float scale_w_inv = 1.0f / static_cast<float>(config.scale_w);

    // Create op trace metadata for the output tensor size
    const uint32_t output_nhw = config.batch_size * output_shape[1] * output_shape[2];
    std::vector<uint32_t> op_trace_metadata(output_nhw, 0);

    // Nested loops for output tensor coordinates with floating point offsets

    uint32_t i = 0;

    // Bilinear upsampling determines the value of an output pixel based on the output pixels coordinates
    // The output pixel coordinates map to input pixel coordinates with the formula
    // input_coord = (output_coord + 0.5) * scale_inv - 0.5
    // However, since one pixel of padding is included at all sides (to avoid negative indices), the offset
    // calculation is adjusted to: input_coord = (output_coord + 0.5) * scale_inv + 0.5
    // The +0.5 comes from -0.5 (from the formula) + 1.0 (accounts for padding) and results in +0.5.
    const float bilinear_starting_offset = 0.5f;

    for (uint32_t b = 0; b < config.batch_size; ++b) {
        float h_offset = (0.5f * scale_h_inv) + bilinear_starting_offset;
        for (uint32_t h = 0; h < output_shape[1]; ++h) {
            // After calculating the input coordinate, the top-left input pixel is determined by flooring the input
            // coordinates
            float w_offset = (0.5f * scale_w_inv) + bilinear_starting_offset;
            for (uint32_t w = 0; w < output_shape[2]; ++w) {
                // Get top-left input coordinate (this is the "top-left index" needed for bilinear interpolation)
                const uint32_t top_left_h = static_cast<uint32_t>(std::floor(h_offset));
                const uint32_t top_left_w = static_cast<uint32_t>(std::floor(w_offset));

                // Calculate flattened input index (top-left coordinate for bilinear interpolation)
                const uint32_t input_idx =
                    (b * padded_input_h * padded_input_w) + (top_left_h * padded_input_w) + top_left_w;
                op_trace_metadata[i++] = input_idx;
                w_offset += scale_w_inv;
            }
            h_offset += scale_h_inv;
        }
    }
    return op_trace_metadata;
}

std::pair<uint32_t, uint32_t> find_minmax_trace_indices(
    const std::vector<uint32_t>& op_trace_metadata, uint32_t start_idx, uint32_t end_idx) {
    TT_ASSERT(
        start_idx <= end_idx && end_idx < op_trace_metadata.size() && start_idx < op_trace_metadata.size() &&
            start_idx >= 0,
        "Error in find_minmax_trace_indices: invalid start or end index");
    auto [min_it, max_it] =
        std::minmax_element(op_trace_metadata.begin() + start_idx, op_trace_metadata.begin() + end_idx + 1);
    return {
        static_cast<uint32_t>(min_it - op_trace_metadata.begin()),
        static_cast<uint32_t>(max_it - op_trace_metadata.begin())};
}

std::vector<ShardBoundary> generate_shard_boundaries(const SlidingWindowConfig& config) {
    auto op_trace_metadata = generate_op_trace_metadata(config);
    std::vector<ShardBoundary> shard_boundaries;

    const uint32_t num_cores = config.num_cores_nhw;
    const uint32_t output_shard_h = config.get_output_shard_y(config.snap_to_tile);

    const uint32_t ceil_padding_w = config.get_ceil_pad_w();
    uint32_t padded_input_w = config.input_hw.second + config.get_pad_w() + ceil_padding_w;

    uint32_t max_index = op_trace_metadata.size();
    if (config.is_transpose) {
        padded_input_w = config.get_transposed_full_input_shape()[2];
    }
    uint32_t dilated_window_h =
        config.window_hw.first + ((config.dilation_hw.first - 1) * (config.window_hw.first - 1));
    uint32_t dilated_window_w =
        config.window_hw.second + ((config.dilation_hw.second - 1) * (config.window_hw.second - 1));
    uint32_t halo_with_pad_len = ((dilated_window_h - 1) * padded_input_w) + dilated_window_w - 1;

    uint32_t output_index_start = 0;
    for (uint32_t core = 0; core < num_cores; ++core) {
        const uint32_t output_index_end = std::min(output_index_start + output_shard_h, max_index) - 1;
        uint32_t input_index_start;
        uint32_t input_index_end;
        if (config.is_bilinear) {
            // For bilinear, find the indices with minimum and maximum values in the output shard range
            // This is because, in the case of bilinear upsampling, the op_trace_metadata tensor is not necessarily
            // monotonically increasing, which means that minimums and maximums (and thus the required input shard
            // boundaries) might not correspond to the first and last output pixel
            if (output_index_start > output_index_end) {
                input_index_start = 1;
                input_index_end = 0;
            } else {
                auto [min_trace_idx, max_trace_idx] =
                    find_minmax_trace_indices(op_trace_metadata, output_index_start, output_index_end);
                input_index_start = op_trace_metadata[min_trace_idx];
                input_index_end = op_trace_metadata[max_trace_idx] + halo_with_pad_len;
            }
        } else {
            input_index_start = op_trace_metadata[std::min(output_index_start, max_index - 1)];
            input_index_end = op_trace_metadata[output_index_end] + halo_with_pad_len;
        }
        if (input_index_start == 0 and output_index_start != 0 && !config.is_bilinear) {
            input_index_start = op_trace_metadata[output_index_end] + 1;
            input_index_end = input_index_start - 1;
            log_debug(
                tt::LogOp,
                "core: {}, output_index_start: {}, output_index_end: {}, input_index_start: {}, input_index_end: {}",
                core,
                output_index_start,
                output_index_end,
                input_index_start,
                input_index_end);
        }
        shard_boundaries.push_back({{output_index_start, output_index_end}, {input_index_start, input_index_end}});
        output_index_start = output_index_end + 1;
    }

    return shard_boundaries;
}

std::vector<PixelMetadata> generate_tensor_metadata(
    const std::vector<bool>& pad_metadata, const SlidingWindowConfig& config, uint32_t shard_height) {
    std::vector<PixelMetadata> tensor_metadata;
    tensor_metadata.reserve(pad_metadata.size());

    uint32_t core_id = 0;
    uint32_t input_reshard_local_idx = 0;

    for (bool is_pad_flag : pad_metadata) {
        if (is_pad_flag) {
            tensor_metadata.push_back(PixelMetadata{true, 0, 0});
        } else {
            tensor_metadata.push_back(PixelMetadata{false, core_id, input_reshard_local_idx});

            input_reshard_local_idx++;
            if (input_reshard_local_idx == shard_height) {
                core_id++;
                input_reshard_local_idx = 0;
            }
        }
    }

    return tensor_metadata;
}

const uint16_t PAD_LOCAL_SENTINAL = 0xFFFF;

using uint32_triplet_t = std::tuple<uint32_t, uint32_t, uint32_t>;
using GatherStep = uint32_triplet_t;
using PerCoreGatherData = std::map<std::pair<uint32_t, uint32_t>, std::vector<GatherStep>>;

uint32_t generate_max_out_nsticks_per_core(const std::vector<ShardBoundary>& shard_boundaries) {
    uint32_t max_out_nsticks_per_core = 0;
    for (auto [_, in_shard] : shard_boundaries) {
        auto [in_start, in_end] = in_shard;
        max_out_nsticks_per_core = std::max(max_out_nsticks_per_core, in_end - in_start + 1);
    }
    return max_out_nsticks_per_core;
}

uint32_t calculate_precise_halo_output_elems(
    const SlidingWindowConfig& config, const std::array<uint32_t, 2>& shard_shape) {
    // Generate metadata for precise calculation
    auto shard_boundaries = generate_shard_boundaries(config);

    // Get precise max sticks per core
    uint32_t max_out_nsticks_per_core = generate_max_out_nsticks_per_core(shard_boundaries);
    return max_out_nsticks_per_core * shard_shape[1];
}

struct GatherHeader {
    uint16_t noc_x;
    uint16_t noc_y;
    uint16_t num_transfers;
};

struct GatherTransfer {
    uint16_t src_id;
    uint16_t dst_id;
    uint16_t size;
};

struct GatherRoute {
    GatherHeader header{};
    std::vector<GatherTransfer> transfers;
};

struct GatherConfig {
    std::vector<GatherRoute> routes;
};

// transfer = noc_x noc_y num_transfers
static void serialize_gather_header(const GatherHeader& header, std::vector<uint16_t>& output) {
    output.push_back(header.noc_x);
    output.push_back(header.noc_y);
    output.push_back(header.num_transfers);
}

// transfer = src_id dst_id size
static void serialize_gather_transfer(const GatherTransfer& transfer, std::vector<uint16_t>& output) {
    output.push_back(transfer.src_id);
    output.push_back(transfer.dst_id);
    output.push_back(transfer.size);
}

// route = header [tranfer0 transfer1 ... ]
static void serialize_gather_route(const GatherRoute& route, std::vector<uint16_t>& output) {
    serialize_gather_header(route.header, output);
    TT_FATAL(route.header.num_transfers == route.transfers.size(), "Number of transfers in route must match header");
    for (const auto& transfer : route.transfers) {
        serialize_gather_transfer(transfer, output);
    }
}

// config = len [route0 route1 ...]
static std::vector<uint16_t> serialize_gather_config(const GatherConfig& config) {
    std::vector<uint16_t> output;
    output.push_back(config.routes.size());
    for (const auto& route : config.routes) {
        TT_FATAL(!route.transfers.empty(), "Expected all routes to have at least one transfer");
        serialize_gather_route(route, output);
    }
    return output;
}

// Flatten a list of configs and ensure they are uniform lengths by padding
static std::vector<std::vector<uint16_t>> serialize_gather_configs(const std::vector<GatherConfig>& configs) {
    std::vector<std::vector<uint16_t>> serialized_configs;
    serialized_configs.reserve(configs.size());
    for (const auto& config : configs) {
        serialized_configs.push_back(serialize_gather_config(config));
    }
    // Pad each core's config to the same length so we can shard it
    size_t max_size = 0;
    for (const auto& config : serialized_configs) {
        max_size = std::max(max_size, config.size());
    }
    for (std::vector<uint16_t>& config : serialized_configs) {
        TT_ASSERT(config.size() <= max_size);
        config.resize(max_size, 0);
    }
    return serialized_configs;
}

struct DestinationTransferPair {
    uint16_t noc_x;
    uint16_t noc_y;
    uint16_t src_id;
    uint16_t dst_id;
    uint16_t size;
};

static std::vector<DestinationTransferPair> flatten_gather_config(const GatherConfig& input) {
    std::vector<DestinationTransferPair> all;
    for (const auto& route : input.routes) {
        for (const auto& t : route.transfers) {
            all.push_back({route.header.noc_x, route.header.noc_y, t.src_id, t.dst_id, t.size});
        }
    }
    return all;
}

// Rebuild config by grouping consecutive transfers that share (noc_x, noc_y)
static GatherConfig reduce_flattened_transfers(const std::vector<DestinationTransferPair>& transfers) {
    GatherConfig output;

    if (transfers.empty()) {
        return output;
    }

    GatherRoute current_route;
    current_route.header.noc_x = transfers[0].noc_x;
    current_route.header.noc_y = transfers[0].noc_y;
    for (const auto& xfer : transfers) {
        bool same_core = (xfer.noc_x == current_route.header.noc_x) && (xfer.noc_y == current_route.header.noc_y);
        if (!same_core) {
            current_route.header.num_transfers = static_cast<uint16_t>(current_route.transfers.size());
            TT_FATAL(current_route.header.num_transfers > 0, "Route cannot have zero transfers");
            output.routes.push_back(current_route);

            current_route = GatherRoute();
            current_route.header.noc_x = xfer.noc_x;
            current_route.header.noc_y = xfer.noc_y;
        }

        GatherTransfer transfer{};
        transfer.src_id = xfer.src_id;
        transfer.dst_id = xfer.dst_id;
        transfer.size = xfer.size;
        current_route.transfers.push_back(transfer);
    }
    current_route.header.num_transfers = static_cast<uint16_t>(current_route.transfers.size());
    TT_FATAL(current_route.header.num_transfers > 0, "Route cannot have zero transfers");
    output.routes.push_back(current_route);

    return output;
}

// Reorder config such that transfers are strongly ordered by their source offset. In order to satisfy this guarantee,
// there may be duplicate routes added to the config.
static GatherConfig reorder_transfers_globally(const GatherConfig& input) {
    std::vector<DestinationTransferPair> all = flatten_gather_config(input);
    if (!all.empty()) {
        // Sort by ascending src_id and tie-break with noc coords
        std::sort(all.begin(), all.end(), [](const DestinationTransferPair& a, const DestinationTransferPair& b) {
            if (a.src_id != b.src_id) {
                return a.src_id < b.src_id;
            }
            if (a.noc_x != b.noc_x) {
                return a.noc_x < b.noc_x;
            }
            return a.noc_y < b.noc_y;
        });
    }
    return reduce_flattened_transfers(all);
}

// Split up transfers that span across blocks of some given block size
static GatherConfig quantize_transfers_along_block_boundaries(const GatherConfig& input, uint32_t block_size) {
    GatherConfig output;
    for (const auto& route : input.routes) {
        GatherRoute new_route;
        new_route.header = route.header;

        for (const auto& transfer : route.transfers) {
            uint32_t src_offset = transfer.src_id;
            uint32_t dst_offset = transfer.dst_id;
            uint32_t length = transfer.size;
            while (length > 0) {
                const uint32_t offset_in_block = src_offset % block_size;
                const uint32_t remaining_in_block = block_size - offset_in_block;
                const uint32_t transfer_size = (length <= remaining_in_block) ? length : remaining_in_block;

                new_route.transfers.push_back(GatherTransfer{src_offset, dst_offset, transfer_size});

                src_offset += transfer_size;
                dst_offset += transfer_size;
                length -= transfer_size;
            }
        }
        new_route.header.num_transfers = new_route.transfers.size();
        output.routes.push_back(new_route);
    }
    return output;
}

const uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;

// Round-robin blocks between two cores
static std::tuple<GatherConfig, GatherConfig, uint32_t> divide_blocks_between_cores(
    const GatherConfig& input, uint32_t block_size) {
    std::vector<DestinationTransferPair> all = flatten_gather_config(input);
    std::vector<DestinationTransferPair> first;
    std::vector<DestinationTransferPair> second;
    int32_t number_of_blocks = 0;
    for (const auto& transfer : all) {
        const auto block_id = transfer.src_id / block_size;
        (block_id % NUM_RISCV_DATA_MOVEMENT_CORES == 0 ? first : second).push_back(transfer);
        number_of_blocks = std::max(number_of_blocks, static_cast<int32_t>(block_id + 1));
    }
    return std::make_tuple(reduce_flattened_transfers(first), reduce_flattened_transfers(second), number_of_blocks);
}

// Round-robin transfers between two cores
static std::pair<GatherConfig, GatherConfig> divide_transfers_between_cores(const GatherConfig& input) {
    std::vector<DestinationTransferPair> all = flatten_gather_config(input);
    std::vector<DestinationTransferPair> first;
    std::vector<DestinationTransferPair> second;
    for (int transfer_id = 0; transfer_id < all.size(); transfer_id++) {
        (transfer_id % NUM_RISCV_DATA_MOVEMENT_CORES == 0 ? first : second).push_back(all[transfer_id]);
    }
    return std::make_pair(reduce_flattened_transfers(first), reduce_flattened_transfers(second));
}

HaloGatherKernelConfig generate_halo_kernel_config_tensors(
    const std::vector<PixelMetadata>& tensor_metadata,
    const std::vector<ShardBoundary>& shard_boundaries,
    bool is_block_sharded,
    bool transpose_mcast,
    bool remote_read,
    IDevice* device,
    uint32_t num_cores_x,
    bool is_in_tiled,
    int block_size) {
    auto core_id_to_noc_coords =
        [is_block_sharded, transpose_mcast, device, num_cores_x](uint32_t core_id) -> CoreCoord {
        auto core_coord = is_block_sharded ? (transpose_mcast ? CoreCoord(core_id, 0) : CoreCoord(0, core_id))
                                           : CoreCoord(core_id % num_cores_x, core_id / num_cores_x);
        return device->worker_core_from_logical_core(core_coord);
    };

    uint32_t num_cores_nhw = shard_boundaries.size();
    PerCoreGatherData
        per_core_gather_data;  // This maps all routes (src_core->dst_core) onto a sequence of operations on the
                               // input sticks that can be padding, local copy/transfer, or remote copy/transfer
    uint32_t core_id = 0;
    for (auto [output_boundary, input_boundary] : shard_boundaries) {
        auto [input_start, input_end] = input_boundary;
        for (uint32_t global_idx = input_start; global_idx <= input_end; ++global_idx) {
            uint32_t dst_core_id = core_id;
            uint32_t local_idx = global_idx - input_start;
            auto [is_pad_stick, src_core_id, src_local_idx] = tensor_metadata[global_idx];
            TT_ASSERT(local_idx < PAD_LOCAL_SENTINAL && src_local_idx < PAD_LOCAL_SENTINAL, "Index overflow");
            if (is_pad_stick) {
                TT_ASSERT(src_local_idx == 0);
                src_core_id = PAD_LOCAL_SENTINAL;
            }
            if (per_core_gather_data.contains({src_core_id, dst_core_id})) {
                auto& [src_start, dst_start, length] = per_core_gather_data[{src_core_id, dst_core_id}].back();
                // src idx is 0 if it is a pad
                if ((src_local_idx == (src_start + length) || is_pad_stick) && local_idx == (dst_start + length)) {
                    ++length;
                    continue;
                }
            }
            // insert new tuple
            per_core_gather_data[{src_core_id, dst_core_id}].push_back({src_local_idx, local_idx, 1});
        }
        ++core_id;
    }

    // construct the config tensors
    // pad_config: length num_cores_nhw - each element (for core i): [dst_start0, length0, dst_start1, length1, ...]
    std::vector<std::vector<uint32_pair_t>> pad_config;
    std::vector<std::pair<uint32_triplet_t, std::vector<uint32_triplet_t>>> local_config;
    std::vector<std::vector<std::pair<uint32_triplet_t, std::vector<uint32_triplet_t>>>> remote_config;
    pad_config.resize(num_cores_nhw);
    local_config.resize(num_cores_nhw);
    remote_config.resize(num_cores_nhw);

    // Split off padding, local transfer, remote transfer operations into their own configs
    for (const auto& [src_dst, data] : per_core_gather_data) {
        auto [src_core_id, dst_core_id] = src_dst;
        bool is_pad = src_core_id == PAD_LOCAL_SENTINAL;
        bool is_local = src_core_id == dst_core_id;
        bool is_remote = !is_local && !is_pad;
        if (is_pad) {
            for (auto [src_start, dst_start, length] : data) {
                pad_config[dst_core_id].push_back({dst_start, length});
            }
        } else if (is_local) {
            CoreCoord noc_xy = core_id_to_noc_coords(dst_core_id);
            local_config[src_core_id].first = {noc_xy.x, noc_xy.y, 3 * data.size()};
            local_config[src_core_id].second = data;
        } else if (is_remote) {
            if (remote_read) {
                CoreCoord noc_xy = core_id_to_noc_coords(src_core_id);
                remote_config[dst_core_id].push_back({{noc_xy.x, noc_xy.y, 3 * data.size()}, data});
            } else {
                CoreCoord noc_xy = core_id_to_noc_coords(dst_core_id);
                remote_config[src_core_id].push_back({{noc_xy.x, noc_xy.y, 3 * data.size()}, data});
            }
        }
    }

    std::vector<GatherConfig> gather_configs(num_cores_nhw);
    for (int core_id = 0; core_id < local_config.size(); core_id++) {
        const auto& config = local_config[core_id];
        const auto& [src_core_id, dst_core_id, num_copies] = config.first;
        std::vector<GatherTransfer> transfers;
        for (const auto& transfer : config.second) {
            const uint16_t src_offset_id = std::get<0>(transfer);
            const uint16_t dst_offset_id = std::get<1>(transfer);
            const uint16_t size = std::get<2>(transfer);
            transfers.emplace_back(src_offset_id, dst_offset_id, size);
        }
        GatherHeader header{src_core_id, dst_core_id, transfers.size()};
        gather_configs[core_id].routes.push_back(GatherRoute{header, transfers});
    }

    for (int core_id = 0; core_id < remote_config.size(); core_id++) {
        for (const auto& destination : remote_config[core_id]) {
            const auto& [src_core_id, dst_core_id, num_copies] = destination.first;
            std::vector<GatherTransfer> transfers;
            for (const auto& transfer : destination.second) {
                const uint16_t src_offset_id = std::get<0>(transfer);
                const uint16_t dst_offset_id = std::get<1>(transfer);
                const uint16_t size = std::get<2>(transfer);
                transfers.emplace_back(src_offset_id, dst_offset_id, size);
            }
            GatherHeader header{src_core_id, dst_core_id, transfers.size()};
            gather_configs[core_id].routes.push_back(GatherRoute{header, transfers});
        }
    }

    const bool use_blocking = is_in_tiled;
    std::vector<GatherConfig> ordered_gather_configs0;
    std::vector<GatherConfig> ordered_gather_configs1;
    std::vector<uint16_t> number_of_blocks_per_core;
    for (const auto& config : gather_configs) {
        if (use_blocking) {
            const auto quantized = quantize_transfers_along_block_boundaries(config, block_size);
            const auto ordered = reorder_transfers_globally(quantized);
            const auto [first, second, number_of_blocks] = divide_blocks_between_cores(ordered, block_size);
            ordered_gather_configs0.push_back(first);
            ordered_gather_configs1.push_back(second);
            number_of_blocks_per_core.push_back(number_of_blocks);
        } else {
            const auto [first, second] = divide_transfers_between_cores(config);
            ordered_gather_configs0.push_back(first);
            ordered_gather_configs1.push_back(second);
        }
    }

    const auto serialized_gather_configs0 = serialize_gather_configs(ordered_gather_configs0);
    const auto serialized_gather_configs1 = serialize_gather_configs(ordered_gather_configs1);

    // Flatten and uniformize the lengths of each config list
    auto flatten_pad_config = [](auto& config) -> std::vector<std::vector<uint16_t>> {
        // find max length
        size_t max_len = 0;
        for (auto& data : config) {
            max_len = std::max(max_len, 2 * data.size());  // each data is 2 * data.size()
        }
        std::vector<std::vector<uint16_t>> flattened_config;
        for (auto& data : config) {
            std::vector<uint16_t> flat_data(max_len, 0);
            uint32_t idx = 0;
            for (auto data_elem : data) {
                auto [dst_start, length] = data_elem;
                flat_data[idx++] = dst_start;
                flat_data[idx++] = length;
            }
            // null plug
            flat_data.emplace_back(0);
            flat_data.emplace_back(0);
            flattened_config.emplace_back(flat_data);
        }
        return flattened_config;
    };

    std::vector<std::vector<uint32_pair_t>> pad_config0(num_cores_nhw);
    std::vector<std::vector<uint32_pair_t>> pad_config1(num_cores_nhw);
    for (int core_idx = 0; core_idx < pad_config.size(); core_idx++) {
        const auto& config = pad_config[core_idx];
        auto middle = config.begin() + config.size() / 2;
        pad_config0[core_idx] = std::vector<uint32_pair_t>(config.begin(), middle);
        pad_config1[core_idx] = std::vector<uint32_pair_t>(middle, config.end());
    }

    auto flattened_pad_config0 = flatten_pad_config(pad_config0);
    auto flattened_pad_config1 = flatten_pad_config(pad_config1);

    auto align_config = [](auto& config, size_t align_granularity = 1, uint16_t align_value = 0) {
        size_t max_len = 0;
        for (auto& core_config : config) {
            max_len = std::max(max_len, core_config.size());
        }
        if (align_granularity > 1) {
            size_t align_amount = max_len % align_granularity;
            max_len = align_amount > 0 ? max_len + align_granularity - align_amount : max_len;
        }
        for (auto& core_config : config) {
            size_t extend_amount = max_len - core_config.size();
            if (extend_amount > 0) {
                std::vector<uint16_t> extend_v(extend_amount, align_value);
                core_config.insert(core_config.end(), extend_v.begin(), extend_v.end());
            }
        }
    };

    align_config(flattened_pad_config0, 2);
    align_config(flattened_pad_config1, 2);

    return HaloGatherKernelConfig{
        flattened_pad_config0,
        flattened_pad_config1,
        serialized_gather_configs0,
        serialized_gather_configs1,
        number_of_blocks_per_core};
}

void visualize_sliding_window_op_config(const std::vector<std::vector<uint16_t>>& config) {
    log_info(tt::LogOp, "========================================");
    log_info(tt::LogOp, "  Sliding Window Op Config Visualization");
    log_info(tt::LogOp, "========================================");
    log_info(tt::LogOp, "Total Cores: {}", config.size());

    // Calculate statistics
    uint32_t total_elements = 0;
    uint32_t max_config_size = 0;
    uint32_t min_config_size = config.empty() ? 0 : config[0].size();

    for (const auto& core_config : config) {
        total_elements += core_config.size();
        max_config_size = std::max(max_config_size, static_cast<uint32_t>(core_config.size()));
        min_config_size = std::min(min_config_size, static_cast<uint32_t>(core_config.size()));
    }

    log_info(tt::LogOp, "Total Elements: {}", total_elements);
    log_info(tt::LogOp, "Max Config Size per Core: {}", max_config_size);
    log_info(tt::LogOp, "Min Config Size per Core: {}", min_config_size);
    log_info(tt::LogOp, "========================================");

    // Visualize each core's configuration
    for (uint32_t core_id = 0; core_id < config.size(); ++core_id) {
        const auto& core_config = config[core_id];
        log_info(tt::LogOp, "");
        log_info(tt::LogOp, "Core #{} (size: {} elements)", core_id, core_config.size());

        if (core_config.empty()) {
            log_info(tt::LogOp, "  [EMPTY]");
            continue;
        }

        // Parse the structure: [num_segments, 0, indices...]
        uint32_t idx = 0;
        uint32_t block_num = 0;

        while (idx < core_config.size()) {
            // Check if we have at least 2 elements for header
            if (idx + 1 >= core_config.size()) {
                break;
            }

            uint32_t num_segments = core_config[idx];
            // core_config[idx + 1] is alignment/padding, we skip it

            // Skip if this looks like padding (all zeros)
            if (num_segments == 0 && (idx + 2 >= core_config.size() || core_config[idx + 2] == 0)) {
                break;
            }

            log_info(tt::LogOp, "  Block {}: {} segments", block_num, num_segments);
            idx += 2;  // Skip num_segments and alignment

            // Display segments (pairs of start-end indices)
            uint32_t segment_count = 0;
            while (segment_count < num_segments && idx + 1 < core_config.size()) {
                uint16_t start_idx = core_config[idx];
                uint16_t end_idx = core_config[idx + 1];
                uint16_t range_size = (end_idx >= start_idx) ? (end_idx - start_idx + 1) : 0;

                log_info(
                    tt::LogOp,
                    "    Segment {}: [{:5d} -> {:5d}] (len: {})",
                    segment_count,
                    start_idx,
                    end_idx,
                    range_size);

                idx += 2;
                segment_count++;
            }

            // If we didn't find all segments, something might be wrong
            if (segment_count < num_segments) {
                log_info(
                    tt::LogOp, "    WARNING: Expected {} segments but only parsed {}", num_segments, segment_count);
                break;
            }

            block_num++;
        }

        // Show remaining padding if any
        if (idx < core_config.size()) {
            uint32_t padding_count = core_config.size() - idx;
            log_info(tt::LogOp, "  [Padding: {} elements]", padding_count);
        }
    }

    log_info(tt::LogOp, "========================================");
}

std::vector<std::vector<uint16_t>> generate_sliding_window_op_config(
    const std::vector<uint32_t>& op_trace_metadata,
    const std::vector<ShardBoundary>& shard_boundaries,
    uint32_t stride_w,
    bool is_conv,
    uint32_t reader0_datums,
    uint32_t reader1_datums,
    bool pad_cores) {
    std::vector<std::vector<uint16_t>> sharded_input_top_left_indices;
    for (const auto& item : shard_boundaries) {
        const auto& [output_shard_start, output_shard_end] = item.output_range;
        const auto& [input_shard_start, input_shard_end] = item.input_range;
        std::vector<uint16_t> local_top_left_indices;
        // sanity check
        if (output_shard_start >= op_trace_metadata.size()) {
            // this core has no output
            sharded_input_top_left_indices.push_back(local_top_left_indices);
            continue;
        }
        TT_ASSERT(input_shard_start == op_trace_metadata[output_shard_start]);
        uint32_t datums_remaining = 0;
        uint32_t last_relative_index = 0;
        uint32_t num_segments = 0;
        bool start_new_block = true;
        uint32_t num_segments_ind = 0;
        bool use_reader0 = true;
        bool split_reader = reader1_datums > 0;

        for (size_t i = output_shard_start; i < output_shard_end + 1; i++) {
            const uint32_t relative_index = op_trace_metadata[i] - op_trace_metadata[output_shard_start];
            if (start_new_block) {
                if (!split_reader) {
                    datums_remaining = reader0_datums;
                } else {
                    datums_remaining = use_reader0 ? reader0_datums : reader1_datums;
                }
                local_top_left_indices.push_back(0);  // number of segments, going to be replaced later
                num_segments_ind = local_top_left_indices.size() - 1;
                local_top_left_indices.push_back(0);  // empty value, used for 32B alignment
                local_top_left_indices.push_back(relative_index);
                start_new_block = false;
            } else {
                if (relative_index != last_relative_index + stride_w) {
                    local_top_left_indices.push_back(last_relative_index);
                    num_segments++;
                    local_top_left_indices.push_back(relative_index);
                }
            }
            datums_remaining--;
            last_relative_index = relative_index;

            if (!datums_remaining || i == output_shard_end) {  // end of block
                local_top_left_indices.push_back(last_relative_index);
                num_segments++;
                local_top_left_indices[num_segments_ind] = num_segments;
                num_segments = 0;
                if (is_conv) {
                    start_new_block = true;
                }
                use_reader0 = !use_reader0;
            }
        }
        sharded_input_top_left_indices.push_back(local_top_left_indices);
    }

    uint32_t indices_length_per_core = sharded_input_top_left_indices[0].size();
    for (uint32_t core_idx = 1; core_idx < shard_boundaries.size(); core_idx++) {
        indices_length_per_core =
            std::max<std::size_t>(sharded_input_top_left_indices[core_idx].size(), indices_length_per_core);
    }
    if (pad_cores) {
        for (uint32_t core_idx = 0; core_idx < shard_boundaries.size(); core_idx++) {
            // Pad indices for this core if not equal to other cores
            if (sharded_input_top_left_indices.size() == core_idx) {
                sharded_input_top_left_indices.push_back(std::vector<uint16_t>());
            }
            TT_FATAL(
                core_idx < sharded_input_top_left_indices.size(),
                "Invalid core_idx {} for sharded_input_top_left_indices",
                core_idx);
            uint32_t indices_length_this_core = sharded_input_top_left_indices[core_idx].size();
            if (indices_length_per_core - indices_length_this_core > 0) {
                std::vector<uint16_t> extend_v(indices_length_per_core - indices_length_this_core, 0);
                sharded_input_top_left_indices[core_idx].insert(
                    sharded_input_top_left_indices[core_idx].end(), extend_v.begin(), extend_v.end());
            }
        }
    }
    return sharded_input_top_left_indices;
}

std::vector<uint16_t> flatten(const std::vector<std::vector<uint16_t>>& input, uint32_t extend_with_zeroes) {
    std::vector<uint16_t> flattened_vector;
    for (auto sub_vec : input) {
        flattened_vector.insert(flattened_vector.end(), sub_vec.begin(), sub_vec.end());
        if (extend_with_zeroes > 0) {
            std::vector<uint16_t> extend_v(extend_with_zeroes, 0);
            flattened_vector.insert(flattened_vector.end(), extend_v.begin(), extend_v.end());
        }
    }
    log_debug(tt::LogOp, "flattened_vector size: {}", flattened_vector.size());
    return flattened_vector;
}

uint32_t get_repeat_factor_for_replicating_nhw_config_across_grid(const ParallelConfig& p_config) {
    switch (p_config.shard_scheme) {
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED: return 1;
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: return p_config.grid.num_cores();
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: {
            TT_ASSERT(p_config.grid.ranges().size() == 1, "BLOCK_SHARDED should have just a single core range");
            uint32_t ncores_y = p_config.grid.ranges().begin()->end_coord.y + 1;
            uint32_t ncores_x = p_config.grid.ranges().begin()->end_coord.x + 1;
            if (p_config.shard_orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR) {
                return ncores_x;
            }
            if (p_config.shard_orientation == tt::tt_metal::ShardOrientation::COL_MAJOR) {
                return ncores_y;
            }
            TT_FATAL(false, "Unsupported shard orientation");
        }
        default: TT_FATAL(false, "Unsupported shard scheme");
    }
}

std::vector<uint16_t> replicate_config(const std::vector<uint16_t>& config_vector, int factor) {
    std::vector<uint16_t> repeat_config;
    for (uint32_t i = 0; i < factor; ++i) {
        repeat_config.insert(repeat_config.end(), config_vector.begin(), config_vector.end());
    }
    return repeat_config;
}

std::vector<uint16_t> remap_nhw_scalar_argument_across_full_grid(
    const std::vector<uint16_t>& config, const ParallelConfig& parallel_config) {
    const auto factor = sliding_window::get_repeat_factor_for_replicating_nhw_config_across_grid(parallel_config);
    if (parallel_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        const auto broadcast_config_per_row = [](const std::vector<uint16_t>& config, uint32_t num_cols) {
            const uint32_t num_rows = static_cast<uint32_t>(config.size());
            std::vector<uint16_t> expanded;
            expanded.reserve(num_rows * num_cols);
            for (uint16_t val : config) {
                for (uint32_t c = 0; c < num_cols; ++c) {
                    expanded.push_back(val);
                }
            }
            return expanded;
        };
        return broadcast_config_per_row(config, factor);
    }
    return sliding_window::replicate_config(config, factor);
}

Tensor construct_on_host_config_tensor(
    const std::vector<std::vector<uint16_t>>& config, const ParallelConfig& p_config, bool store_in_dram) {
    // We need the last dim of tensors to be multiple of 2, pad if needed
    uint32_t extend_with_zeroes = config[0].size() % 2;
    extend_with_zeroes = extend_with_zeroes > 0 ? 2 - extend_with_zeroes : 0;

    ttnn::Shape config_shape(
        {static_cast<uint32_t>(config.size()), static_cast<uint32_t>(config[0].size()) + extend_with_zeroes});
    std::vector<uint16_t> config_vector = flatten(config, extend_with_zeroes);

    const uint32_t factor = store_in_dram ? 1 : get_repeat_factor_for_replicating_nhw_config_across_grid(p_config);
    auto repeat_config = replicate_config(config_vector, factor);

    auto config_buffer = tt::tt_metal::HostBuffer(std::move(repeat_config));
    config_shape = ttnn::Shape({config_shape[0] * factor, config_shape[1]});
    return Tensor(std::move(config_buffer), config_shape, DataType::UINT16, Layout::ROW_MAJOR);
}

Tensor move_config_tensor_to_device(
    const Tensor& config_tensor,
    const ParallelConfig& p_config,
    bool is_block_sharded,
    distributed::MeshDevice* device,
    bool store_in_dram) {
    if (store_in_dram) {
        return config_tensor.to_device(device, MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    }
    auto shard_shape = std::array<uint32_t, 2>({1, (uint32_t)config_tensor.logical_shape()[-1]});
    log_debug(tt::LogOp, "shard_shape: ({}, {})", shard_shape[0], shard_shape[1]);
    auto config_shard_orientation =
        is_block_sharded ? (p_config.shard_orientation == ShardOrientation::COL_MAJOR ? ShardOrientation::ROW_MAJOR
                                                                                      : ShardOrientation::COL_MAJOR)
                         : ShardOrientation::ROW_MAJOR;
    ShardSpec shard_spec(p_config.grid, shard_shape, config_shard_orientation);
    MemoryConfig memory_config{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, shard_spec};
    return config_tensor.to_device(device, memory_config);
}

uint32_t align_buffer(uint32_t size) {
    uint32_t alignment_bytes = tt::tt_metal::hal::get_dram_alignment();
    uint32_t factor = (size + alignment_bytes - 1) / alignment_bytes;
    return factor * alignment_bytes;
};

std::string SlidingWindowConfig::to_string() const {
    return "batch=" + std::to_string(batch_size) + "_ch=" + std::to_string(channels) +
           "_in_h=" + std::to_string(std::get<0>(input_hw)) + "_in_w=" + std::to_string(std::get<1>(input_hw)) +
           "_win_h=" + std::to_string(std::get<0>(window_hw)) + "_win_w=" + std::to_string(std::get<1>(window_hw)) +
           "_stride_h=" + std::to_string(std::get<0>(stride_hw)) +
           "_stride_w=" + std::to_string(std::get<1>(stride_hw)) + "_pad_t=" + std::to_string(padding[0]) +
           "_pad_b=" + std::to_string(padding[1]) + "_pad_l=" + std::to_string(padding[2]) +
           "_pad_r=" + std::to_string(padding[3]) + "_out_pad_h=" + std::to_string(std::get<0>(output_pad_hw)) +
           "_out_pad_w=" + std::to_string(std::get<1>(output_pad_hw)) +
           "_dil_h=" + std::to_string(std::get<0>(dilation_hw)) + "_dil_w=" + std::to_string(std::get<1>(dilation_hw)) +
           "_scale_h=" + std::to_string(scale_h) + "_scale_w=" + std::to_string(scale_w) +
           "_cores_nhw=" + std::to_string(num_cores_nhw) + "_cores_c=" + std::to_string(num_cores_c) +
           "_grid=" + core_range_set.str() + (snap_to_tile ? "_snap_to_tile" : "") + (is_bilinear ? "_bilinear" : "") +
           (is_transpose ? "_transpose" : "") + (ceil_mode ? "_ceil_mode" : "");
}

}  // namespace ttnn::operations::sliding_window

auto fmt::formatter<ttnn::operations::sliding_window::ParallelConfig>::format(
    const ttnn::operations::sliding_window::ParallelConfig& t, format_context& ctx) const -> format_context::iterator {
    std::string shard_scheme_str;
    if (t.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        shard_scheme_str = "HEIGHT_SHARDED";
    } else if (t.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        shard_scheme_str = "BLOCK_SHARDED";
    } else if (t.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        shard_scheme_str = "WIDTH_SHARDED";
    } else {
        shard_scheme_str = "NOT_SHARDED";
    }
    std::string shard_orientation_str;
    if (t.shard_orientation == ShardOrientation::COL_MAJOR) {
        shard_orientation_str = "COL_MAJOR";
    } else if (t.shard_orientation == ShardOrientation::ROW_MAJOR) {
        shard_orientation_str = "ROW_MAJOR";
    } else {
        shard_orientation_str = "INVALID";
    }
    std::string str = fmt::format(
        "ParallelConfig(grid={}, shard_scheme={}, shard_orientation={})",
        t.grid.str(),
        shard_scheme_str,
        shard_orientation_str);
    return fmt::format_to(ctx.out(), "{}", str);
}

auto fmt::formatter<ttnn::operations::sliding_window::SlidingWindowConfig>::format(
    const ttnn::operations::sliding_window::SlidingWindowConfig& t, format_context& ctx) const
    -> format_context::iterator {
    std::string str = fmt::format(
        "SlidingWindowConfig(batch_size={}, input_hw=({},{}), window_hw=({},{}), stride_hw=({},{}), padding=(({}, {}), "
        "({}, {})), output_padding = ({}, {}), ceil_pad_hw=({},{}), "
        "dilation_hw=({},{}), scale_h={}, scale_w={}, num_cores_nhw={}, num_cores_c={}, core_range_set_={}, "
        "is_transpose={})",
        t.batch_size,
        t.input_hw.first,
        t.input_hw.second,
        t.window_hw.first,
        t.window_hw.second,
        t.stride_hw.first,
        t.stride_hw.second,
        t.padding[0],
        t.padding[1],
        t.padding[2],
        t.padding[3],
        t.output_pad_hw.first,
        t.output_pad_hw.second,
        t.get_ceil_pad_h(),
        t.get_ceil_pad_w(),
        t.dilation_hw.first,
        t.dilation_hw.second,
        t.scale_h,
        t.scale_w,
        t.num_cores_nhw,
        t.num_cores_c,
        t.core_range_set.str(),
        t.is_transpose);
    return fmt::format_to(ctx.out(), "{}", str);
}

auto fmt::formatter<ttnn::operations::sliding_window::ShardBoundary>::format(
    const ttnn::operations::sliding_window::ShardBoundary& t, format_context& ctx) const -> format_context::iterator {
    return fmt::format_to(
        ctx.out(),
        "[output_shard=({}, {}), input_shard=({}, {})]",
        t.output_range.start,
        t.output_range.end,
        t.input_range.start,
        t.input_range.end);
}
