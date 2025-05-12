// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>

#include "conv2d_utils.hpp"
#include <tt-metalium/buffer_types.hpp>
#include "tt-metalium/constants.hpp"
#include <tt-metalium/hal.hpp>
#include "tt-metalium/logger.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"

using namespace tt;

namespace ttnn {
namespace operations::conv {
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

uint32_t find_closest_largest_divisor(uint32_t num, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    while (num % divisor != 0) {
        divisor = divisor - 1;
    }
    return divisor;
}

uint32_t find_closest_largest_divisor(uint32_t num1, uint32_t num2, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    while (num1 % divisor != 0 or num2 % divisor != 0) {
        divisor = divisor - 1;
    }
    return divisor;
}

uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    uint32_t padded_num = round_up(num, divisor);
    while ((padded_num - num) >= (int)(padded_num / divisor)) {
        divisor = divisor - 1;
        padded_num = round_up(num, divisor);
    }
    return divisor;
}

uint32_t find_closest_largest_divisor_with_num_padding_and_mult(uint32_t num, uint32_t start_divisor, uint32_t mult) {
    uint32_t divisor = start_divisor;
    uint32_t big_divisor = divisor * mult;
    uint32_t padded_num = round_up(num, big_divisor);
    while ((padded_num - num) >= (int)(padded_num / big_divisor)) {
        divisor = divisor - 1;
        big_divisor = divisor * mult;
        padded_num = round_up(num, big_divisor);
    }
    return divisor;
}

uint32_t get_input_channels_alignment(
    const TensorMemoryLayout input_tensor_memory_layout,
    Layout input_tensor_layout,
    bool is_mm_conv,
    const std::optional<MemoryConfig>& input_memory_config) {
    if (!is_mm_conv && input_tensor_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
        input_tensor_layout == Layout::ROW_MAJOR) {
        if (input_memory_config.has_value() && input_memory_config->is_sharded()) {
            const uint32_t shard_width = input_memory_config->shard_spec()->shape[1];
            if (shard_width % tt::constants::TILE_WIDTH == 0) {
                return tt::constants::TILE_WIDTH;
            } else if (shard_width % 16 == 0) {
                return 16U;
            } else if (shard_width % 8 == 0) {
                return 8U;
            } else {
                return tt::constants::TILE_WIDTH;
            }
        } else {
            // The minimum valid value for input channels alignment is 8.
            // This requirement comes from the L1 alignment, which is 16 bytes.
            // Since the Halo operation outputs data in row-major layout and the smallest data format used is bfloat16
            // (2 bytes per element), we need at least 8 elements (8 * 2 bytes = 16 bytes) in the input channel
            // dimension. This ensures that one channel (or "stick") can be efficiently transferred over the NoC
            // (Network on Chip) in a single, aligned operation.
            return 8U;
        }
    }
    return tt::constants::TILE_WIDTH;
}

uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num1, uint32_t num2, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    uint32_t padded_num1 = round_up(num1, divisor);
    uint32_t padded_num2 = round_up(num2, divisor);
    while ((padded_num1 - num1) >= (padded_num1 / divisor) || (padded_num2 - num2) >= (padded_num2 / divisor)) {
        divisor = divisor - 1;
        padded_num1 = round_up(num1, divisor);
        padded_num2 = round_up(num2, divisor);
    }
    return divisor;
}

// If shard width is tile width, and it is allowed to have half tile shard width, and we have enough cores to do it,
// double number of cores
static uint32_t set_shard_width_to_half_tile_if_possible(
    uint32_t num_cores, uint32_t channels_ntiles, uint32_t max_num_cores, bool width_shard_half_tile_possible) {
    if (width_shard_half_tile_possible && (div_up(channels_ntiles, num_cores) == 1) &&
        (2 * num_cores <= max_num_cores)) {
        return 2 * num_cores;
    }
    return num_cores;
}

ParallelConfig determine_parallel_config(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    const CoreCoord& compute_grid_size,
    ShardOrientation block_shard_orientation,
    bool enable_channels_padding,
    bool is_shard_height_tile_multiple,
    bool is_shard_width_tile_multiple,
    uint32_t act_block_h_override) {
    // Currently, convolution requires multiples of the tile size for both shard height and width,
    // while pooling can accept any height and either a tile multiple or half a tile for width.
    // This approach needs to be modified when other shard dimensions are supported.
    uint32_t effective_tile_height = is_shard_height_tile_multiple ? tt::constants::TILE_HEIGHT : 1;
    uint32_t effective_tile_width = tt::constants::TILE_WIDTH;
    uint32_t out_nhw_ntiles = tt::div_up(batch_size * output_height * output_width, effective_tile_height);
    uint32_t input_channles_ntiles = tt::div_up(input_channels, effective_tile_width);
    uint32_t out_channels_ntiles = tt::div_up(output_channels, effective_tile_width);
    // In case non native activation block height is used, we need to ensure that the amount
    // of work per core in the height dimension is a multiple of the activation block height override.
    uint32_t act_block_h_override_ntiles =
        act_block_h_override == 0 ? 1 : act_block_h_override / tt::constants::TILE_HEIGHT;

    // calculate num_core_nhw and the grid
    uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
    CoreRangeSet grid;
    if (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t num_cores_nhw = find_closest_largest_divisor_with_num_padding_and_mult(
            out_nhw_ntiles, max_num_cores, act_block_h_override_ntiles);
        grid = tt::tt_metal::num_cores_to_corerangeset(num_cores_nhw, compute_grid_size, true);
    } else if (shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        uint32_t start_divisor =
            block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.x : compute_grid_size.y;
        uint32_t num_cores_nhw = find_closest_largest_divisor_with_num_padding_and_mult(
            out_nhw_ntiles, start_divisor, act_block_h_override_ntiles);
        uint32_t start_divisor_c =
            block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.y : compute_grid_size.x;
        uint32_t num_cores_c =
            enable_channels_padding
                ? find_closest_largest_divisor_with_num_padding(
                      out_channels_ntiles, input_channles_ntiles, start_divisor_c)
                : find_closest_largest_divisor(out_channels_ntiles, input_channles_ntiles, start_divisor_c);
        num_cores_c = set_shard_width_to_half_tile_if_possible(
            num_cores_c, input_channles_ntiles, start_divisor_c, !is_shard_width_tile_multiple);
        uint32_t cores_x = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_nhw : num_cores_c;
        uint32_t cores_y = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_c : num_cores_nhw;
        CoreRange core_range = CoreRange(CoreCoord({0, 0}), CoreCoord({cores_x - 1, cores_y - 1}));
        grid = CoreRangeSet({core_range});
    } else if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t num_cores_c = enable_channels_padding
                                   ? find_closest_largest_divisor_with_num_padding(input_channles_ntiles, max_num_cores)
                                   : find_closest_largest_divisor(input_channles_ntiles, max_num_cores);
        num_cores_c = set_shard_width_to_half_tile_if_possible(
            num_cores_c, input_channles_ntiles, max_num_cores, !is_shard_width_tile_multiple);
        grid = tt::tt_metal::num_cores_to_corerangeset(num_cores_c, compute_grid_size, true);
    } else {
        TT_THROW("Conv2d supports Height, Block or Width Sharded Layouts but got {}", shard_layout);
    }

    auto shard_orientation = shard_layout == TensorMemoryLayout::BLOCK_SHARDED
                                 ? block_shard_orientation
                                 : ShardOrientation::ROW_MAJOR;  // NOTE: taking ROW_MAJOR as default orientation for
                                                                 // HEIGHT_SHARDED and WIDTH_SHARDED
    ParallelConfig pconfig = {.grid = grid, .shard_scheme = shard_layout, .shard_orientation = shard_orientation};

    return pconfig;
}

ParallelConfig determine_output_parallel_config(
    const ParallelConfig& input_parallel_config,
    const CoreCoord& compute_grid_size,
    uint32_t out_channels,
    bool is_mm_conv) {
    ParallelConfig output_parallel_config = input_parallel_config;
    if (input_parallel_config.shard_scheme == ttnn::TensorMemoryLayout::WIDTH_SHARDED && !is_mm_conv) {
        uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
        output_parallel_config = {
            .grid = tt::tt_metal::num_cores_to_corerangeset(
                find_closest_largest_divisor_with_num_padding(
                    tt::div_up(out_channels, tt::constants::TILE_WIDTH), max_num_cores),
                compute_grid_size,
                true),
            .shard_scheme = ttnn::TensorMemoryLayout::WIDTH_SHARDED,
            .shard_orientation = input_parallel_config.shard_orientation};
    }
    return output_parallel_config;
}

std::tuple<uint32_t, uint32_t> calculate_output_image_size(
    std::array<uint32_t, 2> input_image_size,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding,
    std::array<uint32_t, 2> dilation) {
    const uint32_t output_height = ((input_image_size[0] - kernel_size[0] - ((kernel_size[0] - 1) * (dilation[0] - 1)) +
                                     (padding[0] + padding[1])) /
                                    stride[0]) +
                                   1;
    const uint32_t output_width = ((input_image_size[1] - kernel_size[1] - ((kernel_size[1] - 1) * (dilation[1] - 1)) +
                                    (padding[2] + padding[3])) /
                                   stride[1]) +
                                  1;
    return {output_height, output_width};
}

uint32_t get_num_cores_nhw_from_parallel_config(const ParallelConfig& pconfig) {
    TT_ASSERT(!pconfig.grid.ranges().empty());
    TT_ASSERT(
        pconfig.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED ||
        pconfig.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED ||
        pconfig.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED);
    auto grid_size = pconfig.grid.bounding_box().grid_size();
    uint32_t num_cores = pconfig.grid.num_cores();
    uint32_t num_cores_nhw = 0;
    if (pconfig.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        return 1;
    }

    if (pconfig.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_cores_nhw = num_cores;
    } else if (pconfig.shard_orientation == ShardOrientation::COL_MAJOR) {
        num_cores_nhw = grid_size.x;
    } else {
        TT_ASSERT(pconfig.shard_orientation == ShardOrientation::ROW_MAJOR);
        num_cores_nhw = grid_size.y;
    }

    TT_ASSERT(num_cores_nhw > 0);
    return num_cores_nhw;
}

uint32_t get_num_cores_channels_from_parallel_config(const ParallelConfig& pconfig) {
    TT_ASSERT(!pconfig.grid.ranges().empty());
    TT_ASSERT(
        pconfig.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED ||
        pconfig.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED ||
        pconfig.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED);
    auto grid_size = pconfig.grid.bounding_box().grid_size();
    uint32_t num_cores_channels = 0;
    if (pconfig.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_cores_channels = 1;
    } else if (pconfig.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        num_cores_channels = pconfig.grid.num_cores();
    } else if (pconfig.shard_orientation == ShardOrientation::COL_MAJOR) {
        num_cores_channels = grid_size.y;
    } else {
        TT_ASSERT(pconfig.shard_orientation == ShardOrientation::ROW_MAJOR);
        num_cores_channels = grid_size.x;
    }
    TT_ASSERT(num_cores_channels > 0);
    return num_cores_channels;
}

MemoryConfig create_sharded_memory_config_from_parallel_config(
    const ttnn::Shape& tensor_shape, const ParallelConfig& parallel_config, uint32_t tile_size) {
    log_debug(
        tt::LogOp,
        "create_sharded_memory_config_from_parallel_config: tensor_shape: {}, parallel_config: {}, tile_size: {}",
        tensor_shape,
        parallel_config,
        tile_size);
    // tensor_shape is [N, H, W, C]
    TT_ASSERT(tensor_shape[0] == 1 && tensor_shape[1] == 1);  // todo: add support for generic non-2d shapes
    // uint32_t channels = tensor_shape[3];
    uint32_t channels = tensor_shape[3];
    uint32_t num_cores_nhw = get_num_cores_nhw_from_parallel_config(parallel_config);
    uint32_t num_cores_channels = get_num_cores_channels_from_parallel_config(parallel_config);
    auto shard_scheme = parallel_config.shard_scheme;
    auto shard_orientation = parallel_config.shard_orientation;

    uint32_t nhw_shape = tensor_shape[0] * tensor_shape[1] * tensor_shape[2];
    uint32_t nhw_padded = nhw_shape;
    if (shard_scheme != TensorMemoryLayout::WIDTH_SHARDED) {
        nhw_padded = round_up(nhw_shape, num_cores_nhw * tile_size);
    }
    uint32_t nhw_shard = nhw_padded / num_cores_nhw;
    TT_FATAL(channels % num_cores_channels == 0, "Channels: {}, num core channels: {}", channels, num_cores_channels);
    uint32_t channel_shard = channels / num_cores_channels;
    auto shard_spec = tt::tt_metal::ShardSpec{parallel_config.grid, {nhw_shard, channel_shard}, shard_orientation};
    log_debug("Calculated Shard Spec = {}", shard_spec);
    return MemoryConfig{shard_scheme, BufferType::L1, shard_spec};
}

OptimizedConvParallelizationConfig determine_conv_op_parallel_config_from_conv_output_mem_config(
    const MemoryConfig& conv_output_mem_config, uint32_t num_cores_nhw, uint32_t num_cores_c) {
    TT_ASSERT(conv_output_mem_config.shard_spec().has_value());
    const auto& shard_spec = conv_output_mem_config.shard_spec().value();
    const auto& shard_shape = shard_spec.shape;
    return {
        .grid_size = shard_spec.grid.bounding_box().grid_size(),
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c = num_cores_c,
        .per_core_out_matrix_height_ntile = div_up(shard_shape[0], tt::constants::TILE_HEIGHT),
        .per_core_out_matrix_width_ntile = div_up(shard_shape[1], tt::constants::TILE_WIDTH),
    };
}

static std::pair<uint32_t, uint32_t> determine_largest_subblock_size(
    uint32_t block_height, uint32_t block_width, bool fp32_accum, bool split_reader_enabled) {
    constexpr std::array<std::pair<uint32_t, uint32_t>, 20> subblocks = {{
        {2, 4}, {4, 2}, {1, 8}, {8, 1}, {1, 7}, {7, 1}, {2, 3}, {3, 2}, {1, 6}, {6, 1},
        {1, 5}, {5, 1}, {2, 2}, {1, 4}, {4, 1}, {1, 3}, {3, 1}, {1, 2}, {2, 1}, {1, 1},
    }};

    uint32_t subblock_h = 0;
    uint32_t subblock_w = 0;
    for (auto [subblock_height, subblock_width] : subblocks) {
        if (fp32_accum && (subblock_height * subblock_width > 4)) {
            continue;
        }

        if (split_reader_enabled && (block_height / subblock_height) < 2) {
            continue;
        }

        if ((block_height % subblock_height == 0) && (block_width % subblock_width == 0)) {
            if (subblock_width != block_width && subblock_height != 1) {
                continue;
            }
            subblock_h = subblock_height;
            subblock_w = subblock_width;
            break;
        }
    }
    TT_FATAL(
        subblock_h > 0 && subblock_w > 0,
        "Could not find valid subblock size for block size {}x{}, split_reader_enabled: {}, fp32_accum: {}",
        block_height,
        block_width,
        split_reader_enabled,
        fp32_accum);
    return {subblock_h, subblock_w};
}

OptimizedConvBlockConfig determine_per_core_conv_block_config(
    const ParallelConfig& parallel_config,
    const OptimizedConvParallelizationConfig& conv_op_parallel_config,
    uint32_t padded_in_channels,
    uint32_t padded_output_height_ntiles_per_core,
    uint32_t act_block_h_override,
    uint32_t act_block_w_div,
    uint32_t window_h,
    uint32_t window_w,
    bool fp32_accum,
    bool split_reader_enabled) {
    if (act_block_h_override > 0) {
        TT_ASSERT(
            act_block_h_override % 32 == 0,
            "Config Error: act_block_h_override must be a multiple of 32 (tile height).");
    }

    uint32_t act_block_h_ntiles = conv_op_parallel_config.per_core_out_matrix_height_ntile;

    if (act_block_h_override > 0) {
        uint32_t act_block_h_override_ntiles = act_block_h_override / constants::TILE_HEIGHT;
        if (padded_output_height_ntiles_per_core % act_block_h_override_ntiles == 0) {
            act_block_h_ntiles = act_block_h_override_ntiles;
        } else {
            uint32_t act_block_h_override_ntiles = act_block_h_override / constants::TILE_HEIGHT;
            if (padded_output_height_ntiles_per_core % act_block_h_override_ntiles == 0) {
                act_block_h_ntiles = act_block_h_override_ntiles;
            } else {
                act_block_h_ntiles =
                    find_closest_largest_divisor(padded_output_height_ntiles_per_core, act_block_h_override_ntiles);
                log_info(
                    LogOp,
                    "act_block_h_override {} is not a valid override for padded_output_height_ntiles_per_core {}, "
                    "instead {} was selected as closest valid option!",
                    act_block_h_override_ntiles,
                    padded_output_height_ntiles_per_core,
                    act_block_h_ntiles);
            }
        }
    }

    auto grid_size = parallel_config.grid.bounding_box().grid_size();
    uint32_t act_c_num_blocks = get_num_cores_channels_from_parallel_config(parallel_config);
    TT_ASSERT(padded_in_channels % act_c_num_blocks == 0);
    uint32_t act_block_w =
        parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED
            ? round_up(padded_in_channels * window_w, 32)
            : round_up((padded_in_channels / act_c_num_blocks) * window_h * window_w, tt::constants::TILE_WIDTH);
    if (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        TT_ASSERT(padded_in_channels % (32 * parallel_config.grid.num_cores() * act_block_w_div) == 0);
        act_block_w = (padded_in_channels * window_h * window_w) / (parallel_config.grid.num_cores() * act_block_w_div);
    }
    TT_ASSERT(act_block_w % 32 == 0);
    uint32_t act_block_w_ntiles = act_block_w / 32;
    uint32_t out_block_h_ntiles = conv_op_parallel_config.per_core_out_matrix_height_ntile;
    uint32_t weight_block_w_ntiles = conv_op_parallel_config.per_core_out_matrix_width_ntile;
    auto [out_subblock_h_ntiles, out_subblock_w_ntiles] =
        determine_largest_subblock_size(act_block_h_ntiles, weight_block_w_ntiles, fp32_accum, split_reader_enabled);
    return {
        .act_block_h_ntiles = act_block_h_ntiles,
        .act_block_w_ntiles = act_block_w_ntiles,
        .out_subblock_h_ntiles = out_subblock_h_ntiles,
        .out_subblock_w_ntiles = out_subblock_w_ntiles};
}

bool use_matmul_for_1x1_conv(
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 4>& padding,
    const std::array<uint32_t, 2>& dilation,
    uint32_t groups,
    const Conv2dConfig& conv_config) {
    bool is_width_sharded =
        (conv_config.shard_layout.has_value() && conv_config.shard_layout.value() == TensorMemoryLayout::WIDTH_SHARDED);
    return kernel_size[0] == 1 && kernel_size[1] == 1 && stride[0] == stride[1] && stride[0] == 1 && padding[0] == 0 &&
           padding[1] == 0 && padding[2] == 0 && padding[3] == 0 && dilation[0] == 1 && dilation[1] == 1 &&
           (not is_width_sharded);
}

bool is_1d_conv(uint32_t kernel_width, uint32_t image_width) { return kernel_width == 1 && image_width == 1; }

bool is_1d_deptwise_conv(
    uint32_t groups,
    uint32_t input_channels,
    uint32_t output_channels,
    uint32_t kernel_width,
    uint32_t image_width,
    bool has_bias) {
    bool is_depthwise_conv = groups == input_channels && groups == output_channels;
    return is_depthwise_conv && is_1d_conv(kernel_width, image_width) && !has_bias;
}

template <typename DeviceType>
DeviceComputeKernelConfig get_conv_default_compute_kernel_config(DeviceType* device) {
    return init_device_compute_kernel_config(device->arch(), std::nullopt, MathFidelity::HiFi4, true, false, false);
}

template <typename T>
static std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> get_conv_padded_input_shape_and_mem_config(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv) {
    ttnn::Tensor input_tensor = input_tensor_;  // tensor to return
    bool input_tensor_on_device = tt::tt_metal::is_device_tensor(input_tensor_);
    bool needs_shard_or_reshard = false;
    if (conv_config.override_sharding_config && conv_config.reshard_if_not_optimal) {
        TT_ASSERT(
            false,
            "Incorrect config provided: reshard_if_not_optimal and override_sharding_config cannot both be set.");
    }

    TT_FATAL(
        (!input_tensor_on_device || input_tensor_.is_sharded()) || conv_config.shard_layout.has_value(),
        "Tensor must be sharded or shard_layout must be set.");

    TensorMemoryLayout shard_layout;
    if (conv_config.shard_layout.has_value()) {
        shard_layout = conv_config.shard_layout.value();
    }

    ParallelConfig input_tensor_parallel_config;
    if (!input_tensor_on_device) {
        needs_shard_or_reshard = true;
    } else {
        const auto& input_memory_config = input_tensor_.memory_config();
        if (!input_memory_config.is_sharded()) {
            needs_shard_or_reshard = true;
        } else {
            const auto input_shard_scheme = input_memory_config.memory_layout();
            const auto input_shard_orientation = input_memory_config.shard_spec().value().orientation;
            const auto input_shard_grid = input_memory_config.shard_spec().value().grid;
            ParallelConfig pconfig = {
                .grid = input_shard_grid,
                .shard_scheme = input_shard_scheme,
                .shard_orientation = input_shard_orientation};
            input_tensor_parallel_config = pconfig;
            if (input_shard_scheme != TensorMemoryLayout::BLOCK_SHARDED &&
                input_shard_orientation != ShardOrientation::ROW_MAJOR) {
                needs_shard_or_reshard = true;
            }
            if (input_shard_scheme != TensorMemoryLayout::HEIGHT_SHARDED &&
                input_shard_scheme != TensorMemoryLayout::BLOCK_SHARDED &&
                input_shard_scheme != TensorMemoryLayout::WIDTH_SHARDED) {
                needs_shard_or_reshard = true;
            }
            if (conv_config.override_sharding_config) {
                TT_FATAL(
                    conv_config.core_grid.has_value(),
                    "If override_sharding_config is set, core_grid must be set as well.");
                TT_FATAL(
                    conv_config.shard_layout.has_value(),
                    "If override_sharding_config is set, shard_layout must be set as well.");
                if (conv_config.core_grid.value() != input_shard_grid) {
                    needs_shard_or_reshard = true;
                }
                if (shard_layout != input_shard_scheme) {
                    needs_shard_or_reshard = true;
                }
                bool input_transpose_shards = input_shard_orientation == ShardOrientation::COL_MAJOR;
                if (shard_layout == TensorMemoryLayout::BLOCK_SHARDED &&
                    conv_config.transpose_shards != input_transpose_shards) {
                    needs_shard_or_reshard = true;
                }
            }
        }
    }

    ParallelConfig parallel_config = input_tensor_parallel_config;
    if (conv_config.reshard_if_not_optimal || needs_shard_or_reshard) {
        auto block_shard_orientation =
            conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;
        ParallelConfig optimal_parallel_config = determine_parallel_config(
            shard_layout,
            batch_size,
            in_channels,
            height,
            width,
            out_channels,
            device->compute_with_storage_grid_size(),
            block_shard_orientation,
            !is_mm_conv,
            true,
            true,
            conv_config.act_block_h_override);

        if (conv_config.override_sharding_config) {
            TT_FATAL(conv_config.core_grid.has_value(), "Core grid must be provided when overriding sharding config");
            // override parallel config
            auto shard_orientation = shard_layout == TensorMemoryLayout::BLOCK_SHARDED ? block_shard_orientation
                                                                                       : ShardOrientation::ROW_MAJOR;
            parallel_config = {
                .grid = conv_config.core_grid.value(),
                .shard_scheme = shard_layout,
                .shard_orientation = shard_orientation};
        } else {
            parallel_config = optimal_parallel_config;
        }
        if (input_tensor_parallel_config != parallel_config) {
            needs_shard_or_reshard = true;
        }
    }
    if (needs_shard_or_reshard) {
        uint32_t input_num_cores_nhw = get_num_cores_nhw_from_parallel_config(parallel_config);
        uint32_t input_num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);

        // TT_ASSERT(input_tensor.get_padded_shape() == input_tensor.get_shape());
        const auto& input_shape = input_tensor.get_logical_shape();
        uint32_t tensor_height = input_shape[0] * input_shape[1] * input_shape[2];
        uint32_t round_up_size = tt::constants::TILE_HEIGHT;
        if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED && input_tensor_.layout() == Layout::ROW_MAJOR) {
            round_up_size = 1;
        }
        uint32_t input_tensor_height_snapped_to_tile = tt::round_up(tensor_height, input_num_cores_nhw * round_up_size);
        const uint32_t input_channels_aligment =
            get_input_channels_alignment(shard_layout, input_tensor_.layout(), is_mm_conv, std::nullopt);
        TT_ASSERT(input_tensor_height_snapped_to_tile >= tensor_height);
        uint32_t input_tensor_width_snapped_to_channels_alignment =
            tt::round_up(input_shape[3], input_num_cores_c * input_channels_aligment);

        auto input_padded_shape = ttnn::Shape(
            {1,
             1,
             input_tensor_height_snapped_to_tile,
             input_tensor_width_snapped_to_channels_alignment});  // TODO: resolve ttnn::types::Shape and
                                                                  // tt::tt_metal::LegacyShape issue to clean up next
                                                                  // line
        MemoryConfig input_tensor_sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            ttnn::Shape({input_padded_shape[0], input_padded_shape[1], input_padded_shape[2], input_padded_shape[3]}),
            parallel_config,
            round_up_size);

        return {input_padded_shape, input_tensor_sharded_memory_config, needs_shard_or_reshard};
    } else {
        return {input_tensor.get_logical_shape(), input_tensor.memory_config(), needs_shard_or_reshard};
    }
}

ttnn::Shape flatten_4d_shape(const ttnn::Shape& input_shape) {
    TT_FATAL(input_shape.size() == 4, "Expected 4D shape");
    const uint32_t nhw = input_shape[0] * input_shape[1] * input_shape[2];
    const uint32_t channels = input_shape[3];
    return ttnn::Shape{1, 1, nhw, channels};
}

template <typename T>
std::tuple<ttnn::Tensor, ParallelConfig, ParallelConfig> shard_or_reshard_tensor_if_required(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv,
    bool auto_shard) {
    ttnn::Tensor input_tensor = input_tensor_;  // tensor to return
    bool input_tensor_on_device = tt::tt_metal::is_device_tensor(input_tensor_);
    auto compute_grid_size = device->compute_with_storage_grid_size();

    auto [input_padded_shape, input_tensor_sharded_memory_config, needs_shard_or_reshard] =
        get_conv_padded_input_shape_and_mem_config(
            device, input_tensor_, conv_config, batch_size, height, width, in_channels, out_channels, is_mm_conv);
    ParallelConfig parallel_config = {
        .grid = input_tensor_sharded_memory_config.shard_spec().value().grid,
        .shard_scheme = input_tensor_sharded_memory_config.memory_layout(),
        .shard_orientation = input_tensor_sharded_memory_config.shard_spec().value().orientation};

    ParallelConfig output_parallel_config =
        determine_output_parallel_config(parallel_config, compute_grid_size, out_channels, is_mm_conv);

    // We can have flat and unflattened (n, h, w, c) tensors here
    const auto flattened_input_shape = flatten_4d_shape(input_tensor.get_logical_shape());
    const auto flattened_padded_input_shape = flatten_4d_shape(input_tensor.get_padded_shape());

    input_tensor = ttnn::reshape(input_tensor, flattened_input_shape, flattened_padded_input_shape);
    const ttnn::Shape input_shape = flattened_input_shape;

    if (needs_shard_or_reshard) {
        uint32_t tensor_height = input_shape[2];
        uint32_t tensor_width = input_shape[3];
        if (!input_tensor_on_device) {
            if (input_padded_shape[-2] != tensor_height || input_padded_shape[-1] != tensor_width) {
                input_tensor = ttnn::pad(
                    input_tensor,
                    tt::tt_metal::Array4D(
                        {input_shape[0], input_shape[1], input_padded_shape[-2], input_padded_shape[-1]}),
                    tt::tt_metal::Array4D({0, 0, 0, 0}),
                    0);
            }
        }

        // In case we are in auto sharded codepath and convolution maps to matmul
        // Skip sharding of the input tensor and run the matmul out of interleaved tensor.
        bool auto_shard_mm = auto_shard && is_mm_conv;
        if (input_tensor_on_device) {
            if (is_mm_conv && input_tensor.layout() == Layout::ROW_MAJOR &&
                parallel_config.shard_scheme != TensorMemoryLayout::HEIGHT_SHARDED) {
                // Workaround #13979 ttnn::tilize doesn't support BLOCK_SHARDED layout
                input_tensor =
                    ttnn::to_layout(input_tensor, Layout::TILE, std::nullopt, std::nullopt, input_tensor.device());
            }
            if (!auto_shard_mm) {
                ttnn::MemoryConfig input_tensor_sharded_memory_config_to_layout = input_tensor_sharded_memory_config;
                if (!input_tensor.is_sharded()) {
                    // In case we need to run Interleaved2Sharded switch fron physical sharding
                    // to logical sharding, in order to get smaller allocation size of sharded buffer.
                    input_tensor_sharded_memory_config_to_layout =
                        input_tensor_sharded_memory_config_to_layout.with_shard_spec(tt::tt_metal::ShardSpec(
                            input_tensor_sharded_memory_config.shard_spec().value().grid,
                            input_tensor_sharded_memory_config.shard_spec().value().shape,
                            input_tensor_sharded_memory_config.shard_spec().value().shape,
                            input_tensor_sharded_memory_config.shard_spec().value().orientation));
                }
                Tensor resharded_input_tensor =
                    ttnn::to_memory_config(input_tensor, input_tensor_sharded_memory_config_to_layout, std::nullopt);
                if (conv_config.deallocate_activation) {
                    input_tensor.deallocate(/*force*/ true);
                    resharded_input_tensor = ttnn::move(resharded_input_tensor);
                }
                input_tensor = resharded_input_tensor;
            }
        } else {
            input_tensor = ttnn::to_device(
                input_tensor, device, (auto_shard_mm ? ttnn::DRAM_MEMORY_CONFIG : input_tensor_sharded_memory_config));
        }
    }
    return {input_tensor, parallel_config, output_parallel_config};
}

void validate_weight_and_bias_tensors(
    const ttnn::Tensor& weight_tensor, std::optional<const ttnn::Tensor>& bias_tensor) {
    TT_ASSERT(!ttnn::has_storage_type_of(weight_tensor, ttnn::DEVICE_STORAGE_TYPE));
    TT_ASSERT(weight_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_ASSERT(weight_tensor.get_logical_shape().rank() == 4);
    // TODO: enable this assert
    // TT_ASSERT(weight_tensor.get_shape() == weight_tensor.get_padded_shape());
    if (bias_tensor.has_value()) {
        TT_ASSERT(!ttnn::has_storage_type_of(bias_tensor.value(), ttnn::DEVICE_STORAGE_TYPE));
        TT_ASSERT(bias_tensor.value().get_logical_shape().rank() == 4);
        TT_ASSERT(bias_tensor.value().get_layout() == Layout::ROW_MAJOR);
        // TODO: enable this assert
        // TT_ASSERT(bias_tensor.value().get_shape() == bias_tensor.value().get_padded_shape());
    }
}

ttnn::operations::matmul::MatmulProgramConfig determine_matmul_op_config_from_conv_op_config(
    OptimizedConvParallelizationConfig conv_parallelization_config,
    OptimizedConvBlockConfig conv_blocking_config,
    bool height_sharded,
    const string& activation,
    bool transpose_mcast,
    uint32_t grid_size_along_c) {
    if (height_sharded) {
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig matmul_config = {
            .compute_with_storage_grid_size = conv_parallelization_config.grid_size,
            .in0_block_w = conv_blocking_config.act_block_w_ntiles,
            .out_subblock_h = conv_blocking_config.out_subblock_h_ntiles,
            .out_subblock_w = conv_blocking_config.out_subblock_w_ntiles,
            .out_block_h = conv_parallelization_config.per_core_out_matrix_height_ntile,
            .out_block_w = conv_parallelization_config.per_core_out_matrix_width_ntile,
            .per_core_M = conv_parallelization_config.per_core_out_matrix_height_ntile,
            .per_core_N = conv_parallelization_config.per_core_out_matrix_width_ntile,
            .fuse_batch = true,
            .mcast_in0 = false};
        if (activation != "") {
            matmul_config.fused_activation = ttnn::operations::unary::utils::string_to_unary_with_param(activation);
        }
        return matmul_config;
    } else {
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig matmul_config = {
            .compute_with_storage_grid_size = conv_parallelization_config.grid_size,
            .in0_block_w = conv_blocking_config.act_block_w_ntiles,
            .out_subblock_h = conv_blocking_config.out_subblock_h_ntiles,
            .out_subblock_w = conv_blocking_config.out_subblock_w_ntiles,
            .out_block_h = conv_parallelization_config.per_core_out_matrix_height_ntile,
            .out_block_w = conv_parallelization_config.per_core_out_matrix_width_ntile,
            .per_core_M = conv_parallelization_config.per_core_out_matrix_height_ntile,
            .per_core_N = conv_parallelization_config.per_core_out_matrix_width_ntile,
            .transpose_mcast = transpose_mcast};
        if (activation != "") {
            matmul_config.fused_activation = ttnn::operations::unary::utils::string_to_unary_with_param(activation);
        }
        return matmul_config;
    }
}

Conv2dConfig determine_conv_config_for_auto_shard(
    const Conv2dConfig& conv_config,
    bool is_mm_conv,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t weights_width,
    uint32_t input_height,
    uint32_t input_width,
    const CoreCoord& compute_grid_size,
    Layout input_tensor_layout,
    std::optional<const MemoryConfig> input_memory_config,
    const std::array<uint32_t, 2>& kernel_size,
    const uint32_t groups,
    const bool enable_bias,
    const DeviceComputeKernelConfig& compute_config) {
    // If the input tensor is already sharded, or the conv_config has a specified shard layout, we don't need to do
    // anything.
    if ((input_memory_config.has_value() && input_memory_config.value().is_sharded()) ||
        conv_config.shard_layout.has_value()) {
        return conv_config;
    }

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    struct core_count_and_size {
        uint32_t core_count;
        uint32_t size;
        Conv2dConfig conv_config;
    };

    const bool conv_is_1d = is_1d_conv(kernel_size[1], input_width);
    const bool conv_is_1d_deptwise =
        is_1d_deptwise_conv(groups, in_channels, out_channels, kernel_size[1], input_width, enable_bias);

    auto get_l1_usage_for_sharding = [&](TensorMemoryLayout shard_layout,
                                         const Conv2dConfig& conv_config_in) -> core_count_and_size {
        Conv2dConfig conv_config = conv_config_in;
        conv_config.shard_layout = shard_layout;
        if (conv_config.act_block_h_override == 0) {
            // Set act_block_h_override to min value to
            // be conservative with L1 memory usage.
            conv_config.act_block_h_override = constants::TILE_HEIGHT;
            if (conv_config.enable_split_reader) {
                // Split reader needs at least 2 tiles in height to work.
                conv_config.act_block_h_override *= 2;
            }
        }

        const uint32_t input_channels_alignment =
            get_input_channels_alignment(shard_layout, input_tensor_layout, is_mm_conv, std::nullopt);
        const uint32_t in_channels_aligned = round_up(in_channels, input_channels_alignment);
        const uint32_t output_channels_padded = round_up(out_channels, constants::TILE_WIDTH);
        // Note: These are not exact shapes for weights as prepare_conv_weights will pad the weights depending on the
        // conv2d params, but these are good enough for L1 usage estimation.
        const ttnn::Shape weights_shape(
            {1, 1, in_channels_aligned * kernel_size[0] * kernel_size[1], output_channels_padded});

        const ParallelConfig input_parallel_config = determine_parallel_config(
            conv_config.shard_layout.value(),
            batch_size,
            in_channels,
            output_height,
            output_width,
            out_channels,
            compute_grid_size,
            shard_orientation,
            !is_mm_conv,
            true,
            true,
            conv_config.act_block_h_override);

        const ParallelConfig output_parallel_config = determine_output_parallel_config(
            input_parallel_config,
            compute_grid_size,
            out_channels,
            is_mm_conv && conv_config.shard_layout != TensorMemoryLayout::WIDTH_SHARDED);

        const uint32_t in_channels_padded = tt::round_up(
            in_channels, get_num_cores_channels_from_parallel_config(input_parallel_config) * input_channels_alignment);
        auto [opt_conv_op_parallel_config, opt_conv_op_block_config, conv_out_memory_config] = get_conv_configs(
            conv_config,
            compute_config,
            input_parallel_config,
            output_parallel_config,
            in_channels_padded,
            out_channels,
            batch_size,
            output_height,
            output_width,
            kernel_size,
            compute_grid_size);

        if (conv_config.act_block_w_div == 1 && conv_config.shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            uint32_t width_sharded_num_cores = input_parallel_config.grid.num_cores();
            // Set act_block_w_div to max value to
            // be conservative with L1 memory usage.
            // act_block_w_div == 1 is currently the default value.
            conv_config.act_block_w_div = tt::div_up(in_channels, width_sharded_num_cores * constants::TILE_WIDTH);
        }

        conv_op_l1_usage l1_usage = calculate_L1_usage(
            compute_config,
            opt_conv_op_block_config,
            opt_conv_op_parallel_config,
            weights_shape,
            kernel_size,
            conv_config,
            conv_out_memory_config,
            enable_bias,
            conv_is_1d_deptwise);

        // Since we don't have L1 usage for halo output (input to conv2d)
        // use approx input tensor size per core as a proxy.
        uint32_t input_nhw = tt::div_up(batch_size * input_height * input_width, tt::constants::TILE_HEIGHT);
        uint32_t input_c = tt::div_up(in_channels_aligned, tt::constants::TILE_WIDTH);
        uint32_t approx_input_size =
            input_nhw * input_c * tt::tile_size(datatype_to_dataformat_converter(conv_config.dtype));
        uint32_t approx_input_size_per_core = approx_input_size / input_parallel_config.grid.num_cores();

        l1_usage.tensor_allocation_size += approx_input_size_per_core;
        log_debug(
            tt::LogOp,
            "L1 usage for {}: {}, {}",
            conv_config.shard_layout,
            l1_usage.tensor_allocation_size,
            l1_usage.CB_allocation_size);
        return core_count_and_size{
            .core_count = input_parallel_config.grid.num_cores(),
            .size = l1_usage.CB_allocation_size + l1_usage.tensor_allocation_size,
            .conv_config = conv_config};
    };

    core_count_and_size height = get_l1_usage_for_sharding(TensorMemoryLayout::HEIGHT_SHARDED, conv_config);

    // 1d deptwise convs support only height sharding
    if (conv_is_1d_deptwise) {
        return height.conv_config;
    }

    const core_count_and_size block = get_l1_usage_for_sharding(TensorMemoryLayout::BLOCK_SHARDED, conv_config);
    const core_count_and_size width = get_l1_usage_for_sharding(TensorMemoryLayout::WIDTH_SHARDED, conv_config);

    core_count_and_size& winning_config = height;
    // Make sure that BS not only has smaller size but provides at least some slicing along the channels.
    // In case we have BS that would slice the tensor only along the HS conv2d code would fail later on.
    if (block.size < winning_config.size && block.core_count > compute_grid_size.x) {
        winning_config = block;
    }
    if (width.size < winning_config.size && !is_mm_conv) {
        winning_config = width;
    }

    log_debug(LogOp, "Core counts H: {} B: {}, W: {}", height.core_count, block.core_count, width.core_count);
    log_debug(
        LogOp, "Selected shard layout: {}, size: {}", winning_config.conv_config.shard_layout, winning_config.size);

    return winning_config.conv_config;
}

std::tuple<OptimizedConvParallelizationConfig, OptimizedConvBlockConfig, MemoryConfig> get_conv_configs(
    const Conv2dConfig& conv_config,
    const DeviceComputeKernelConfig& compute_config,
    const ParallelConfig& input_parallel_config,
    const ParallelConfig& output_parallel_config,
    uint32_t in_channels_padded,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t output_height,
    uint32_t output_width,
    std::array<uint32_t, 2> kernel_size,
    const CoreCoord& compute_grid) {
    uint32_t round_up_size = tt::constants::TILE_HEIGHT;
    uint32_t nhw_out = batch_size * output_height * output_width;
    uint32_t out_channels_padded = tt::round_up(
        out_channels, get_num_cores_channels_from_parallel_config(output_parallel_config) * tt::constants::TILE_WIDTH);
    MemoryConfig conv_out_memory_config = create_sharded_memory_config_from_parallel_config(
        ttnn::Shape({1, 1, nhw_out, out_channels_padded}), output_parallel_config, round_up_size);
    ParallelConfig largest_parallel_config =
        output_parallel_config.grid.num_cores() > input_parallel_config.grid.num_cores() ? output_parallel_config
                                                                                         : input_parallel_config;

    OptimizedConvParallelizationConfig opt_conv_op_parallel_config =
        determine_conv_op_parallel_config_from_conv_output_mem_config(
            conv_out_memory_config,
            get_num_cores_nhw_from_parallel_config(largest_parallel_config),
            get_num_cores_channels_from_parallel_config(largest_parallel_config));

    uint32_t nhw_out_padded_ntile_per_core =
        conv_out_memory_config.shard_spec().value().shape[0] / tt::constants::TILE_HEIGHT;

    OptimizedConvBlockConfig opt_conv_op_block_config = determine_per_core_conv_block_config(
        input_parallel_config,
        opt_conv_op_parallel_config,
        in_channels_padded,
        nhw_out_padded_ntile_per_core,
        conv_config.act_block_h_override,
        conv_config.act_block_w_div,
        kernel_size[0],
        kernel_size[1],
        get_fp32_dest_acc_en(compute_config),
        conv_config.enable_split_reader && input_parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED);
    return {opt_conv_op_parallel_config, opt_conv_op_block_config, conv_out_memory_config};
}

conv_op_l1_usage conv2d::calculate_L1_usage(
    const DeviceComputeKernelConfig& compute_kernel_config,
    const OptimizedConvBlockConfig& block_config,
    const OptimizedConvParallelizationConfig& pconfig,
    const ttnn::Shape& weights_shape,
    std::array<uint32_t, 2> kernel_size,
    const Conv2dConfig& conv_config,
    const MemoryConfig& output_memory_config,
    const bool enable_bias,
    bool is_1d_depthwise_conv) {
    bool untilize_out = conv_config.output_layout == Layout::ROW_MAJOR;

    // Output of halo op is always ROW_MAJOR, so input for convs is eighter DataType::FLOAT32 or DataType::BFLOAT16
    const DataType input_dtype = conv_config.dtype == DataType::FLOAT32 ? DataType::FLOAT32 : DataType::BFLOAT16;
    uint32_t input_tile_size = tt::tile_size(datatype_to_dataformat_converter(input_dtype));
    uint32_t weights_tile_size = tt::tile_size(datatype_to_dataformat_converter(conv_config.weights_dtype));
    uint32_t bias_tile_size = 0;
    if (enable_bias) {
        bias_tile_size = tt::tile_size(datatype_to_dataformat_converter(conv_config.weights_dtype));
    }
    uint32_t output_tile_size = tt::tile_size(datatype_to_dataformat_converter(conv_config.dtype));

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), compute_kernel_config);

    uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    uint32_t act_block_num_tiles = block_config.act_block_h_ntiles * act_block_w_ntiles;

    uint32_t weight_matrix_height = weights_shape[2];
    uint32_t weight_matrix_width = weights_shape[3];
    uint32_t weight_matrix_height_ntiles = weight_matrix_height / tt::constants::TILE_HEIGHT;
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / tt::constants::TILE_WIDTH;

    uint32_t per_core_out_matrix_width_ntiles = pconfig.per_core_out_matrix_width_ntile;
    uint32_t per_core_out_matrix_height_ntiles = pconfig.per_core_out_matrix_height_ntile;

    uint32_t num_blocks_act_h_per_core =
        (per_core_out_matrix_height_ntiles + act_block_h_ntiles - 1) / act_block_h_ntiles;
    uint32_t out_block_h_ntiles_padded = num_blocks_act_h_per_core * act_block_h_ntiles;

    TensorMemoryLayout sharding_scheme = conv_config.shard_layout.value();
    if (sharding_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t conv_output_c_per_core = per_core_out_matrix_width_ntiles * tt::constants::TILE_WIDTH;

        uint32_t output_size_per_core_in_bytes = per_core_out_matrix_width_ntiles * per_core_out_matrix_height_ntiles *
                                                 tt::tile_size(datatype_to_dataformat_converter(conv_config.dtype));

        uint32_t act_block_num_bytes = act_block_num_tiles * input_tile_size;
        uint32_t tilized_act_block_num_bytes = act_block_num_tiles * output_tile_size;

        uint32_t weight_block_w_ntiles = per_core_out_matrix_width_ntiles;
        uint32_t weight_block_num_tiles =
            weight_block_w_ntiles * act_block_w_ntiles;  // act_block_w_ntiles == weight_block_h_ntiles
        uint32_t weight_block_num_bytes = weight_block_num_tiles * weights_tile_size;

        uint32_t bias_block_num_bytes = per_core_out_matrix_width_ntiles * bias_tile_size;

        uint32_t out_block_num_tiles = per_core_out_matrix_height_ntiles * per_core_out_matrix_width_ntiles;

        uint32_t num_blocks_act_w = weight_matrix_height_ntiles / act_block_w_ntiles;

        packer_l1_acc = packer_l1_acc && ((enable_bias && num_blocks_act_w > 1) || (num_blocks_act_w > 2));

        auto interm_dtype =
            packer_l1_acc ? (fp32_dest_acc_en ? DataType::FLOAT32 : DataType::BFLOAT16) : conv_config.dtype;

        uint32_t partial_tile_size = tt::tile_size(datatype_to_dataformat_converter(interm_dtype));

        uint32_t partials_block_num_bytes = out_block_num_tiles * partial_tile_size;

        // ACT CB
        uint32_t act_cb_size = tilized_act_block_num_bytes;
        tt::log_debug(tt::LogOp, "Act CB Size: {}", act_cb_size);

        // WEIGHTS CB
        uint32_t weights_cb_size = weight_block_num_bytes;
        tt::log_debug(tt::LogOp, "Weights CB Size: {}", weights_cb_size);

        // BIAS CB
        uint32_t bias_cb_size = bias_block_num_bytes;
        tt::log_debug(tt::LogOp, "Bias CB Size: {}", bias_cb_size);

        // L1 CB
        uint32_t l1_scratchpad_cb_size = conv2d::l1_scratchpad_CB_size;
        tt::log_debug(tt::LogOp, "L1 CB Size: {}", l1_scratchpad_cb_size);

        // ACT ROW MAJOR CB
        uint32_t row_major_act_cb_size = act_block_num_bytes;
        tt::log_debug(tt::LogOp, "Act row major CB Size: {}", row_major_act_cb_size);

        // MATMUL PARTIALs CB
        uint32_t matmul_partials_cb_size = partials_block_num_bytes;
        if (!untilize_out && interm_dtype == conv_config.dtype) {
            matmul_partials_cb_size = 0;
        } else {
            tt::log_debug(tt::LogOp, "Matmul partial CB Size: {}", matmul_partials_cb_size);
        }

        // TILIZED ACT CB
        uint32_t tilized_act_cb_size = tilized_act_block_num_bytes;
        tt::log_debug(tt::LogOp, "Tilized act CB Size: {}", tilized_act_cb_size);

        uint32_t total_CB_size = act_cb_size + weights_cb_size + bias_cb_size + l1_scratchpad_cb_size +
                                 row_major_act_cb_size + matmul_partials_cb_size + tilized_act_cb_size;

        tt::log_debug(tt::LogOp, "Total CB Size: {}", total_CB_size);

        return conv2d::conv_op_l1_usage{
            .tensor_allocation_size = output_size_per_core_in_bytes, .CB_allocation_size = total_CB_size};
    } else if (sharding_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t output_size = per_core_out_matrix_height_ntiles * per_core_out_matrix_width_ntiles * output_tile_size;

        uint32_t bias_block_num_bytes = per_core_out_matrix_width_ntiles * bias_tile_size;

        uint32_t conv_act_c_blocks = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;

        uint32_t weight_block_w_ntiles = per_core_out_matrix_width_ntiles;
        uint32_t weight_block_h_ntiles = (is_1d_depthwise_conv) ? act_block_h_ntiles : act_block_w_ntiles;

        uint32_t act_block_cb_ntiles = act_block_h_ntiles * act_block_w_ntiles;

        uint32_t act_block_cb_size = act_block_cb_ntiles * input_tile_size;
        uint32_t tilzed_act_cb_size = act_block_cb_ntiles * output_tile_size;

        uint32_t output_block_ntiles = out_block_h_ntiles_padded * per_core_out_matrix_width_ntiles;

        uint32_t num_blocks_act_w = weight_matrix_height_ntiles / act_block_w_ntiles;
        uint32_t num_blocks_act_h = per_core_out_matrix_height_ntiles / act_block_h_ntiles;
        uint32_t in0_num_blocks_w =
            num_blocks_act_w * conv_act_c_blocks;  // Fold outer c_block loop together with weight_block_num_tiles = 9

        packer_l1_acc = packer_l1_acc && ((enable_bias && in0_num_blocks_w > 1) || (in0_num_blocks_w > 2));

        auto interm_dtype =
            packer_l1_acc ? (fp32_dest_acc_en ? DataType::FLOAT32 : DataType::BFLOAT16) : conv_config.dtype;

        uint32_t partial_tile_size = tt::tile_size(datatype_to_dataformat_converter(interm_dtype));

        uint32_t act_block_split_last_ntiles = 0;
        uint32_t act_block_split_ntiles = act_block_cb_ntiles;
        if (conv_config.enable_split_reader) {
            uint32_t act_block_h_nsubblocks = block_config.act_block_h_ntiles / block_config.out_subblock_h_ntiles;
            uint32_t act_block_split_last_ntiles = act_block_cb_ntiles / 2;
            uint32_t act_block_split_ntiles = act_block_cb_ntiles - act_block_split_last_ntiles;
        }

        if (conv_config.enable_act_double_buffer) {
            act_block_split_last_ntiles *= 2;
            act_block_split_ntiles *= 2;
        }
        // ACT CB
        uint32_t act_cb_size = act_block_split_ntiles * input_tile_size;
        tt::log_debug(tt::LogOp, "Act CB Size: {}", act_cb_size);

        // WEIGHTS CB
        uint32_t weights_cb_size = weight_block_h_ntiles * weight_block_w_ntiles * weights_tile_size;
        if (num_blocks_act_h > 1) {
            weights_cb_size *= kernel_size[0];
        }
        if (num_blocks_act_h <= 1 && conv_config.enable_weights_double_buffer) {
            weights_cb_size *= 2;
        }
        tt::log_debug(tt::LogOp, "Weights CB Size: {}", weights_cb_size);

        // BIAS CB
        uint32_t bias_cb_size = bias_block_num_bytes;
        tt::log_debug(tt::LogOp, "Bias CB Size: {}", bias_cb_size);

        // L1 CB
        uint32_t l1_scratchpad_cb_size = conv2d::l1_scratchpad_CB_size;
        tt::log_debug(tt::LogOp, "L1 CB Size: {}", l1_scratchpad_cb_size);

        // SPLIT READER CB
        uint32_t split_second_act_reader_cb_size = act_block_split_last_ntiles * input_tile_size;
        tt::log_debug(tt::LogOp, "Split reader CB Size: {}", split_second_act_reader_cb_size);

        // MATMUL PARTIALS CB
        uint32_t matmul_partials_cb_size = output_block_ntiles * partial_tile_size;
        if (!untilize_out && interm_dtype == conv_config.dtype) {
            matmul_partials_cb_size = 0;
        }
        if (is_1d_depthwise_conv) {
            matmul_partials_cb_size = output_tile_size;
        }
        if (matmul_partials_cb_size != 0) {
            tt::log_debug(tt::LogOp, "Matmul partials CB Size: {}", matmul_partials_cb_size);
        }
        // TILIZED ACT CB
        uint32_t tilized_act_cb_size = tilzed_act_cb_size;
        tt::log_debug(tt::LogOp, "Tilized act CB Size: {}", tilized_act_cb_size);

        // TEMP SUM CB
        uint32_t temp_sum_cb_size = 0;
        if (is_1d_depthwise_conv) {
            temp_sum_cb_size = output_tile_size;
            tt::log_debug(tt::LogOp, "Temp sum CB Size: {}", temp_sum_cb_size);
        }
        uint32_t total_CB_size = act_cb_size + weights_cb_size + bias_cb_size + l1_scratchpad_cb_size +
                                 split_second_act_reader_cb_size + matmul_partials_cb_size + tilized_act_cb_size +
                                 temp_sum_cb_size;
        return conv2d::conv_op_l1_usage{.tensor_allocation_size = output_size, .CB_allocation_size = total_CB_size};
    } else if (sharding_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        auto output_shard_shape = output_memory_config.shard_spec().value().shape;

        uint32_t output_size = 0;
        if (untilize_out) {
            uint32_t per_core_out_width_aligned = pconfig.per_core_out_matrix_width_ntile * tt::constants::TILE_WIDTH;
            if (conv_config.dtype == DataType::BFLOAT16) {
                per_core_out_width_aligned *= 2;
            } else if (conv_config.dtype == DataType::FLOAT32) {
                per_core_out_width_aligned *= 4;
            }
            output_size = tt::round_up(per_core_out_width_aligned, tt::tt_metal::hal::get_l1_alignment()) *
                          pconfig.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT;
        } else {
            output_size = per_core_out_matrix_height_ntiles * per_core_out_matrix_width_ntiles * output_tile_size;
        }

        uint32_t bias_block_num_bytes = per_core_out_matrix_width_ntiles * bias_tile_size;

        uint32_t conv_act_c_blocks = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;

        uint32_t weight_block_w_ntiles = per_core_out_matrix_width_ntiles;
        uint32_t weight_block_h_ntiles = act_block_w_ntiles;

        uint32_t tilized_act_block_cb_size = act_block_h_ntiles * act_block_w_ntiles * output_tile_size;
        uint32_t row_major_act_cb_size = act_block_h_ntiles * act_block_w_ntiles * input_tile_size;

        uint32_t output_block_ntiles = per_core_out_matrix_height_ntiles * per_core_out_matrix_width_ntiles;

        uint32_t num_blocks_act_w = 1;
        uint32_t num_blocks_act_h = per_core_out_matrix_height_ntiles / act_block_h_ntiles;
        uint32_t in0_num_blocks_w =
            num_blocks_act_w * conv_act_c_blocks;  // Fold outer c_block loop together with weight_block_num_tiles = 9

        packer_l1_acc = packer_l1_acc && ((enable_bias && in0_num_blocks_w > 1) || (in0_num_blocks_w > 2));

        auto interm_dtype =
            packer_l1_acc ? (fp32_dest_acc_en ? DataType::FLOAT32 : DataType::BFLOAT16) : conv_config.dtype;

        uint32_t partial_tile_size = tt::tile_size(datatype_to_dataformat_converter(interm_dtype));

        // ACT CB
        uint32_t act_cb_size = tilized_act_block_cb_size;
        if (conv_config.enable_act_double_buffer) {
            act_cb_size *= 2;
        }
        tt::log_debug(tt::LogOp, "Act CB Size: {}", act_cb_size);

        // WEIGHTS CB
        uint32_t weights_cb_size = weight_block_h_ntiles * weight_block_w_ntiles * weights_tile_size;
        if (conv_config.enable_weights_double_buffer) {
            weights_cb_size *= 2;
        }
        tt::log_debug(tt::LogOp, "Weights CB Size: {}", weights_cb_size);

        // BIAS CB
        uint32_t bias_cb_size = bias_block_num_bytes;
        tt::log_debug(tt::LogOp, "Bias CB Size: {}", bias_cb_size);

        // L1 CB
        uint32_t l1_scratchpad_cb_size = conv2d::l1_scratchpad_CB_size;
        tt::log_debug(tt::LogOp, "L1 CB Size: {}", l1_scratchpad_cb_size);

        // ACT ROW MAJOR CB
        tt::log_debug(tt::LogOp, "Act row major CB Size: {}", row_major_act_cb_size);

        // MATMUL PARTIALS CB
        uint32_t matmul_partials_cb_size = output_block_ntiles * partial_tile_size;
        if (!untilize_out && interm_dtype == conv_config.dtype) {
            matmul_partials_cb_size = 0;
        } else {
            tt::log_debug(tt::LogOp, "Matmul partials CB Size: {}", matmul_partials_cb_size);
        }

        // TILIZED ACT CB
        uint32_t tilized_act_cb_size = tilized_act_block_cb_size;
        tt::log_debug(tt::LogOp, "Tilized act CB Size: {}", tilized_act_cb_size);

        bool need_unpad_after_untilize =
            output_shard_shape[1] * output_shard_shape[0] <
            per_core_out_matrix_height_ntiles * per_core_out_matrix_width_ntiles * tt::constants::TILE_HW;

        tt::log_debug(tt::LogOp, "Need Unpad after untilize: {}", need_unpad_after_untilize);

        // UNTILIZED UNPADDED OUT CB
        uint32_t untilized_unpadded_out_cb_size = 0;
        if (need_unpad_after_untilize && untilize_out) {
            untilized_unpadded_out_cb_size = output_block_ntiles * output_tile_size;
            tt::log_debug(tt::LogOp, "Untilized unapadded out CB Size: {}", untilized_unpadded_out_cb_size);
        }
        uint32_t total_CB_size = act_cb_size + weights_cb_size + bias_cb_size + l1_scratchpad_cb_size +
                                 row_major_act_cb_size + matmul_partials_cb_size + tilized_act_cb_size +
                                 untilized_unpadded_out_cb_size;
        return conv2d::conv_op_l1_usage{.tensor_allocation_size = output_size, .CB_allocation_size = total_CB_size};
    }
    TT_THROW("Invalid shard layout {}", sharding_scheme);
}

bool conv2d::determine_packer_l1_acc(bool packer_l1_acc, bool enable_bias, uint32_t in0_num_blocks_w) {
    return packer_l1_acc && ((enable_bias && in0_num_blocks_w > 1) || (in0_num_blocks_w > 2));
}

template std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> get_conv_padded_input_shape_and_mem_config<IDevice>(
    IDevice* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv);

template std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> get_conv_padded_input_shape_and_mem_config<MeshDevice>(
    MeshDevice* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv);

template std::tuple<ttnn::Tensor, ParallelConfig, ParallelConfig> shard_or_reshard_tensor_if_required<IDevice>(
    IDevice* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv,
    bool auto_shard);

template std::tuple<ttnn::Tensor, ParallelConfig, ParallelConfig> shard_or_reshard_tensor_if_required<MeshDevice>(
    MeshDevice* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channel,
    bool is_mm_conv,
    bool auto_shard);

template DeviceComputeKernelConfig get_conv_default_compute_kernel_config<tt::tt_metal::IDevice>(
    tt::tt_metal::IDevice* device);

template DeviceComputeKernelConfig get_conv_default_compute_kernel_config<ttnn::MeshDevice>(ttnn::MeshDevice* device);

std::ostream& operator<<(std::ostream& os, const Conv2dConfig& config) {
    tt::stl::reflection::operator<<(os, config);
    return os;
}

}  // namespace operations::conv
}  // namespace ttnn
