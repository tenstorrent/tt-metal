// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include <cstdint>
#include <optional>

#include "conv2d_utils.hpp"
#include "common/constants.hpp"
#include "common/logger.hpp"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/common/core_coord.hpp"

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

uint32_t find_closest_common_largest_divisor(uint32_t num1, uint32_t num2, uint32_t start_divisor) {
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

ParallelConfig determine_parallel_config_non_tile_mul_width(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    const CoreCoord& compute_grid_size,
    ShardOrientation block_shard_orientation) {
    uint32_t effective_tile_height = 1;
    uint32_t effective_tile_width = 1;
    CoreRangeSet grid;
    uint32_t out_nhw_ntiles = tt::round_up(batch_size * output_height * output_width, tt::constants::TILE_HEIGHT);
    uint32_t start_divisor =
        block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.x : compute_grid_size.y;
    uint32_t num_cores_nhw = find_closest_largest_divisor_with_num_padding(out_nhw_ntiles, start_divisor);

    uint32_t start_divisor_c =
        block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.y : compute_grid_size.x;
    uint32_t num_cores_c = find_closest_common_largest_divisor(output_channels, input_channels, start_divisor_c);
    uint32_t cores_x = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_nhw : num_cores_c;
    uint32_t cores_y = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_c : num_cores_nhw;
    CoreRange core_range = CoreRange(CoreCoord({0, 0}), CoreCoord({cores_x - 1, cores_y - 1}));
    grid = CoreRangeSet({core_range});
    auto shard_orientation =
        shard_layout == TensorMemoryLayout::BLOCK_SHARDED ? block_shard_orientation : ShardOrientation::ROW_MAJOR;
    ParallelConfig pconfig = {.grid = grid, .shard_scheme = shard_layout, .shard_orientation = block_shard_orientation};

    return pconfig;
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
    bool is_out_tiled) {
    uint32_t effective_tile_height = is_out_tiled ? tt::constants::TILE_HEIGHT : 1;
    uint32_t effective_tile_width = is_out_tiled ? tt::constants::TILE_WIDTH : 1;
    uint32_t out_nhw_ntiles =
        tt::round_up(batch_size * output_height * output_width, tt::constants::TILE_HEIGHT) / effective_tile_height;
    uint32_t input_channles_ntiles = tt::div_up(input_channels, effective_tile_width);
    uint32_t out_channels_ntiles = tt::div_up(output_channels, effective_tile_width);

    // calculate num_core_nhw and the grid
    uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
    CoreRangeSet grid;
    if (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t num_cores_nhw = find_closest_largest_divisor_with_num_padding(out_nhw_ntiles, max_num_cores);
        grid = num_cores_to_corerangeset(num_cores_nhw, compute_grid_size, true);
    } else if (shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        uint32_t start_divisor =
            block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.x : compute_grid_size.y;
        uint32_t num_cores_nhw = find_closest_largest_divisor_with_num_padding(out_nhw_ntiles, start_divisor);
        uint32_t start_divisor_c =
            block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.y : compute_grid_size.x;
        uint32_t num_cores_c =
            enable_channels_padding
                ? find_closest_largest_divisor_with_num_padding(
                      out_channels_ntiles, input_channles_ntiles, start_divisor_c)
                : find_closest_largest_divisor(out_channels_ntiles, input_channles_ntiles, start_divisor_c);
        uint32_t cores_x = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_nhw : num_cores_c;
        uint32_t cores_y = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_c : num_cores_nhw;
        CoreRange core_range = CoreRange(CoreCoord({0, 0}), CoreCoord({cores_x - 1, cores_y - 1}));
        grid = CoreRangeSet({core_range});
    } else if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t num_cores_c = enable_channels_padding
                                   ? find_closest_largest_divisor_with_num_padding(input_channles_ntiles, max_num_cores)
                                   : find_closest_largest_divisor(input_channles_ntiles, max_num_cores);
        grid = num_cores_to_corerangeset(num_cores_c, compute_grid_size, true);
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
            .grid = num_cores_to_corerangeset(
                find_closest_largest_divisor_with_num_padding(
                    tt::div_up(out_channels, tt::constants::TILE_WIDTH), max_num_cores),
                compute_grid_size,
                true),
            .shard_scheme = ttnn::TensorMemoryLayout::WIDTH_SHARDED,
            .shard_orientation = input_parallel_config.shard_orientation};
    }
    return output_parallel_config;
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
    uint32_t channels = tensor_shape.with_tile_padding()[3];
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
    TT_ASSERT(channels % num_cores_channels == 0, "Channels: {}, num core channels: {}", channels, num_cores_channels);
    uint32_t channel_shard = channels / num_cores_channels;
    auto shard_spec = ShardSpec{parallel_config.grid, {nhw_shard, channel_shard}, shard_orientation};
    log_debug("Calculated Shard Spec = {}", shard_spec);
    return MemoryConfig{shard_scheme, BufferType::L1, shard_spec};
}

OptimizedConvParallelizationConfig determine_conv_op_parallel_config_from_conv_output_mem_config(
    const MemoryConfig& conv_output_mem_config, uint32_t num_cores_nhw, uint32_t num_cores_c) {
    TT_ASSERT(conv_output_mem_config.shard_spec.has_value());
    const auto& shard_spec = conv_output_mem_config.shard_spec.value();
    const auto& shard_shape = shard_spec.shape;
    uint32_t per_core_out_matrix_height_ntiles = div_up(shard_shape[0], 32);
    return {
        .grid_size = shard_spec.grid.bounding_box().grid_size(),
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c = num_cores_c,
        .per_core_out_matrix_height = shard_shape[0],
        .per_core_out_matrix_width = shard_shape[1],
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
    TT_ASSERT(subblock_h > 0 && subblock_w > 0);
    return {subblock_h, subblock_w};
}

OptimizedConvBlockConfig determine_per_core_conv_block_config(
    const ParallelConfig& parallel_config,
    const OptimizedConvParallelizationConfig& conv_op_parallel_config,
    uint32_t padded_in_channels,
    uint32_t padded_output_height_ntiles,
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

    uint32_t act_block_h_ntiles =
        div_up(conv_op_parallel_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT);

    if (act_block_h_override > 0) {
        uint32_t act_block_h_override_ntiles = act_block_h_override / constants::TILE_HEIGHT;
        if (padded_output_height_ntiles % act_block_h_override_ntiles == 0) {
            act_block_h_ntiles = act_block_h_override_ntiles;
        } else {
            log_info(
                LogOp,
                "act_block_h_override {} is not a valid override for padded_output_height_ntiles {}, override will "
                "be ignored",
                act_block_h_override_ntiles,
                padded_output_height_ntiles);
        }
    }

    auto grid_size = parallel_config.grid.bounding_box().grid_size();
    uint32_t act_c_num_blocks = parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED ? 1
                                : parallel_config.shard_orientation == ShardOrientation::COL_MAJOR ? grid_size.y
                                                                                                   : grid_size.x;
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
    uint32_t out_block_h_ntiles =
        div_up(conv_op_parallel_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT);
    uint32_t weight_block_w_ntiles =
        div_up(conv_op_parallel_config.per_core_out_matrix_width, tt::constants::TILE_WIDTH);
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
    const std::array<uint32_t, 2>& padding,
    const std::array<uint32_t, 2>& dilation,
    uint32_t groups,
    const Conv2dConfig& conv_config) {
    bool is_width_sharded =
        (conv_config.shard_layout.has_value() && conv_config.shard_layout.value() == TensorMemoryLayout::WIDTH_SHARDED);
    return kernel_size[0] == 1 && kernel_size[1] == 1 && stride[0] == stride[1] && stride[0] == 1 && padding[0] == 0 &&
           padding[1] == 0 && dilation[0] == 1 && dilation[1] == 1 && groups == 1 && (not is_width_sharded);
}

// Implements a heuristic for selecting shard layout based on how many tenix cores are available
// for each shard.
static TensorMemoryLayout select_shard_spec(
    bool is_mm_conv,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t weights_width,
    uint32_t input_width,
    ShardOrientation shard_orientation,
    const CoreCoord& compute_grid_size) {
    auto get_core_count_for_sharding = [&](TensorMemoryLayout shard_layout) {
        return determine_parallel_config(
                   shard_layout,
                   batch_size,
                   in_channels,
                   output_height,
                   output_width,
                   out_channels,
                   compute_grid_size,
                   shard_orientation,
                   !is_mm_conv)
            .grid.num_cores();
    };

    // 1d convs support only height sharding
    const bool is_conv1d = weights_width == 1 && input_width == 1;

    const uint32_t cc_height = get_core_count_for_sharding(TensorMemoryLayout::HEIGHT_SHARDED);
    // matmul doesn't support width sharding
    const uint32_t cc_width =
        !is_mm_conv && !is_conv1d ? get_core_count_for_sharding(TensorMemoryLayout::WIDTH_SHARDED) : 0;
    const uint32_t cc_block = !is_conv1d ? get_core_count_for_sharding(TensorMemoryLayout::BLOCK_SHARDED) : 0;

    uint32_t max_cc = cc_block;
    TensorMemoryLayout shard_layout = TensorMemoryLayout::BLOCK_SHARDED;

    // Prefer block sharding over height sharding but make sure that we got at least
    // some blocking on width dimension as well.
    // Also for larger number of cores pefer block sharding, as it will divide weights along
    // the cores.
    const uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
    const uint32_t tree_quarter_cores = static_cast<uint32_t>(0.75f * max_num_cores);
    if ((cc_height > max_cc && max_cc < tree_quarter_cores) ||
        (cc_height == max_cc && cc_height <= compute_grid_size.x)) {
        shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;
        max_cc = cc_height;
    }

    if (cc_width >= max_cc) {
        shard_layout = TensorMemoryLayout::WIDTH_SHARDED;
        max_cc = cc_width;
    }

    if (shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        // For large number of input channels prefer width sharding
        // even if it has less cores.
        // For BH we probably need to adjust this, or even better we make block sharding
        // more configurable regarding L1 memory usage.
        if (cc_width >= 40 && in_channels > 1280) {
            shard_layout = TensorMemoryLayout::WIDTH_SHARDED;
            log_debug(LogOp, "Switching to WIDTH_SHARDED layout due to large in_channels");
            max_cc = cc_width;
        }
    }
    log_debug(LogOp, "Selected shard layout: {}, num cores: {}", shard_layout, max_cc);

    return shard_layout;
}

template <typename T>
static std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool, bool> get_conv_padded_input_shape_and_mem_config(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv,
    bool is_non_tile_mul_width) {
    ttnn::Tensor input_tensor = input_tensor_;  // tensor to return
    bool input_tensor_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_);
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
            const auto input_shard_scheme = input_memory_config.memory_layout;
            const auto input_shard_orientation = input_memory_config.shard_spec.value().orientation;
            const auto input_shard_grid = input_memory_config.shard_spec.value().grid;
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

    // shallow conv variriant not supported
    // out_channels <= 256 incorrect output from pack_untilize_dst if output > 256 Tracking --> #14236
    // bf8 not supported due to limation of sharding dim multipl of 32
    bool use_non_tile_height = (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) && out_channels <= 256 &&
                               conv_config.act_block_h_override == 0 &&
                               (conv_config.dtype == DataType::BFLOAT16 || conv_config.dtype == DataType::FLOAT32) &&
                               conv_config.output_layout == Layout::ROW_MAJOR &&
                               conv_config.input_channels_alignment != 16;  // shalow conv varient

    ParallelConfig parallel_config = input_tensor_parallel_config;
    if (conv_config.reshard_if_not_optimal || needs_shard_or_reshard) {
        auto block_shard_orientation =
            conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;
        ParallelConfig optimal_parallel_config;
        if (is_non_tile_mul_width) {
            optimal_parallel_config = determine_parallel_config_non_tile_mul_width(
                shard_layout,
                batch_size,
                in_channels,
                height,
                width,
                out_channels,
                device->compute_with_storage_grid_size(),
                block_shard_orientation);
        } else {
            optimal_parallel_config = determine_parallel_config(
                shard_layout,
                batch_size,
                in_channels,
                height,
                width,
                out_channels,
                device->compute_with_storage_grid_size(),
                block_shard_orientation,
                !is_mm_conv,
                !use_non_tile_height);
        }

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

        // TT_ASSERT(input_tensor.get_legacy_shape() == input_tensor.get_shape());
        uint32_t tensor_height =
            input_tensor.get_shape()[0] * input_tensor.get_shape()[1] * input_tensor.get_shape()[2];
        uint32_t round_up_size = tt::constants::TILE_HEIGHT;
        if ((use_non_tile_height || shard_layout == TensorMemoryLayout::WIDTH_SHARDED) &&
            input_tensor_.layout() == Layout::ROW_MAJOR) {
            round_up_size = 1;
        }
        uint32_t input_tensor_height_snapped_to_tile = tt::round_up(tensor_height, input_num_cores_nhw * round_up_size);
        TT_ASSERT(input_tensor_height_snapped_to_tile >= tensor_height);
        uint32_t input_tensor_width_snapped_to_channels_alignment =
            tt::round_up(input_tensor.get_shape()[3], input_num_cores_c * conv_config.input_channels_alignment);
        if (is_non_tile_mul_width) {
            input_tensor_width_snapped_to_channels_alignment =
                tt::round_up(input_tensor.get_shape()[3], conv_config.input_channels_alignment);
        }

        auto input_padded_shape = ttnn::Shape(std::array<uint32_t, 4>{
            1,
            1,
            input_tensor_height_snapped_to_tile,
            input_tensor_width_snapped_to_channels_alignment});  // TODO: resolve ttnn::types::Shape and
                                                                 // tt::tt_metal::LegacyShape issue to clean up next
                                                                 // line
        MemoryConfig input_tensor_sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            ttnn::Shape(std::array<uint32_t, 4>{
                input_padded_shape[0], input_padded_shape[1], input_padded_shape[2], input_padded_shape[3]}),
            parallel_config,
            round_up_size);

        return {input_padded_shape, input_tensor_sharded_memory_config, needs_shard_or_reshard, use_non_tile_height};
    } else {
        return {input_tensor.shape(), input_tensor.memory_config(), needs_shard_or_reshard, use_non_tile_height};
    }
}

template <typename T>
std::tuple<ttnn::Tensor, ParallelConfig, ParallelConfig, bool> shard_or_reshard_tensor_if_required(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv,
    bool auto_shard,
    bool is_non_tile_mul_width) {
    ttnn::Tensor input_tensor = input_tensor_;  // tensor to return
    bool input_tensor_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_);
    auto compute_grid_size = device->compute_with_storage_grid_size();

    auto [input_padded_shape, input_tensor_sharded_memory_config, needs_shard_or_reshard, use_non_tile_height] =
        get_conv_padded_input_shape_and_mem_config(
            device,
            input_tensor_,
            conv_config,
            batch_size,
            height,
            width,
            in_channels,
            out_channels,
            is_mm_conv,
            is_non_tile_mul_width);
    ParallelConfig parallel_config = {
        .grid = input_tensor_sharded_memory_config.shard_spec.value().grid,
        .shard_scheme = input_tensor_sharded_memory_config.memory_layout,
        .shard_orientation = input_tensor_sharded_memory_config.shard_spec.value().orientation};

    ParallelConfig output_parallel_config =
        determine_output_parallel_config(parallel_config, compute_grid_size, out_channels, is_mm_conv);

    if (needs_shard_or_reshard) {
        if (input_tensor.get_shape()[0] != 1 or input_tensor.get_shape()[1] != 1) {
            // reshape to [1, 1, N*H*W, C]
            input_tensor = ttnn::reshape(
                input_tensor,
                ttnn::SimpleShape(std::array<uint32_t, 4>{
                    1,
                    1,
                    input_tensor.get_shape()[0] * input_tensor.get_shape()[1] * input_tensor.get_shape()[2],
                    input_tensor.get_shape()[3]}));
        }

        uint32_t tensor_height = input_tensor.get_shape()[2];
        uint32_t tensor_width = input_tensor.get_shape()[3];

        if (!input_tensor_on_device) {
            if (input_padded_shape[-2] != tensor_height || input_padded_shape[-1] != tensor_width) {
                input_tensor = ttnn::pad(
                    input_tensor,
                    tt::tt_metal::Array4D(
                        {input_tensor.get_shape()[0],
                         input_tensor.get_shape()[1],
                         input_padded_shape[-2],
                         input_padded_shape[-1]}),
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
                auto resharded_input_tensor =
                    ttnn::to_memory_config(input_tensor, input_tensor_sharded_memory_config, std::nullopt);
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
    return {input_tensor, parallel_config, output_parallel_config, use_non_tile_height};
}

void validate_weight_and_bias_tensors(
    const ttnn::Tensor& weight_tensor, std::optional<const ttnn::Tensor>& bias_tensor) {
    TT_ASSERT(!ttnn::has_storage_type_of(weight_tensor, ttnn::DEVICE_STORAGE_TYPE));
    TT_ASSERT(weight_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_ASSERT(weight_tensor.get_shape().rank() == 4);
    // TODO: enable this assert
    // TT_ASSERT(weight_tensor.get_shape() == weight_tensor.get_legacy_shape());
    if (bias_tensor.has_value()) {
        TT_ASSERT(!ttnn::has_storage_type_of(bias_tensor.value(), ttnn::DEVICE_STORAGE_TYPE));
        TT_ASSERT(bias_tensor.value().get_shape().rank() == 4);
        TT_ASSERT(bias_tensor.value().get_layout() == Layout::ROW_MAJOR);
        // TODO: enable this assert
        // TT_ASSERT(bias_tensor.value().get_shape() == bias_tensor.value().get_legacy_shape());
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
            .out_block_h = div_up(conv_parallelization_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT),
            .out_block_w = div_up(conv_parallelization_config.per_core_out_matrix_width, tt::constants::TILE_WIDTH),
            .per_core_M = div_up(conv_parallelization_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT),
            .per_core_N = div_up(conv_parallelization_config.per_core_out_matrix_width, tt::constants::TILE_WIDTH),
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
            .out_block_h = div_up(conv_parallelization_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT),
            .out_block_w = div_up(conv_parallelization_config.per_core_out_matrix_width, tt::constants::TILE_WIDTH),
            .per_core_M = div_up(conv_parallelization_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT),
            .per_core_N = div_up(conv_parallelization_config.per_core_out_matrix_width, tt::constants::TILE_WIDTH),
            .transpose_mcast = transpose_mcast};
        if (activation != "") {
            matmul_config.fused_activation = ttnn::operations::unary::utils::string_to_unary_with_param(activation);
        }
        return matmul_config;
    }
}

void adjust_conv_op_config_for_auto_shard_if_necessary(
    bool is_mm_conv,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t weights_width,
    uint32_t input_width,
    const CoreCoord& compute_grid_size,
    Conv2dConfig& conv_config,
    Layout input_tensor_layout,
    std::optional<const MemoryConfig> input_memory_config) {
    // If the input tensor is already sharded, or the conv_config has a specified shard layout, we don't need to do
    // anything.
    if ((input_memory_config.has_value() && input_memory_config.value().is_sharded()) ||
        conv_config.shard_layout.has_value()) {
        return;
    }

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;
    conv_config.shard_layout = select_shard_spec(
        is_mm_conv,
        batch_size,
        in_channels,
        out_channels,
        output_height,
        output_width,
        weights_width,
        input_width,
        shard_orientation,
        compute_grid_size);

    if (conv_config.act_block_h_override == 0) {
        if (in_channels <= constants::TILE_WIDTH / 2 && conv_config.input_channels_alignment == constants::TILE_WIDTH &&
            !is_mm_conv && conv_config.shard_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
            input_tensor_layout == Layout::ROW_MAJOR) {
            log_debug(LogOp, "Auto shard, enable shallow conv");
            // height sharded, non matmul conv, with input channels <= 16, and default setting for
            // input_channels_alignment
            // Currently data-movement ops have too many restrictions to support shallow convs with tiled input.
            conv_config.input_channels_alignment = constants::TILE_WIDTH / 2;
        }

        // Set act_block_h_override to min value to
        // be conservative with L1 memory usage.
        conv_config.act_block_h_override = constants::TILE_HEIGHT;
    }

    if (conv_config.act_block_w_div == 1 && conv_config.shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t width_sharded_num_cores = determine_parallel_config(
                                               TensorMemoryLayout::WIDTH_SHARDED,
                                               batch_size,
                                               in_channels,
                                               output_height,
                                               output_width,
                                               out_channels,
                                               compute_grid_size,
                                               shard_orientation,
                                               !is_mm_conv)
                                               .grid.num_cores();
        // Set act_block_w_div to max value to
        // be conservative with L1 memory usage.
        // act_block_w_div == 1 is currently the default value.
        conv_config.act_block_w_div = tt::div_up(in_channels, width_sharded_num_cores * constants::TILE_WIDTH);
    }
}

template std::tuple<ttnn::Tensor, ParallelConfig, ParallelConfig, bool> shard_or_reshard_tensor_if_required<Device>(
    Device* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv,
    bool auto_shard,
    bool is_non_tile_mul_width);

template std::tuple<ttnn::Tensor, ParallelConfig, ParallelConfig, bool> shard_or_reshard_tensor_if_required<MeshDevice>(
    MeshDevice* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channel,
    bool is_mm_conv,
    bool auto_shard,
    bool is_non_tile_mul_width);

std::ostream& operator<<(std::ostream& os, const Conv2dConfig& config) {
    tt::stl::reflection::operator<<(os, config);
    return os;
}

}  // namespace operations::conv
}  // namespace ttnn
