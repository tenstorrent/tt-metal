// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <tuple>

#include "conv2d_utils.hpp"
#include <tt-metalium/buffer_types.hpp>
#include "conv2d/conv2d_op_program_factory_common.hpp"
#include "tt-metalium/constants.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/fold/fold.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::conv {
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;
using ttnn::prim::conv_op_l1_usage;

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
    uint32_t padded_num = tt::round_up(num, divisor);
    while ((padded_num - num) >= (int)(padded_num / divisor)) {
        divisor = divisor - 1;
        padded_num = tt::round_up(num, divisor);
    }
    return divisor;
}

uint32_t find_closest_largest_divisor_with_num_padding_and_mult(uint32_t num, uint32_t start_divisor, uint32_t mult) {
    uint32_t divisor = start_divisor;
    uint32_t big_divisor = divisor * mult;
    uint32_t padded_num = tt::round_up(num, big_divisor);
    while ((padded_num - num) >= (int)(padded_num / divisor) && divisor > 1) {
        divisor = divisor - 1;
        big_divisor = divisor * mult;
        padded_num = tt::round_up(num, big_divisor);
    }
    return divisor;
}

uint32_t get_input_channels_alignment(
    const TensorMemoryLayout input_tensor_memory_layout,
    Layout input_tensor_layout,
    bool sliced_op,
    bool is_mm_conv,
    const std::optional<MemoryConfig>& input_memory_config) {
    if (!is_mm_conv && input_tensor_memory_layout != TensorMemoryLayout::WIDTH_SHARDED &&
        (input_tensor_layout == Layout::ROW_MAJOR || sliced_op)) {
        if (input_memory_config.has_value() && input_memory_config->is_sharded()) {
            const uint32_t shard_width = input_memory_config->shard_spec()->shape[1];
            // NOLINTNEXTLINE(bugprone-branch-clone)
            if (shard_width % tt::constants::TILE_WIDTH == 0) {
                return tt::constants::TILE_WIDTH;
            }
            if (shard_width % 16 == 0) {
                return 16U;
            }
            if (shard_width % 8 == 0) {
                return 8U;
            }
            return tt::constants::TILE_WIDTH;
        }
        // The minimum valid value for input channels alignment is 8.
        // This requirement comes from the L1 alignment, which is 16 bytes.
        // Since the Halo operation outputs data in row-major layout and the smallest data format used is bfloat16
        // (2 bytes per element), we need at least 8 elements (8 * 2 bytes = 16 bytes) in the input channel
        // dimension. This ensures that one channel (or "stick") can be efficiently transferred over the NoC
        // (Network on Chip) in a single, aligned operation.
        return tt::tt_metal::hal::get_l1_alignment() / 2;
    }
    return tt::constants::TILE_WIDTH;
}

CoreCoord get_output_compute_grid_size(
    const CoreCoord& device_compute_grid_size,
    const Conv2dConfig& conv_config,
    const ParallelConfig& input_parallel_config) {
    CoreCoord output_compute_grid_size = device_compute_grid_size;
    if (conv_config.override_output_sharding_config) {
        TT_FATAL(
            conv_config.core_grid.has_value(),
            "When override_output_sharding_config is set to true, core_grid must have a value.");
        TT_FATAL(
            input_parallel_config.shard_scheme == ttnn::TensorMemoryLayout::BLOCK_SHARDED,
            "Output sharding config override is only supported for BLOCK_SHARDED layout.");
        auto override_compute_grid_size = conv_config.core_grid.value().bounding_box().grid_size();
        TT_FATAL(
            device_compute_grid_size.x >= override_compute_grid_size.x &&
                device_compute_grid_size.y >= override_compute_grid_size.y,
            "Invalid core grid override: {}x{} for device compute grid size: {}x{}",
            override_compute_grid_size.x,
            override_compute_grid_size.y,
            device_compute_grid_size.x,
            device_compute_grid_size.y);
        TT_FATAL(
            (input_parallel_config.shard_orientation == ShardOrientation::ROW_MAJOR
                 ? override_compute_grid_size.y
                 : override_compute_grid_size.x) == get_num_cores_nhw_from_parallel_config(input_parallel_config),
            "NHW cores must match for input and output when overriding the grid size.");
        output_compute_grid_size = override_compute_grid_size;
    }

    return output_compute_grid_size;
}

uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num1, uint32_t num2, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    uint32_t padded_num1 = tt::round_up(num1, divisor);
    uint32_t padded_num2 = tt::round_up(num2, divisor);
    while ((padded_num1 - num1) >= (padded_num1 / divisor) || (padded_num2 - num2) >= (padded_num2 / divisor)) {
        divisor = divisor - 1;
        padded_num1 = tt::round_up(num1, divisor);
        padded_num2 = tt::round_up(num2, divisor);
    }
    return divisor;
}

ParallelConfig determine_parallel_config(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    uint32_t input_channels_alignment,
    const CoreCoord& compute_grid_size,
    ShardOrientation block_shard_orientation,
    bool enable_channels_padding,
    bool is_shard_height_tile_multiple,
    bool is_shard_width_tile_multiple,
    uint32_t act_block_h_override) {
    // Currently, convolution requires multiples of the tile size for both shard height and width,
    // while pooling can accept any height and either a tile multiple or half a tile for width.
    uint32_t effective_tile_height = is_shard_height_tile_multiple ? tt::constants::TILE_HEIGHT : 1;
    uint32_t effective_tile_width =
        is_shard_width_tile_multiple ? tt::constants::TILE_WIDTH : tt::constants::TILE_WIDTH / 2;
    uint32_t out_nhw_ntiles = tt::div_up(batch_size * output_height * output_width, effective_tile_height);

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
        uint32_t input_channels_blocks = tt::div_up(input_channels, input_channels_alignment);
        uint32_t start_divisor =
            block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.x : compute_grid_size.y;
        uint32_t num_cores_nhw = find_closest_largest_divisor_with_num_padding_and_mult(
            out_nhw_ntiles, start_divisor, act_block_h_override_ntiles);
        uint32_t start_divisor_c =
            block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.y : compute_grid_size.x;
        uint32_t num_cores_c =
            enable_channels_padding
                ? find_closest_largest_divisor_with_num_padding(input_channels_blocks, start_divisor_c)
                : find_closest_largest_divisor(out_channels_ntiles, input_channels_blocks, start_divisor_c);

        uint32_t cores_x = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_nhw : num_cores_c;
        uint32_t cores_y = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_c : num_cores_nhw;
        CoreRange core_range = CoreRange(CoreCoord({0, 0}), CoreCoord({cores_x - 1, cores_y - 1}));
        grid = CoreRangeSet({core_range});
    } else if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t input_channels_ntiles = tt::div_up(input_channels, effective_tile_width);
        uint32_t num_cores_c = enable_channels_padding
                                   ? find_closest_largest_divisor_with_num_padding(input_channels_ntiles, max_num_cores)
                                   : find_closest_largest_divisor(input_channels_ntiles, max_num_cores);
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
    ShardOrientation block_shard_orientation,
    bool is_mm_conv) {
    ParallelConfig output_parallel_config = input_parallel_config;
    if (!is_mm_conv) {
        const uint32_t out_channels_ntiles = tt::div_up(out_channels, tt::constants::TILE_WIDTH);
        if (input_parallel_config.shard_scheme == ttnn::TensorMemoryLayout::WIDTH_SHARDED) {
            uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
            output_parallel_config.grid = tt::tt_metal::num_cores_to_corerangeset(
                find_closest_largest_divisor_with_num_padding(out_channels_ntiles, max_num_cores),
                compute_grid_size,
                true);
        } else if (input_parallel_config.shard_scheme == ttnn::TensorMemoryLayout::BLOCK_SHARDED) {
            const uint32_t start_divisor_c =
                block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.y : compute_grid_size.x;
            uint32_t num_cores_c = find_closest_largest_divisor_with_num_padding(out_channels_ntiles, start_divisor_c);
            const uint32_t num_cores_nhw = get_num_cores_nhw_from_parallel_config(input_parallel_config);
            const uint32_t cores_x =
                block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_nhw : num_cores_c;
            const uint32_t cores_y =
                block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_c : num_cores_nhw;
            CoreRange core_range = CoreRange(CoreCoord({0, 0}), CoreCoord({cores_x - 1, cores_y - 1}));
            output_parallel_config.grid = CoreRangeSet({core_range});
        }
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

std::tuple<uint32_t, uint32_t> calculate_ct2d_output_image_size(
    std::array<uint32_t, 2> input_image_size,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding,
    std::array<uint32_t, 2> output_padding,
    std::array<uint32_t, 2> dilation) {
    uint32_t output_height = ((input_image_size[0] - 1) * stride[0]) - (padding[0] + padding[1]) +
                             (dilation[0] * (kernel_size[0] - 1)) + output_padding[0] + 1;

    uint32_t output_width = ((input_image_size[1] - 1) * stride[1]) - (padding[2] + padding[3]) +
                            (dilation[1] * (kernel_size[1] - 1)) + output_padding[1] + 1;
    return {output_height, output_width};
}

uint32_t get_num_cores_nhw(
    const CoreRangeSet& cores, TensorMemoryLayout shard_layout, ShardOrientation shard_orientation) {
    TT_ASSERT(!cores.ranges().empty());
    TT_ASSERT(
        shard_layout == TensorMemoryLayout::HEIGHT_SHARDED || shard_layout == TensorMemoryLayout::BLOCK_SHARDED ||
        shard_layout == TensorMemoryLayout::WIDTH_SHARDED);
    auto grid_size = cores.bounding_box().grid_size();
    uint32_t num_cores = cores.num_cores();
    uint32_t num_cores_nhw = 0;
    if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        return 1;
    }

    if (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_cores_nhw = num_cores;
    } else if (shard_orientation == ShardOrientation::COL_MAJOR) {
        num_cores_nhw = grid_size.x;
    } else {
        TT_ASSERT(shard_orientation == ShardOrientation::ROW_MAJOR);
        num_cores_nhw = grid_size.y;
    }
    TT_ASSERT(num_cores_nhw > 0);
    return num_cores_nhw;
}
uint32_t get_num_cores_nhw_from_parallel_config(const ParallelConfig& pconfig) {
    return get_num_cores_nhw(pconfig.grid, pconfig.shard_scheme, pconfig.shard_orientation);
}

uint32_t get_num_cores_channels(
    const CoreRangeSet& cores, TensorMemoryLayout shard_layout, ShardOrientation shard_orientation) {
    TT_ASSERT(!cores.ranges().empty());
    TT_ASSERT(
        shard_layout == TensorMemoryLayout::HEIGHT_SHARDED || shard_layout == TensorMemoryLayout::BLOCK_SHARDED ||
        shard_layout == TensorMemoryLayout::WIDTH_SHARDED);
    auto grid_size = cores.bounding_box().grid_size();
    uint32_t num_cores_channels = 0;
    if (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_cores_channels = 1;
    } else if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        num_cores_channels = cores.num_cores();
    } else if (shard_orientation == ShardOrientation::COL_MAJOR) {
        num_cores_channels = grid_size.y;
    } else {
        TT_ASSERT(shard_orientation == ShardOrientation::ROW_MAJOR);
        num_cores_channels = grid_size.x;
    }
    TT_ASSERT(num_cores_channels > 0);
    return num_cores_channels;
}

uint32_t get_num_cores_channels_from_parallel_config(const ParallelConfig& pconfig) {
    return get_num_cores_channels(pconfig.grid, pconfig.shard_scheme, pconfig.shard_orientation);
}

MemoryConfig create_sharded_memory_config_from_parallel_config(
    const ttnn::Shape& tensor_shape, const ParallelConfig& parallel_config, uint32_t tile_size) {
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
        nhw_padded = tt::round_up(nhw_shape, num_cores_nhw * tile_size);
    }
    uint32_t nhw_shard = nhw_padded / num_cores_nhw;
    TT_FATAL(channels % num_cores_channels == 0, "Channels: {}, num core channels: {}", channels, num_cores_channels);
    uint32_t channel_shard = channels / num_cores_channels;
    auto shard_spec = tt::tt_metal::ShardSpec{parallel_config.grid, {nhw_shard, channel_shard}, shard_orientation};
    return MemoryConfig{shard_scheme, BufferType::L1, shard_spec};
}

Conv2dParallelizationConfig determine_conv_op_parallel_config_from_conv_output_mem_config(
    const MemoryConfig& conv_output_mem_config,
    uint32_t num_cores_nhw,
    uint32_t num_cores_c_in,
    uint32_t num_cores_c_out) {
    TT_ASSERT(conv_output_mem_config.shard_spec().has_value());
    const auto& shard_spec = conv_output_mem_config.shard_spec().value();
    const auto& shard_shape = shard_spec.shape;
    return {
        .grid_size = shard_spec.grid.bounding_box().grid_size(),
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c_in = num_cores_c_in,
        .num_cores_c_out = num_cores_c_out,
        .per_core_out_matrix_height_ntile = tt::div_up(shard_shape[0], tt::constants::TILE_HEIGHT),
        .per_core_out_matrix_width_ntile = tt::div_up(shard_shape[1], tt::constants::TILE_WIDTH),
    };
}

static std::pair<uint32_t, uint32_t> determine_largest_subblock_size(
    uint32_t block_height, uint32_t block_width, bool fp32_accum) {
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
        "Could not find valid subblock size for block size {}x{}, fp32_accum: {}",
        block_height,
        block_width,
        fp32_accum);
    return {subblock_h, subblock_w};
}

Conv2dBlockConfig determine_per_core_conv_block_config(
    const ParallelConfig& parallel_config,
    const Conv2dParallelizationConfig& conv_op_parallel_config,
    uint32_t padded_in_channels,
    uint32_t padded_output_height_ntiles_per_core,
    uint32_t act_block_h_override,
    uint32_t act_block_w_div,
    uint32_t window_h,
    uint32_t window_w,
    uint32_t output_width,
    bool fp32_accum,
    bool full_inner_dim,
    bool enable_activation_reuse,
    bool is_1d_depthwise_conv) {
    if (act_block_h_override > 0) {
        TT_ASSERT(
            act_block_h_override % 32 == 0,
            "Config Error: act_block_h_override must be a multiple of 32 (tile height).");
    }

    uint32_t act_block_h_ntiles = conv_op_parallel_config.per_core_out_matrix_height_ntile;

    if (act_block_h_override > 0) {
        uint32_t act_block_h_override_ntiles = act_block_h_override / tt::constants::TILE_HEIGHT;
        if (padded_output_height_ntiles_per_core % act_block_h_override_ntiles == 0) {
            act_block_h_ntiles = act_block_h_override_ntiles;
        } else {
            uint32_t act_block_h_override_ntiles = act_block_h_override / tt::constants::TILE_HEIGHT;
            if (padded_output_height_ntiles_per_core % act_block_h_override_ntiles == 0) {
                act_block_h_ntiles = act_block_h_override_ntiles;
            } else {
                act_block_h_ntiles =
                    find_closest_largest_divisor(padded_output_height_ntiles_per_core, act_block_h_override_ntiles);
                log_warning(
                    tt::LogOp,
                    "act_block_h_override {} is not a valid override for padded_output_height_ntiles_per_core {}, "
                    "instead {} was selected as closest valid option!",
                    act_block_h_override_ntiles,
                    padded_output_height_ntiles_per_core,
                    act_block_h_ntiles);
            }
        }
    }

    if (parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED && enable_activation_reuse) {
        const uint32_t output_image_width_ntiles = tt::div_up(output_width, tt::constants::TILE_HEIGHT);
        TT_FATAL(
            act_block_h_ntiles > output_image_width_ntiles,
            "Activation reuse needs act block h ({}) to be bigger than output image width in tiles ({}) for the "
            "optimization to give boost",
            act_block_h_ntiles,
            output_image_width_ntiles);
    }

    uint32_t act_c_num_blocks = get_num_cores_channels_from_parallel_config(parallel_config);
    TT_ASSERT(padded_in_channels % act_c_num_blocks == 0);
    uint32_t act_block_w = 0;
    if (parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        // For 1D depthwise conv, each channel is processed independently, so activation
        // block width doesn't need to be multiplied by the kernel window size
        if (is_1d_depthwise_conv) {
            act_block_w = tt::round_up(padded_in_channels, tt::constants::TILE_WIDTH);
        } else {
            act_block_w = enable_activation_reuse
                              ? tt::round_up(padded_in_channels * window_h * window_w, tt::constants::TILE_WIDTH)
                              : tt::round_up(padded_in_channels * window_w, tt::constants::TILE_WIDTH);
        }
    } else if (parallel_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        act_block_w = tt::round_up(
            padded_in_channels / act_c_num_blocks * window_w * (full_inner_dim ? window_h : 1),
            tt::constants::TILE_WIDTH);

    } else if (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        TT_ASSERT(
            padded_in_channels % (32 * parallel_config.grid.num_cores() * act_block_w_div) == 0,
            "Padded In Channels = {}, num_cores = {}, act_block_w_div = {}",
            padded_in_channels,
            parallel_config.grid.num_cores(),
            act_block_w_div);
        act_block_w = (padded_in_channels * window_h * window_w) / (parallel_config.grid.num_cores() * act_block_w_div);
    }

    TT_ASSERT(act_block_w % tt::constants::TILE_HEIGHT == 0);
    uint32_t act_block_w_ntiles = act_block_w / tt::constants::TILE_HEIGHT;
    uint32_t weight_block_w_ntiles = conv_op_parallel_config.per_core_out_matrix_width_ntile;
    auto [out_subblock_h_ntiles, out_subblock_w_ntiles] =
        determine_largest_subblock_size(act_block_h_ntiles, weight_block_w_ntiles, fp32_accum);
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
    uint32_t /*groups*/,
    const Conv2dConfig& conv_config) {
    bool is_width_sharded =
        (conv_config.shard_layout.has_value() && conv_config.shard_layout.value() == TensorMemoryLayout::WIDTH_SHARDED);
    return kernel_size[0] == 1 && kernel_size[1] == 1 && stride[0] == stride[1] && stride[0] == 1 && padding[0] == 0 &&
           padding[1] == 0 && padding[2] == 0 && padding[3] == 0 && dilation[0] == 1 && dilation[1] == 1 &&
           (not is_width_sharded);
}

bool is_1d_conv(uint32_t kernel_height, uint32_t image_height) { return kernel_height == 1 && image_height == 1; }

bool is_1d_depthwise_conv(
    uint32_t groups,
    uint32_t input_channels,
    uint32_t output_channels,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t image_height,
    bool has_bias) {
    bool is_depthwise_conv = groups == input_channels && groups == output_channels;
    // Only use 1D depthwise path when kernel is truly pointwise (1x1)
    // is_1d_conv checks height=1, we also need kernel_width=1 to ensure no spatial computation
    return is_depthwise_conv && is_1d_conv(kernel_height, image_height) && kernel_width == 1 && !has_bias;
}

SkipMcast conv_skip_mcast(const Conv2dParallelizationConfig& parallelization_config, TensorMemoryLayout memory_layout) {
    bool skip_act_mcast = false;
    bool skip_weights_mcast = false;
    if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED || memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        skip_act_mcast = parallelization_config.num_cores_c_out == 1 && parallelization_config.num_cores_c_in == 1;
        skip_weights_mcast = parallelization_config.num_cores_nhw == 1;
    } else {
        skip_act_mcast = (parallelization_config.num_cores_nhw * parallelization_config.num_cores_c_out) == 1;
        skip_weights_mcast = skip_act_mcast;
    }
    return SkipMcast{skip_act_mcast, skip_weights_mcast};
}

DeviceComputeKernelConfig get_conv_default_compute_kernel_config(
    MeshDevice* device, DataType input_dtype, DataType weight_dtype) {
    // Default fp32_dest_acc to true if both inputs are FP32, false otherwise
    bool default_fp32_acc = (input_dtype == DataType::FLOAT32 && weight_dtype == DataType::FLOAT32);
    return init_device_compute_kernel_config(
        device->arch(), std::nullopt, MathFidelity::HiFi4, true, default_fp32_acc, false);
}

std::tuple<ttnn::Shape, ttnn::MemoryConfig> determine_input_memory_config(
    TensorMemoryLayout shard_layout,
    ShardOrientation block_shard_orientation,
    uint32_t batch_size,
    ttnn::Shape input_tensor_shape,
    ttnn::Shape output_tensor_shape,
    bool is_mm_conv,
    CoreCoord compute_grid_size,
    Layout input_tensor_layout,
    BufferType input_tensor_buffer_type,
    const std::optional<ParallelConfig>& input_tensor_parallel_config,
    std::optional<uint32_t> act_block_h_override,
    bool enable_channels_padding,
    bool is_shard_height_tile_multiple,
    bool is_shard_width_tile_multiple) {
    const uint32_t input_channels_alignment = get_input_channels_alignment(
        shard_layout, input_tensor_layout, input_tensor_buffer_type == BufferType::DRAM, is_mm_conv, std::nullopt);
    ParallelConfig parallel_config;
    if (input_tensor_parallel_config.has_value()) {
        parallel_config = input_tensor_parallel_config.value();
    } else {
        parallel_config = determine_parallel_config(
            shard_layout,
            batch_size,
            input_tensor_shape[3],
            output_tensor_shape[1],
            output_tensor_shape[2],
            output_tensor_shape[3],
            input_channels_alignment,
            compute_grid_size,
            block_shard_orientation,
            enable_channels_padding && !is_mm_conv,
            is_shard_height_tile_multiple,
            is_shard_width_tile_multiple,
            act_block_h_override.value_or(0));
    }
    uint32_t input_num_cores_nhw = get_num_cores_nhw_from_parallel_config(parallel_config);
    uint32_t input_num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);

    uint32_t tensor_height = input_tensor_shape[0] * input_tensor_shape[1] * input_tensor_shape[2];
    uint32_t round_up_size = tt::constants::TILE_HEIGHT;
    if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED && input_tensor_layout == Layout::ROW_MAJOR) {
        round_up_size = 1;
    }
    uint32_t input_tensor_height_snapped_to_tile = tt::round_up(tensor_height, input_num_cores_nhw * round_up_size);

    uint32_t input_tensor_width_snapped_to_channels_alignment =
        tt::round_up(input_tensor_shape[3], input_num_cores_c * input_channels_alignment);

    auto input_padded_shape =
        ttnn::Shape({1, 1, input_tensor_height_snapped_to_tile, input_tensor_width_snapped_to_channels_alignment});

    MemoryConfig input_tensor_sharded_memory_config =
        create_sharded_memory_config_from_parallel_config(input_padded_shape, parallel_config, round_up_size);

    return {input_padded_shape, input_tensor_sharded_memory_config};
};

std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> get_conv_padded_input_shape_and_mem_config(
    MeshDevice* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv) {
    const ttnn::Tensor& input_tensor = input_tensor_;  // tensor to return
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

    TensorMemoryLayout shard_layout{};
    if (conv_config.shard_layout.has_value()) {
        shard_layout = conv_config.shard_layout.value();
    }

    const ttnn::MemoryConfig& input_memory_config = input_tensor_.memory_config();
    const tt::tt_metal::TensorMemoryLayout input_shard_scheme = input_memory_config.memory_layout();
    const uint32_t input_channels_alignment = get_input_channels_alignment(
        input_shard_scheme, input_tensor_.layout(), false, is_mm_conv, input_memory_config);

    ParallelConfig input_tensor_parallel_config;
    if (!input_tensor_on_device) {
        needs_shard_or_reshard = true;
    } else {
        if (!input_memory_config.is_sharded()) {
            needs_shard_or_reshard = true;
        } else {
            const tt::tt_metal::ShardSpec& input_shard_spec = input_memory_config.shard_spec().value();
            const tt::tt_metal::ShardOrientation input_shard_orientation = input_shard_spec.orientation;
            const CoreRangeSet input_shard_grid = input_shard_spec.grid;
            ParallelConfig pconfig = {
                .grid = input_shard_grid,
                .shard_scheme = input_shard_scheme,
                .shard_orientation = input_shard_orientation};
            input_tensor_parallel_config = pconfig;
            if (input_shard_scheme != TensorMemoryLayout::BLOCK_SHARDED &&
                input_shard_orientation != ShardOrientation::ROW_MAJOR) {
                needs_shard_or_reshard = true;
            }

            // Check if input channels alignment is satisfied
            if (input_shard_spec.shape[1] % input_channels_alignment != 0) {
                needs_shard_or_reshard = true;
            }

            if (is_mm_conv && input_shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
                uint32_t num_cores_c = input_shard_orientation == ShardOrientation::ROW_MAJOR
                                           ? input_shard_grid.bounding_box().grid_size().x
                                           : input_shard_grid.bounding_box().grid_size().y;
                if (in_channels != num_cores_c * input_shard_spec.shape[1]) {
                    needs_shard_or_reshard = true;
                    shard_layout = TensorMemoryLayout::BLOCK_SHARDED;
                }
            }

            // Additional check for mm convs to ensure shard height is multiple of TILE_HEIGHT since tiling requires
            // that
            if (is_mm_conv && (input_shard_spec.shape[0] % tt::constants::TILE_HEIGHT != 0)) {
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

    auto block_shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;
    ParallelConfig parallel_config = input_tensor_parallel_config;
    if (conv_config.reshard_if_not_optimal || needs_shard_or_reshard) {
        ParallelConfig optimal_parallel_config = determine_parallel_config(
            shard_layout,
            batch_size,
            in_channels,
            height,
            width,
            out_channels,
            input_channels_alignment,
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
        auto [input_padded_shape, input_tensor_sharded_memory_config] = determine_input_memory_config(
            shard_layout,
            block_shard_orientation,
            batch_size,
            input_tensor.logical_shape(),
            input_tensor.padded_shape(),
            is_mm_conv,
            device->compute_with_storage_grid_size(),
            input_tensor.layout(),
            BufferType::L1,
            parallel_config,
            conv_config.act_block_h_override);
        return {input_padded_shape, input_tensor_sharded_memory_config, needs_shard_or_reshard};
    }
    return {input_tensor.logical_shape(), input_tensor.memory_config(), needs_shard_or_reshard};
}

ttnn::Shape flatten_4d_shape(const ttnn::Shape& input_shape) {
    TT_FATAL(input_shape.size() == 4, "Expected 4D shape");
    const uint32_t nhw = input_shape[0] * input_shape[1] * input_shape[2];
    const uint32_t channels = input_shape[3];
    return ttnn::Shape{1, 1, nhw, channels};
}

std::tuple<ttnn::Tensor, ParallelConfig, ParallelConfig> shard_or_reshard_tensor_if_required(
    MeshDevice* device,
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

    auto output_compute_grid_size = get_output_compute_grid_size(compute_grid_size, conv_config, parallel_config);
    ParallelConfig output_parallel_config = determine_output_parallel_config(
        parallel_config, output_compute_grid_size, out_channels, parallel_config.shard_orientation, is_mm_conv);

    // We can have flat and unflattened (n, h, w, c) tensors here
    const auto flattened_input_shape = flatten_4d_shape(input_tensor.logical_shape());
    const auto flattened_padded_input_shape = flatten_4d_shape(input_tensor.padded_shape());

    input_tensor = ttnn::reshape(input_tensor, flattened_input_shape, flattened_padded_input_shape);
    const ttnn::Shape& input_shape = flattened_input_shape;

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
                Tensor input_tensor_tilized = ttnn::to_layout(input_tensor, Layout::TILE);
                if (conv_config.deallocate_activation && !input_tensor.memory_config().is_dram()) {
                    input_tensor.deallocate(/*force*/ true);
                    input_tensor_tilized = ttnn::move(input_tensor_tilized);
                }
                input_tensor = input_tensor_tilized;
            }
            if (!auto_shard_mm) {
                ttnn::MemoryConfig input_tensor_sharded_memory_config_to_layout = input_tensor_sharded_memory_config;
                tt::tt_metal::Alignment alignment = {};
                if (!input_tensor.is_sharded()) {
                    // In case we need to run Interleaved2Sharded, adjust the shard spec,
                    // in order to get smaller allocation size of sharded buffer.
                    const auto& shard_spec = input_tensor_sharded_memory_config.shard_spec().value();
                    input_tensor_sharded_memory_config_to_layout =
                        input_tensor_sharded_memory_config_to_layout.with_shard_spec(
                            tt::tt_metal::ShardSpec(shard_spec.grid, shard_spec.shape, shard_spec.orientation));
                    alignment = tt::tt_metal::Alignment{shard_spec.shape[0], shard_spec.shape[1]};
                }
                Tensor resharded_input_tensor = tt::tt_metal::create_device_tensor(
                    TensorSpec(
                        input_tensor.logical_shape(),
                        tt::tt_metal::TensorLayout(
                            input_tensor.dtype(),
                            tt::tt_metal::PageConfig(input_tensor.layout()),
                            input_tensor_sharded_memory_config_to_layout,
                            alignment)),
                    input_tensor.device());
                ttnn::to_memory_config(
                    input_tensor, input_tensor_sharded_memory_config_to_layout, std::nullopt, resharded_input_tensor);
                if (conv_config.deallocate_activation && !input_tensor.memory_config().is_dram()) {
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

ttnn::operations::matmul::MatmulProgramConfig determine_matmul_op_config_from_conv_op_config(
    Conv2dParallelizationConfig conv_parallelization_config,
    Conv2dBlockConfig conv_blocking_config,
    bool height_sharded,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    bool transpose_mcast,
    uint32_t /*grid_size_along_c*/) {
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
        if (activation.has_value()) {
            matmul_config.fused_activation = activation.value();
        }
        return matmul_config;
    }
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
    if (activation.has_value()) {
        matmul_config.fused_activation = activation.value();
    }
    return matmul_config;
}

core_count_and_size calculate_L1_usage_for_conv_op(
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t output_height,
    uint32_t output_width,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 4>& padding,
    const std::array<uint32_t, 2>& dilation,
    uint32_t groups,
    bool enable_bias,
    DataType input_datatype,
    DataType output_datatype,
    Layout input_layout,
    CoreCoord compute_grid_size,
    bool is_mm_conv,
    TensorMemoryLayout shard_layout,
    DeviceComputeKernelConfig compute_config,
    const Conv2dConfig& conv_config_in,
    const std::optional<ttnn::MemoryConfig>& _halo_input_memory_config) {
    Conv2dConfig conv_config = conv_config_in;
    conv_config.shard_layout = shard_layout;

    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;

    // Output of halo op is always ROW_MAJOR, so input for convs is either DataType::FLOAT32 or DataType::BFLOAT16
    const tt::tt_metal::DataType conv_input_dtype = (input_datatype == tt::tt_metal::DataType::FLOAT32)
                                                        ? tt::tt_metal::DataType::FLOAT32
                                                        : tt::tt_metal::DataType::BFLOAT16;
    const uint32_t input_datum_size = conv_input_dtype == tt::tt_metal::DataType::FLOAT32 ? 4 : 2;

    const bool conv_is_1d_depthwise = is_1d_depthwise_conv(
        groups, in_channels, out_channels, kernel_size[0], kernel_size[1], input_height, enable_bias);

    const uint32_t input_channels_alignment =
        get_input_channels_alignment(shard_layout, input_layout, false, is_mm_conv, std::nullopt);
    const uint32_t in_channels_aligned = tt::round_up(in_channels, input_channels_alignment);
    const uint32_t output_channels_padded = tt::round_up(out_channels, tt::constants::TILE_WIDTH);
    // Note: These are not exact shapes for weights as prepare_conv_weights will pad the weights depending on the
    // conv2d params, but these are good enough for L1 usage estimation.
    const ttnn::Shape weights_shape(
        {1, 1, in_channels_aligned * kernel_size[0] * kernel_size[1], output_channels_padded});

    const ParallelConfig input_parallel_config = _halo_input_memory_config.has_value() ? ParallelConfig{
            .grid = _halo_input_memory_config->shard_spec().value().grid,
            .shard_scheme = _halo_input_memory_config->memory_layout(),
            .shard_orientation = _halo_input_memory_config->shard_spec().value().orientation}
    : determine_parallel_config(
        conv_config.shard_layout.value(),
        batch_size,
        in_channels,
        output_height,
        output_width,
        out_channels,
        input_channels_alignment,
        compute_grid_size,
        shard_orientation,
        !is_mm_conv,
        true,
        true,
        conv_config.act_block_h_override);

    auto output_compute_grid_size = get_output_compute_grid_size(compute_grid_size, conv_config, input_parallel_config);
    const ParallelConfig output_parallel_config = determine_output_parallel_config(
        input_parallel_config,
        output_compute_grid_size,
        out_channels,
        shard_orientation,
        is_mm_conv /* && conv_config.shard_layout != TensorMemoryLayout::WIDTH_SHARDED*/);

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
        compute_grid_size,
        conv_is_1d_depthwise);

    SlidingWindowConfig sliding_window_config{
        .input_hw = {input_height, input_width}, .window_hw = {kernel_size[0], kernel_size[1]}};
    conv_op_l1_usage l1_usage = calculate_L1_usage(
        compute_config,
        opt_conv_op_block_config,
        opt_conv_op_parallel_config,
        weights_shape,
        sliding_window_config,
        dilation,
        conv_config,
        input_datatype,
        output_datatype,
        output_width,
        enable_bias,
        conv_is_1d_depthwise,
        in_channels_padded);

    auto halo_input_memory_config = _halo_input_memory_config.has_value()
                                        ? _halo_input_memory_config.value()
                                        : std::get<1>(determine_input_memory_config(
                                              conv_config.shard_layout.value(),
                                              shard_orientation,
                                              batch_size,
                                              ttnn::Shape({batch_size, input_height, input_width, in_channels}),
                                              ttnn::Shape({batch_size, output_height, output_width, out_channels}),
                                              is_mm_conv,
                                              compute_grid_size,
                                              input_layout,
                                              BufferType::L1,
                                              input_parallel_config,
                                              conv_config.act_block_h_override));

    auto halo_input_shard_shape = halo_input_memory_config.shard_spec().value().shape;

    // Create SlidingWindowConfig for precise halo calculation
    SlidingWindowConfig halo_config;
    halo_config.batch_size = batch_size;
    halo_config.input_hw = {input_height, input_width};
    halo_config.window_hw = {kernel_size[0], kernel_size[1]};
    halo_config.stride_hw = {stride[0], stride[1]};
    halo_config.padding = padding;
    halo_config.dilation_hw = {dilation[0], dilation[1]};
    halo_config.num_cores_nhw = get_num_cores_nhw_from_parallel_config(input_parallel_config);
    halo_config.core_range_set = halo_input_memory_config.shard_spec().value().grid;
    halo_config.snap_to_tile = true;

    uint32_t precise_input_size_per_core =
        sliding_window::calculate_precise_halo_output_elems(halo_config, halo_input_shard_shape);

    log_trace(
        tt::LogOp,
        "L1 usage for {}: Input {}, Output {}, CBs {}, Halo Output : {}",
        conv_config.shard_layout,
        halo_input_memory_config,
        l1_usage.tensor_allocation_size,
        l1_usage.CB_allocation_size,
        precise_input_size_per_core);

    uint32_t input_size_divider = (input_datatype == tt::tt_metal::DataType::BFLOAT8_B) ? 2 : 1;

    return core_count_and_size{
        .core_count = std::max(input_parallel_config.grid.num_cores(), output_parallel_config.grid.num_cores()),
        .halo_input_size =
            halo_input_shard_shape[0] * halo_input_shard_shape[1] * input_datum_size / input_size_divider,
        .halo_output_size = precise_input_size_per_core * input_datum_size,
        .conv_op_size = l1_usage.CB_allocation_size + l1_usage.tensor_allocation_size,
        .total_size = l1_usage.CB_allocation_size + l1_usage.tensor_allocation_size +
                      precise_input_size_per_core * input_datum_size,
        .conv_config = conv_config};
};

Conv2dConfig determine_conv_config_for_auto_shard(
    const Conv2dConfig& conv_config,
    bool is_mm_conv,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t /*weights_width*/,
    uint32_t input_height,
    uint32_t input_width,
    const CoreCoord& compute_grid_size,
    Layout input_layout,
    tt::tt_metal::DataType input_datatype,
    tt::tt_metal::DataType output_datatype,
    std::optional<const MemoryConfig> input_memory_config,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& dilation,
    const std::array<uint32_t, 4>& padding,
    uint32_t groups,
    bool enable_bias,
    const DeviceComputeKernelConfig& compute_config) {
    // If the input tensor is already sharded, or the conv_config has a specified shard layout, we don't need to do
    // anything.

    log_debug(
        tt::LogOp,
        "Auto sharding Input={}x{}, Output={}x{}, Padding={}, Input, Output DTYPE = {},{}",
        input_height,
        input_width,
        output_height,
        output_width,
        padding,
        input_datatype,
        output_datatype);
    if ((input_memory_config.has_value() && input_memory_config.value().is_sharded()) ||
        conv_config.shard_layout.has_value()) {
        return conv_config;
    }
    const bool conv_is_1d_depthwise = is_1d_depthwise_conv(
        groups, in_channels, out_channels, kernel_size[0], kernel_size[1], input_height, enable_bias);

    auto get_l1_usage_for_sharding = [&](TensorMemoryLayout shard_layout, const Conv2dConfig& conv_config) {
        return calculate_L1_usage_for_conv_op(
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            enable_bias,
            input_datatype,
            output_datatype,
            input_layout,
            compute_grid_size,
            is_mm_conv,
            shard_layout,
            compute_config,
            conv_config);
    };
    core_count_and_size height = get_l1_usage_for_sharding(TensorMemoryLayout::HEIGHT_SHARDED, conv_config);

    // 1d depthwise convs support only height sharding
    if (conv_is_1d_depthwise) {
        return height.conv_config;
    }

    const core_count_and_size block = get_l1_usage_for_sharding(TensorMemoryLayout::BLOCK_SHARDED, conv_config);
    const core_count_and_size width = get_l1_usage_for_sharding(TensorMemoryLayout::WIDTH_SHARDED, conv_config);

    core_count_and_size& winning_config = height;
    // Make sure that BS not only has smaller size but provides at least some slicing along the channels.
    // In case we have BS that would slice the tensor only along the HS conv2d code would fail later on.
    if (block.total_size < winning_config.total_size && block.core_count >= compute_grid_size.x) {
        winning_config = block;
    }
    if (width.total_size < winning_config.total_size && !is_mm_conv) {
        winning_config = width;
    }

    log_trace(
        tt::LogOp,
        "Selected shard layout: {}, size: {}",
        winning_config.conv_config.shard_layout,
        winning_config.total_size);
    return winning_config.conv_config;
}

std::tuple<Conv2dParallelizationConfig, Conv2dBlockConfig, MemoryConfig> get_conv_configs(
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
    const CoreCoord& /*compute_grid*/,
    bool is_1d_depthwise_conv) {
    uint32_t round_up_size = tt::constants::TILE_HEIGHT;
    uint32_t nhw_out = batch_size * output_height * output_width;
    uint32_t out_channels_padded = tt::round_up(
        out_channels, get_num_cores_channels_from_parallel_config(output_parallel_config) * tt::constants::TILE_WIDTH);
    MemoryConfig conv_out_memory_config = create_sharded_memory_config_from_parallel_config(
        ttnn::Shape({1, 1, nhw_out, out_channels_padded}), output_parallel_config, round_up_size);

    Conv2dParallelizationConfig opt_conv_op_parallel_config =
        determine_conv_op_parallel_config_from_conv_output_mem_config(
            conv_out_memory_config,
            get_num_cores_nhw_from_parallel_config(input_parallel_config),
            get_num_cores_channels_from_parallel_config(input_parallel_config),
            get_num_cores_channels_from_parallel_config(output_parallel_config));

    uint32_t nhw_out_padded_ntile_per_core =
        conv_out_memory_config.shard_spec().value().shape[0] / tt::constants::TILE_HEIGHT;

    Conv2dBlockConfig opt_conv_op_block_config = determine_per_core_conv_block_config(
        input_parallel_config,
        opt_conv_op_parallel_config,
        in_channels_padded,
        nhw_out_padded_ntile_per_core,
        conv_config.act_block_h_override,
        conv_config.act_block_w_div,
        kernel_size[0],
        kernel_size[1],
        output_width,
        get_fp32_dest_acc_en(compute_config),
        conv_config.full_inner_dim,
        conv_config.enable_activation_reuse,
        is_1d_depthwise_conv);
    return {opt_conv_op_parallel_config, opt_conv_op_block_config, conv_out_memory_config};
}

}  // namespace ttnn::operations::conv

namespace ttnn::prim {

using ttnn::operations::sliding_window::SlidingWindowConfig;

conv_op_l1_usage calculate_L1_usage(
    const DeviceComputeKernelConfig& compute_kernel_config,
    const Conv2dBlockConfig& block_config,
    const Conv2dParallelizationConfig& pconfig,
    const ttnn::Shape& weights_shape,
    const SlidingWindowConfig& sliding_window_config,
    std::array<uint32_t, 2> dilation,
    const Conv2dConfig& conv_config,
    const DataType input_datatype,
    const DataType output_datatype,
    const uint32_t output_image_width,
    const bool enable_bias,
    bool is_1d_depthwise_conv,
    uint32_t input_channels_padded,
    bool skip_act_cb_create) {
    // Input shard doesn't affect L1 usage calculation.
    std::array<uint32_t, 2> dummy_input_shard_shape = {0, 0};
    std::vector<CBInfo> cb_info = get_cb_info(
        compute_kernel_config,
        block_config,
        pconfig,
        weights_shape,
        {sliding_window_config.window_hw.first, sliding_window_config.window_hw.second},
        {sliding_window_config.input_hw.first, sliding_window_config.input_hw.second},
        dilation,
        conv_config,
        input_datatype,
        output_datatype,
        dummy_input_shard_shape,
        output_image_width,
        enable_bias,
        is_1d_depthwise_conv,
        skip_act_cb_create,
        input_channels_padded);
    uint32_t total_CB_size = 0;
    uint32_t output_size = 0;
    for (const CBInfo& cb : cb_info) {
        if (!cb.is_globally_allocated) {
            total_CB_size += cb.cb_size_per_core();
        }
        if (cb.name == Conv2dCb::OUT) {
            output_size = cb.cb_size_per_core();
        }
    }
    log_trace(tt::LogOp, "Conv L1 Size Estimation, Total CB size: {}, Output Size: {}", total_CB_size, output_size);

    return conv_op_l1_usage{.tensor_allocation_size = output_size, .CB_allocation_size = total_CB_size};
}

bool determine_packer_l1_acc(bool packer_l1_acc, bool enable_bias, uint32_t in0_num_blocks_w) {
    return packer_l1_acc && ((enable_bias && in0_num_blocks_w > 1) || (in0_num_blocks_w > 2));
}

}  // namespace ttnn::prim

namespace ttnn::operations::conv {

bool auto_enable_kernel_folding(
    const ttnn::MemoryConfig& input_memory_config,
    Layout input_layout,
    const DataType& /*input_dtype*/,
    std::optional<bool> enable_folding_,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2>& kernel_size,
    std::array<uint32_t, 2>& stride,
    std::array<uint32_t, 2>& dilation,
    std::array<uint32_t, 4>& padding_n4) {
    if (!enable_folding_.has_value()) {
        if (stride[0] != kernel_size[0] || stride[1] != kernel_size[1]) {
            return false;
        }
        if (stride[0] == 1 && stride[1] == 1) {
            return false;
        }
        if (dilation[0] != 1 || dilation[1] != 1) {
            return false;
        }
        auto input_padded_height = input_height + padding_n4[0] + padding_n4[1];
        auto input_padded_width = input_width + padding_n4[2] + padding_n4[3];

        if (input_padded_height % stride[0] != 0 || input_padded_width % stride[1] != 0) {
            return false;
        }
        auto is_zero_padding = padding_n4[0] == 0 && padding_n4[1] == 0 && padding_n4[2] == 0 && padding_n4[3] == 0;
        return (
            (input_memory_config.is_dram() && (input_layout == Layout::ROW_MAJOR || is_zero_padding)) ||
            (input_memory_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED));
    }
    return enable_folding_.value();
}
Tensor fold_input_tensor_if_required(
    const ttnn::Tensor& input_tensor,
    MeshDevice* device,
    uint32_t& batch_size,
    uint32_t& input_height,
    uint32_t& input_width,
    uint32_t& in_channels,
    std::array<uint32_t, 2>& kernel_size,
    std::array<uint32_t, 2>& stride,
    std::array<uint32_t, 2>& dilation,
    std::array<uint32_t, 4>& padding_n4,
    bool& mm_conv,
    Conv2dConfig& conv_config) {
    // Conv DRAM would fold the input tensor, but conv_config.enable_kernel_stride_folding would stil be true as weights
    // also need to be folded.
    conv_config.enable_kernel_stride_folding = auto_enable_kernel_folding(
        input_tensor.memory_config(),
        input_tensor.layout(),
        input_tensor.dtype(),
        conv_config.enable_kernel_stride_folding,
        input_height,
        input_width,
        kernel_size,
        stride,
        dilation,
        padding_n4);

    if (conv_config.enable_kernel_stride_folding.value() && (stride[0] > 1 || stride[1] > 1)) {
        auto folding_result = compute_kernel_stride_folding_params(
            input_height, input_width, in_channels, kernel_size, stride, padding_n4, conv_config);
        auto folded_input_tensor = fold_tensor(
            input_tensor, device, stride, kernel_size, padding_n4, batch_size, input_height, input_width, in_channels);
        if (conv_config.deallocate_activation && !input_tensor.memory_config().is_dram()) {
            auto tensor_to_deallocate = input_tensor;
            tensor_to_deallocate.deallocate(true);
        }

        // Update the input tensor parameters to the folding result
        input_height = folding_result.input_height;
        input_width = folding_result.input_width;
        in_channels = folding_result.in_channels;
        stride = folding_result.stride;
        kernel_size = folding_result.kernel_size;
        padding_n4 = folding_result.padding_n4;  // Padding is zero after folding
        mm_conv = folding_result.mm_conv;
        return folded_input_tensor;
    }
    return input_tensor;
}

ttnn::Tensor fold_tensor(
    const ttnn::Tensor& tensor,
    MeshDevice* device,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 4> padding_n4,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t in_channels) {
    // Validation checks
    TT_FATAL(
        stride[0] <= kernel_size[0] && stride[1] <= kernel_size[1],
        "Conv2D kernel stride folding: Stride must be less than or equal to kernel size");

    // Move to device if needed
    ttnn::Tensor tensor_on_device = tensor;
    if (!tt::tt_metal::is_device_tensor(tensor_on_device)) {
        tensor_on_device = ttnn::to_device(tensor_on_device, device, ttnn::DRAM_MEMORY_CONFIG);
    }

    // Reshape tensor from flattened 4D shape (e.g., [1, 1, N*H*W, C]) back to original 4D shape [N, H, W, C] before
    // folding
    const auto& current_shape = tensor_on_device.logical_shape();
    bool needs_reshape =
        (current_shape.rank() == 4 && (current_shape[1] != input_height || current_shape[2] != input_width));
    if (needs_reshape) {
        const auto unflattened_shape = ttnn::Shape{batch_size, input_height, input_width, in_channels};
        tensor_on_device = ttnn::reshape(tensor_on_device, unflattened_shape, unflattened_shape);
    }

    // Core folding operation
    tensor_on_device = ttnn::fold(tensor_on_device, stride[0], stride[1], false, std::nullopt, padding_n4);

    return tensor_on_device;
}

KernelStrideFoldingResult compute_kernel_stride_folding_params(
    uint32_t input_height,
    uint32_t input_width,
    uint32_t in_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 4> padding_n4,
    const Conv2dConfig& /*conv_config*/) {
    // Calculate padded dimensions first - this is what the folding operation will see
    uint32_t padded_height = input_height + padding_n4[0] + padding_n4[1];
    uint32_t padded_width = input_width + padding_n4[2] + padding_n4[3];

    TT_FATAL(
        padded_height % stride[0] == 0,
        "Padded input height ({}) must be divisible by stride ({}) for kernel stride folding.",
        padded_height,
        stride[0]);
    TT_FATAL(
        padded_width % stride[1] == 0,
        "Padded input width ({}) must be divisible by stride ({}) for kernel stride folding.",
        padded_width,
        stride[1]);

    // Update dimensions for folded operation based on padded dimensions
    uint32_t folded_height = padded_height / stride[0];
    uint32_t folded_width = padded_width / stride[1];
    uint32_t folded_channels = in_channels * stride[0] * stride[1];

    auto kernel_h = tt::div_up(kernel_size[0], stride[0]);
    auto kernel_w = tt::div_up(kernel_size[1], stride[1]);

    return KernelStrideFoldingResult{
        .input_height = folded_height,
        .input_width = folded_width,
        .in_channels = folded_channels,
        .stride = {1, 1},
        .kernel_size = {kernel_h, kernel_w},
        .padding_n4 = {0, 0, 0, 0},  // Padding is zero after folding
        .mm_conv = (kernel_size[0] == stride[0] && kernel_size[1] == stride[1])};
}

std::ostream& operator<<(std::ostream& os, const Conv2dConfig& config) {
    tt::stl::reflection::operator<<(os, config);
    return os;
}

void tilize_with_optional_deallocation(Tensor& input_tensor_on_device, bool deallocate) {
    if (input_tensor_on_device.layout() != Layout::TILE) {
        Tensor input_tensor_tilized = ttnn::to_layout(input_tensor_on_device, Layout::TILE);
        if (deallocate) {
            input_tensor_on_device.deallocate(/*force*/ true);
        }
        input_tensor_on_device = std::move(input_tensor_tilized);
    }
}

Conv2dExecutionPath determine_conv2d_execution_path(
    const ttnn::Tensor& input_tensor, const std::optional<const Conv2dSliceConfig>& slice_config) {
    return determine_conv2d_execution_path(input_tensor.memory_config().is_l1(), slice_config);
}
Conv2dExecutionPath determine_conv2d_execution_path(
    bool input_is_in_L1, const std::optional<const Conv2dSliceConfig>& slice_config) {
    // If slice config explicitly specifies L1_FULL, use L1 path
    if (slice_config.has_value() && slice_config->slice_type == Conv2dSliceConfig::SliceType::L1_FULL) {
        return Conv2dExecutionPath::L1;
    }

    // If no slice config and input is already on device in L1, use L1 path
    if (!slice_config.has_value() && input_is_in_L1) {
        return Conv2dExecutionPath::L1;
    }

    // Otherwise, use DRAM path
    return Conv2dExecutionPath::DRAM;
}

}  // namespace ttnn::operations::conv
