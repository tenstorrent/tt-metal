// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "parallel_config_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::conv {

using sliding_window::ParallelConfig;


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

uint32_t get_num_cores_nhw_from_parallel_config(const ParallelConfig& pconfig) {
    return get_num_cores_nhw(pconfig.grid, pconfig.shard_scheme, pconfig.shard_orientation);
}


uint32_t get_num_cores_channels_from_parallel_config(const ParallelConfig& pconfig) {
    return get_num_cores_channels(pconfig.grid, pconfig.shard_scheme, pconfig.shard_orientation);
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


ttnn::Shape flatten_4d_shape(const ttnn::Shape& input_shape) {
    TT_FATAL(input_shape.size() == 4, "Expected 4D shape");
    const uint32_t nhw = input_shape[0] * input_shape[1] * input_shape[2];
    const uint32_t channels = input_shape[3];
    return ttnn::Shape{1, 1, nhw, channels};
}

}  // namespace ttnn::operations::conv
