// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pool_utils.hpp"
#include <limits>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/assert.hpp>

#include "tt-metalium/constants.hpp"
#include "tt-metalium/hal.hpp"
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"

namespace ttnn {
namespace operations::pool {
// Return a single bf16 scalar for the pool type in u32 (packed in the least 16 bits)
uint32_t get_bf16_pool_scalar(Pool2DType pool_type, uint32_t kernel_size_hw) {
    float value;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: value = 1.; break;
        case Pool2DType::AVG_POOL2D: value = 1. / (float)kernel_size_hw; break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    return bfloat16(value).to_packed();
}

// Return a single bf16 init value for the pool type in u32 (packed in the least 16 bits)
uint32_t get_bf16_pool_init_value(Pool2DType pool_type) {
    float value;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: value = -std::numeric_limits<float>::infinity(); break;
        case Pool2DType::AVG_POOL2D: value = 0.; break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    return bfloat16(value).to_packed();
}

std::map<std::string, std::string> get_defines(Pool2DType pool_type) {
    std::map<std::string, std::string> defines;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: defines["REDUCE_OP"] = "PoolType::MAX"; break;
        case Pool2DType::AVG_POOL2D: defines["REDUCE_OP"] = "PoolType::SUM"; break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";

    return defines;
}

using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

uint32_t find_closest_largest_divisor(uint32_t num, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    while (num % divisor != 0) {
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
    while ((padded_num - num) >= (int)(padded_num / big_divisor)) {
        divisor = divisor - 1;
        big_divisor = divisor * mult;
        padded_num = tt::round_up(num, big_divisor);
    }
    return divisor;
}

std::optional<ParallelConfig> determine_parallel_config(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t channels,
    uint32_t output_height,
    uint32_t output_width,
    const CoreCoord& compute_grid_size,
    ShardOrientation block_shard_orientation,
    bool enable_channels_padding,
    bool is_shard_height_tile_multiple,
    bool is_shard_width_tile_multiple,
    uint32_t act_block_h_override) {
    uint32_t effective_tile_height = is_shard_height_tile_multiple ? tt::constants::TILE_HEIGHT : 1;
    uint32_t effective_tile_width = tt::constants::TILE_WIDTH;
    uint32_t out_nhw_ntiles = tt::div_up(batch_size * output_height * output_width, effective_tile_height);

    // In case non native activation block height is used, we need to ensure that the amount
    // of work per core in the height dimension is a multiple of the activation block height override.
    uint32_t act_block_h_override_ntiles =
        act_block_h_override == 0 ? 1 : act_block_h_override / tt::constants::TILE_HEIGHT;

    // calculate num_core_nhw and the grid
    uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;

    CoreRangeSet grid;
    if (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        if (channels != tt::constants::TILE_WIDTH / 2 && channels % tt::constants::TILE_WIDTH != 0) {
            return std::nullopt;
        }

        uint32_t num_cores_nhw = find_closest_largest_divisor_with_num_padding_and_mult(
            out_nhw_ntiles, max_num_cores, act_block_h_override_ntiles);
        grid = tt::tt_metal::num_cores_to_corerangeset(num_cores_nhw, compute_grid_size, true);
    } else if (shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        if (!enable_channels_padding && channels % (tt::constants::TILE_WIDTH / 2) != 0) {
            return std::nullopt;
        }

        uint32_t start_divisor =
            block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.x : compute_grid_size.y;
        uint32_t num_cores_nhw = find_closest_largest_divisor_with_num_padding_and_mult(
            out_nhw_ntiles, start_divisor, act_block_h_override_ntiles);

        uint32_t start_divisor_c =
            block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.y : compute_grid_size.x;
        uint32_t tilesize = tt::constants::TILE_WIDTH;
        // regardless of enable_channels_padding, tilesize can be set to 16 as if num of channels
        // does not exceed (number of width cores)*16
        if (channels <= start_divisor_c * tt::constants::TILE_WIDTH / 2) {
            tilesize = tt::constants::TILE_WIDTH / 2;
        } else if (enable_channels_padding) {
            // num of channels exceeds (number of width cores)*16
            tilesize = tt::constants::TILE_WIDTH;
        } else if (channels % tt::constants::TILE_WIDTH == 0) {
            tilesize = tt::constants::TILE_WIDTH;
        } else {
            tilesize = tt::constants::TILE_WIDTH / 2;
        }
        uint32_t channles_ntiles = tt::div_up(channels, tilesize);
        uint32_t num_cores_c = enable_channels_padding
                                   ? find_closest_largest_divisor_with_num_padding(channles_ntiles, start_divisor_c)
                                   : find_closest_largest_divisor(channles_ntiles, start_divisor_c);

        uint32_t cores_x = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_nhw : num_cores_c;
        uint32_t cores_y = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_c : num_cores_nhw;
        CoreRange core_range = CoreRange(CoreCoord({0, 0}), CoreCoord({cores_x - 1, cores_y - 1}));
        grid = CoreRangeSet({core_range});
    } else if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        if (!enable_channels_padding && channels % (tt::constants::TILE_WIDTH / 2) != 0) {
            return std::nullopt;
        }

        uint32_t tilesize = tt::constants::TILE_WIDTH;
        if (channels <= max_num_cores * tt::constants::TILE_WIDTH / 2) {
            tilesize = tt::constants::TILE_WIDTH / 2;
        } else if (enable_channels_padding) {
            tilesize = tt::constants::TILE_WIDTH;
        } else if (channels % tt::constants::TILE_WIDTH == 0) {
            tilesize = tt::constants::TILE_WIDTH;
        } else {
            tilesize = tt::constants::TILE_WIDTH / 2;
        }
        uint32_t channles_ntiles = tt::div_up(channels, tilesize);
        uint32_t num_cores_c = enable_channels_padding
                                   ? find_closest_largest_divisor_with_num_padding(channles_ntiles, max_num_cores)
                                   : find_closest_largest_divisor(channles_ntiles, max_num_cores);

        grid = tt::tt_metal::num_cores_to_corerangeset(num_cores_c, compute_grid_size, true);
    } else {
        TT_THROW("Pool2d supports Height, Block or Width Sharded Layouts but got {}", shard_layout);
    }

    auto shard_orientation = shard_layout == TensorMemoryLayout::BLOCK_SHARDED
                                 ? block_shard_orientation
                                 : ShardOrientation::ROW_MAJOR;  // NOTE: taking ROW_MAJOR as default orientation for
                                                                 // HEIGHT_SHARDED and WIDTH_SHARDED
    ParallelConfig pconfig = {.grid = grid, .shard_scheme = shard_layout, .shard_orientation = shard_orientation};

    return pconfig;
}

uint32_t calculate_L1_usage(
    const Tensor& input,
    const uint32_t kernel_h,
    const uint32_t kernel_w,
    const uint32_t out_h,
    const uint32_t out_w,
    const MemoryConfig& input_memory,
    const MemoryConfig& output_memory,
    Pool2DType pool_type) {
    const auto input_shape = input.get_padded_shape();

    auto in_dtype = input.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input.dtype();
    tt::DataFormat in_df = datatype_to_dataformat_converter(in_dtype);
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = in_nbytes;

    auto pconfig = input.memory_config();
    auto grid_size = input_memory.shard_spec().value().grid.bounding_box().grid_size();
    uint32_t num_shards_c = 0;
    if (pconfig.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_shards_c = 1;
    } else if (pconfig.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        num_shards_c = input_memory.shard_spec().value().grid.num_cores();
    } else if (input_memory.shard_spec().value().orientation == ShardOrientation::COL_MAJOR) {
        num_shards_c = grid_size.y;
    } else {
        num_shards_c = grid_size.x;
    }

    uint32_t kernel_size_hw = kernel_h * kernel_w;  // number of valid rows, to read
    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / num_shards_c / tt::constants::TILE_WIDTH);

    const bool is_partial_tile = (input_shape[3] / num_shards_c) == 16;

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    const bool is_wide_reduction = in_ntiles_c > MAX_TILES_PER_REDUCTION;

    const bool is_large_kernel =
        is_partial_tile ? kernel_size_hw > tt::constants::TILE_HEIGHT / 2 : kernel_size_hw > tt::constants::TILE_HEIGHT;

    // ToDo: enable 32 sticks per tile for reduction for all cases.
    const uint32_t max_rows_for_reduction =
        (!is_partial_tile && !is_large_kernel) ? tt::constants::TILE_HEIGHT : tt::constants::TILE_HEIGHT / 2;
    const bool is_blackhole = tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE;

    if (input_shape[3] < tt::constants::TILE_WIDTH) {
        TT_FATAL(input_shape[3] == 16, "Error");
    }

    uint32_t nblocks = 1;

    // CBs
    uint32_t multi_buffering_factor = 2;

    uint32_t split_reader = 1;

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    uint32_t in_scalar_cb_config_size = in_scalar_cb_npages * in_scalar_cb_pagesize;

    // For avgpool, instantiate and use this CB, which consists of 1s. We don't want to divide
    // twice by kernel size for large kernel case.
    uint32_t in_one_cb_size = 0;
    if (pool_type == Pool2DType::AVG_POOL2D) {
        uint32_t in_one_cb_pagesize = tile_size(in_df);
        uint32_t in_one_cb_npages = 1;
        in_one_cb_size = in_one_cb_pagesize * in_one_cb_npages;
    }

    uint32_t clear_value_cb_size = 0;
    const bool avg_pool_on_blackhole = is_blackhole && pool_type == Pool2DType::AVG_POOL2D;
    if (max_rows_for_reduction == tt::constants::TILE_HEIGHT || avg_pool_on_blackhole) {
        // CB storing just "clear value" (-inf for maxpool, 0 for avgpool)
        // is needed only if we use more then 16 sticks per tile for reduction.
        clear_value_cb_size = tile_size(in_df);
    }

    uint32_t in_cb_sz = 0;
    // uint32_t in_nblocks_c = 1;
    if (is_wide_reduction) {
        in_cb_sz = MAX_TILES_PER_REDUCTION * tt::constants::TILE_HW;
        // in_nblocks_c = std::ceil((float)in_ntiles_c / MAX_TILES_PER_REDUCTION);
    } else {
        in_cb_sz = in_ntiles_c * tt::constants::TILE_HW;
    }

    uint32_t in_cb_page_padded = tt::round_up(
        in_cb_sz,
        tt::constants::TILE_HW);  // NOTE: ceil to tile size since triscs work with tilesize instead of pagesize
    uint32_t in_cb_pagesize = in_nbytes * in_cb_page_padded;
    uint32_t in_cb_npages = multi_buffering_factor * nblocks;
    uint32_t in_cb_config_0_size = in_cb_npages * in_cb_pagesize;
    uint32_t in_cb_config_1_size = 0;

    if (split_reader) {
        in_cb_config_1_size = in_cb_npages * in_cb_pagesize;
    }

    // after reduction
    uint32_t out_cb_pagesize = std::min(tt::constants::TILE_WIDTH, output_memory.shard_spec().value().shape[1]) *
                               out_nbytes;  // there is just one row of channels after each reduction (or 1 block
                                            // of c if its greater than 8 tiles)
    uint32_t out_cb_npages = output_memory.shard_spec().value().shape[0] * in_ntiles_c;
    uint32_t out_cb_config_size = out_cb_npages * out_cb_pagesize;

    uint32_t max_pool_partials_cb_config_size = 0;
    if (is_large_kernel) {
        uint32_t max_pool_partials_cb_pagesize = out_cb_pagesize;
        uint32_t max_pool_partials_cb_npages = nblocks;
        max_pool_partials_cb_config_size = max_pool_partials_cb_npages * max_pool_partials_cb_pagesize;
    }

    return in_scalar_cb_config_size + in_one_cb_size + clear_value_cb_size + in_cb_config_0_size + in_cb_config_1_size +
           out_cb_config_size /* global, involved */
           + max_pool_partials_cb_config_size;
}

std::optional<ParallelConfig> determine_pool_config_for_auto_shard(
    const Tensor& input_tensor,
    const SlidingWindowConfig& sliding_window_config,
    uint32_t channels,
    Pool2DType pool_type) {
    auto batch_size = sliding_window_config.batch_size;
    auto output_shape = sliding_window_config.get_output_shape();
    auto compute_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    struct l1_usage_config {
        uint32_t l1_usage;
        std::optional<ParallelConfig> config;
    };

    auto get_memconfig = [&](const ParallelConfig& parallel_config) {
        uint32_t nhw = batch_size * output_shape[1] * output_shape[2];
        uint32_t out_channel_padded = tt::round_up(
            channels, conv::get_num_cores_channels_from_parallel_config(parallel_config) * tt::constants::TILE_WIDTH);
        return conv::create_sharded_memory_config_from_parallel_config(
            ttnn::Shape({1, 1, nhw, out_channel_padded}), parallel_config, tt::constants::TILE_HEIGHT);
    };

    auto calc_l1_usage_inner = [&](TensorMemoryLayout layout, ShardOrientation orientation) -> l1_usage_config {
        auto input_parallel_config = pool::determine_parallel_config(
            layout,
            batch_size,
            channels,
            output_shape[1],
            output_shape[2],
            compute_grid_size,
            orientation,
            false,
            false);

        if (!input_parallel_config.has_value()) {
            return {std::numeric_limits<uint32_t>::max(), input_parallel_config};
        }
        uint32_t l1_usage = calculate_L1_usage(
            input_tensor,
            sliding_window_config.window_hw.first,
            sliding_window_config.window_hw.second,
            sliding_window_config.get_output_shape()[1],
            sliding_window_config.get_output_shape()[2],
            get_memconfig(input_parallel_config.value()),
            get_memconfig(input_parallel_config.value()),
            pool_type);

        return {l1_usage, input_parallel_config};
    };

    auto l1_config_height = calc_l1_usage_inner(TensorMemoryLayout::HEIGHT_SHARDED, ShardOrientation::ROW_MAJOR);
    auto l1_config_width = calc_l1_usage_inner(TensorMemoryLayout::WIDTH_SHARDED, ShardOrientation::ROW_MAJOR);
    auto l1_config_block = calc_l1_usage_inner(TensorMemoryLayout::BLOCK_SHARDED, ShardOrientation::ROW_MAJOR);

    uint32_t l1_usage_height = l1_config_height.l1_usage;
    uint32_t l1_usage_width = l1_config_width.l1_usage;
    uint32_t l1_usage_block = l1_config_block.l1_usage;

    if (l1_usage_height > l1_usage_width) {
        if (l1_usage_width > l1_usage_block) {
            return l1_config_block.config;
        }
        return l1_config_width.config;
    }
    if (l1_usage_height > l1_usage_block) {
        return l1_config_block.config;
    }
    return l1_config_height.config;
}

}  // namespace operations::pool
}  // namespace ttnn
