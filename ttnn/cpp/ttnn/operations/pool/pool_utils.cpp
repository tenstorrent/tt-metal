// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pool_utils.hpp"
#include <limits>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/assert.hpp>
#include <sys/types.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>

#include "pool2d_utils.hpp"
#include <tt-metalium/buffer_constants.hpp>
#include "tt-metalium/constants.hpp"
#include "tt-metalium/hal.hpp"
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
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"

using namespace tt;
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

    tt::DataFormat in_df = datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat out_df = in_df;
    uint32_t in_nbytes = datum_size(in_df);
    uint32_t out_nbytes = datum_size(out_df);

    auto pconfig = input.memory_config();
    auto grid_size = input.shard_spec().value().grid.bounding_box().grid_size();
    uint32_t num_shards_c = 0;
    if (pconfig.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_shards_c = 1;
    } else if (pconfig.memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        num_shards_c = input.shard_spec().value().grid.num_cores();
    } else if (input.shard_spec().value().orientation == ShardOrientation::COL_MAJOR) {
        num_shards_c = grid_size.y;
    } else {
        num_shards_c = grid_size.x;
    }

    uint32_t in_nbytes_c = input_shape[3] / num_shards_c * in_nbytes;  // row of input (channels)

    tt::DataFormat indices_df =
        tt::DataFormat::RawUInt16;  // datatype_to_dataformat_converter(reader_indices.get_dtype());
    uint32_t indices_nbytes = datum_size(indices_df);

    uint32_t kernel_size_hw = kernel_h * kernel_w;  // number of valid rows, to read
    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / num_shards_c / tt::constants::TILE_WIDTH);

    uint32_t max_rows_for_reduction = 16;

    // Hardware can do reduction of 8 tiles at a time.
    // CB sizes can be restricted to this in case input channels are more than 256 to perform reduction iteratively.
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
    const bool is_large_kernel = kernel_size_hw > max_rows_for_reduction;
    const bool is_wide_reduction = in_ntiles_c > MAX_TILES_PER_REDUCTION;

    uint32_t nblocks = 1;
    // TT_FATAL(nblocks == 1, "Multiple blocks not yet supported");

    // distributing out_hw across the grid
    uint32_t out_nhw_per_core = output_memory.shard_spec.value().shape[0];

    // TODO: support generic nblocks
    TT_FATAL(
        out_nhw_per_core % nblocks == 0,
        "number of sticks per core ({}) should be divisible by nblocks ({})",
        out_nhw_per_core,
        nblocks);

    // CBs
    uint32_t multi_buffering_factor = 2;

    uint32_t split_reader = 1;

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_pagesize = tile_size(in_df);
    uint32_t in_scalar_cb_npages = 1;
    uint32_t in_scalar_cb_config_size = in_scalar_cb_npages * in_scalar_cb_pagesize;

    // incoming data is the input cb instead of raw l1/dram addr
    // this input shard has halo and padding inserted.
    auto raw_in_cb_id = tt::CBIndex::c_2;
    uint32_t raw_in_cb_npages = input_memory.shard_spec.value().shape[0];
    uint32_t raw_in_cb_pagesize = in_nbytes_c;
    uint32_t raw_in_cb_config_size = raw_in_cb_npages * raw_in_cb_pagesize;

    // For avgpool, instantiate and use this CB, which consists of 1s. We don't want to divide
    // twice by kernel size for large kernel case.
    uint32_t in_one_cb_size = 0;
    if (pool_type == Pool2DType::AVG_POOL2D) {
        uint32_t in_one_cb_pagesize = tile_size(in_df);
        uint32_t in_one_cb_npages = 1;
        in_one_cb_size = in_one_cb_pagesize * in_one_cb_npages;
    }

    // reader indices
    uint32_t in_reader_indices_cb_pagesize =
        tt::round_up(out_nhw_per_core * indices_nbytes, 4);  // pagesize needs to be multiple of 4
    uint32_t in_reader_indices_cb_npages = 1;
    uint32_t in_reader_indices_cb_config_size = in_reader_indices_cb_npages * in_reader_indices_cb_pagesize;

    uint32_t in_cb_sz = 0;
    uint32_t in_nblocks_c = 1;
    if (is_wide_reduction) {
        in_cb_sz = MAX_TILES_PER_REDUCTION * tt::constants::TILE_HW;
        in_nblocks_c = std::ceil((float)in_ntiles_c / MAX_TILES_PER_REDUCTION);
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
    uint32_t out_cb_pagesize = std::min(tt::constants::TILE_WIDTH, output_memory.shard_spec.value().shape[1]) *
                               out_nbytes;  // there is just one row of channels after each reduction (or 1 block
                                            // of c if its greater than 8 tiles)
    uint32_t out_cb_npages = output_memory.shard_spec.value().shape[0] * in_ntiles_c;
    uint32_t out_cb_config_size = out_cb_npages * out_cb_pagesize;

    uint32_t max_pool_partials_cb_config_size = 0;
    if (is_large_kernel) {
        uint32_t max_pool_partials_cb_pagesize = out_cb_pagesize;
        uint32_t max_pool_partials_cb_npages = nblocks;
        max_pool_partials_cb_config_size = max_pool_partials_cb_npages * max_pool_partials_cb_pagesize;
    }

    return in_scalar_cb_config_size + raw_in_cb_config_size + in_one_cb_size + in_reader_indices_cb_config_size +
           in_cb_config_0_size + in_cb_config_1_size + out_cb_config_size + max_pool_partials_cb_config_size;
}

sliding_window::ParallelConfig determine_pool_config_for_auto_shard(
    const Tensor& input_tensor,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t channels,
    Pool2DType pool_type) {
    auto batch_size = sliding_window_config.batch_size;
    auto output_shape = sliding_window_config.get_output_shape();
    auto compute_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    struct l1_usage_config {
        uint32_t l1_usage;
        sliding_window::ParallelConfig config;
    };

    auto get_memconfig = [&](const ParallelConfig& parallel_config) {
        uint32_t nhw = batch_size * output_shape[1] * output_shape[2];
        uint32_t out_channeal_padded = tt::round_up(
            channels, conv::get_num_cores_channels_from_parallel_config(parallel_config) * tt::constants::TILE_WIDTH);
        return conv::create_sharded_memory_config_from_parallel_config(
            ttnn::Shape({1, 1, nhw, out_channeal_padded}), parallel_config, tt::constants::TILE_HEIGHT);
    };

    auto calc_l1_usage_inner = [&](TensorMemoryLayout layout, ShardOrientation orientation) -> l1_usage_config {
        auto input_parallel_config = conv::determine_parallel_config(
            layout,
            batch_size,
            channels,
            output_shape[1],
            output_shape[2],
            channels,
            compute_grid_size,
            orientation,
            false,
            false);
        auto output_parallel_config =
            conv::determine_output_parallel_config(input_parallel_config, compute_grid_size, channels, false);

        uint32_t l1_usage = calculate_L1_usage(
            input_tensor,
            sliding_window_config.window_hw.first,
            sliding_window_config.window_hw.second,
            sliding_window_config.get_output_shape()[1],
            sliding_window_config.get_output_shape()[2],
            get_memconfig(input_parallel_config),
            get_memconfig(output_parallel_config),
            pool_type);

        return {l1_usage, input_parallel_config};
    };

    auto l1_config_height = calc_l1_usage_inner(TensorMemoryLayout::HEIGHT_SHARDED, ShardOrientation::ROW_MAJOR);
    auto l1_config_width = calc_l1_usage_inner(TensorMemoryLayout::WIDTH_SHARDED, ShardOrientation::ROW_MAJOR);
    auto l1_config_block = calc_l1_usage_inner(TensorMemoryLayout::BLOCK_SHARDED, ShardOrientation::COL_MAJOR);

    uint32_t l1_usage_height = l1_config_height.l1_usage;
    uint32_t l1_usage_width = l1_config_width.l1_usage;
    uint32_t l1_usage_block = l1_config_block.l1_usage;

    uint32_t ncores_block = l1_config_block.config.grid.num_cores();

    uint32_t winning_l1_usage = l1_usage_height;
    auto winning_config = l1_config_height.config;
    // Make sure that BS not only has smaller size but provides at least some slicing along the channels.
    // In case we have BS that would slice the tensor only along the HS conv2d code would fail later on.
    if (l1_usage_block < l1_usage_height && ncores_block > compute_grid_size.x) {
        winning_l1_usage = l1_usage_block;
        winning_config = l1_config_block.config;
    }
    if (l1_usage_width < winning_l1_usage) {
        winning_config = l1_config_width.config;
    }

    return winning_config;
}

}  // namespace operations::pool
}  // namespace ttnn
