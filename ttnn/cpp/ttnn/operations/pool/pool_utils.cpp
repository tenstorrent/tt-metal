// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "pool_utils.hpp"
#include <limits>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/assert.hpp>

#include "tt-metalium/constants.hpp"
#include "tt-metalium/hal.hpp"

#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
namespace ttnn::operations::pool {
// Return a single bf16 scalar for the pool type in u32 (packed in the least 16 bits)
// For the maxpool it is 1, for the avg pool it is 1/kernel_size or the divisor override used to initialize compile
// time argument sent to kernels. If there are multiple scalars needed call get_bf16_avg_pool_config_scalars
uint32_t get_bf16_pool_scalar(
    Pool2DType pool_type, uint32_t kernel_h, uint32_t kernel_w, std::optional<int32_t> divisor_override) {
    float value;

    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: value = 1.; break;
        case Pool2DType::AVG_POOL2D:
            if (divisor_override.has_value()) {
                value = 1. / (float)divisor_override.value();
            } else {
                value = 1. / (float)(kernel_h * kernel_w);
            }
            break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    return bfloat16(value).to_packed() << 16;
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

bool is_pool_op_one_scalar_per_core(
    Pool2DType pool_type,
    bool ceil_mode,
    uint32_t ceil_h,
    uint32_t ceil_w,
    bool count_include_pad,
    uint32_t pad_h,
    uint32_t pad_w,
    std::optional<int32_t> divisor_override) {
    return pool_type != Pool2DType::AVG_POOL2D || divisor_override.has_value() ||
           ((ceil_mode == false || (ceil_h == 0 && ceil_w == 0)) &&
            (count_include_pad == true || (pad_h == 0 && pad_w == 0)));
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

std::optional<ParallelConfig> determine_valid_parallel_config(
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
    if (shard_layout != TensorMemoryLayout::HEIGHT_SHARDED && shard_layout != TensorMemoryLayout::BLOCK_SHARDED &&
        shard_layout != TensorMemoryLayout::WIDTH_SHARDED) {
        TT_THROW("Pool2d supports Height, Block or Width Sharded Layouts but got {}", shard_layout);
    }
    auto pconfig = conv::determine_parallel_config(
        shard_layout,
        batch_size,
        channels,
        output_height,
        output_width,
        channels,
        tt::constants::TILE_WIDTH,
        compute_grid_size,
        block_shard_orientation,
        enable_channels_padding,
        is_shard_height_tile_multiple,
        is_shard_width_tile_multiple,
        act_block_h_override);

    // pooling can accept any height and either a tile multiple or half a tile for width.
    if (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t num_cores_c = 1;
        uint32_t c_per_core = channels / num_cores_c;
        if (c_per_core != 16 && c_per_core % tt::constants::TILE_WIDTH != 0) {
            return std::nullopt;
        }
    } else if (shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        auto grid_x = pconfig.grid.ranges()[0].end_coord.x - pconfig.grid.ranges()[0].start_coord.x + 1;
        auto grid_y = pconfig.grid.ranges()[0].end_coord.y - pconfig.grid.ranges()[0].start_coord.y + 1;
        uint32_t num_cores_c = block_shard_orientation == ShardOrientation::COL_MAJOR ? grid_y : grid_x;
        uint32_t c_per_core = channels / num_cores_c;
        if (c_per_core != 16 && c_per_core % tt::constants::TILE_WIDTH != 0) {
            return std::nullopt;
        }
    } else if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t num_cores_c = pconfig.grid.num_cores();
        uint32_t c_per_core = channels / num_cores_c;
        if (c_per_core != 16 && c_per_core % tt::constants::TILE_WIDTH != 0) {
            return std::nullopt;
        }
    }

    return pconfig;
}

FactoryParameters get_factory_parameters(
    uint32_t num_shards_c, const Tensor& input, uint32_t kernel_h, uint32_t kernel_w, Pool2DType pool_type) {
    uint32_t multi_buffering_factor = 2;
    bool split_reader = true;

    const auto& input_shape = input.padded_shape();

    auto dtype = input.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input.dtype();
    tt::DataFormat data_format = datatype_to_dataformat_converter(dtype);
    uint32_t nbytes = datum_size(data_format);

    uint32_t kernel_size_hw = kernel_h * kernel_w;  // number of valid rows, to read
    // for medium kernels with sizes 16 < kernel_size_hw < 32 we tilize an entire tile even if some rows are unused,
    // so the in_cb height must be equal to the TILE_HEIGHT, but for kernels spanning only one face we set the
    // face_r_dim to only tilize the necessary number of rows, thus we can make the in_cb height smaller
    uint32_t num_tilized_rows = kernel_size_hw <= 16 ? kernel_size_hw : tt::constants::TILE_HEIGHT;
    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)input_shape[3] / num_shards_c / tt::constants::TILE_WIDTH);

    bool is_avg_pool = pool_type == Pool2DType::AVG_POOL2D;
    const bool is_partial_tile = (input_shape[3] / num_shards_c) == 16;
    const uint32_t max_rows_for_reduction =
        !is_partial_tile ? tt::constants::TILE_HEIGHT : tt::constants::TILE_HEIGHT / 2;
    const bool is_large_kernel = kernel_size_hw > max_rows_for_reduction;
    const uint32_t MAX_TILES_PER_REDUCTION = (is_avg_pool && is_large_kernel) ? 4 : 8;
    const bool is_wide_reduction = in_ntiles_c > MAX_TILES_PER_REDUCTION;

    return FactoryParameters{
        .multi_buffering_factor = multi_buffering_factor,
        .split_reader = split_reader,
        .nbytes = nbytes,
        .data_format = data_format,
        .in_ntiles_c = in_ntiles_c,
        .is_avg_pool = is_avg_pool,
        .max_rows_for_reduction = max_rows_for_reduction,
        .is_large_kernel = is_large_kernel,
        .MAX_TILES_PER_REDUCTION = MAX_TILES_PER_REDUCTION,
        .is_wide_reduction = is_wide_reduction,
        .num_tilized_rows = num_tilized_rows};
}

uint32_t calculate_L1_usage(
    const Tensor& input,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t ceil_pad_h,
    uint32_t ceil_pad_w,
    bool ceil_mode,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t out_h,
    uint32_t out_w,
    const MemoryConfig& input_memory,
    const MemoryConfig& output_memory,
    Pool2DType pool_type,
    bool count_include_pad,
    std::optional<int32_t> divisor_override) {
    const auto grid_size = input_memory.shard_spec().value().grid.bounding_box().grid_size();
    uint32_t num_shards_c = 0;
    if (input_memory.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_shards_c = 1;
    } else if (input_memory.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        num_shards_c = input_memory.shard_spec().value().grid.num_cores();
    } else if (input_memory.shard_spec().value().orientation == ShardOrientation::COL_MAJOR) {
        num_shards_c = grid_size.y;
    } else {
        num_shards_c = grid_size.x;
    }

    FactoryParameters params = get_factory_parameters(num_shards_c, input, kernel_h, kernel_w, pool_type);

    bool one_scalar_per_core = is_pool_op_one_scalar_per_core(
        pool_type, ceil_mode, ceil_pad_h, ceil_pad_w, count_include_pad, pad_h, pad_w, divisor_override);

    // scalar CB as coefficient of reduce
    uint32_t in_scalar_cb_pagesize = tt::constants::TILE_HW * params.nbytes;
    uint32_t in_scalar_cb_npages = params.multi_buffering_factor;
    uint32_t in_scalar_cb_size_0 = in_scalar_cb_npages * in_scalar_cb_pagesize;
    uint32_t in_scalar_cb_size_1 = 0;

    if (pool_type == Pool2DType::AVG_POOL2D && params.split_reader && !one_scalar_per_core) {
        in_scalar_cb_size_1 = in_scalar_cb_npages * in_scalar_cb_pagesize;
    }

    uint32_t clear_value_cb_size = tt::constants::TILE_HW * params.nbytes;

    uint32_t in_cb_sz = 0;
    if (params.is_wide_reduction) {
        in_cb_sz = params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * params.num_tilized_rows;
    } else {
        in_cb_sz = params.in_ntiles_c * tt::constants::TILE_WIDTH * params.num_tilized_rows;
    }

    uint32_t in_cb_page_padded = tt::round_up(in_cb_sz, tt::constants::TILE_HW);
    uint32_t in_cb_pagesize = params.nbytes * in_cb_page_padded;
    uint32_t in_cb_npages = params.multi_buffering_factor;
    uint32_t in_cb_config_0_size = in_cb_npages * in_cb_pagesize;
    uint32_t in_cb_config_1_size = 0;

    if (params.split_reader) {
        in_cb_config_1_size = in_cb_npages * in_cb_pagesize;
    }

    // after reduction
    uint32_t out_cb_pagesize =
        std::min(tt::constants::TILE_WIDTH, output_memory.shard_spec().value().shape[1]) * params.nbytes;
    uint32_t out_cb_npages = output_memory.shard_spec().value().shape[0] * params.in_ntiles_c;
    uint32_t out_cb_config_size = out_cb_npages * out_cb_pagesize;

    uint32_t alignment_bytes = tt::tt_metal::hal::get_dram_alignment();
    auto align = [alignment_bytes](uint32_t size) {
        uint32_t factor = (size + alignment_bytes - 1) / alignment_bytes;
        return factor * alignment_bytes;
    };

    return in_scalar_cb_size_0 + in_scalar_cb_size_1 + clear_value_cb_size + in_cb_config_0_size + in_cb_config_1_size +
           align(out_cb_config_size) /* global, involved */;
}

std::optional<ParallelConfig> determine_pool_config_for_auto_shard(
    const Tensor& input_tensor,
    const SlidingWindowConfig& sliding_window_config,
    uint32_t channels,
    Pool2DType pool_type,
    bool count_include_pad,
    std::optional<int32_t> divisor_override) {
    uint32_t batch_size = sliding_window_config.batch_size;
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

    bool is_in_tiled = input_tensor.layout() == ttnn::TILE_LAYOUT;

    auto calc_l1_usage_inner = [&](TensorMemoryLayout layout, ShardOrientation orientation) -> l1_usage_config {
        auto input_parallel_config = pool::determine_valid_parallel_config(
            layout,
            batch_size,
            channels,
            output_shape[1],
            output_shape[2],
            compute_grid_size,
            orientation,
            false,
            false,
            is_in_tiled,  // if input is tiled we need the shard width to be a tile multiple,
            0);

        if (!input_parallel_config.has_value()) {
            return {std::numeric_limits<uint32_t>::max(), input_parallel_config};
        }
        uint32_t l1_usage = calculate_L1_usage(
            input_tensor,
            sliding_window_config.get_pad_h(),
            sliding_window_config.get_pad_w(),
            sliding_window_config.get_ceil_pad_h(),
            sliding_window_config.get_ceil_pad_w(),
            sliding_window_config.ceil_mode,
            sliding_window_config.window_hw.first,
            sliding_window_config.window_hw.second,
            sliding_window_config.get_output_shape()[1],
            sliding_window_config.get_output_shape()[2],
            get_memconfig(input_parallel_config.value()),
            get_memconfig(input_parallel_config.value()),
            pool_type,
            count_include_pad,
            divisor_override);

        return {.l1_usage = l1_usage, .config = input_parallel_config};
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

// pool specific validations are done in validate_pool2d, but we want to validate basic inputs to ensure
// they are sensical to avoid problems in sliding window config, halo and other setup procedures
void validate_input_params(
    const Tensor& input_tensor,
    uint32_t batch_size,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t channels,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    uint32_t pad_top,
    uint32_t pad_bottom,
    uint32_t pad_left,
    uint32_t pad_right,
    uint32_t dilation_h,
    uint32_t dilation_w,
    bool is_in_tiled) {
    // dimension value validation
    TT_FATAL(batch_size > 0, "Pool2D: Batch size must be greater than 0, got {}", batch_size);
    TT_FATAL(input_h > 0, "Pool2D: Input height must be greater than 0, got {}", input_h);
    TT_FATAL(input_w > 0, "Pool2D: Input width must be greater than 0, got {}", input_w);
    TT_FATAL(channels > 0, "Pool2D: Channels must be greater than 0, got {}", channels);

    // tensor shape validation against provided NHWC dimensions
    const uint32_t nhw = batch_size * input_h * input_w;
    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(
        input_shape[0] == 1 && input_shape[1] == 1 && input_shape[2] == nhw && input_shape[3] == channels,
        "Input tensor shape {} does not match expected shape (1, 1, {}, {})",
        input_shape,
        nhw,
        channels);

    if (is_in_tiled) {
        const uint32_t padded_channels = tt::round_up(channels, tt::constants::TILE_WIDTH);
        const uint32_t padded_nhw = tt::round_up(nhw, tt::constants::TILE_HEIGHT);
        const auto& padded_input_shape = input_tensor.padded_shape();
        TT_FATAL(
            padded_input_shape[0] == 1 && padded_input_shape[1] == 1 && padded_input_shape[2] == padded_nhw &&
                padded_input_shape[3] == padded_channels,
            "Padded input tensor shape {} does not match expected shape (1, 1, {}, {})",
            padded_input_shape,
            padded_nhw,
            padded_channels);
    }

    // kernel size validation
    TT_FATAL(
        kernel_size[0] > 0 && kernel_size[1] > 0,
        "Pool2D: Kernel size must be greater than 0 in both dimensions, got ({}, {})",
        kernel_size[0],
        kernel_size[1]);

    // stride validation
    TT_FATAL(
        stride[0] > 0 && stride[1] > 0,
        "Pool2D: Stride must be greater than 0 in both dimensions, got ({}, {})",
        stride[0],
        stride[1]);

    // dilation validation
    TT_FATAL(
        dilation_h > 0 && dilation_w > 0,
        "Pool2D: Dilation must be greater than 0 in both dimensions, got ({}, {})",
        dilation_h,
        dilation_w);

    // check that padding is not excessive (should not be more than half the kernel size)
    TT_FATAL(
        pad_top <= kernel_size[0] / 2 && pad_bottom <= kernel_size[0] / 2 && pad_left <= kernel_size[1] / 2 &&
            pad_right <= kernel_size[1] / 2,
        "Pool2D: Padding ({}, {}, {}, {}) should not exceed half of kernel size ({}, {})",
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        kernel_size[0],
        kernel_size[1]);

    // ensure effective kernel size (with dilation) doesn't exceed padded input
    uint32_t effective_kernel_h = dilation_h * (kernel_size[0] - 1) + 1;
    uint32_t effective_kernel_w = dilation_w * (kernel_size[1] - 1) + 1;
    uint32_t padded_input_h = input_h + pad_top + pad_bottom;
    uint32_t padded_input_w = input_w + pad_left + pad_right;
    TT_FATAL(
        effective_kernel_h <= padded_input_h && effective_kernel_w <= padded_input_w,
        "Pool2D: Effective kernel size ({}, {}) cannot exceed padded input size ({}, {})",
        effective_kernel_h,
        effective_kernel_w,
        padded_input_h,
        padded_input_w);
}

}  // namespace ttnn::operations::pool
