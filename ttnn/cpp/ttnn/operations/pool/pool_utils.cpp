// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "pool_utils.hpp"
#include <limits>
#include <tt-metalium/bfloat16.hpp>
#include <tt_stl/assert.hpp>

#include "tt-metalium/constants.hpp"

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
    // TODO: #27672: Truncation should be removed once we figure a root cause of regression without it
    return std::bit_cast<uint16_t>(bfloat16::truncate(value)) << 16;
}

// Return a single bf16 init value for the pool type in u32 (packed in the least 16 bits)
uint32_t get_bf16_pool_init_value(Pool2DType pool_type) {
    float value;
    switch (pool_type) {
        case Pool2DType::MAX_POOL2D: value = -std::numeric_limits<float>::infinity(); break;
        case Pool2DType::AVG_POOL2D: value = 0.; break;
        default: TT_FATAL(false, "Unsupported pool operation type");
    }
    // TODO: #27672: Truncation should be removed once we figure a root cause of regression without it
    return std::bit_cast<uint16_t>(bfloat16::truncate(value));
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
           ((!ceil_mode || (ceil_h == 0 && ceil_w == 0)) && (count_include_pad || (pad_h == 0 && pad_w == 0)));
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

    return pconfig;
}

FactoryParameters get_factory_parameters(
    uint32_t num_shards_c,
    const DataType& input_dtype,
    const DataType& output_dtype,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t in_channels,
    Pool2DType pool_type,
    bool return_indices,
    const Layout& output_layout) {
    uint32_t multi_buffering_factor = 2;
    bool split_reader = true;
    TT_FATAL((split_reader && return_indices) || !return_indices, "split_reader must be true for MPWI");

    auto dtype = input_dtype == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_dtype;
    tt::DataFormat data_format = datatype_to_dataformat_converter(dtype);
    tt::DataFormat index_format = datatype_to_dataformat_converter(DataType::UINT16);
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output_dtype);
    uint32_t nbytes = datum_size(data_format);
    uint32_t index_nbytes = datum_size(index_format);

    uint32_t kernel_size_hw = kernel_h * kernel_w;  // number of valid rows, to read
    // for medium kernels with sizes 16 < kernel_size_hw < 32 we tilize an entire tile even if some rows are unused,
    // so the in_cb height must be equal to the TILE_HEIGHT, but for kernels spanning only one face we set the
    // face_r_dim to only tilize the necessary number of rows, thus we can make the in_cb height smaller
    uint32_t num_tilized_rows =
        kernel_size_hw <= tt::constants::FACE_WIDTH ? kernel_size_hw : tt::constants::TILE_HEIGHT;
    uint32_t in_ntiles_c = (uint32_t)std::ceil((float)in_channels / num_shards_c / tt::constants::TILE_WIDTH);
    // For TILE_LAYOUT output, we need to align to TILE_WIDTH instead of FACE_WIDTH
    uint32_t effective_tile_width_for_output =
        (output_layout == Layout::TILE) ? tt::constants::TILE_WIDTH : tt::constants::FACE_WIDTH;
    uint32_t out_ntiles_c = (uint32_t)std::ceil((float)in_channels / num_shards_c / effective_tile_width_for_output);

    bool is_avg_pool = pool_type == Pool2DType::AVG_POOL2D;
    const bool last_tile_is_partial =
        (in_channels / num_shards_c) % tt::constants::TILE_WIDTH != 0 &&
        (in_channels / num_shards_c) % tt::constants::TILE_WIDTH <= tt::constants::FACE_WIDTH;
    const uint32_t max_rows_for_reduction =
        !last_tile_is_partial ? tt::constants::TILE_HEIGHT : tt::constants::TILE_HEIGHT / 2;
    const bool is_large_kernel = kernel_size_hw > max_rows_for_reduction;
    if (return_indices) {
        TT_FATAL(
            !is_avg_pool && !is_large_kernel,
            "Currently only small full width max pool is supported with return_indices");
    }
    const uint32_t MAX_TILES_PER_REDUCTION = return_indices ? 1 : (is_avg_pool && is_large_kernel) ? 4 : 8;
    const bool is_wide_reduction = in_ntiles_c > MAX_TILES_PER_REDUCTION;

    return FactoryParameters{
        .multi_buffering_factor = multi_buffering_factor,
        .split_reader = split_reader,
        .nbytes = nbytes,
        .index_nbytes = index_nbytes,
        .data_format = data_format,
        .index_format = index_format,
        .output_data_format = output_data_format,
        .in_ntiles_c = in_ntiles_c,
        .out_ntiles_c = out_ntiles_c,
        .is_avg_pool = is_avg_pool,
        .max_rows_for_reduction = max_rows_for_reduction,
        .is_large_kernel = is_large_kernel,
        .MAX_TILES_PER_REDUCTION = MAX_TILES_PER_REDUCTION,
        .is_wide_reduction = is_wide_reduction,
        .num_tilized_rows = num_tilized_rows,
    };
}

uint32_t calculate_L1_usage(
    DataType input_dtype,
    uint32_t in_channels,
    uint32_t pad_h,
    uint32_t pad_w,
    uint32_t ceil_pad_h,
    uint32_t ceil_pad_w,
    bool ceil_mode,
    bool return_indices,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t /*out_h*/,
    uint32_t /*out_w*/,
    const MemoryConfig& input_memory,
    const MemoryConfig& output_memory,
    Pool2DType pool_type,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    const Layout& output_layout,
    const DataType& output_dtype,
    bool config_tensor_in_dram) {
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

    FactoryParameters params = get_factory_parameters(
        num_shards_c,
        input_dtype,
        output_dtype,
        kernel_h,
        kernel_w,
        in_channels,
        pool_type,
        return_indices,
        output_layout);

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
    if (return_indices) {
        in_cb_sz = params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    } else {
        if (params.is_wide_reduction) {
            in_cb_sz = params.MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH * params.num_tilized_rows;
        } else {
            in_cb_sz = params.in_ntiles_c * tt::constants::TILE_WIDTH * params.num_tilized_rows;
        }
    }

    uint32_t in_cb_page_padded = tt::round_up(in_cb_sz, tt::constants::TILE_HW);
    uint32_t in_cb_pagesize = params.nbytes * in_cb_page_padded;
    uint32_t in_cb_npages = params.multi_buffering_factor;
    uint32_t in_cb_config_0_size = in_cb_npages * in_cb_pagesize;
    uint32_t in_cb_config_1_size = 0;

    if (params.split_reader) {
        in_cb_config_1_size = in_cb_npages * in_cb_pagesize;
    }

    uint32_t total_mpwi_cb_size = 0;
    if (return_indices) {
        // Add tile temporary CBs for return_indices
        uint32_t tile_elems = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        uint32_t idx_tile_size = params.index_nbytes * tile_elems * 1;  // 1 page
        uint32_t data_tile_size = params.nbytes * tile_elems * 1;       // 1 page
        // 1 data sized tile (pack_tmp_cb) and 5 index sized tiles (in_idx, pack_idx_tmp, right_inc, down_left_wrap_inc,
        // up_left_wrap_inc)
        total_mpwi_cb_size = (5 * idx_tile_size) + data_tile_size;
    }

    uint32_t out_cb_pagesize;
    uint32_t out_cb_npages;
    const bool is_output_tiled = output_layout == Layout::TILE;

    if (is_output_tiled) {
        out_cb_pagesize = tt::tile_size(datatype_to_dataformat_converter(output_dtype));
        out_cb_npages = output_memory.shard_spec().value().shape[0] * output_memory.shard_spec().value().shape[1] /
                        tt::constants::TILE_HW;
    } else {
        out_cb_pagesize =
            std::min(static_cast<uint32_t>(tt::constants::FACE_WIDTH), output_memory.shard_spec().value().shape[1]) *
            params.nbytes;
        out_cb_npages = output_memory.shard_spec().value().shape[0] * params.out_ntiles_c;
    }
    uint32_t out_cb_config_size = out_cb_npages * out_cb_pagesize;

    uint32_t pre_tilize_cb_size = 0;

    if (is_output_tiled) {
        const uint32_t pre_tilize_cb_pagesize = params.in_ntiles_c * tt::constants::TILE_HW * params.nbytes;
        const uint32_t pre_tilize_cb_npages = 1;
        pre_tilize_cb_size = pre_tilize_cb_pagesize * pre_tilize_cb_npages;
    }

    uint32_t out_idx_cb_config_size = 0;
    if (return_indices) {
        uint32_t out_cb_pagesize =
            std::min(static_cast<uint32_t>(tt::constants::FACE_WIDTH), output_memory.shard_spec().value().shape[1]) *
            params.index_nbytes;
        out_idx_cb_config_size = out_cb_npages * out_cb_pagesize;
    }
    uint32_t config_tensor_l1_CB_size = 0;
    if (config_tensor_in_dram) {
        auto output_shard_shape = output_memory.shard_spec().value().shape;
        config_tensor_l1_CB_size =
            (output_shard_shape[0] * 6) + 2;  // Worst case of 6 Bytes per output elem for reader indices
        if (!one_scalar_per_core) {
            config_tensor_l1_CB_size +=
                output_shard_shape[0] * 6;  // Additional 6 Bytes per output elem for avg pool scalar config tensor
        }
    }
    log_trace(
        tt::LogOp,
        "L1 Usage Breakdown: in_scalar_cb_size_0 = {}, in_scalar_cb_size_1 = {}, clear_value_cb_size = {}, "
        "in_cb_config_0_size = {}, in_cb_config_1_size = {}, total_mpwi_cb_size = {}, pre_tilize_cb_size = {}, "
        "config_tensor_l1_CB_size = {} "
        "out_cb_config_size = {}, out_idx_cb_config_size = {}",
        in_scalar_cb_size_0,
        in_scalar_cb_size_1,
        clear_value_cb_size,
        in_cb_config_0_size,
        in_cb_config_1_size,
        total_mpwi_cb_size,
        pre_tilize_cb_size,
        config_tensor_l1_CB_size,
        sliding_window::align_buffer(out_cb_config_size),
        sliding_window::align_buffer(out_idx_cb_config_size));

    return in_scalar_cb_size_0 + in_scalar_cb_size_1 + clear_value_cb_size + in_cb_config_0_size + in_cb_config_1_size +
           total_mpwi_cb_size + pre_tilize_cb_size + config_tensor_l1_CB_size +
           sliding_window::align_buffer(out_cb_config_size) + sliding_window::align_buffer(out_idx_cb_config_size);
}

std::optional<ParallelConfig> determine_pool_config_for_auto_shard(
    const DataType& input_dtype,
    const Layout& input_layout,
    CoreCoord compute_grid_size,
    const SlidingWindowConfig& sliding_window_config,
    uint32_t channels,
    Pool2DType pool_type,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    bool return_indices,
    const Layout& output_layout,
    const DataType& output_dtype,
    bool config_tensor_in_dram) {
    uint32_t batch_size = sliding_window_config.batch_size;
    auto output_shape = sliding_window_config.get_output_shape();

    struct l1_usage_config {
        uint32_t l1_usage{};
        std::optional<ParallelConfig> config;
    };

    auto get_memconfig = [&](const ParallelConfig& parallel_config) {
        uint32_t nhw = batch_size * output_shape[1] * output_shape[2];
        uint32_t out_channel_padded = tt::round_up(
            channels, conv::get_num_cores_channels_from_parallel_config(parallel_config) * tt::constants::TILE_WIDTH);
        return conv::create_sharded_memory_config_from_parallel_config(
            ttnn::Shape({1, 1, nhw, out_channel_padded}), parallel_config, tt::constants::TILE_HEIGHT);
    };

    bool is_in_tiled = input_layout == ttnn::TILE_LAYOUT;
    bool is_out_tiled = output_layout == ttnn::TILE_LAYOUT;

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
            is_out_tiled,
            is_in_tiled || is_out_tiled,  // if input/output is tiled we need the shard width to be a tile multiple,
            0);

        if (!input_parallel_config.has_value()) {
            return {std::numeric_limits<uint32_t>::max(), input_parallel_config};
        }
        uint32_t l1_usage = calculate_L1_usage(
            input_dtype,
            sliding_window_config.channels,
            sliding_window_config.get_pad_h(),
            sliding_window_config.get_pad_w(),
            sliding_window_config.get_ceil_pad_h(),
            sliding_window_config.get_ceil_pad_w(),
            sliding_window_config.ceil_mode,
            return_indices,
            sliding_window_config.window_hw.first,
            sliding_window_config.window_hw.second,
            sliding_window_config.get_output_shape()[1],
            sliding_window_config.get_output_shape()[2],
            get_memconfig(input_parallel_config.value()),
            get_memconfig(input_parallel_config.value()),
            pool_type,
            count_include_pad,
            divisor_override,
            output_layout,
            output_dtype,
            config_tensor_in_dram);

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

    // Support both (1, 1, nhw, c) and (n, h, w, c) formats
    bool is_flattened_format =
        (input_shape[0] == 1 && input_shape[1] == 1 && input_shape[2] == nhw && input_shape[3] == channels);

    if (is_in_tiled && is_flattened_format) {
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
        pad_top <= kernel_size[0] / 2 && pad_bottom <= kernel_size[0] / 2 && pad_left <= kernel_size[1] / 2,
        "Pool2D: Padding ({}, {}, {}) should not exceed half of kernel size ({}, {})",
        pad_top,
        pad_bottom,
        pad_left,
        kernel_size[0],
        kernel_size[1]);

    // ensure effective kernel size (with dilation) doesn't exceed padded input
    uint32_t effective_kernel_h = (dilation_h * (kernel_size[0] - 1)) + 1;
    uint32_t effective_kernel_w = (dilation_w * (kernel_size[1] - 1)) + 1;
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
