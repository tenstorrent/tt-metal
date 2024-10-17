// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d.hpp"
#include <sys/types.h>
#include <cstdint>

#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/pool/downsample/device/downsample_op.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt;
namespace ttnn {
namespace operations::conv {
using sliding_window::SlidingWindowConfig;
using sliding_window::ParallelConfig;

namespace conv2d {

uint32_t find_closest_largest_divisor(uint32_t num, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    while (num % divisor != 0) divisor = divisor - 1;
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

uint32_t find_closest_common_largest_divisor(uint32_t num1, uint32_t num2, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    while (num1 % divisor != 0 or num2 % divisor != 0) divisor = divisor - 1;
    return divisor;
}

// Converts convolution weights to tilized 2d matrix layout.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout(
    Tensor conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype){
        return tt::tt_metal::convert_conv_weight_tensor_to_tiled_layout(conv_weight_tensor, in1_block_h, in1_block_w, output_dtype);
    }

// Converts convolution weights to tilized 2d matrix layout with special block height padding
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(
    Tensor conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype){
        return tt::tt_metal::convert_conv_weight_tensor_to_special_padding_tiled_layout(conv_weight_tensor, in1_block_h, in1_block_w, output_dtype);
    }

// Converts convolution weights to grouped layout with padded zeros
Tensor convert_conv_weight_tensor_to_grouped_layout(Tensor conv_weight_tensor, uint32_t num_groups, DataType output_dtype){
       return tt::tt_metal::convert_conv_weight_tensor_to_grouped_layout(conv_weight_tensor, num_groups, output_dtype);
}

template <typename T>
ParallelConfig determine_parallel_config(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    T * device,
    ShardOrientation block_shard_orientation,
    bool is_out_tiled) {

    uint32_t conv_out_2d_matrix_height = batch_size * output_height * output_width;
    // pad height to 32
    conv_out_2d_matrix_height = tt::round_up(conv_out_2d_matrix_height, 32);
    uint32_t conv_out_2d_matrix_height_ntiles = 0;
    uint32_t conv_out_2d_matrix_width_ntiles = 0;
    if (is_out_tiled) {
        conv_out_2d_matrix_height_ntiles = (int)(conv_out_2d_matrix_height / 32);
        conv_out_2d_matrix_width_ntiles = (int)(tt::round_up(output_channels, 32) / 32);
    } else {
        conv_out_2d_matrix_height_ntiles = conv_out_2d_matrix_height;
        conv_out_2d_matrix_width_ntiles = output_channels;
    }
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    std::vector<uint32_t> device_grid_size = {
        (uint32_t)compute_with_storage_grid_size.x, (uint32_t)compute_with_storage_grid_size.y};
    CoreCoord device_grid_size_coord = {
        (std::size_t)compute_with_storage_grid_size.x, (std::size_t)compute_with_storage_grid_size.y};
    uint32_t max_num_cores = device_grid_size[0] * device_grid_size[1];

    auto calculate_num_cores_nhw = [&]() {

        if(shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            return find_closest_largest_divisor(conv_out_2d_matrix_height_ntiles, max_num_cores);
        } else if(shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            return find_closest_largest_divisor_with_num_padding(conv_out_2d_matrix_height_ntiles, device_grid_size[0]);
        } else if(shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            return 1u;
        }

        TT_THROW("Conv2d supports Height, Block or Width Sharded Layouts but got {}",shard_layout);
        return 0u;
    };

    auto calculate_grid = [&](uint32_t num_cores_nhw) {
        if (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            CoreRangeSet grid = num_cores_to_corerange_set(num_cores_nhw, device_grid_size_coord, true);
            return grid;

        } else if(shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            uint32_t num_cores_channels = find_closest_common_largest_divisor(
                conv_out_2d_matrix_width_ntiles, std::ceil((double)input_channels / (double)tt::constants::TILE_WIDTH), max_num_cores);
             log_debug(LogOp, "Num cores for Width Sharding : {}", num_cores_channels);
            CoreRangeSet grid = num_cores_to_corerange_set(num_cores_channels, device_grid_size_coord, true);
            return grid;

        } else if(shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            uint32_t total_cores_for_channels =
                block_shard_orientation == ShardOrientation::COL_MAJOR ? device_grid_size[1] : device_grid_size[0];
            uint32_t num_cores_channels = find_closest_common_largest_divisor(
                conv_out_2d_matrix_width_ntiles, std::ceil((double)input_channels / (double)tt::constants::TILE_WIDTH), total_cores_for_channels);
            uint32_t cores_x =
                block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_nhw : num_cores_channels;
            uint32_t cores_y =
                block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_channels : num_cores_nhw;
            CoreRange core_range = CoreRange(CoreCoord({0, 0}), CoreCoord({cores_x - 1, cores_y - 1}));
            CoreRangeSet grid = CoreRangeSet({core_range});
            return grid;

        } else {
           TT_THROW("Conv2d supports Height, Block or Width Sharded Layouts but got {}", shard_layout);
            return CoreRangeSet({});
        }
    };

    uint32_t num_cores_nhw = calculate_num_cores_nhw();
    const CoreRangeSet& grid = calculate_grid(num_cores_nhw);
    auto shard_orientation = shard_layout == TensorMemoryLayout::BLOCK_SHARDED ? block_shard_orientation : ShardOrientation::ROW_MAJOR;
    ParallelConfig pconfig = {.grid = grid, .shard_scheme = shard_layout, .shard_orientation = shard_orientation};
    return pconfig;
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
    if(pconfig.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
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
    } else if(pconfig.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
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
    const ttnn::Shape& tensor_shape, ParallelConfig& parallel_config, uint32_t tile_size) {

    log_debug(tt::LogOp, "create_sharded_memory_config_from_parallel_config: tensor_shape: {}, parallel_config: {}, tile_size: {}", tensor_shape, parallel_config, tile_size);
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
    if(shard_scheme != TensorMemoryLayout::WIDTH_SHARDED) {
        nhw_padded = round_up(nhw_shape, num_cores_nhw * tile_size);
    }
    uint32_t nhw_shard = nhw_padded / num_cores_nhw;
    TT_ASSERT(channels % num_cores_channels == 0, "Channels: {}, num core channels: {}", channels, num_cores_channels);
    uint32_t channel_shard = channels / num_cores_channels;
    auto shard_spec = ShardSpec{parallel_config.grid, {nhw_shard, channel_shard}, shard_orientation};
    return MemoryConfig{shard_scheme, BufferType::L1, shard_spec};
}


OptimizedConvParallelizationConfig determine_conv_op_parallel_config_from_conv_output_mem_config(
    const MemoryConfig& conv_output_mem_config, uint32_t num_cores_nhw, uint32_t num_cores_c) {
    TT_ASSERT(conv_output_mem_config.shard_spec.has_value());
    const auto& shard_spec = conv_output_mem_config.shard_spec.value();
    const auto& shard_shape = shard_spec.shape;
    TT_ASSERT(conv_output_mem_config.memory_layout == TensorMemoryLayout::WIDTH_SHARDED || shard_shape[0] % 32 == 0);
    TT_ASSERT(shard_shape[1] % 32 == 0);
    return {
        .grid_size = shard_spec.grid.bounding_box().grid_size(),
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c = num_cores_c,
        .per_core_out_matrix_height_ntiles = tt::round_up(shard_shape[0], 32) / 32,
        .per_core_out_matrix_width_ntiles = shard_shape[1] / 32,
    };
}

std::pair<uint32_t, uint32_t> determine_largest_subblock_size(
    uint32_t block_height, uint32_t block_width, bool fp32_accum) {
    std::vector<std::pair<uint32_t, uint32_t>> subblocks = {
        {2, 4}, {4, 2}, {1, 8}, {8, 1}, {1, 7}, {7, 1}, {2, 3}, {3, 2}, {1, 6}, {6, 1},
        {1, 5}, {5, 1}, {2, 2}, {1, 4}, {4, 1}, {1, 3}, {3, 1}, {1, 2}, {2, 1}, {1, 1},
    };
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
    TT_ASSERT(subblock_h > 0 && subblock_w > 0);
    return {subblock_h, subblock_w};
}

OptimizedConvBlockConfig determine_per_core_conv_block_config(
    const ParallelConfig& parallel_config,
    const OptimizedConvParallelizationConfig& conv_op_parallel_config,
    uint32_t padded_in_channels,
    uint32_t act_block_h_override,
    uint32_t act_block_w_div,
    uint32_t window_h,
    uint32_t window_w,
    bool fp32_accum,
    bool use_shallow_conv_variant) {
    if (act_block_h_override > 0) {
        TT_ASSERT(
            act_block_h_override % 32 == 0,
            "Config Error: act_block_h_override must be a multiple of 32 (tile height).");
    }
    auto grid_size = parallel_config.grid.bounding_box().grid_size();
    uint32_t act_block_h_ntiles = act_block_h_override > 0 ? act_block_h_override / 32
                                                           : conv_op_parallel_config.per_core_out_matrix_height_ntiles;
    uint32_t act_block_w = parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED
                               ? round_up(padded_in_channels * window_w, 32)
                               : padded_in_channels;
    if(parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        act_block_w = (padded_in_channels * window_h * window_w)/(parallel_config.grid.num_cores() * act_block_w_div);
    }
    TT_ASSERT(act_block_w % 32 == 0);
    uint32_t act_block_w_ntiles = act_block_w / 32;
    uint32_t act_c_num_blocks = parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED ? 1
                                : parallel_config.shard_orientation == ShardOrientation::COL_MAJOR ? grid_size.y
                                                                                                   : grid_size.x;
    uint32_t out_block_h_ntiles = conv_op_parallel_config.per_core_out_matrix_height_ntiles;
    uint32_t weight_block_w_ntiles = conv_op_parallel_config.per_core_out_matrix_width_ntiles;
    auto [out_subblock_h_ntiles, out_subblock_w_ntiles] =
        determine_largest_subblock_size(act_block_h_ntiles, weight_block_w_ntiles, fp32_accum);
    if (use_shallow_conv_variant && ((act_block_h_ntiles / out_subblock_h_ntiles) % 2 != 0)) {
        TT_ASSERT(parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED);
        // TODO: do a proper fix and remove this temporary hack for shallow conv
        TT_ASSERT(act_block_h_ntiles % 2 == 0);
        out_subblock_h_ntiles = act_block_h_ntiles / 2;
        TT_ASSERT((out_subblock_h_ntiles * out_subblock_w_ntiles) <= 8);
    }
    return {
        .act_block_h_ntiles = act_block_h_ntiles,
        .act_block_w_ntiles = act_block_w_ntiles,
        .out_subblock_h_ntiles = out_subblock_h_ntiles,
        .out_subblock_w_ntiles = out_subblock_w_ntiles};
}

// Implements a heuristic for selecting shard layout based on the input tensors shapes
// and stride.
static TensorMemoryLayout select_shard_layout(
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride) {
    // ToDo: enhance shard layout selection logic to check
    // which sharding scheme maximizes core count (and consequently available L1 space)

    TensorMemoryLayout shard_layout;

    // 1d convs support only height sharding
    bool is_conv1d = width == 1 && kernel_size[1] == 1;
    // block and width sharding support very few configurations of kernel size and stride
    // which are encoded below.
    bool is_width_or_block_sharding_valid =
        (kernel_size[0] == 3 && kernel_size[1] == 3 && (stride[0] == 1 || stride[0] == 2)) ||
        (kernel_size[0] == 1 && kernel_size[1] == 1 && stride[0] == 2);

    if (is_conv1d || !is_width_or_block_sharding_valid) {
        log_debug(LogOp, "Conv which can only be supported by TensorMemoryLayout::HEIGHT_SHARDED");
        shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    } else {
        float nhw = height * width * batch_size;
        float ratio = nhw / in_channels;
        log_debug(LogOp, "NHW: {}, C: {}, ratio: {}", nhw, in_channels, ratio);

        if (ratio > 8.0f) {
            shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;
            log_debug(LogOp, "Shard layout: TensorMemoryLayout::HEIGHT_SHARDED");
        } else if (ratio < 0.4f) {
            shard_layout = TensorMemoryLayout::WIDTH_SHARDED;
            log_debug(LogOp, "Shard layout: TensorMemoryLayout::WIDTH_SHARDED");
        } else {
            log_debug(LogOp, "Shard layout: TensorMemoryLayout::BLOCK_SHARDED");
            shard_layout = TensorMemoryLayout::BLOCK_SHARDED;
        }
    }

    return shard_layout;
}

template <typename T>
std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> get_conv_padded_input_shape_and_mem_config(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride) {
    ttnn::Tensor input_tensor = input_tensor_;  // tensor to return
    bool input_tensor_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_);
    bool needs_shard_or_reshard = false;
    if (conv_config.override_sharding_config && conv_config.reshard_if_not_optimal) {
        TT_ASSERT(
            false,
            "Incorrect config provided: reshard_if_not_optimal and override_sharding_config cannot both be set.");
    }

    TensorMemoryLayout shard_layout;
    if (conv_config.shard_layout.has_value()) {
        shard_layout = conv_config.shard_layout.value();
    } else {
        shard_layout = select_shard_layout(batch_size, height, width, in_channels, kernel_size, stride);
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
                TT_FATAL(conv_config.core_grid.has_value(), "If override_sharding_config is set, core_grid must be set as well.");
                TT_FATAL(conv_config.shard_layout.has_value(), "If override_sharding_config is set, shard_layout must be set as well.");
                if (conv_config.core_grid.value() != input_shard_grid) {
                    needs_shard_or_reshard = true;
                }
                if(shard_layout!=input_shard_scheme) {
                    needs_shard_or_reshard = true;
                }
                bool input_transpose_shards = input_shard_orientation == ShardOrientation::COL_MAJOR;
                if (shard_layout == TensorMemoryLayout::BLOCK_SHARDED && conv_config.transpose_shards != input_transpose_shards) {
                    needs_shard_or_reshard = true;
                }
            }
        }
    }
    ParallelConfig parallel_config = input_tensor_parallel_config;
    if (conv_config.reshard_if_not_optimal || needs_shard_or_reshard) {
        auto block_shard_orientation =
            conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;
        const ParallelConfig& optimal_parallel_config = determine_parallel_config(
            shard_layout, batch_size, in_channels, height, width, out_channels, device, block_shard_orientation);

        if (conv_config.override_sharding_config) {
            TT_FATAL(conv_config.core_grid.has_value(), "Error");
            // override parallel config
            auto shard_orientation = shard_layout == TensorMemoryLayout::BLOCK_SHARDED
                                         ? block_shard_orientation
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
        // TT_ASSERT(input_tensor.get_legacy_shape() == input_tensor.get_shape());
        uint32_t tensor_height = input_tensor.get_shape()[0] * input_tensor.get_shape()[1] * input_tensor.get_shape()[2];
        uint32_t input_tensor_height_snapped_to_tile = (shard_layout == TensorMemoryLayout::WIDTH_SHARDED)? tensor_height : tt::round_up(tensor_height, input_num_cores_nhw * 32);
        TT_ASSERT(input_tensor_height_snapped_to_tile >= tensor_height);
        uint32_t tensor_width = input_tensor.get_shape()[3];
        uint32_t input_tensor_width_snapped_to_channels_alignment =
            tt::round_up(tensor_width, conv_config.input_channels_alignment);
        TT_ASSERT(input_tensor_width_snapped_to_channels_alignment >= tensor_width);

        auto input_padded_shape = ttnn::Shape(std::array<uint32_t, 4>{
            1,
            1,
            input_tensor_height_snapped_to_tile,
            input_tensor_width_snapped_to_channels_alignment});  // TODO: resolve ttnn::types::Shape and
                                                                 // tt::tt_metal::LegacyShape issue to clean up next line
        auto input_tensor_sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            ttnn::Shape(std::array<uint32_t, 4>{
                input_padded_shape[0], input_padded_shape[1], input_padded_shape[2], input_padded_shape[3]}),
            parallel_config,
            32);
        return {input_padded_shape, input_tensor_sharded_memory_config, needs_shard_or_reshard};
    } else {
        return {input_tensor.shape(), input_tensor.memory_config(), needs_shard_or_reshard};
    }
}

template <typename T>
std::tuple<ttnn::Tensor, ParallelConfig, bool> shard_or_reshard_tensor_if_required(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride) {
    ttnn::Tensor input_tensor = input_tensor_;  // tensor to return
    bool input_tensor_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_);

    auto [input_padded_shape, input_tensor_sharded_memory_config, needs_shard_or_reshard] =
        get_conv_padded_input_shape_and_mem_config(
            device, input_tensor_, conv_config, batch_size, height, width, in_channels, out_channels, kernel_size, stride);
    ParallelConfig parallel_config = {
        .grid = input_tensor_sharded_memory_config.shard_spec.value().grid,
        .shard_scheme = input_tensor_sharded_memory_config.memory_layout,
        .shard_orientation = input_tensor_sharded_memory_config.shard_spec.value().orientation
    };
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
                    tt::tt_metal::Array4D({input_tensor.get_shape()[0],
                     input_tensor.get_shape()[1],
                     input_padded_shape[-2],
                     input_padded_shape[-1]}),
                    tt::tt_metal::Array4D({0, 0, 0, 0}),
                    0);
            }
        }

        if (input_tensor_on_device) {
            auto resharded_input_tensor = ttnn::to_memory_config(
                input_tensor, input_tensor_sharded_memory_config, std::nullopt);
            if (conv_config.deallocate_activation) {
                input_tensor.deallocate();
                resharded_input_tensor = ttnn::operations::core::reallocate(resharded_input_tensor, resharded_input_tensor.memory_config());
            }
            input_tensor = resharded_input_tensor;
        } else {
            input_tensor = ttnn::operations::core::to_device(
                input_tensor, device, input_tensor_sharded_memory_config);
        }
    }
    return {input_tensor, parallel_config, needs_shard_or_reshard};
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

template <typename T>
std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    T * device,
    uint32_t groups,
    uint32_t act_block_h_ntiles,
    uint32_t input_width) {

    validate_weight_and_bias_tensors(weight_tensor, bias_tensor);
    ttnn::Tensor weight_tensor_;  // tensor to return
    ttnn::Tensor bias_tensor_;

    auto original_weights_shape = weight_tensor.get_shape();
    uint32_t original_weights_out_channels = original_weights_shape[0];
    uint32_t original_weights_in_channels = original_weights_shape[1];
    uint32_t original_weights_window_h = original_weights_shape[2];
    uint32_t original_weights_window_w = original_weights_shape[3];

    bool is_conv1d = original_weights_window_w == 1 && input_width == 1;
    bool is_depthwise_conv = groups == original_weights_out_channels && original_weights_in_channels == 1;

    weight_tensor_ = weight_tensor;

    // Convert weight tensor to 0 padded shape if groups > 1
    if (!is_conv1d and groups > 1) {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, groups, weights_bias_dtype);
    }
    else if (is_conv1d and groups > 1) {
        if (is_depthwise_conv) {
            weight_tensor_ = convert_conv_weight_tensor_to_depthwise_layout(weight_tensor_, act_block_h_ntiles, weights_bias_dtype);
            weight_block_h_ntiles = act_block_h_ntiles;
        }
        else{
           weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, groups, weights_bias_dtype);
        }
    }

    auto weights_shape = weight_tensor_.get_shape();
    uint32_t out_channels = weights_shape[0];
    uint32_t in_channels = weights_shape[1];
    uint32_t window_h = weights_shape[2];
    uint32_t window_w = weights_shape[3];
    uint32_t out_channel_padding = tt::round_up(out_channels, 32) - out_channels;
    tt::tt_metal::LegacyShape weights_channels_padded_shape = tt::tt_metal::LegacyShape(std::array<uint32_t, 4>(
        {tt::round_up(out_channels, 32), tt::round_up(in_channels, input_channels_alignment), window_h, window_w}));
    if (weights_bias_dtype == DataType::BFLOAT8_B) {
        TT_ASSERT(weight_tensor_.get_dtype() == DataType::FLOAT32);
        if (bias_tensor.has_value()) {
            TT_ASSERT(bias_tensor.value().get_dtype() == DataType::FLOAT32);
        }
    } else {
        // TODO: fix the need to check this. We should be able to accept any datatype and convert
        TT_ASSERT(weight_tensor_.get_dtype() == weights_bias_dtype);
        if (bias_tensor.has_value()) {
            TT_ASSERT(bias_tensor.value().get_dtype() == weights_bias_dtype);
        }
    }
    weight_tensor_ = ttnn::pad(weight_tensor_, weights_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    // for conv op, pad the weights to block shape
    if (parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_special_padding_tiled_layout(
            weight_tensor_, weight_block_h_ntiles, weight_block_w_ntiles, weights_bias_dtype);
    } else {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_tiled_layout(
            weight_tensor_, weight_block_h_ntiles, weight_block_w_ntiles, weights_bias_dtype);
    }

    uint32_t weight_matrix_height = in_channels * window_h * window_w;
    int32_t weight_matrix_height_padding = weight_tensor_.shape()[2] - weight_matrix_height;
    TT_FATAL(weight_matrix_height_padding >= 0," Matrix Height Padding can't be negative");

    // convert_conv_weight_tensor adds the padding to the base shape.
    // Reshape the weights to remove padding from the base shape.
    weight_tensor_.set_shape(
        ttnn::Shape(std::array<uint32_t,4>{1, 1, weight_matrix_height, out_channels},
        std::array<std::array<uint32_t, 2>, 4>{
            std::array<uint32_t, 2>{0, 0},
            std::array<uint32_t, 2>{0, 0},
            std::array<uint32_t, 2>{0, weight_matrix_height_padding},
            std::array<uint32_t, 2>{0, out_channel_padding}
    }));

    weight_tensor_ = ttnn::operations::core::to_device(weight_tensor_, device, std::nullopt);
    if (bias_tensor.has_value()) {
        bias_tensor_ = bias_tensor.value();
        auto bias_shape = bias_tensor_.get_shape();
        TT_ASSERT(bias_shape[3] == out_channels && bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1);
        tt::tt_metal::LegacyShape bias_channels_padded_shape = tt::tt_metal::LegacyShape(
            std::array<uint32_t, 4>({1, 1, 32, tt::round_up(out_channels, weight_block_w_ntiles * 32)}));
        bias_tensor_ = ttnn::pad(bias_tensor_, bias_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), 0);
        bias_tensor_ = ttnn::to_layout(
            bias_tensor_, Layout::TILE, std::nullopt, std::nullopt, (T*)nullptr);
        if (bias_tensor_.get_dtype() != weights_bias_dtype) {
            bias_tensor_ = ttnn::to_dtype(bias_tensor_, weights_bias_dtype);
        }
        bias_tensor_ = ttnn::operations::core::to_device(bias_tensor_, device, std::nullopt);
    }

    return {weight_tensor_, bias_tensor.has_value() ? bias_tensor_ : std::optional<ttnn::Tensor>()};
}

ttnn::operations::matmul::MatmulProgramConfig determine_matmul_op_config_from_conv_op_config(
    OptimizedConvParallelizationConfig conv_parallelization_config,
    OptimizedConvBlockConfig conv_blocking_config,
    bool height_sharded,
    string activation,
    bool transpose_mcast,
    uint32_t grid_size_along_c) {
    if (height_sharded) {
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig matmul_config = {
            .compute_with_storage_grid_size = conv_parallelization_config.grid_size,
            .in0_block_w = conv_blocking_config.act_block_w_ntiles,
            .out_subblock_h = conv_blocking_config.out_subblock_h_ntiles,
            .out_subblock_w = conv_blocking_config.out_subblock_w_ntiles,
            .per_core_M = conv_parallelization_config.per_core_out_matrix_height_ntiles,
            .per_core_N = conv_parallelization_config.per_core_out_matrix_width_ntiles,
            .fuse_batch = true,
            .mcast_in0 = false};
        if (activation != "") {
            matmul_config.fused_activation = ttnn::operations::unary::utils::string_to_unary_with_param(activation);
        }
        return matmul_config;
    } else {
        TT_ASSERT(conv_blocking_config.act_block_w_ntiles % grid_size_along_c == 0);
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig matmul_config = {
            .compute_with_storage_grid_size = conv_parallelization_config.grid_size,
            .in0_block_w = conv_blocking_config.act_block_w_ntiles / grid_size_along_c,
            .out_subblock_h = conv_blocking_config.out_subblock_h_ntiles,
            .out_subblock_w = conv_blocking_config.out_subblock_w_ntiles,
            .per_core_M = conv_parallelization_config.per_core_out_matrix_height_ntiles,
            .per_core_N = conv_parallelization_config.per_core_out_matrix_width_ntiles,
            .transpose_mcast = transpose_mcast};
        if (activation != "") {
            matmul_config.fused_activation = ttnn::operations::unary::utils::string_to_unary_with_param(activation);
        }
        return matmul_config;
    }
}

template<typename T>
std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> conv2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    T * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    std::optional<const Conv2dConfig> conv_config_,
    const std::optional<const MemoryConfig> memory_config) {

    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    uint32_t output_height = ((input_height - kernel_size[0] - ((kernel_size[0] - 1 ) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
    uint32_t output_width = ((input_width - kernel_size[1] - ((kernel_size[0] - 1 ) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;
    auto [input_tensor_post_tm, parallel_config, tensor_manipulated] = shard_or_reshard_tensor_if_required(
        device, input_tensor, conv_config, batch_size, output_height, output_width, in_channels, out_channels, kernel_size, stride);
    if (tensor_manipulated) {
        if (conv_config.deallocate_activation) {
            ttnn::Tensor input_tensor_ = input_tensor;  // TODO: allow in place modification of inputs to the op
            input_tensor_.deallocate();
            // ttnn::operations::core::deallocate(input_tensor_);
        }
        conv_config.deallocate_activation = true;
    }
    auto conv_out_memory_config = create_sharded_memory_config_from_parallel_config(
        ttnn::Shape(std::array<uint32_t, 4>{1, 1, batch_size * output_height * output_width, tt::round_up(out_channels, 32)}),
        parallel_config,
        32);
    auto opt_conv_op_parallel_config = determine_conv_op_parallel_config_from_conv_output_mem_config(
        conv_out_memory_config, get_num_cores_nhw_from_parallel_config(parallel_config),
        get_num_cores_channels_from_parallel_config(parallel_config));
    auto opt_conv_op_block_config = determine_per_core_conv_block_config(
        parallel_config,
        opt_conv_op_parallel_config,
        tt::round_up(in_channels, conv_config.input_channels_alignment),
        conv_config.act_block_h_override,
        conv_config.act_block_w_div,
        kernel_size[0],
        kernel_size[1],
        conv_config.fp32_dest_acc_enabled,
        conv_config.input_channels_alignment == 16);
    bool weight_is_on_device = ttnn::is_tensor_on_device_or_multidevice(weight_tensor);
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
    if (!weight_is_on_device) {
        // prepare weights in desired layout and move to device
        tie(weight_tensor_on_device, bias_tensor_on_device) = prepare_conv_weights_biases_and_move_to_device(
            weight_tensor,
            bias_tensor,
            conv_config.input_channels_alignment,
            conv_config.weights_dtype,
            opt_conv_op_block_config.act_block_w_ntiles,
            opt_conv_op_block_config.out_subblock_w_ntiles,
            parallel_config,
            device,
            groups,
            opt_conv_op_block_config.act_block_h_ntiles,
            input_width);
    }
    // if 1x1 conv w/ stride 1, convert input tensor to tile layout if required
    bool use_matmul_for_1x1_conv = kernel_size[0] == 1 && kernel_size[1] == 1 && stride[0] == stride[1] && stride[0] == 1 &&
                                   padding[0] == 0 && padding[1] == 0 && dilation[0] == 1 && dilation[1] == 1 &&
                                   groups == 1;
    Tensor input_tensor_post_tm_out;
    if (use_matmul_for_1x1_conv) {
        input_tensor_post_tm_out = ttnn::to_layout(
            input_tensor_post_tm, Layout::TILE, conv_config.dtype, input_tensor_post_tm.memory_config(), device);
        if (conv_config.deallocate_activation) {
            input_tensor_post_tm.deallocate();
            // ttnn::operations::core::deallocate(input_tensor_post_tm);
        }
        input_tensor_post_tm = input_tensor_post_tm_out;
    }
    // call optimized conv op or matmul micro op
    bool input_is_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_post_tm);
    TT_ASSERT(input_is_on_device);
    DeviceComputeKernelConfig compute_kernel_config = ttnn::init_device_compute_kernel_config(
        device->arch(),
        std::nullopt,
        conv_config.math_fidelity,
        conv_config.math_approx_mode_enabled,
        conv_config.fp32_dest_acc_enabled,
        conv_config.packer_l1_accum_enabled);

    if (!use_matmul_for_1x1_conv) {
        // call halo op
        SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_height, input_width},
            .window_hw = {kernel_size[0], kernel_size[1]},
            .stride_hw = {stride[0], stride[1]},
            .pad_hw = {padding[0], padding[1]},
            .dilation_hw = {dilation[0], dilation[1]},
            .num_cores_nhw = opt_conv_op_parallel_config.num_cores_nhw,
            .core_range_set = input_tensor_post_tm.memory_config().shard_spec.value().grid,
            .snap_to_tile = true
        };

        bool bypass_halo = (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED &&
            sliding_window_config.pad_hw.first==0 &&
            sliding_window_config.pad_hw.second==0
            );
        if(bypass_halo) {
            // call conv micro op
            auto conv_output = optimized_conv_new(
                input_tensor_post_tm,
                weight_tensor_on_device,
                bias_tensor_on_device,
                sliding_window_config,
                out_channels,
                groups,
                conv_config.output_layout == Layout::ROW_MAJOR,
                conv_config.activation == "relu",
                conv_config.math_fidelity,
                opt_conv_op_parallel_config,
                opt_conv_op_block_config,
                conv_out_memory_config,
                conv_config.dtype,
                {batch_size, input_height, input_width, in_channels},
                conv_config.input_channels_alignment == 16,
                compute_kernel_config,
                conv_config.enable_act_double_buffer,
                conv_config.enable_split_reader,
                conv_config.enable_subblock_padding);
            if (conv_config.deallocate_activation) {
                ttnn::operations::core::deallocate(input_tensor_post_tm);
            }
            return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
        }
        auto halo_output = ttnn::halo(
            DefaultQueueId,
            input_tensor_post_tm,
            sliding_window_config,
            0,
            false,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            0,
            input_tensor_post_tm.memory_config());
        if (conv_config.deallocate_activation) {
            ttnn::operations::core::deallocate(input_tensor_post_tm);
        }
        if (conv_config.reallocate_halo_output) {
            auto move_output = ttnn::operations::core::reallocate(halo_output, halo_output.memory_config());
            ttnn::operations::core::deallocate(halo_output);
            halo_output = move_output;
        }

        // call conv micro op
        auto conv_output = optimized_conv_new(
            halo_output,
            weight_tensor_on_device,
            bias_tensor_on_device,
            sliding_window_config,
            out_channels,
            groups,
            conv_config.output_layout == Layout::ROW_MAJOR,
            conv_config.activation == "relu",
            conv_config.math_fidelity,
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            conv_out_memory_config,
            conv_config.dtype,
            {batch_size, input_height, input_width, in_channels},
            conv_config.input_channels_alignment == 16,
            compute_kernel_config,
            conv_config.enable_act_double_buffer,
            conv_config.enable_split_reader,
            conv_config.enable_subblock_padding);
        ttnn::operations::core::deallocate(halo_output);

        if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
            conv_output = ttnn::to_memory_config(conv_output, memory_config.value(), std::nullopt);
        }

        return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    } else {
        // run conv as matmul
        uint32_t num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);
        auto matmul_program_config = determine_matmul_op_config_from_conv_op_config(
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
            conv_config.activation,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            num_cores_c);
        Tensor matmul_input = input_tensor_post_tm;
        if (stride[0] > 1) {
            // run downsample
            matmul_input = ttnn::operations::downsample::downsample(
                input_tensor_post_tm, {batch_size, input_height, input_width, stride[0], stride[1]});
            if (conv_config.deallocate_activation) {
                ttnn::operations::core::deallocate(input_tensor_post_tm);
            }
        }
        auto matmul_output = ttnn::operations::matmul::matmul(
            matmul_input,
            weight_tensor_on_device,
            bias_tensor_on_device,
            ttnn::operations::matmul::Matmul{
            matmul_program_config,
            /*bcast_batch=*/std::nullopt,
            conv_out_memory_config,
            conv_config.dtype,
            compute_kernel_config});
        if (conv_config.deallocate_activation) {
            ttnn::operations::core::deallocate(matmul_input);
        }

        if (memory_config.has_value() && memory_config.value() != matmul_output.memory_config()) {
            matmul_output = ttnn::to_memory_config(matmul_output, memory_config.value(), std::nullopt);
        }

        return {matmul_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    }
}

template ParallelConfig determine_parallel_config<Device>(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    Device * device,
    ShardOrientation block_shard_orientation,
    bool is_out_tiled);

template ParallelConfig determine_parallel_config<MeshDevice>(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    MeshDevice * device,
    ShardOrientation block_shard_orientation,
    bool is_out_tiled);

template std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> get_conv_padded_input_shape_and_mem_config<Device>(
    Device* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride);

template std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool> get_conv_padded_input_shape_and_mem_config<MeshDevice>(
    MeshDevice * device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride);

template std::tuple<ttnn::Tensor, ParallelConfig, bool> shard_or_reshard_tensor_if_required<Device>(
    Device* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride);

template std::tuple<ttnn::Tensor, ParallelConfig, bool> shard_or_reshard_tensor_if_required<MeshDevice>(
    MeshDevice * device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride);

template std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device<Device>(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    Device * device,
    uint32_t groups, uint32_t act_block_h_ntiles, uint32_t input_width);

template std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device<MeshDevice>(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    MeshDevice * device,
    uint32_t groups, uint32_t act_block_h_ntiles, uint32_t input_width);

template std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> conv2d<Device>(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    Device * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    std::optional<const Conv2dConfig> conv_config_,
    const std::optional<const MemoryConfig> memory_config);

template std::tuple<ttnn::Tensor, uint32_t, uint32_t, ttnn::Tensor, std::optional<ttnn::Tensor>> conv2d<MeshDevice>(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    std::optional<const Conv2dConfig> conv_config_,
    const std::optional<const MemoryConfig> memory_config);

}  // namespace conv2d
}  // namespace operations
}  // namespace ttnn
