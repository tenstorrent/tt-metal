// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
using sliding_window::ParallelConfig;
using sliding_window::SlidingWindowConfig;

conv::conv_op_l1_usage calculate_L1_usage(
    const DeviceComputeKernelConfig& compute_kernel_config,
    const conv::OptimizedConvBlockConfig& block_config,
    const conv::OptimizedConvParallelizationConfig& pconfig,
    std::array<uint32_t, 2> kernel_size,
    const conv::Conv2dConfig& conv_config,
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
        get_compute_kernel_config_args(tt::tt_metal::hal.get_arch(), compute_kernel_config);

    uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    uint32_t act_block_num_tiles = block_config.act_block_h_ntiles * act_block_w_ntiles;

    uint32_t weight_matrix_height = 0;  // weights_shape[2];
    uint32_t weight_matrix_width = 0;   // weights_shape[3];
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
        uint32_t weight_block_num_tiles = 0;
        // weight_block_w_ntiles * act_block_w_ntiles;  // act_block_w_ntiles == weight_block_h_ntiles
        uint32_t weight_block_num_bytes = weight_block_num_tiles * weights_tile_size;

        uint32_t bias_block_num_bytes = per_core_out_matrix_width_ntiles * bias_tile_size;

        uint32_t out_block_num_tiles = per_core_out_matrix_height_ntiles * per_core_out_matrix_width_ntiles;

        uint32_t num_blocks_act_w = weight_matrix_height_ntiles / act_block_w_ntiles;

        packer_l1_acc = packer_l1_acc && ((enable_bias && num_blocks_act_w > 1) || (num_blocks_act_w > 2));

        auto interm_dtype =
            packer_l1_acc ? (fp32_dest_acc_en ? DataType::FLOAT32 : DataType::BFLOAT16) : conv_config.dtype;

        uint32_t partial_tile_size = tt::tile_size(datatype_to_dataformat_converter(interm_dtype));

        uint32_t partials_block_num_bytes = out_block_num_tiles * partial_tile_size;

        // CB 0
        uint32_t act_cb_0_size = tilized_act_block_num_bytes;
        tt::log_debug(tt::LogOp, "CB0 Size: {}", act_cb_0_size);

        // CB 1
        uint32_t weights_cb_1_size = weight_block_num_bytes;
        tt::log_debug(tt::LogOp, "CB1 Size: {}", weights_cb_1_size);

        // CB 2
        uint32_t bias_cb_2_size = bias_block_num_bytes;
        tt::log_debug(tt::LogOp, "CB2 Size: {}", bias_cb_2_size);

        // CB 5
        uint32_t l1_scratchpad_cb_5_size = conv::l1_scratchpad_CB_size;
        tt::log_debug(tt::LogOp, "CB5 Size: {}", l1_scratchpad_cb_5_size);

        // CB 6
        uint32_t row_major_act_cb_6_size = act_block_num_bytes;
        tt::log_debug(tt::LogOp, "CB6 Size: {}", row_major_act_cb_6_size);

        // CB 24
        uint32_t matmul_partial_cb_24_size = partials_block_num_bytes;
        if (interm_dtype == conv_config.dtype) {
            matmul_partial_cb_24_size = 0;
        } else {
            tt::log_debug(tt::LogOp, "CB24 Size: {}", matmul_partial_cb_24_size);
        }

        // CB 25
        uint32_t tilized_act_cb_25_size = tilized_act_block_num_bytes;
        tt::log_debug(tt::LogOp, "CB25 Size: {}", tilized_act_cb_25_size);

        uint32_t total_CB_size = act_cb_0_size + weights_cb_1_size + bias_cb_2_size + l1_scratchpad_cb_5_size +
                                 row_major_act_cb_6_size + matmul_partial_cb_24_size + tilized_act_cb_25_size;

        tt::log_debug(tt::LogOp, "Total CB Size: {}", total_CB_size);

        return conv::conv_op_l1_usage{
            .tensor_allocation_size = output_size_per_core_in_bytes, .CB_allocation_size = total_CB_size};
    } else if (sharding_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t output_size = per_core_out_matrix_height_ntiles * per_core_out_matrix_width_ntiles * output_tile_size;

        uint32_t bias_block_num_bytes = per_core_out_matrix_width_ntiles * bias_tile_size;

        uint32_t conv_act_c_blocks = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;

        uint32_t weight_block_w_ntiles = 0;  // per_core_out_matrix_width_ntiles;
        uint32_t weight_block_h_ntiles = 0;  //(is_1d_depthwise_conv) ? act_block_h_ntiles : act_block_w_ntiles;

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
        // CB 0
        uint32_t act_cb_0_size = act_block_split_ntiles * input_tile_size;
        tt::log_debug(tt::LogOp, "CB0 Size: {}", act_cb_0_size);

        // CB 1
        uint32_t weights_cb_1_size = weight_block_h_ntiles * weight_block_w_ntiles * weights_tile_size;
        if (num_blocks_act_h > 1) {
            weights_cb_1_size *= kernel_size[0];
        }
        if (num_blocks_act_h <= 1 && conv_config.enable_weights_double_buffer) {
            weights_cb_1_size *= 2;
        }
        tt::log_debug(tt::LogOp, "CB1 Size: {}", weights_cb_1_size);

        // CB 2
        uint32_t bias_cb_2_size = bias_block_num_bytes;
        tt::log_debug(tt::LogOp, "CB2 Size: {}", bias_cb_2_size);

        uint32_t l1_scratchpad_cb_5_size = conv::l1_scratchpad_CB_size;
        tt::log_debug(tt::LogOp, "CB5 Size: {}", l1_scratchpad_cb_5_size);

        uint32_t split_second_act_reader_cb_7_size = 0;

        split_second_act_reader_cb_7_size = act_block_split_last_ntiles * input_tile_size;
        tt::log_debug(tt::LogOp, "CB7 Size: {}", split_second_act_reader_cb_7_size);

        // CB 24
        uint32_t matmul_partials_cb_24_size = output_block_ntiles * partial_tile_size;
        if (untilize_out == false && interm_dtype == conv_config.dtype) {
            matmul_partials_cb_24_size = 0;
        }
        if (is_1d_depthwise_conv) {
            matmul_partials_cb_24_size = output_tile_size;
        }
        if (matmul_partials_cb_24_size != 0) {
            tt::log_debug(tt::LogOp, "CB24 Size: {}", matmul_partials_cb_24_size);
        }
        // CB 25
        uint32_t tilized_act_cb_25_size = tilzed_act_cb_size;
        tt::log_debug(tt::LogOp, "CB25 Size: {}", tilized_act_cb_25_size);

        uint32_t temp_sum_cb_27_size = 0;
        if (is_1d_depthwise_conv) {
            temp_sum_cb_27_size = output_tile_size;
            tt::log_debug(tt::LogOp, "CB27 Size: {}", temp_sum_cb_27_size);
        }
        uint32_t total_CB_size = act_cb_0_size + weights_cb_1_size + bias_cb_2_size + l1_scratchpad_cb_5_size +
                                 split_second_act_reader_cb_7_size + matmul_partials_cb_24_size +
                                 tilized_act_cb_25_size + temp_sum_cb_27_size;
        return conv::conv_op_l1_usage{.tensor_allocation_size = output_size, .CB_allocation_size = total_CB_size};
    } else if (sharding_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        auto output_shard_shape = output_memory_config.shard_spec.value().shape;

        uint32_t output_size = 0;
        if (untilize_out) {
            uint32_t per_core_out_width_aligned = pconfig.per_core_out_matrix_width_ntile * tt::constants::TILE_WIDTH;
            if (conv_config.dtype == DataType::BFLOAT16) {
                per_core_out_width_aligned *= 2;
            } else if (conv_config.dtype == DataType::FLOAT32) {
                per_core_out_width_aligned *= 4;
            }
            output_size =
                round_up(per_core_out_width_aligned, tt::tt_metal::hal.get_alignment(tt::tt_metal::HalMemType::L1)) *
                pconfig.per_core_out_matrix_height_ntile * tt::constants::TILE_HEIGHT;
        } else {
            output_size = per_core_out_matrix_height_ntiles * per_core_out_matrix_width_ntiles * output_tile_size;
        }

        uint32_t bias_block_num_bytes = per_core_out_matrix_width_ntiles * bias_tile_size;

        uint32_t conv_act_c_blocks = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;

        uint32_t weight_block_w_ntiles = 0;  // per_core_out_matrix_width_ntiles;
        uint32_t weight_block_h_ntiles = 0;  // act_block_w_ntiles;

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

        // CB 0
        uint32_t act_cb_0_size = tilized_act_block_cb_size;
        if (conv_config.enable_act_double_buffer) {
            act_cb_0_size *= 2;
        }
        tt::log_debug(tt::LogOp, "CB0 Size: {}", act_cb_0_size);

        // CB 1
        uint32_t weights_cb_1_size = weight_block_h_ntiles * weight_block_w_ntiles * weights_tile_size;
        if (conv_config.enable_weights_double_buffer) {
            weights_cb_1_size *= 2;
        }
        tt::log_debug(tt::LogOp, "CB1 Size: {}", weights_cb_1_size);

        // CB 2
        uint32_t bias_cb_2_size = bias_block_num_bytes;
        tt::log_debug(tt::LogOp, "CB2 Size: {}", bias_cb_2_size);

        // CB 5
        uint32_t l1_scratchpad_cb_5_size = conv::l1_scratchpad_CB_size;
        tt::log_debug(tt::LogOp, "CB5 Size: {}", l1_scratchpad_cb_5_size);

        // CB 6
        uint32_t cb6_size = row_major_act_cb_size;
        tt::log_debug(tt::LogOp, "CB6 Size: {}", cb6_size);

        // CB 24
        uint32_t matmul_partials_cb_24_size = output_block_ntiles * partial_tile_size;
        if (untilize_out == false && interm_dtype == conv_config.dtype) {
            matmul_partials_cb_24_size = 0;
        } else {
            tt::log_debug(tt::LogOp, "CB24 Size: {}", matmul_partials_cb_24_size);
        }

        // CB 25
        uint32_t tilized_act_cb_25_size = tilized_act_block_cb_size;
        tt::log_debug(tt::LogOp, "CB25 Size: {}", tilized_act_cb_25_size);

        bool need_unpad_after_untilize =
            output_shard_shape[1] * output_shard_shape[0] <
            per_core_out_matrix_height_ntiles * per_core_out_matrix_width_ntiles * tt::constants::TILE_HW;

        tt::log_debug(tt::LogOp, "Need Unpad after untilize: {}", need_unpad_after_untilize);

        uint32_t cb28_size = 0;
        if (need_unpad_after_untilize && untilize_out) {
            cb28_size = output_block_ntiles * output_tile_size;
            tt::log_debug(tt::LogOp, "CB28 Size: {}", cb28_size);
        }
        uint32_t total_CB_size = act_cb_0_size + weights_cb_1_size + bias_cb_2_size + l1_scratchpad_cb_5_size +
                                 cb6_size + matmul_partials_cb_24_size + tilized_act_cb_25_size + cb28_size;
        return conv::conv_op_l1_usage{.tensor_allocation_size = output_size, .CB_allocation_size = total_CB_size};
    }
    TT_THROW("Invalid shard layout {}", sharding_scheme);
}

TensorMemoryLayout determine_pool_config_for_auto_shard(
    uint32_t batch_size,
    uint32_t channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t input_height,
    uint32_t input_width,
    const CoreCoord& compute_grid_size,
    Layout input_tensor_layout,
    const std::array<uint32_t, 2>& kernel_size,
    const DeviceComputeKernelConfig& compute_config,
    const DataType& input_dtype) {
    ShardOrientation shard_orientation = ShardOrientation::COL_MAJOR;
    // const bool is_out_tiled = conv_config.output_layout == Layout::TILE;

    struct core_count_and_size {
        uint32_t core_count;
        uint32_t size;
        conv::Conv2dConfig conv_config;
    };

    // 1d deptwise convs support only height sharding
    if (conv::is_1d_deptwise_conv(1 /*groups*/, channels, channels, kernel_size[1], input_width)) {
        return TensorMemoryLayout::HEIGHT_SHARDED;
    }

    auto get_l1_usage_for_sharding = [&](TensorMemoryLayout shard_layout) -> core_count_and_size {
        conv::Conv2dConfig conv_config = conv::Conv2dConfig();
        conv_config.dtype = input_dtype;
        conv_config.shard_layout = shard_layout;
        if (conv_config.act_block_h_override == 0) {
            conv_config.act_block_h_override = tt::constants::TILE_HEIGHT;
            if (channels < tt::constants::TILE_WIDTH && shard_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
                input_tensor_layout == Layout::ROW_MAJOR) {
                log_debug(tt::LogOp, "Auto shard, enable shallow conv");
                // height sharded, non matmul conv, with input channels < 32, and default setting for
                // input_channels_alignment
                // Currently data-movement ops have too many restrictions to support shallow convs with tiled input.
                conv_config.input_channels_alignment = 8;
            } else if (shard_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
                conv_config.input_channels_alignment = tt::constants::TILE_WIDTH;
            }

            // Set act_block_h_override to min value to
            // be conservative with L1 memory usage.
            uint32_t act_block_h_override = tt::constants::TILE_HEIGHT;
        }

        const uint32_t in_channels_padded = ccl::cmd::round_up(channels, conv_config.input_channels_alignment);
        const uint32_t output_channels_padded = ccl::cmd::round_up(channels, tt::constants::TILE_WIDTH);
        // Note: These are not exact shapes for weights as prepare_conv_weights will pad the weights depending on the
        // conv2d params, but these are good enough for L1 usage estimation.
        // const ttnn::Shape weights_shape(
        //     {1, 1, in_channels_padded * kernel_size[0] * kernel_size[1], output_channels_padded});

        const sliding_window::ParallelConfig input_parallel_config = conv::determine_parallel_config(
            shard_layout,
            batch_size,
            channels,
            output_height,
            output_width,
            channels,
            compute_grid_size,
            shard_orientation,
            false,
            false);

        const sliding_window::ParallelConfig output_parallel_config =
            conv::determine_output_parallel_config(input_parallel_config, compute_grid_size, channels, false);

        auto [opt_conv_op_parallel_config, opt_conv_op_block_config, conv_out_memory_config] = conv::get_conv_configs(
            conv_config,
            compute_config,
            input_parallel_config,
            output_parallel_config,
            channels,
            channels,
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
            conv_config.act_block_w_div = tt::div_up(channels, width_sharded_num_cores * tt::constants::TILE_WIDTH);
        }

        conv::conv_op_l1_usage l1_usage = pool::calculate_L1_usage(
            compute_config,
            opt_conv_op_block_config,
            opt_conv_op_parallel_config,
            kernel_size,
            conv_config,
            conv_out_memory_config,
            false,
            false);

        // Since we don't have L1 usage for halo output (input to conv2d)
        // use approx input tensor size per core as a proxy.
        uint32_t input_nhw = tt::div_up(batch_size * input_height * input_width, tt::constants::TILE_HEIGHT);
        uint32_t input_c = tt::div_up(in_channels_padded, tt::constants::TILE_WIDTH);
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

    core_count_and_size height = get_l1_usage_for_sharding(TensorMemoryLayout::HEIGHT_SHARDED);
    const core_count_and_size block = get_l1_usage_for_sharding(TensorMemoryLayout::BLOCK_SHARDED);
    const core_count_and_size width = get_l1_usage_for_sharding(TensorMemoryLayout::WIDTH_SHARDED);

    core_count_and_size& winning_config = height;
    // Make sure that BS not only has smaller size but provides at least some slicing along the channels.
    // In case we have BS that would slice the tensor only along the HS conv2d code would fail later on.
    if (block.size < winning_config.size && block.core_count > compute_grid_size.x) {
        winning_config = block;
    }
    if (width.size < winning_config.size) {
        winning_config = width;
    }

    log_debug(tt::LogOp, "Core counts H: {} B: {}, W: {}", height.core_count, block.core_count, width.core_count);
    log_debug(
        tt::LogOp, "Selected shard layout: {}, size: {}", winning_config.conv_config.shard_layout, winning_config.size);

    return winning_config.conv_config.shard_layout.value();
}

}  // namespace operations::pool
}  // namespace ttnn
