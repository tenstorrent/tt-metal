// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "tt-metalium/circular_buffer_config.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/compute_throttle_utils.hpp"

namespace ttnn::operations::conv {
namespace conv2d {

tt::tt_metal::operation::ProgramWithCallbacks multi_core_optimized_conv_width_sharded_v2_impl(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const ttnn::Shape& ashape,
    std::optional<const Tensor> bias,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    const sliding_window::ParallelConfig& parallel_config,
    const std::vector<uint32_t>& op_trace_metadata,
    const std::vector<sliding_window::ShardBoundary>& shard_boundaries,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    bool has_bias,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool enable_subblock_padding);

tt::tt_metal::operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_v2_impl(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const ttnn::Shape& ashape,
    std::optional<const Tensor> bias,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    const sliding_window::ParallelConfig& parallel_config,
    const std::vector<uint32_t>& op_trace_metadata,
    const std::vector<sliding_window::ShardBoundary>& shard_boundaries,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    bool has_bias,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    bool transpose_mcast,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding) {
    tt::tt_metal::IDevice* device = a.device();
    TT_FATAL(a.layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_FATAL(a.memory_config().is_sharded(), "Conv activation must be sharded.");
    TT_FATAL(output_channels <= b.padded_shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    const uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    const uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    const uint32_t weight_block_w_ntiles = parallelization_config.per_core_out_matrix_width_ntile;
    const uint32_t out_block_h_ntiles = parallelization_config.per_core_out_matrix_height_ntile;
    const uint32_t out_subblock_h_ntiles = block_config.out_subblock_h_ntiles;
    const uint32_t out_subblock_w_ntiles = block_config.out_subblock_w_ntiles;

    const bool skip_mcast = is_singlecore_skip_mcast(parallelization_config, a.memory_config().memory_layout());

    const tt::DataFormat tilized_act_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t out_subblock_h_ntiles_padded = out_subblock_h_ntiles;
    if (enable_subblock_padding) {
        uint32_t max_num_subblock = fp32_dest_acc_en ? 4 : 8;
        uint32_t max_subblock_h = fp32_dest_acc_en ? 4 : 8;

        TT_FATAL(
            act_block_h_ntiles == out_block_h_ntiles, "to pad subblock, the number of blocks on height dim must be 1");

        if ((out_subblock_w_ntiles * out_subblock_h_ntiles <= max_num_subblock / 2) and
            (out_subblock_w_ntiles == weight_block_w_ntiles) and (act_block_h_ntiles == out_block_h_ntiles)) {
            uint32_t num_subblock_h = act_block_h_ntiles / out_subblock_h_ntiles;
            uint32_t num_iter = max_subblock_h - out_subblock_h_ntiles;
            uint32_t new_out_subblock_h = out_subblock_h_ntiles;
            uint32_t preferred_out_subblock_h = out_subblock_h_ntiles;

            for (uint32_t i = 0; i < num_iter; ++i) {
                new_out_subblock_h += 1;
                uint32_t new_num_subblock_h = (act_block_h_ntiles + new_out_subblock_h - 1) / new_out_subblock_h;

                if (new_num_subblock_h < num_subblock_h and
                    (out_subblock_w_ntiles * new_out_subblock_h <= max_num_subblock)) {
                    num_subblock_h = new_num_subblock_h;
                    preferred_out_subblock_h = new_out_subblock_h;
                }
            }
            out_subblock_h_ntiles_padded = preferred_out_subblock_h;
        }
    }

    TT_FATAL(
        out_block_h_ntiles >= act_block_h_ntiles,
        "Output block height (in # of tiles) ({}) should be greater than or equal to activation block height (in # of "
        "tiles) ({})",
        out_block_h_ntiles,
        act_block_h_ntiles);

    // Tensor b has weights and it should be tiled layout after converting conv weights into weight matrix
    TT_FATAL(b.layout() == Layout::TILE, "Conv weights should be in tiled layout");
    TT_FATAL(b.padded_shape()[0] == 1, "Conv weight matrix shape is invalid");
    TT_FATAL(b.padded_shape()[1] == 1, "Conv weight matrix shape is invalid");
    uint32_t weight_matrix_height = b.padded_shape()[2];
    uint32_t weight_matrix_width = b.padded_shape()[3];
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / tt::constants::TILE_WIDTH;

    const std::array<uint32_t, 2> shard_shape = a.shard_spec().value().shape;

    // parallelization config
    const auto& p_config = parallelization_config;
    uint32_t num_cores_x = p_config.grid_size.x;
    uint32_t num_cores_y = p_config.grid_size.y;
    uint32_t total_num_cores = num_cores_x * num_cores_y;

    uint32_t per_core_out_matrix_width_ntiles = parallelization_config.per_core_out_matrix_width_ntile;
    uint32_t per_core_out_matrix_height_ntiles = parallelization_config.per_core_out_matrix_height_ntile;
    const bool block_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool height_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;

    uint32_t conv_act_c_blocks;
    uint32_t input_channels_padded;
    if (block_sharded) {
        conv_act_c_blocks =
            a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR ? num_cores_x : num_cores_y;
        if (transpose_mcast) {
            TT_FATAL(conv_act_c_blocks == num_cores_y, "Expected conv_act_c_blocks to be equal to height of grid");
            input_channels_padded = shard_shape[1] * num_cores_y;
        } else {
            TT_FATAL(conv_act_c_blocks == num_cores_x, "Expected conv_act_c_blocks to be equal to width of grid");
            input_channels_padded = shard_shape[1] * num_cores_x;
        }
    } else {
        conv_act_c_blocks = 1;
        input_channels_padded = shard_shape[1];
    }
    const ttnn::Shape ashape_with_channels_padded({ashape[0], ashape[1], ashape[2], input_channels_padded});
    uint32_t conv_act_size_w = ashape_with_channels_padded[2];
    const uint32_t conv_act_size_c = ashape_with_channels_padded[3];
    const uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;   // filter_h
    const uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
    uint32_t pad_w = (uint32_t)sliding_window_config.get_pad_w();
    const uint32_t dilation_h = (uint32_t)sliding_window_config.dilation_hw.first;
    const uint32_t dilation_w = (uint32_t)sliding_window_config.dilation_hw.second;
    const uint32_t stride_w = (uint32_t)sliding_window_config.stride_hw.second;

    if (sliding_window_config.is_transpose) {
        auto input_shape = sliding_window_config.get_transposed_full_input_shape();
        conv_act_size_w = input_shape[2];
        pad_w = 0;
    }

    const bool is_conv_1d_depthwise_conv =
        is_1d_deptwise_conv(groups, ashape[3], output_channels, filter_w, ashape[2], has_bias);
    if ((block_sharded || is_conv_1d_depthwise_conv) && enable_split_reader) {
        enable_split_reader = false;
        log_warning(tt::LogOp, "Split reader is not supported for block sharded or 1d depthwise conv");
    }

    TT_FATAL(input_channels_padded >= ashape[3], "Incorrect padding of input channels!");
    // check is for 16-byte alignment
    TT_FATAL(
        // Since fp16 is smalleset data format used for halo output, 8 input_channels is enough for 16 byte alignment
        input_channels_padded % 8 == 0,
        "Expected input channels to be padded for 16 byte alignment in L1 ({} % 16 != 0)",
        input_channels_padded);
    if (enable_split_reader) {
        TT_FATAL(
            (act_block_h_ntiles / block_config.out_subblock_h_ntiles) >= 2,
            "split reader needs to have at leaset two subblocks");
        TT_FATAL(
            block_config.act_block_h_ntiles % block_config.out_subblock_h_ntiles == 0,
            "Out_block_h must be divisible by out_subblock_h!");
    }

    // Compute the 2d matrix shape
    auto [act_matrix_shape, act_matrix_shape_unpadded] =
        optimized_conv_op_utils::compute_opt_conv_activation_as_mm_shape(
            ashape_with_channels_padded,
            sliding_window_config,
            parallelization_config.num_cores_nhw,
            out_block_h_ntiles);
    TT_FATAL(act_matrix_shape.size() == 3, "act_matrix_shape should have be of size 3");
    TT_FATAL(act_matrix_shape[0] == 1, "act_matrix_shape should have 1 as the first dimension");
    uint32_t act_matrix_height = (uint32_t)act_matrix_shape[1];
    uint32_t act_matrix_width = (uint32_t)act_matrix_shape[2];
    if (block_sharded) {
        act_matrix_width =
            tt::round_up((input_channels_padded / conv_act_c_blocks) * filter_w * filter_h, tt::constants::TILE_WIDTH) *
            conv_act_c_blocks;
    }

    const uint32_t act_matrix_height_unpadded = (uint32_t)act_matrix_shape_unpadded[1];

    if (has_bias) {
        if (is_conv_1d_depthwise_conv) {
            TT_THROW("Bias is not supported for depthwise conv1d");
        }
        // Tensor bias is of shape {output_channels}
        TT_FATAL(bias.has_value(), "Error");
        TT_FATAL(bias.value().buffer() != nullptr, "Error");
        auto bias_shape_without_padding = bias.value().logical_shape();
        TT_FATAL(bias_shape_without_padding[0] == 1, "Bias should have batch == 1");
    }

    // matrix multiplication shape check valid for all convs except depthwise conv1d
    TT_FATAL(
        act_matrix_width == weight_matrix_height || is_conv_1d_depthwise_conv,
        "The width of tensor a {} needs to match the height of tensor b {}",
        act_matrix_width,
        weight_matrix_height);
    // Tile size divisibility checks
    TT_FATAL(
        act_matrix_height % tt::constants::TILE_HEIGHT == 0, "Height of activation matrix needs to be divisible by 32");
    TT_FATAL(
        act_matrix_width % tt::constants::TILE_WIDTH == 0, "Width of activation matrix needs to be divisible by 32");
    TT_FATAL(
        weight_matrix_height % tt::constants::TILE_HEIGHT == 0, "Height of weight matrix needs to be divisible by 32");
    TT_FATAL(
        weight_matrix_width % tt::constants::TILE_WIDTH == 0, "Width of weight matrix needs to be divisible by 32");

    // Device compatibility checks
    TT_FATAL(
        a.storage_type() == StorageType::DEVICE && b.storage_type() == StorageType::DEVICE,
        "Operands to large matmul need to be on device!");
    TT_FATAL(a.device() == b.device(), "Operands to conv need to be on the same device!");
    TT_FATAL(
        a.buffer() != nullptr && b.buffer() != nullptr, "Operands to conv need to be allocated in buffers on device!");
    if (has_bias) {
        TT_FATAL(bias.value().storage_type() == StorageType::DEVICE, "Bias should be on device");
        TT_FATAL(bias.value().device() == a.device(), "Bias should be on the same device as act tensor");
    }

    // Convert tensor dims to tile dims
    const uint32_t act_matrix_height_ntiles = act_matrix_height / tt::constants::TILE_HEIGHT;
    const uint32_t act_matrix_width_ntiles = act_matrix_width / tt::constants::TILE_WIDTH;

    TT_FATAL(
        act_matrix_height_ntiles % act_block_h_ntiles == 0,
        "act_matrix_height_ntiles {} should be divisible by act_block_h_ntiles {}",
        act_matrix_height_ntiles,
        act_block_h_ntiles);
    TT_FATAL(
        act_matrix_width_ntiles % act_block_w_ntiles == 0,
        "act_matrix_width_ntiles {} should be divisible by act_block_w_ntiles {}",
        act_matrix_width_ntiles,
        act_block_w_ntiles);
    TT_FATAL(
        weight_matrix_width_ntiles % weight_block_w_ntiles == 0,
        "weight_matrix_width_ntiles {} should be divisible by weight_block_w_ntiles {}",
        weight_matrix_width_ntiles,
        weight_block_w_ntiles);
    TT_FATAL(
        act_matrix_height_ntiles % out_block_h_ntiles == 0,
        "act_matrix_height_ntiles {} should be divisible by out_block_h_ntiles {}",
        act_matrix_height_ntiles,
        out_block_h_ntiles);

    const uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    const uint32_t num_blocks_out_h = act_matrix_height_ntiles / out_block_h_ntiles;
    const uint32_t num_blocks_act_w = block_sharded ? 1 : filter_h;
    const uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

    // act block info
    uint32_t act_block_w_datums = act_matrix_width / num_blocks_act_w;
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;

    uint32_t act_block_h_nsubblocks = block_config.act_block_h_ntiles / block_config.out_subblock_h_ntiles;
    uint32_t act_block_h_nsubblocks_split = act_block_h_nsubblocks;
    uint32_t act_block_h_nsubblocks_split_last = 0;
    if (enable_split_reader) {
        act_block_h_nsubblocks_split_last = act_block_h_nsubblocks / 2;
        act_block_h_nsubblocks_split = act_block_h_nsubblocks - act_block_h_nsubblocks_split_last;
    }
    uint32_t act_block_h_datums_split =
        act_block_h_nsubblocks_split * out_subblock_h_ntiles * tt::constants::TILE_HEIGHT;
    uint32_t act_block_h_datums_split_last =
        act_block_h_nsubblocks_split_last * out_subblock_h_ntiles * tt::constants::TILE_HEIGHT;

    uint32_t act_block_num_tiles_split = act_block_h_nsubblocks_split * out_subblock_h_ntiles * act_block_w_ntiles;
    uint32_t act_block_num_tiles_split_last =
        act_block_h_nsubblocks_split_last * out_subblock_h_ntiles * act_block_w_ntiles;

    // weight block info
    uint32_t weight_block_w_datums = weight_matrix_width / num_blocks_weight_w;
    TT_FATAL(
        weight_block_w_ntiles % out_subblock_w_ntiles == 0,
        "weight_block_w_ntiles {} should be divisible by weight_block_w_ntiles {}",
        weight_block_w_ntiles,
        out_subblock_w_ntiles);
    uint32_t weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
    uint32_t weight_block_h_ntiles = is_conv_1d_depthwise_conv ? act_block_h_ntiles : act_block_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles;

    // writer of conv op partially removes padding on the width
    // it removes the padding done for block width but it doesn't remove padding done for tiled width
    uint32_t output_channels_padded_to_tile_width = tt::round_up(output_channels, tt::constants::TILE_WIDTH);
    TT_FATAL(
        output_channels_padded_to_tile_width <= weight_matrix_width,
        "output_channels_padded_to_tile_width {} should be less than or equal to weight_matrix_width {}",
        output_channels_padded_to_tile_width,
        weight_matrix_width);
    uint32_t num_blocks_output_w =
        (uint32_t)std::ceil((double)output_channels_padded_to_tile_width / (double)weight_block_w_datums);
    uint32_t last_block_width_datums = (output_channels_padded_to_tile_width % weight_block_w_datums == 0)
                                           ? weight_block_w_datums
                                           : (output_channels_padded_to_tile_width % weight_block_w_datums);
    TT_FATAL(
        last_block_width_datums % tt::constants::TILE_WIDTH == 0,
        "last_block_width_datums {} should be divisible by TILE_WIDTH",
        last_block_width_datums);

    TT_FATAL(output.is_sharded(), "Output buffer must be sharded!");

    // out
    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;
    TT_FATAL(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    TT_FATAL(
        act_block_h_ntiles % out_subblock_h_ntiles == 0,
        "act_block_h_ntiles {} should be divisible by out_subblock_h_ntiles {}",
        act_block_h_ntiles,
        out_subblock_h_ntiles);

    const uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    const uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    const uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    const uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    const uint32_t in0_num_blocks_w = block_sharded ? conv_act_c_blocks : num_blocks_act_w;

    // weight
    const uint32_t weight_dram_addr = b.buffer()->address();

    // bias
    tt::tt_metal::Buffer* bias_buffer = nullptr;
    uint32_t bias_dram_addr = 0;
    uint32_t bias_ntiles = 0;
    if (has_bias) {
        bias_buffer = bias.value().buffer();
        bias_dram_addr = bias_buffer->address();
        bias_ntiles =
            bias.value().padded_shape()[3] / tt::constants::TILE_WIDTH;  // TODO: support non tile multiple sizes
    }

    uint32_t output_height_padded_to_tile_height = tt::round_up(act_matrix_height_unpadded, tt::constants::TILE_HEIGHT);
    uint32_t output_height_num_tiles = output_height_padded_to_tile_height / tt::constants::TILE_HEIGHT;
    TT_FATAL(
        output_height_num_tiles <= act_matrix_height_ntiles,
        "output_height_num_tiles {} should be less than or equal to act_matrix_height_ntiles {}",
        output_height_num_tiles,
        act_matrix_height_ntiles);

    const uint32_t window_outer = block_sharded ? 1 : num_blocks_act_w;
    const uint32_t window_inner = block_sharded ? filter_h : filter_h * filter_w / num_blocks_act_w;
    log_debug(tt::LogOp, "window_outer: {}, window_inner: {}", window_outer, window_inner);

    TT_FATAL(
        weight_matrix_width_ntiles % per_core_out_matrix_width_ntiles == 0,
        "weight_matrix_width_ntiles {} should be divisible by per_core_out_matrix_width_ntiles {}",
        weight_matrix_width_ntiles,
        per_core_out_matrix_width_ntiles);
    TT_FATAL(
        per_core_out_matrix_width_ntiles % weight_block_w_ntiles == 0,
        "per_core_out_matrix_width_ntiles {} should be divisible by weight_block_w_ntiles {}",
        per_core_out_matrix_width_ntiles,
        weight_block_w_ntiles);
    uint32_t num_blocks_weight_w_per_core = per_core_out_matrix_width_ntiles / weight_block_w_ntiles;
    if (height_sharded) {
        TT_FATAL(
            num_blocks_weight_w_per_core == num_blocks_weight_w,
            "num_blocks_weight_w_per_core {} should be equal to num_blocks_weight_w {}",
            num_blocks_weight_w_per_core,
            num_blocks_weight_w);
    }
    uint32_t num_weight_slices_width = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    uint32_t total_num_cores_per_weight_slice = 0;
    uint32_t total_num_cores_per_act_slice = 0;  // only used when (BLOCK_SHARDING && !transpose_mcast)
    if (block_sharded) {
        if (transpose_mcast) {
            TT_FATAL(
                num_cores_y % num_weight_slices_width == 0,
                "num_cores_y {} should be divisible by num_weight_slices_width {}",
                num_cores_y,
                num_weight_slices_width);
            uint32_t num_cores_y_per_weight_slice_width = num_cores_y / num_weight_slices_width;
            total_num_cores_per_weight_slice = num_cores_y_per_weight_slice_width * num_cores_x;
        } else {
            TT_FATAL(
                num_cores_x % num_weight_slices_width == 0,
                "num_cores_x {} should be divisible by num_weight_slices_width {}",
                num_cores_x,
                num_weight_slices_width);
            uint32_t num_cores_x_per_weight_slice_width = num_cores_x / num_weight_slices_width;
            uint32_t num_act_slices_height = act_matrix_height_ntiles / per_core_out_matrix_height_ntiles;
            total_num_cores_per_act_slice = num_cores_x * num_cores_y / num_act_slices_height;
            log_debug(tt::LogOp, "total_num_cores_per_act_slice: {}", total_num_cores_per_act_slice);
            total_num_cores_per_weight_slice = num_cores_x_per_weight_slice_width * num_cores_y;
        }
        TT_FATAL(
            total_num_cores_per_weight_slice * per_core_out_matrix_height_ntiles == act_matrix_height_ntiles,
            "total_num_cores_per_weight_slice {} * per_core_out_matrix_height_ntiles {} should be equal to "
            "act_matrix_height_ntiles {}",
            total_num_cores_per_weight_slice,
            per_core_out_matrix_height_ntiles,
            act_matrix_height_ntiles);
    } else {
        TT_FATAL(
            num_cores_y % num_weight_slices_width == 0,
            "num_cores_y {} should be divisible by num_weight_slices_width {}",
            num_cores_y,
            num_weight_slices_width);
        uint32_t num_cores_y_per_weight_slice_width = num_cores_y / num_weight_slices_width;
        total_num_cores_per_weight_slice = num_cores_y_per_weight_slice_width * num_cores_x;
        TT_FATAL(
            total_num_cores * per_core_out_matrix_height_ntiles >= act_matrix_height_ntiles,
            "total_num_cores {} * per_core_out_matrix_height_ntiles {} should be greater than or equal to "
            "act_matrix_height_ntiles {}",
            total_num_cores,
            per_core_out_matrix_height_ntiles,
            act_matrix_height_ntiles);
    }
    TT_FATAL(
        per_core_out_matrix_height_ntiles % act_block_h_ntiles == 0,
        "per_core_out_matrix_height_ntiles {} should be divisible by act_block_h_ntiles {}",
        per_core_out_matrix_height_ntiles,
        act_block_h_ntiles);
    uint32_t num_blocks_act_h_per_core = per_core_out_matrix_height_ntiles / act_block_h_ntiles;
    // TT_FATAL(per_core_out_matrix_height_ntiles % out_block_h_ntiles == 0);
    // uint32_t num_blocks_out_h_per_core = per_core_out_matrix_height_ntiles / out_block_h_ntiles;
    uint32_t num_blocks_out_h_per_core =
        (per_core_out_matrix_height_ntiles + out_block_h_ntiles - 1) / out_block_h_ntiles;
    bool act_height_sliced = per_core_out_matrix_height_ntiles < act_matrix_height_ntiles;
    if (not act_height_sliced) {
        TT_FATAL(
            num_blocks_act_h_per_core == num_blocks_act_h,
            "num_blocks_act_h_per_core {} should be equal to num_blocks_act_h {}",
            num_blocks_act_h_per_core,
            num_blocks_act_h);
        TT_FATAL(
            num_blocks_out_h_per_core == num_blocks_out_h,
            "num_blocks_out_h_per_core {} should be equal to num_blocks_out_h {}",
            num_blocks_out_h_per_core,
            num_blocks_out_h);
        TT_FATAL(num_cores_x == 1, "num_cores_x {} should be equal to 1", num_cores_x);
    }
    uint32_t act_block_h_datums_last_block =
        (per_core_out_matrix_height_ntiles - (num_blocks_act_h_per_core - 1) * act_block_h_ntiles) *
        tt::constants::TILE_HEIGHT;

    std::vector<std::vector<uint16_t>> conv_sharded_input_top_left_indices =
        ttnn::operations::sliding_window::generate_sliding_window_op_config(
            op_trace_metadata,
            shard_boundaries,
            stride_w,
            true,
            enable_split_reader ? act_block_h_datums_split : act_block_h_datums,
            enable_split_reader ? act_block_h_datums_split_last : 0);

    // create sharded ttnn config tensors
    DataType indices_tt_dtype = DataType::UINT16;
    // For 2d convs, each core in a column or row share the same specs
    CoreCoord grid_size = parallel_config.grid.bounding_box().grid_size();

    Tensor conv_reader_indices_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        conv_sharded_input_top_left_indices, parallel_config);
    conv_reader_indices_tensor = ttnn::operations::sliding_window::move_config_tensor_to_device(
        conv_reader_indices_tensor, parallel_config, block_sharded, a.device());

    const tt::tt_metal::DeviceStorage& conv_reader_indices_storage = conv_reader_indices_tensor.device_storage();

    TT_FATAL(act_matrix_height_ntiles % per_core_out_matrix_height_ntiles == 0, "Error");
    uint32_t total_active_num_cores_per_weight_slice = act_matrix_height_ntiles / per_core_out_matrix_height_ntiles;
    TT_FATAL(total_active_num_cores_per_weight_slice <= total_num_cores_per_weight_slice, "Error");
    uint32_t total_noop_cores = total_num_cores_per_weight_slice - total_active_num_cores_per_weight_slice;
    uint32_t total_active_num_cores = total_active_num_cores_per_weight_slice * num_weight_slices_width;
    if (block_sharded) {
        TT_FATAL(total_noop_cores == 0, "Error");
        TT_FATAL(total_active_num_cores == total_num_cores, "Error");
    }

    if (has_bias) {
        TT_FATAL(bias_ntiles == weight_matrix_width_ntiles, "Error");
    }
    uint32_t bias_ntiles_per_core = bias_ntiles / num_weight_slices_width;

    CoreRange all_cores(CoreCoord(0, 0), CoreCoord(num_cores_x - 1, num_cores_y - 1));
    TT_FATAL(total_active_num_cores >= num_cores_x, "Error");
    uint32_t num_active_cores_x = num_cores_x;
    uint32_t num_active_cores_y_with_full_x = total_active_num_cores / num_cores_x;
    uint32_t num_active_cores_x_last_y = total_active_num_cores % num_cores_x;
    TT_FATAL(
        (num_active_cores_x * num_active_cores_y_with_full_x) + num_active_cores_x_last_y == total_active_num_cores,
        "Error");

    std::set<CoreRange> all_active_cores_set;
    all_active_cores_set.insert(
        CoreRange(CoreCoord(0, 0), CoreCoord(num_active_cores_x - 1, num_active_cores_y_with_full_x - 1)));
    if (num_active_cores_x_last_y > 0) {
        all_active_cores_set.insert(CoreRange(
            CoreCoord(0, num_active_cores_y_with_full_x),
            CoreCoord(num_active_cores_x_last_y - 1, num_active_cores_y_with_full_x)));
    }
    CoreRangeSet all_active_cores(all_active_cores_set);
    std::set<CoreRange> noop_cores_set;
    if (total_noop_cores > 0) {
        TT_FATAL(
            total_noop_cores == num_cores_x - num_active_cores_x_last_y,
            "Expected total_noop_cores {} to be equal to num_cores_x {} - num_active_cores_x_last_y {}",
            total_noop_cores,
            num_cores_x,
            num_active_cores_x_last_y);
        noop_cores_set.insert(CoreRange(
            CoreCoord(num_active_cores_x_last_y, num_active_cores_y_with_full_x),
            CoreCoord(num_cores_x - 1, num_active_cores_y_with_full_x)));
    }
    CoreRangeSet noop_cores(noop_cores_set);

    // Mcast cores
    // If total_num_cores, there is no mcasting
    CoreCoord top_left_core = {(std::size_t)0, (std::size_t)0};
    CoreCoord top_left_core_plus_one = {(std::size_t)1, (std::size_t)1};
    CoreCoord bottom_right_core = {(std::size_t)num_cores_x - 1, (std::size_t)num_cores_y - 1};
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto top_left_core_plus_one_physical = device->worker_core_from_logical_core(top_left_core_plus_one);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    CoreRange mcast_sender_cores(top_left_core, top_left_core);  // If single core, this kernel doesn't do mcasting
    CoreRangeSet mcast_receiver_cores;
    uint32_t weights_mcast_sender_semaphore_id{};
    uint32_t weights_mcast_receiver_semaphore_id{};
    uint32_t act_mcast_sender_semaphore_id = 0;
    uint32_t act_mcast_receiver_semaphore_id = 0;
    std::vector<uint32_t> act_mcast_noc_y;
    if (block_sharded) {
        // 2D mcast
        if (transpose_mcast && !skip_mcast) {
            mcast_sender_cores = CoreRange(top_left_core, CoreCoord(0, num_cores_y - 1));
            mcast_receiver_cores = CoreRange(CoreCoord(1, 0), bottom_right_core);
        } else if (!skip_mcast) {
            mcast_sender_cores = CoreRange(top_left_core, CoreCoord(num_cores_x - 1, 0));
            mcast_receiver_cores = CoreRange(CoreCoord(0, 1), bottom_right_core);
        }
        weights_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
        weights_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
    } else {
        // 1D mcast
        if (total_num_cores > 1) {
            std::set<CoreRange> mcast_receiver_set;
            if (num_cores_x > 1) {
                mcast_receiver_set.insert(CoreRange(CoreCoord(1, 0), CoreCoord(num_active_cores_x - 1, 0)));
            }
            if (num_cores_y > 1) {
                if (num_active_cores_y_with_full_x >= 2) {
                    mcast_receiver_set.insert(CoreRange(
                        CoreCoord(0, 1), CoreCoord(num_active_cores_x - 1, num_active_cores_y_with_full_x - 1)));
                }
                if (num_active_cores_x_last_y > 0) {
                    mcast_receiver_set.insert(CoreRange(
                        CoreCoord(0, num_active_cores_y_with_full_x),
                        CoreCoord(num_active_cores_x_last_y - 1, num_active_cores_y_with_full_x)));
                }
            }
            mcast_receiver_cores = mcast_receiver_set;
            weights_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
            weights_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
        }
    }

    std::vector<uint32_t> reader_rt_args;
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_compile_time_args;

    uint32_t conv_act_c_read_bytes = conv_act_size_c * a.element_size() / conv_act_c_blocks;
    uint32_t act_block_w_extra_align_bytes =
        block_sharded
            ? (tt::round_up(shard_shape[1] * filter_h * filter_w, tt::constants::TILE_WIDTH) -
               (shard_shape[1] * filter_h * filter_w)) *
                  a.element_size()
            : (tt::round_up(shard_shape[1] * filter_w, tt::constants::TILE_WIDTH) - (shard_shape[1] * filter_w)) *
                  a.element_size();
    const uint32_t act_block_w_extra_align_scalars = act_block_w_extra_align_bytes / a.element_size();
    // When using block float format, we must handle cases where the data doesn't align to 16-scalar boundaries.
    // If act_block_w_extra_align_bytes contains a number of scalars that isn't a multiple of 16,
    // we need to zero out the temporary circular buffers used during the tiling process.
    // Failing to do this could allow residual junk data in L1 memory to corrupt valid input data.
    const bool needs_act_block_zero_out =
        act_block_w_extra_align_scalars % 16 != 0 && tt::tt_metal::is_block_float(output.dtype());

    const uint32_t tilized_act_tile_size = tt::tt_metal::detail::TileSize(tilized_act_df);

    // Only enable packer l1 accumulation when there are in0_num_blocks_w > 2, otherwise
    // unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
    // does a spill and reload, so need more than 2 blocks to use l1 acc for packer
    // For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
    const bool packer_l1_acc_en = determine_packer_l1_acc(packer_l1_acc, has_bias, in0_num_blocks_w);

    Conv2dConfig conv_config = Conv2dConfig{
        .weights_dtype = b.dtype(),
        .shard_layout = a.memory_config().memory_layout(),
        .output_layout = (untilize_out ? Layout::ROW_MAJOR : Layout::TILE),
        .enable_act_double_buffer = enable_act_double_buffer,
        .enable_weights_double_buffer = enable_weights_double_buffer,
        .enable_split_reader = enable_split_reader};
    std::vector<CBInfo> cb_info = get_cb_info(
        compute_kernel_config,
        block_config,
        p_config,
        b.padded_shape(),
        {filter_h, filter_w},
        conv_config,
        a.dtype(),
        output.dtype(),
        shard_shape,
        has_bias,
        is_conv_1d_depthwise_conv,
        skip_mcast);

    access_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).page_size = conv_sharded_input_top_left_indices[0].size();

    // call function to allocate circular buffers
    allocate_cbs(cb_info, program, all_cores, a, output, conv_reader_indices_tensor);

    const tt::tt_metal::CBHandle cb_sharded_act = get_cb_info_by_name(cb_info, Conv2dCb::ACT_SHARDED).handle;
    const tt::tt_metal::CBHandle cb_output = get_cb_info_by_name(cb_info, Conv2dCb::OUT).handle;
    const bool partials_cb_uses_output = get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).is_globally_allocated;
    log_debug(tt::LogOp, "partials_cb_uses_output: {}", partials_cb_uses_output);
    const tt::tt_metal::CBHandle cb_partials = get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).handle;

    std::string reader_kernel;
    std::string compute_kernel =
        "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize_col_major_out_blocks.cpp";
    std::string writer_mcast_sender_kernel;
    std::string writer_mcast_receiver_kernel;

    // Input should always be sharded in this conv; always use reader kernel for input shard with halo and padding
    if (filter_h >= 1 and filter_w >= 1) {
        if (!is_conv_1d_depthwise_conv && block_sharded) {
            // Block sharded conv
            reader_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp";
            writer_mcast_sender_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
            writer_mcast_receiver_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";
            act_mcast_sender_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);
            act_mcast_receiver_semaphore_id = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);

            if (transpose_mcast) {
                act_mcast_noc_y.reserve(num_cores_y);
                for (uint32_t core_idx_y = 0; core_idx_y < num_cores_y; ++core_idx_y) {
                    act_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
                }
            } else {
                // NOTE: using same var for x as well, this is intentional
                act_mcast_noc_y.reserve(num_cores_x);
                for (int32_t core_idx_x = 0; core_idx_x < num_cores_x; ++core_idx_x) {
                    act_mcast_noc_y.push_back(device->worker_core_from_logical_core({(uint32_t)core_idx_x, 0}).x);
                }
            }
        } else if (is_conv_1d_depthwise_conv) {
            // 1D Depthwise Conv (height sharded)
            TT_FATAL(
                act_block_w_datums == tt::round_up(conv_act_size_c * filter_w, tt::constants::TILE_WIDTH), "Error");

            compute_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/compute_depthwise_conv1d.cpp";
            reader_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_depthwise_conv1d.cpp";
            writer_mcast_sender_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
            writer_mcast_receiver_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";

        } else {
            // Height sharded conv
            TT_FATAL(
                act_block_w_datums == tt::round_up(conv_act_size_c * filter_w, tt::constants::TILE_WIDTH), "Error");

            reader_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp";

            writer_mcast_sender_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
            writer_mcast_receiver_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";
        }
    } else {
        TT_THROW("Sharded input not supported for this conv yet!");
    }

    uint32_t reader_arg_act_block_h_datums = (enable_split_reader ? act_block_h_datums_split : act_block_h_datums);
    TT_FATAL(reader_arg_act_block_h_datums % 2 == 0, "2 Indices are packed in one uint32_t word.");

    reader_compile_time_args = {
        (uint32_t)dilation_h,
        (uint32_t)dilation_w,
        (uint32_t)stride_w,
        (uint32_t)conv_act_c_read_bytes,
        (uint32_t)window_outer,
        (uint32_t)window_inner,
        (uint32_t)(enable_split_reader ? act_block_num_tiles_split / conv_act_c_blocks
                                       : act_block_num_tiles / conv_act_c_blocks),
        (uint32_t)filter_h,
        (uint32_t)filter_w,
        (uint32_t)conv_act_size_w + (pad_w),
        (uint32_t)act_block_w_extra_align_bytes,                          // only used for 1d systolic variant
        (uint32_t)num_blocks_act_h_per_core,                              // act_num_blocks_h
        (uint32_t)act_block_num_tiles,                                    // act_block_num_tiles
        (uint32_t)conv_act_c_blocks,                                      // act_w_num_outer
        (uint32_t)(transpose_mcast ? num_cores_y - 1 : num_cores_x - 1),  // act_mcast_num_dests
        (uint32_t)(transpose_mcast ? num_cores_y - 1 : num_cores_x - 1),  // act_mcast_num_cores
        (uint32_t)act_mcast_sender_semaphore_id,
        (uint32_t)act_mcast_receiver_semaphore_id,
        (uint32_t)act_block_num_tiles * tilized_act_tile_size,  // act_mcast_sender_size_bytes
        (uint32_t)(transpose_mcast ? 1 : 0),
        (uint32_t)needs_act_block_zero_out,  // zero_out_act_cb
        get_cb_info_by_name(cb_info, Conv2dCb::ACT).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_SHARDED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_TILIZED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16).index,
        get_cb_info_by_name(cb_info, Conv2dCb::L1_ARRAY).index,
    };

    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> writer_mcast_sender_defines;
    std::map<std::string, std::string> compute_defines;
    if (total_num_cores == 1) {
        writer_mcast_sender_defines["SKIP_MCAST"] = "1";
    }
    if (skip_mcast) {
        reader_defines["SKIP_MCAST"] = "1";
    }
    if (has_bias) {
        writer_defines["FUSE_BIAS"] = "1";
        writer_mcast_sender_defines["FUSE_BIAS"] = "1";
        compute_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == unary::UnaryOpType::RELU) {
            compute_defines["PACK_RELU"] = "1";
        } else {
            compute_defines.merge(ttnn::operations::unary::utils::get_defines(
                fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i"));
        }
    }

    if (block_sharded) {
        compute_defines["BLOCK_SHARDED"] = "1";
    }

    if (enable_split_reader) {
        reader_defines["SPLIT_READER"] = "1";
        compute_defines["SPLIT_READER"] = "1";
    }

    if (packer_l1_acc_en) {
        compute_defines["PACKER_L1_ACC"] = "1";
    }
    if (weight_block_w_ntiles <= 8) {
        compute_defines["PACKER_UNTILIZE"] = "1";
    }
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), total_num_cores, compute_defines, ttnn::get_throttle_level(compute_kernel_config));

    for (auto elem : compute_defines) {
        log_debug(tt::LogOp, "compute_defines: {} = {}", elem.first, elem.second);
    }

    writer_compile_time_args = {
        get_cb_info_by_name(cb_info, Conv2dCb::WEIGHTS).index,
        get_cb_info_by_name(cb_info, Conv2dCb::BIAS).index,
        (uint32_t)(bias_buffer == nullptr ? 0 : (bias_buffer->buffer_type() == BufferType::DRAM ? 1 : 0)),
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_SECOND_READER).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_SHARDED).index,
        get_cb_info_by_name(cb_info, Conv2dCb::READER_INDICES).index,
        num_blocks_act_w,
        weight_block_num_tiles,
        conv_act_c_blocks,
        weight_block_h_ntiles,
        weight_block_w_ntiles,
        weight_matrix_width_ntiles,
        weight_matrix_width_ntiles * weight_block_h_ntiles,
        weight_block_w_ntiles,

        // bias
        bias_ntiles_per_core,

        num_blocks_act_h_per_core,
        num_blocks_weight_w_per_core};

    if (enable_split_reader) {
        std::vector<uint32_t> split_reader_args = {
            (uint32_t)(conv_act_c_read_bytes > 0),
            (uint32_t)act_block_num_tiles_split_last / conv_act_c_blocks,
            (uint32_t)conv_act_c_read_bytes,
            (uint32_t)filter_w,                       // weight_size_w
            (uint32_t)(conv_act_size_w + pad_w),      // conv_act_size_w_padded
            (uint32_t)act_block_w_extra_align_bytes,  // only used for 1d systolic variant
            (uint32_t)needs_act_block_zero_out,
            (uint32_t)dilation_h,
            (uint32_t)dilation_w,
            (uint32_t)stride_w};
        writer_compile_time_args.insert(
            writer_compile_time_args.end(), split_reader_args.begin(), split_reader_args.end());
    } else {
        std::vector<uint32_t> split_reader_args(10, 0);
        writer_compile_time_args.insert(
            writer_compile_time_args.end(), split_reader_args.begin(), split_reader_args.end());
    }

    std::vector<uint32_t> compute_kernel_args = {
        act_block_w_ntiles,
        act_num_subblocks,
        act_block_num_tiles,
        act_subblock_num_tiles,
        act_subblock_h_ntiles,

        weight_num_subblocks,
        weight_block_num_tiles,
        weight_block_w_ntiles,

        num_blocks_act_h_per_core,
        in0_num_blocks_w,
        num_blocks_weight_w_per_core,

        out_subblock_h_ntiles_padded,
        out_subblock_w_ntiles,
        out_subblock_num_tiles,

        height_sharded,
        untilize_out,

        bias_ntiles_per_core,

        get_cb_info_by_name(cb_info, Conv2dCb::BIAS).index,
        skip_mcast ? get_cb_info_by_name(cb_info, Conv2dCb::ACT_TILIZED).index
                   : get_cb_info_by_name(cb_info, Conv2dCb::ACT).index,
        get_cb_info_by_name(cb_info, Conv2dCb::WEIGHTS).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_ROW_MAJOR_BFLOAT16).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_SECOND_READER).index,
        get_cb_info_by_name(cb_info, Conv2dCb::MATMUL_PARTIALS).index,
        get_cb_info_by_name(cb_info, Conv2dCb::ACT_TILIZED).index,

        get_cb_info_by_name(cb_info, Conv2dCb::OUT).index,
        get_cb_info_by_name(cb_info, Conv2dCb::TEMP_SUM).index,
        partials_cb_uses_output};

    const tt::tt_metal::NOC writer_mcast_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());
    const tt::tt_metal::NOC reader_noc =
        writer_mcast_noc == tt::tt_metal::NOC::NOC_0 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0;
    auto writer_mcast_sender_id = CreateKernel(
        program,
        writer_mcast_sender_kernel,
        mcast_sender_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_mcast_noc,
            .compile_args = writer_compile_time_args,
            .defines = writer_mcast_sender_defines});

    tt::tt_metal::KernelHandle writer_mcast_receiver_id = -1;
    if (total_num_cores > 1) {
        writer_mcast_receiver_id = CreateKernel(
            program,
            writer_mcast_receiver_kernel,
            mcast_receiver_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = writer_mcast_noc,
                .compile_args = writer_compile_time_args,
                .defines = writer_defines});
    }

    auto reader_id = CreateKernel(
        program,
        reader_kernel,
        all_active_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .compile_args = reader_compile_time_args,
            .defines = reader_defines});

    // Compile compute kernel for active cores only
    // Compile blank kernel for noop cores
    auto compute_id = CreateKernel(
        program,
        compute_kernel,
        all_active_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    for (uint32_t core_i = 0; core_i < total_active_num_cores; core_i++) {
        uint32_t core_x_i = core_i % num_cores_x;
        uint32_t core_y_i = core_i / num_cores_x;
        CoreRange core(CoreCoord(core_x_i, core_y_i), CoreCoord(core_x_i, core_y_i));
        bool noop_core = false;

        // per core specific args
        uint32_t weight_slice_i;
        if (block_sharded && transpose_mcast || !block_sharded) {
            weight_slice_i = core_i / total_num_cores_per_weight_slice;
        } else {
            weight_slice_i = core_i % total_num_cores_per_act_slice;
        }
        uint32_t out_start_tile_id_w = weight_slice_i * per_core_out_matrix_width_ntiles;
        uint32_t bias_tile_offset = weight_slice_i * per_core_out_matrix_width_ntiles;
        if (has_bias) {
            TT_FATAL(
                bias_tile_offset < bias_ntiles,
                "bias_tile_offset {} should be less than bias_ntiles {}",
                bias_tile_offset,
                bias_ntiles);
        }

        if (block_sharded) {
            bool reader_is_noc_0 = reader_noc == tt::tt_metal::NOC::NOC_0;

            if (transpose_mcast) {
                CoreCoord bottom_core = {(std::size_t)core_x_i, (std::size_t)num_cores_y - 1};
                auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);

                uint32_t act_mcast_dest_noc_start_x = bottom_core_physical.x;
                uint32_t act_mcast_dest_noc_start_y =
                    reader_is_noc_0 ? top_left_core_physical.y : bottom_core_physical.y;
                uint32_t act_mcast_dest_noc_end_x = bottom_core_physical.x;
                uint32_t act_mcast_dest_noc_end_y = reader_is_noc_0 ? bottom_core_physical.y : top_left_core_physical.y;
                reader_rt_args = {
                    (uint32_t)noop_core,

                    // mcast args
                    act_mcast_dest_noc_start_x,
                    act_mcast_dest_noc_start_y,
                    act_mcast_dest_noc_end_x,
                    act_mcast_dest_noc_end_y,
                    core_y_i,                          // act_mcast_sender_id (goes down the column)
                    (uint32_t)bottom_core_physical.x,  // act_mcast_sender_noc_x
                };
                reader_rt_args.insert(
                    reader_rt_args.end(), act_mcast_noc_y.begin(), act_mcast_noc_y.end());  // act_mcast_sender_noc_y
            } else {
                CoreCoord core = {core_x_i, core_y_i};
                auto core_physical = device->worker_core_from_logical_core(core);
                CoreCoord bottom_right_core = {(std::size_t)num_cores_x - 1, (std::size_t)num_cores_y - 1};
                auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

                uint32_t act_mcast_dest_noc_start_x =
                    reader_is_noc_0 ? top_left_core_physical.x : bottom_right_core_physical.x;
                uint32_t act_mcast_dest_noc_start_y = core_physical.y;
                uint32_t act_mcast_dest_noc_end_x =
                    reader_is_noc_0 ? bottom_right_core_physical.x : top_left_core_physical.x;
                uint32_t act_mcast_dest_noc_end_y = core_physical.y;
                reader_rt_args = {
                    (uint32_t)noop_core,

                    // mcast args
                    act_mcast_dest_noc_start_x,
                    act_mcast_dest_noc_start_y,
                    act_mcast_dest_noc_end_x,
                    act_mcast_dest_noc_end_y,
                    core_x_i,                   // act_mcast_sender_id (goes along the row)
                    (uint32_t)core_physical.y,  // act_mcast_sender_noc_x
                };
                reader_rt_args.insert(
                    reader_rt_args.end(), act_mcast_noc_y.begin(), act_mcast_noc_y.end());  // act_mcast_sender_noc_y
            }
        } else {
            reader_rt_args = {(uint32_t)noop_core};
        }
        SetRuntimeArgs(program, reader_id, core, reader_rt_args);

        std::vector<uint32_t> sender_rt_args = {
            weight_dram_addr,
            bias_dram_addr,
            out_start_tile_id_w,

            // bias
            bias_tile_offset,

            (uint32_t)noop_core};

        if (block_sharded) {
            // 2D mcast
            if (transpose_mcast) {
                CoreCoord right_core = {(std::size_t)num_cores_x - 1, (std::size_t)core_y_i};
                auto right_core_physical = device->worker_core_from_logical_core(right_core);
                if (core_x_i == 0) {
                    // sender
                    if (writer_mcast_noc == tt::tt_metal::NOC::NOC_0) {
                        sender_rt_args.push_back(top_left_core_plus_one_physical.x);  // weights_mcast_dest_noc_start_x
                        sender_rt_args.push_back(right_core_physical.y);              // weights_mcast_dest_noc_start_y
                        sender_rt_args.push_back(bottom_right_core_physical.x);       // weights_mcast_dest_noc_end_x
                        sender_rt_args.push_back(right_core_physical.y);              // weights_mcast_dest_noc_end_y
                    } else {
                        sender_rt_args.push_back(bottom_right_core_physical.x);       // weights_mcast_dest_noc_start_x
                        sender_rt_args.push_back(right_core_physical.y);              // weights_mcast_dest_noc_start_y
                        sender_rt_args.push_back(top_left_core_plus_one_physical.x);  // weights_mcast_dest_noc_end_x
                        sender_rt_args.push_back(right_core_physical.y);              // weights_mcast_dest_noc_end_y
                    }

                    sender_rt_args.push_back(num_cores_x - 1);  // weights_mcast_num_dests
                    sender_rt_args.push_back(num_cores_x - 1);  // weights_mcast_num_cores
                    sender_rt_args.push_back(weights_mcast_sender_semaphore_id);
                    sender_rt_args.push_back(weights_mcast_receiver_semaphore_id);
                    sender_rt_args.push_back(output.buffer()->aligned_page_size());

                    SetRuntimeArgs(program, writer_mcast_sender_id, core, sender_rt_args);
                } else {
                    std::vector<uint32_t> receiver_rt_args{
                        (uint32_t)noop_core,
                        top_left_core_physical.x,  // weights_mcast_sender_noc_x
                        right_core_physical.y,     // weights_mcast_sender_noc_y
                        weights_mcast_sender_semaphore_id,
                        weights_mcast_receiver_semaphore_id};

                    SetRuntimeArgs(program, writer_mcast_receiver_id, core, receiver_rt_args);
                }
            } else {
                CoreCoord top_core = {(std::size_t)core_x_i, 0};
                auto top_core_physical = device->worker_core_from_logical_core(top_core);
                if (core_y_i == 0) {
                    // sender
                    if (writer_mcast_noc == tt::tt_metal::NOC::NOC_0) {
                        sender_rt_args.push_back(top_core_physical.x);                // weights_mcast_dest_noc_start_x
                        sender_rt_args.push_back(top_left_core_plus_one_physical.y);  // weights_mcast_dest_noc_start_y
                        sender_rt_args.push_back(top_core_physical.x);                // weights_mcast_dest_noc_end_x
                        sender_rt_args.push_back(bottom_right_core_physical.y);       // weights_mcast_dest_noc_end_y
                    } else {
                        sender_rt_args.push_back(top_core_physical.x);                // weights_mcast_dest_noc_start_x
                        sender_rt_args.push_back(bottom_right_core_physical.y);       // weights_mcast_dest_noc_start_y
                        sender_rt_args.push_back(top_core_physical.x);                // weights_mcast_dest_noc_end_x
                        sender_rt_args.push_back(top_left_core_plus_one_physical.y);  // weights_mcast_dest_noc_end_y
                    }

                    sender_rt_args.push_back(num_cores_y - 1);  // weights_mcast_num_dests
                    sender_rt_args.push_back(num_cores_y - 1);  // weights_mcast_num_cores
                    sender_rt_args.push_back(weights_mcast_sender_semaphore_id);
                    sender_rt_args.push_back(weights_mcast_receiver_semaphore_id);

                    SetRuntimeArgs(program, writer_mcast_sender_id, core, sender_rt_args);
                } else {
                    std::vector<uint32_t> receiver_rt_args{
                        (uint32_t)noop_core,
                        top_core_physical.x,       // weights_mcast_sender_noc_x
                        top_left_core_physical.y,  // weights_mcast_sender_noc_y
                        weights_mcast_sender_semaphore_id,
                        weights_mcast_receiver_semaphore_id};
                    SetRuntimeArgs(program, writer_mcast_receiver_id, core, receiver_rt_args);
                }
            }
        } else {
            // 1D mcast
            if (core_x_i == 0 and core_y_i == 0) {
                // sender
                if (writer_mcast_noc == tt::tt_metal::NOC::NOC_0) {
                    sender_rt_args.push_back(top_left_core_physical.x);      // weights_mcast_dest_noc_start_x
                    sender_rt_args.push_back(top_left_core_physical.y);      // weights_mcast_dest_noc_start_y
                    sender_rt_args.push_back(bottom_right_core_physical.x);  // weights_mcast_dest_noc_end_x
                    sender_rt_args.push_back(bottom_right_core_physical.y);  // weights_mcast_dest_noc_end_y
                } else {
                    sender_rt_args.push_back(bottom_right_core_physical.x);  // weights_mcast_dest_noc_start_x
                    sender_rt_args.push_back(bottom_right_core_physical.y);  // weights_mcast_dest_noc_start_y
                    sender_rt_args.push_back(top_left_core_physical.x);      // weights_mcast_dest_noc_end_x
                    sender_rt_args.push_back(top_left_core_physical.y);      // weights_mcast_dest_noc_end_y
                }
                sender_rt_args.push_back(total_active_num_cores - 1);  // weights_mcast_num_dests
                sender_rt_args.push_back(total_num_cores - 1);         // weights_mcast_num_cores
                sender_rt_args.push_back(weights_mcast_sender_semaphore_id);
                sender_rt_args.push_back(weights_mcast_receiver_semaphore_id);

                SetRuntimeArgs(program, writer_mcast_sender_id, core, sender_rt_args);
            } else {
                std::vector<uint32_t> receiver_rt_args{
                    (uint32_t)noop_core,
                    top_left_core_physical.x,  // weights_mcast_sender_noc_x
                    top_left_core_physical.y,  // weights_mcast_sender_noc_y
                    weights_mcast_sender_semaphore_id,
                    weights_mcast_receiver_semaphore_id};

                SetRuntimeArgs(program, writer_mcast_receiver_id, core, receiver_rt_args);
            }
        }

    }  // for num_cores

    std::vector<CoreCoord> mcast_sender_cores_vec =
        grid_to_cores(mcast_sender_cores.start_coord, mcast_sender_cores.end_coord, true);
    // Capture conv_reader_indices_storage to cache this with the program
    auto override_runtime_arguments_callback =
        [mcast_sender_cores = mcast_sender_cores_vec,
         writer_mcast_sender_id = writer_mcast_sender_id,
         cb_sharded_act = cb_sharded_act,
         cb_output = cb_output,
         cb_partials = cb_partials,
         partials_cb_uses_output = partials_cb_uses_output,
         has_bias = has_bias,
         conv_reader_indices_storage = conv_reader_indices_storage](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            // Reader config indices is an optional static sharded tensor, so no need to update address
            TT_FATAL(output_tensors.size() == 1, "Error");

            auto src_buffer_a = input_tensors.at(0).buffer();
            auto src_buffer_b = input_tensors.at(1).buffer();

            std::optional<tt::tt_metal::Buffer*> src_buffer_c = std::nullopt;
            if (has_bias) {
                src_buffer_c = optional_input_tensors.at(0).value().buffer();
                TT_FATAL(src_buffer_c.value() != nullptr, "Error");
            }

            auto& writer_sender_kernel_args_by_core = GetRuntimeArgs(program, writer_mcast_sender_id);
            for (const auto& core : mcast_sender_cores) {
                auto& runtime_args = writer_sender_kernel_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer_b->address();
                if (has_bias) {
                    runtime_args[1] = (*src_buffer_c)->address();
                }
            }

            UpdateDynamicCircularBufferAddress(program, cb_sharded_act, *src_buffer_a);
            UpdateDynamicCircularBufferAddress(program, cb_output, *output_tensors.at(0).buffer());
            if (partials_cb_uses_output) {
                UpdateDynamicCircularBufferAddress(program, cb_partials, *output_tensors.at(0).buffer());
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

tt::tt_metal::operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_v2_new(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    const std::optional<unary::UnaryWithParam>& fused_activation,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    DataType output_dtype,
    std::array<std::uint32_t, 4> input_tensor_shape,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    Tensor& output,
    bool enable_act_double_buffer,
    bool enable_weights_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    ttnn::operations::sliding_window::ParallelConfig parallel_config{
        .grid = a.shard_spec().value().grid,
        .shard_scheme = a.memory_config().memory_layout(),
        .shard_orientation = a.shard_spec().value().orientation};

    std::vector<uint32_t> op_trace_metadata =
        ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    std::vector<sliding_window::ShardBoundary> shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config, op_trace_metadata);

    if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        return multi_core_optimized_conv_width_sharded_v2_impl(
            program,
            a,
            b,
            ttnn::Shape(input_tensor_shape),
            bias,
            sliding_window_config,
            parallel_config,
            op_trace_metadata,
            shard_boundaries,
            output_channels,
            groups,
            untilize_out,
            bias.has_value(),
            fused_activation,
            parallelization_config,
            block_config,
            output,
            compute_kernel_config.value(),
            enable_act_double_buffer,
            enable_weights_double_buffer,
            enable_subblock_padding);
    }
    return multi_core_optimized_conv_sharded_v2_impl(
        program,
        a,
        b,
        ttnn::Shape(input_tensor_shape),
        bias,
        sliding_window_config,
        parallel_config,
        op_trace_metadata,
        shard_boundaries,
        output_channels,
        groups,
        untilize_out,
        bias.has_value(),
        fused_activation,
        parallelization_config,
        block_config,
        a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR,
        output,
        compute_kernel_config.value(),
        enable_act_double_buffer,
        enable_weights_double_buffer,
        enable_split_reader,
        enable_subblock_padding);
}
}  // namespace conv2d

}  // namespace ttnn::operations::conv
