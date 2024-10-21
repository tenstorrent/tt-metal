// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace ttnn::operations::conv  {

namespace conv2d {

using namespace tt;

const uint32_t act_cb = CB::c_in0;
const uint32_t weight_cb = CB::c_in1;
const uint32_t bias_cb = CB::c_in2;
const uint32_t sharded_act_cb = CB::c_in3;
const uint32_t cb_for_reader_indices = CB::c_in4;
const uint32_t cb_for_l1_array = CB::c_in5;
const uint32_t act_cb_row_major_bfloat16 = CB::c_in6;
const uint32_t act_cb_second_reader = CB::c_in7;
const uint32_t matmul_partials_cb = CB::c_intermed0;
const uint32_t tilize_mode_tilized_act_cb = CB::c_intermed1;
const uint32_t untilize_mode_reblock_cb = CB::c_intermed2;
const uint32_t out0_cb = CB::c_out0;

operation::ProgramWithCallbacks multi_core_optimized_conv_width_sharded_v2_impl(
    tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const ttnn::Shape& ashape,
    std::optional<const Tensor> bias,
    const std::optional<const Tensor> conv_reader_indices,
    sliding_window::SlidingWindowConfig sliding_window_config,
    uint32_t output_channels,
    uint32_t groups,
    bool untilize_out,
    bool has_bias,
    bool fuse_relu,
    const OptimizedConvParallelizationConfig& parallelization_config,
    const OptimizedConvBlockConfig& block_config,
    bool use_shallow_conv_variant,
    bool transpose_mcast,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config,
    bool enable_act_double_buffer,
    bool enable_split_reader,
    bool enable_subblock_padding) {
    bool pass = true;
    enable_split_reader = false;
    tt_metal::Device* device = a.device();
    TT_FATAL(a.get_layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_FATAL(a.memory_config().is_sharded(), "Conv activation must be sharded.");
    TT_FATAL(output_channels <= b.get_legacy_shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    uint32_t weight_block_w_ntiles = parallelization_config.per_core_out_matrix_width_ntiles;
    uint32_t out_block_h_ntiles = parallelization_config.per_core_out_matrix_height_ntiles;
    uint32_t out_subblock_h_ntiles = block_config.out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles = block_config.out_subblock_w_ntiles;

    // out_subblock_h_ntiles = 8;

    tt::DataFormat act_df = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    tt::DataFormat weight_df = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    tt::DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    tt::DataFormat bias_df =
        has_bias ? tt_metal::datatype_to_dataformat_converter(bias.value().get_dtype()) : tt::DataFormat::Float16_b;
    tt::DataFormat tilized_act_df = out_df;

    log_debug(LogOp, "act_df: {}", act_df);
    log_debug(LogOp, "weight_df: {}", weight_df);
    log_debug(LogOp, "out_df: {}", out_df);
    log_debug(LogOp, "bias_df: {}", bias_df);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    if (fp32_dest_acc_en and (out_subblock_h_ntiles * out_subblock_w_ntiles > 4)) {
        if (out_subblock_w_ntiles >= 4) {
            out_subblock_h_ntiles = 1;
            out_subblock_w_ntiles = tt::tt_metal::find_max_block_size(out_subblock_w_ntiles, 4);
        } else {
            while (out_subblock_h_ntiles * out_subblock_w_ntiles > 4) {
                uint32_t div = tt::tt_metal::find_max_divisor(out_subblock_h_ntiles, out_subblock_h_ntiles - 1);
                out_subblock_h_ntiles = tt::tt_metal::find_max_block_size(out_subblock_h_ntiles, div);
            }
        }
    }
    // it is bad for compute, pad act_block_h_ntiles
    uint32_t max_num_subblock = fp32_dest_acc_en ? 4 : 8;
    uint32_t max_subblock_h = fp32_dest_acc_en ? 4 : 8;
    uint32_t act_block_h_ntiles_padded = act_block_h_ntiles;
    uint32_t out_subblock_h_ntiles_padded = out_subblock_h_ntiles;
    // bool enable_subblock_padding = false;
    // bool enable_split_reader = false;
    // enable_act_double_buffer = false;
    if (enable_subblock_padding) {
        TT_FATAL(act_block_h_ntiles == out_block_h_ntiles, "to pad subblock, the number of blocks on height dim must be 1");

        if ((out_subblock_w_ntiles * out_subblock_h_ntiles <= max_num_subblock / 2) and (out_subblock_w_ntiles == weight_block_w_ntiles) and (act_block_h_ntiles == out_block_h_ntiles)) {
            uint32_t num_subblock_h = act_block_h_ntiles / out_subblock_h_ntiles;
            uint32_t num_iter = max_subblock_h - out_subblock_h_ntiles;
            uint32_t new_out_subblock_h = out_subblock_h_ntiles;
            uint32_t preferred_out_subblock_h = out_subblock_h_ntiles;

            for (uint32_t i=0; i < num_iter; ++i) {
                new_out_subblock_h += 1;
                uint32_t new_num_subblock_h = (act_block_h_ntiles + new_out_subblock_h - 1) / new_out_subblock_h;

                if (new_num_subblock_h < num_subblock_h and (out_subblock_w_ntiles * new_out_subblock_h <= max_num_subblock)) {
                    num_subblock_h = new_num_subblock_h;
                    preferred_out_subblock_h = new_out_subblock_h;
                }
            }
            out_subblock_h_ntiles_padded = preferred_out_subblock_h;
            act_block_h_ntiles_padded = out_subblock_h_ntiles_padded * num_subblock_h;
        }
    }

    // TT_FATAL(out_block_h_ntiles == act_block_h_ntiles, "Error"); // TODO: fix output block sizing
    TT_FATAL(
        out_block_h_ntiles >= act_block_h_ntiles,
        "Output block height (in # of tiles) ({}) should be greater than or equal to activation block height (in # of "
        "tiles) ({})", out_block_h_ntiles, act_block_h_ntiles);

    // Tensor b has weights and it should be tiled layout after converting conv weights into weight matrix
    TT_FATAL(b.get_layout() == Layout::TILE, "Conv weights should be in tiled layout");
    TT_FATAL(b.get_legacy_shape()[0] == 1, "Conv weight matrix shape is invalid");
    TT_FATAL(b.get_legacy_shape()[1] == 1, "Conv weight matrix shape is invalid");
    uint32_t weight_matrix_height = b.get_legacy_shape()[2];
    uint32_t weight_matrix_width = b.get_legacy_shape()[3];
    uint32_t weight_matrix_height_ntiles = weight_matrix_height / TILE_HEIGHT;
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / TILE_WIDTH;

    // Partitions conv inner dim into blocks to support sharding along this dim
    // TODO: Only 2D convs with sharded input use this, but we can uplift to support generically
    // TODO: Only updated variables which is affected, but there may be more that needs to account for this
    // TODO: Loop naming in reader, writer, and compute kernels could also be cleaned up
    // TODO: Can conv_act_c_blocks be same as num_blocks_act_w?
    auto shard_shape = a.shard_spec().value().shape;

    // parallelization config
    const auto& p_config = parallelization_config;
    uint32_t num_cores_x = p_config.grid_size.x;
    uint32_t num_cores_y = p_config.grid_size.y;
    uint32_t total_num_cores = p_config.num_cores_c;
    TT_FATAL(num_cores_x < 13, "Error");
    TT_FATAL(num_cores_y < 10, "Error");
    uint32_t per_core_out_matrix_height_ntiles = p_config.per_core_out_matrix_height_ntiles;
    uint32_t per_core_out_matrix_width_ntiles = p_config.per_core_out_matrix_width_ntiles;

    // weight_width_sliced determines is 1d-sysarr-conv or 2d-sysarr-conv
    bool weight_width_sliced = per_core_out_matrix_width_ntiles < weight_matrix_width_ntiles;
    // uint32_t conv_act_c_blocks = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    uint32_t input_channels_padded = shard_shape[1] * total_num_cores;
    // TT_FATAL(conv_act_c_blocks == p_config.num_cores_c, "Error");
    TT_FATAL(input_channels_padded >= ashape[3], "Incorrect padding of input channels!");
    // check is for 16-byte alignment
    TT_FATAL(
        input_channels_padded % 16 == 0,
        "Expected input channels to be padded for 16 byte alignment in L1");  // TODO: For bfp16, check if its divisible
                                                                              // by 8 not 16.
    // Always use split reader for first conv in resnet which has input channels = 16
    // TODO: Expose option to split readers for 1D convs to python?
    // bool split_reader = use_shallow_conv_variant;
    if (enable_split_reader) {
        TT_FATAL(not weight_width_sliced, "split reader does not work with 2d conv");
        TT_FATAL((act_block_h_ntiles / block_config.out_subblock_h_ntiles) >= 2, "split reader needs to have at leaset two subblocks");
    }
    bool split_reader = use_shallow_conv_variant or enable_split_reader;
    if (split_reader) {
        TT_FATAL(
            block_config.act_block_h_ntiles % block_config.out_subblock_h_ntiles == 0,
            "Out_block_h must be divisible by out_subblock_h!");
    }

    ttnn::Shape ashape_with_channels_padded(std::vector<uint32_t>{ashape[0], ashape[1], ashape[2], input_channels_padded});

    uint32_t conv_act_size_h = ashape_with_channels_padded[1];
    uint32_t conv_act_size_w = ashape_with_channels_padded[2];
    uint32_t conv_act_size_c = ashape_with_channels_padded[3];

    uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;  // filter_h
    uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
    uint32_t stride_h = (uint32_t)sliding_window_config.stride_hw.first;
    uint32_t stride_w = (uint32_t)sliding_window_config.stride_hw.second;
    uint32_t dilation_h = (uint32_t)sliding_window_config.dilation_hw.first;
    uint32_t dilation_w = (uint32_t)sliding_window_config.dilation_hw.second;

    uint32_t pad_h = (uint32_t)sliding_window_config.pad_hw.first;
    uint32_t pad_w = (uint32_t)sliding_window_config.pad_hw.second;

    uint32_t input_size_h = conv_act_size_h + (pad_h*2);
    uint32_t input_size_w = conv_act_size_w + (pad_w*2);


    // Compute the 2d matrix shape
    auto [act_matrix_shape, act_matrix_shape_unpadded] =
        optimized_conv_op_utils::compute_opt_conv_activation_as_mm_shape(
            ashape_with_channels_padded.value, sliding_window_config, out_block_h_ntiles);
    TT_FATAL(act_matrix_shape.size() == 3, "Error");
    TT_FATAL(act_matrix_shape[0] == 1, "Error");
    uint32_t act_matrix_height = (uint32_t)act_matrix_shape[1];
    uint32_t act_matrix_width = (uint32_t)act_matrix_shape[2];
    uint32_t act_matrix_height_unpadded = (uint32_t)act_matrix_shape_unpadded[1];
    uint32_t act_matrix_width_unpadded = (uint32_t)act_matrix_shape_unpadded[2];

    // TODO: Move all these TT_FATALs/checks to validate?

    if (has_bias) {
        // Tensor bias is of shape {output_channels}
        TT_FATAL(bias.has_value(), "Error");
        TT_FATAL(bias.value().buffer() != nullptr, "Error");
        auto bias_shape_without_padding = bias.value().get_logical_shape();
        TT_FATAL(bias_shape_without_padding[0] == 1, "Bias should have batch == 1");
    }

    // Normal matrix shape check
    TT_FATAL(act_matrix_width == weight_matrix_height, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_FATAL(act_matrix_height % TILE_HEIGHT == 0, "Height of activation matrix needs to be divisible by 32");
    TT_FATAL(act_matrix_width % TILE_WIDTH == 0, "Width of activation matrix needs to be divisible by 32");
    TT_FATAL(weight_matrix_height % TILE_HEIGHT == 0, "Height of weight matrix needs to be divisible by 32");
    TT_FATAL(weight_matrix_width % TILE_WIDTH == 0, "Width of weight matrix needs to be divisible by 32");

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
    uint32_t act_matrix_height_ntiles = act_matrix_height / TILE_HEIGHT;
    uint32_t act_matrix_width_ntiles = act_matrix_width / TILE_WIDTH;

    TT_FATAL(act_matrix_height_ntiles % act_block_h_ntiles == 0, "Error");
    TT_FATAL(act_matrix_width_ntiles % act_block_w_ntiles == 0, "Error");
    TT_FATAL(weight_matrix_width_ntiles % weight_block_w_ntiles == 0, "Error");
    TT_FATAL(act_matrix_height_ntiles % out_block_h_ntiles == 0, "Error");

    uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    uint32_t num_blocks_out_h = act_matrix_height_ntiles / out_block_h_ntiles;
    uint32_t num_blocks_act_w = act_matrix_width_ntiles / act_block_w_ntiles;
    uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

    TT_FATAL(num_blocks_act_w%total_num_cores==0, "{} {}", num_blocks_act_w, total_num_cores);
    uint32_t per_core_num_blocks_act_w = num_blocks_act_w/total_num_cores;


    // act block info
    uint32_t act_block_w_datums = act_matrix_width / num_blocks_act_w;
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;

    uint32_t act_block_h_nsubblocks = block_config.act_block_h_ntiles / block_config.out_subblock_h_ntiles;
    uint32_t act_block_h_nsubblocks_split = act_block_h_nsubblocks;
    uint32_t act_block_h_nsubblocks_split_last = 0;
    if (split_reader) {
        act_block_h_nsubblocks_split_last = act_block_h_nsubblocks / 2;
        act_block_h_nsubblocks_split = act_block_h_nsubblocks - act_block_h_nsubblocks_split_last;
    }
    uint32_t act_block_h_datums_split = act_block_h_nsubblocks_split * out_subblock_h_ntiles * TILE_HEIGHT;
    uint32_t act_block_h_datums_split_last = act_block_h_nsubblocks_split_last * out_subblock_h_ntiles * TILE_HEIGHT;

    uint32_t act_block_num_tiles_split = act_block_h_nsubblocks_split * out_subblock_h_ntiles * act_block_w_ntiles;
    uint32_t act_block_num_tiles_split_last = act_block_h_nsubblocks_split_last * out_subblock_h_ntiles * act_block_w_ntiles;

    log_debug(LogOp, "act_block_h_nsubblocks_split: {}", act_block_h_nsubblocks_split);
    log_debug(LogOp, "act_block_h_nsubblocks_split_last: {}", act_block_h_nsubblocks_split_last);
    log_debug(LogOp, "act_block_h_datums_split: {}", act_block_h_datums_split);
    log_debug(LogOp, "act_block_h_datums_split_last: {}", act_block_h_datums_split_last);
    log_debug(LogOp, "act_block_num_tiles_split: {}", act_block_num_tiles_split);
    log_debug(LogOp, "act_block_num_tiles_split_last: {}", act_block_num_tiles_split_last);
    log_debug(LogOp, "act_block_w_datums: {}", act_block_w_datums);
    log_debug(LogOp, "conv_act_size_c: {}", conv_act_size_c);
    log_debug(LogOp, "filter_h: {}", filter_h);
    log_debug(LogOp, "filter_w: {}", filter_w);
    log_debug(LogOp, "dilation_h: {}", dilation_h);
    log_debug(LogOp, "dilation_w: {}", dilation_w);



    // TT_FATAL(
    //     (act_block_w_datums == round_up(conv_act_size_c * filter_w, TILE_WIDTH)) ||
    //     ((act_block_w_datums <= conv_act_size_c)
    //      && (conv_act_size_c % act_block_w_datums == 0)
    //      ));

    // weight block info
    uint32_t weight_block_w_datums = weight_matrix_width / num_blocks_weight_w;
    TT_FATAL(weight_block_w_ntiles % out_subblock_w_ntiles == 0, "Error");
    uint32_t weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
    uint32_t weight_block_h_ntiles = act_block_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles;
    uint32_t weight_block_in_channels_ntiles = input_channels_padded/(32*total_num_cores*per_core_num_blocks_act_w);
    TT_FATAL(input_channels_padded>=(TILE_HEIGHT*total_num_cores), "Error");
    TT_FATAL(input_channels_padded%(TILE_HEIGHT*total_num_cores)==0, "Error");

    uint32_t num_groups = num_blocks_act_h * num_blocks_act_w * num_blocks_weight_w;
    // writer of conv op partially removes padding on the width
    // it removes the padding done for block width but it doesn't remove padding done for tiled width
    uint32_t output_channels_padded_to_tile_width = round_up(output_channels, TILE_WIDTH);
    TT_FATAL(output_channels_padded_to_tile_width <= weight_matrix_width, "Error");
    uint32_t output_width_num_tiles = output_channels_padded_to_tile_width / TILE_WIDTH;
    uint32_t num_blocks_output_w =
        (uint32_t) std::ceil((double)output_channels_padded_to_tile_width / (double)weight_block_w_datums);
    uint32_t last_block_width_datums = (output_channels_padded_to_tile_width % weight_block_w_datums == 0)
                                           ? weight_block_w_datums
                                           : (output_channels_padded_to_tile_width % weight_block_w_datums);
    TT_FATAL(last_block_width_datums % TILE_WIDTH == 0, "Error");

    // sanity check
    TT_FATAL(num_blocks_output_w == num_blocks_weight_w, "Error");

    uint32_t out_block_h_datums = out_block_h_ntiles * TILE_HEIGHT;

    tt_metal::Buffer* src0_dram_buffer = a.buffer();
    tt_metal::Buffer* src1_dram_buffer = b.buffer();

    tt_metal::Buffer* dst_dram_buffer = output.buffer();
    TT_FATAL(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;
    TT_FATAL(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    // act
    uint32_t act_dram_addr = src0_dram_buffer->address();
    auto act_dram_noc_xy = src0_dram_buffer->noc_coordinates();
    uint32_t act_noc_x = act_dram_noc_xy.x;
    uint32_t act_noc_y = act_dram_noc_xy.y;

    TT_FATAL(act_matrix_width_ntiles % act_block_w_ntiles == 0, "Error");
    TT_FATAL(act_block_h_ntiles % out_subblock_h_ntiles == 0, "Error");
    // TT_FATAL(out_block_h_ntiles % out_subblock_h_ntiles == 0, "Error");
    uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    // weight
    uint32_t weight_dram_addr = src1_dram_buffer->address();

    // bias
    tt_metal::Buffer* bias_buffer = nullptr;
    uint32_t bias_dram_addr = 0;
    uint32_t bias_ntiles = 0;
    bool bias_in_dram = true;
    if (has_bias) {
        bias_buffer = bias.value().buffer();
        bias_dram_addr = bias_buffer->address();
        bias_ntiles =
            bias.value().get_legacy_shape()[3] / constants::TILE_WIDTH;  // TODO: support non tile multiple sizes
        bias_in_dram = bias_buffer->buffer_type() == BufferType::DRAM;
    }

    auto output_shape = sliding_window_config.get_output_shape();
    uint32_t conv_output_size_h = output_shape[1];
    uint32_t conv_output_size_w = output_shape[2];


    std::map<string, string> reader_defines;

    if (act_matrix_height_unpadded < act_matrix_height) {
        reader_defines["ACT_BLOCK_HEIGHT_PADDING"] = "1";
    }

    // if (conv_act_c_blocks > 1) {
    //     reader_defines["ACT_W_OUTER_BLOCKS"] = "1";
    // }

    uint32_t output_height_padded_to_tile_height = round_up(act_matrix_height_unpadded, TILE_HEIGHT);
    uint32_t output_height_num_tiles = output_height_padded_to_tile_height / TILE_HEIGHT;
    TT_FATAL(output_height_num_tiles <= act_matrix_height_ntiles, "Error");

    uint32_t src_dram_act_buffer_size_bytes = src0_dram_buffer->size();
    uint32_t src_dram_weight_buffer_size_bytes = src1_dram_buffer->size();
    uint32_t dst_l1_act_buffer_size_bytes =
        out_block_h_ntiles * act_block_w_ntiles * tt::tt_metal::detail::TileSize(act_df);
    uint32_t dst_l1_weight_buffer_size_bytes =
        weight_block_h_ntiles * weight_block_w_ntiles * tt::tt_metal::detail::TileSize(weight_df);


    //Number of bytes to be read from the channel dimension in one block.
    uint32_t conv_act_c_read_bytes = conv_act_size_c * a.element_size() / (total_num_cores*per_core_num_blocks_act_w);

    //Compute only on cores that have an input shard.
    CoreRangeSet all_cores = a.memory_config().shard_spec.value().grid;


    // log info for debugging opts
    {
        log_debug(LogOp, "input_channels_padded: {}", input_channels_padded);
        log_debug(LogOp, "grid_size: {}", p_config.grid_size);
        log_debug(LogOp, "packer_l1: {}", packer_l1_acc);
        log_debug(LogOp, "split_reader: {}", split_reader);
        log_debug(LogOp, "enable_act_double_buffer: {}", enable_act_double_buffer);
        log_debug(LogOp, "enable block padding: {}", (per_core_out_matrix_height_ntiles % act_block_h_ntiles != 0));
        log_debug(LogOp, "enable subblock padding: {}", enable_subblock_padding);
        log_debug(LogOp, "per_core_out_matrix_height_ntiles: {}", per_core_out_matrix_height_ntiles);
        log_debug(LogOp, "act_block_h_ntiles_padded: {}", act_block_h_ntiles_padded);
        log_debug(LogOp, "act_block_w_ntiles: {}", act_block_w_ntiles);
        log_debug(LogOp, "weight_block_w_ntiles: {}", weight_block_w_ntiles);
        log_debug(LogOp, "out_subblock_h_ntiles_padded: {}", out_subblock_h_ntiles_padded);
        log_debug(LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
    }

    // For debug
    {
        log_debug(LogOp, "OP Name : multi_core_optimized_conv_width_sharded_v2_");
        log_debug(LogOp, "split readers: {}", split_reader);
        log_debug(LogOp, "conv_act_size_h: {}", conv_act_size_h);
        log_debug(LogOp, "conv_act_size_w: {}", conv_act_size_w);
        log_debug(LogOp, "act_matrix_height: {}", act_matrix_height);
        log_debug(LogOp, "act_matrix_width: {}", act_matrix_width);
        log_debug(LogOp, "act_matrix_height_unpadded: {}", act_matrix_height_unpadded);
        log_debug(LogOp, "act_matrix_width_unpadded: {}", act_matrix_width_unpadded);
        log_debug(LogOp, "act_matrix_height_ntiles: {}", act_matrix_height_ntiles);
        log_debug(LogOp, "act_matrix_width_ntiles: {}", act_matrix_width_ntiles);
        log_debug(LogOp, "weight_matrix_width_ntiles: {}", weight_matrix_width_ntiles);
        log_debug(LogOp, "per_core_out_matrix_height_ntiles: {}", per_core_out_matrix_height_ntiles);
        log_debug(LogOp, "per_core_out_matrix_width_ntiles: {}", per_core_out_matrix_width_ntiles);
        log_debug(LogOp, "per_core_num_blocks_act_w: {}", per_core_num_blocks_act_w);

        log_debug(LogOp, "num_blocks_act_h: {}", num_blocks_act_h);
        log_debug(LogOp, "num_blocks_act_w: {}", num_blocks_act_w);
        log_debug(LogOp, "num_blocks_weight_w: {}", num_blocks_weight_w);
        log_debug(LogOp, "num_blocks_out_h: {}", num_blocks_out_h);
        log_debug(LogOp, "act_dram_addr: {}", act_dram_addr);

        log_debug(LogOp, "conv_act_c_read_bytes: {}",conv_act_c_read_bytes);
        log_debug(LogOp, "act_block_h_ntiles: {}", act_block_h_ntiles);
        log_debug(LogOp, "act_block_h_datums: {}", act_block_h_datums);
        log_debug(LogOp, "act_block_w_ntiles: {}", act_block_w_ntiles);
        log_debug(LogOp, "act_block_w_datums: {}", act_block_w_datums);
        log_debug(LogOp, "out_block_h_ntiles: {}", out_block_h_ntiles);
        log_debug(LogOp, "act_num_subblocks: {}", act_num_subblocks);
        log_debug(LogOp, "act_block_num_tiles: {}", act_block_num_tiles);
        log_debug(LogOp, "act_subblock_h_ntiles: {}", act_subblock_h_ntiles);
        log_debug(LogOp, "act_subblock_num_tiles: {}", act_subblock_num_tiles);
        log_debug(LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(LogOp, "weight_dram_addr: {}", weight_dram_addr);
        log_debug(LogOp, "weight_num_subblocks: {}", weight_num_subblocks);
        log_debug(LogOp, "weight_block_num_tiles: {}", weight_block_num_tiles);
        log_debug(LogOp, "weight_block_w_ntiles: {}", weight_block_w_ntiles);
        log_debug(LogOp, "weight_block_h_ntiles: {}", weight_block_h_ntiles);
        log_debug(LogOp, "weight_block_in_channels_ntiles: {}", weight_block_in_channels_ntiles);
        log_debug(LogOp, "has_bias: {}", has_bias);
        log_debug(LogOp, "bias_dram_addr: {}", bias_dram_addr);
        log_debug(LogOp, "bias_ntiles: {}", bias_ntiles);
        log_debug(LogOp, "out_dram_addr: {}", out_dram_addr);
        log_debug(LogOp, "out_subblock_h_ntiles: {}", out_subblock_h_ntiles);
        log_debug(LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
        log_debug(LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(LogOp, "num_groups: {}", num_groups);
        log_debug(LogOp, "math_fidelity: {}", math_fidelity);
        log_debug(LogOp, "math_approx_mode: {}", math_approx_mode);
        log_debug(LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);
        log_debug(LogOp, "packer_l1_acc: {}", packer_l1_acc);
        log_debug(LogOp, "all_cores: {}",all_cores.str());
    }



    std::string compute_kernel_path = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize_col_major_out_blocks.cpp";
    std::string activation_kernel_path = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/activation_reader_width_sharded.cpp";
    std::string weights_kernel_path = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/weights_reader_width_sharded.cpp";

    std::vector<uint32_t> reader_rt_args;
    std::vector<uint32_t> activation_kernel_compile_args;
    std::vector<uint32_t> weights_kernel_compile_args;
    std::vector<uint32_t> writer_rt_args;
    std::vector<uint32_t> writer_compile_time_args;
    std::vector<uint32_t> compute_kernel_args;
    bool tilize_in0 = false;

    uint32_t act_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, 0);   //0==INVALID
    uint32_t act_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, 0); //0==INVALID.

    CoreCoord act_mcast_start_core_logical(0,0);
    CoreCoord act_mcast_end_core_logical(p_config.grid_size.x-1,p_config.grid_size.y-1);
    auto act_mcast_start = device->worker_core_from_logical_core(act_mcast_start_core_logical);
    auto act_mcast_end = device->worker_core_from_logical_core(act_mcast_end_core_logical);
    TT_FATAL(act_block_h_datums % 2 == 0, "2 Indices are packed in one uint32_t word.");

    activation_kernel_compile_args = {
        (uint32_t)0, // Never in DRAM
        (uint32_t)stride_h,
        (uint32_t)stride_w,
        (uint32_t)dilation_h,
        (uint32_t)dilation_w,
        (uint32_t)input_size_w,
        (uint32_t)conv_act_c_read_bytes,
        (uint32_t)filter_h, //Input filter window height
        (uint32_t)filter_w, //Input filter window width
        (uint32_t)act_block_h_datums,
        (uint32_t)act_block_num_tiles,
        (uint32_t)total_num_cores,
        (uint32_t)per_core_num_blocks_act_w,
        (uint32_t)act_mcast_sender_semaphore,
        (uint32_t)act_mcast_receiver_semaphore,
        (uint32_t)act_mcast_start.x,
        (uint32_t)act_mcast_start.y,
        (uint32_t)act_mcast_end.x,
        (uint32_t)act_mcast_end.y,
        (uint32_t)act_block_num_tiles*tt_metal::detail::TileSize(tilized_act_df),
        (uint32_t)total_num_cores
    };
    weights_kernel_compile_args = {
        weight_cb,                                                  //cb_id_weight
        act_block_w_ntiles/(filter_h*filter_w),           //core_in_channels_ntiles
        filter_h*filter_w,                                //window_size_hw
        weight_block_w_ntiles,                                      //weight_block_width_ntiles
        weight_block_num_tiles,                                     //weight_block_num_tiles
        weight_matrix_width_ntiles,                                 //weight_matrix_width_ntiles
        (weight_matrix_width_ntiles * input_channels_padded)/32,    //weight_next_channel_stride_h
        weight_matrix_width_ntiles*weight_block_in_channels_ntiles, //weight_next_block_this_core_stride_h
        weight_matrix_width_ntiles*weight_block_in_channels_ntiles* //weight_next_block_other_core_stride_h
                                         per_core_num_blocks_act_w,
        total_num_cores,                                            //other_core_weight_height_blocks
        per_core_num_blocks_act_w,                                  //this_core_weight_height_blocks
        bias_cb,
        bias_in_dram
    };

    uint32_t num_weight_slices_width = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    uint32_t num_blocks_act_h_per_core = (per_core_out_matrix_height_ntiles + act_block_h_ntiles - 1) / act_block_h_ntiles;
    uint32_t num_blocks_weight_w_per_core = per_core_out_matrix_width_ntiles / weight_block_w_ntiles;
    uint32_t bias_ntiles_per_core = bias_ntiles / num_weight_slices_width;

    std::map<string, string> writer_defines;
    std::map<string, string> writer_mcast_sender_defines;
    std::map<string, string> compute_defines;

    compute_defines["WIDTH_SHARDED"] = "1";

    if (output.memory_config().is_sharded()) {
        writer_defines["SHARDED_OUT"] = "1";
        writer_mcast_sender_defines["SHARDED_OUT"] = "1";
    }
    if (total_num_cores == 1) {
        writer_mcast_sender_defines["SKIP_MCAST"] = "1";
    }
    if (has_bias) {
        writer_defines["FUSE_BIAS"] = "1";
        writer_mcast_sender_defines["FUSE_BIAS"] = "1";
        compute_defines["FUSE_BIAS"] = "1";
    }

    if (fuse_relu) {
        compute_defines["PACK_RELU"] = "1";
    }

    if (split_reader) {
        reader_defines["SPLIT_READER"] = "1";
        compute_defines["SPLIT_READER"] = "1";
    }

    if (packer_l1_acc) {
        compute_defines["PACKER_L1_ACC"] = "1";
    }

    compute_kernel_args = {
        act_block_w_ntiles,           //in0_block_w
        act_num_subblocks,            //in0_num_sublocks
        act_block_num_tiles,          //in0_block_num_tiles,
        act_subblock_num_tiles,       //in0_sublock_num_tiles
        act_subblock_h_ntiles,        //in0_subblock_h


        weight_num_subblocks,         //in1_num_sublocks
        weight_block_num_tiles,       //in1_block_num_tiles,
        weight_block_w_ntiles,        //in1_block_w

        num_blocks_act_h_per_core,    //in0_num_blocks_h
        num_blocks_act_w,             //in0_num_blocks_w,
        num_blocks_weight_w_per_core, //in1_num_blocks_w

        out_subblock_h_ntiles_padded, //out_sublock_h
        out_subblock_w_ntiles,        //out_sublock_w
        out_subblock_num_tiles,       //out_sublock_num_tiles


        tilize_in0,                   //tilize_in0
        untilize_out,                 //untilize_out

        bias_ntiles_per_core,

        total_num_cores,              //in0_nblocks_w_tilize. Repeat tilize after all cores have done one round of MCAST.
    };

    bool packer_l1_acc_en = packer_l1_acc && ((has_bias && num_blocks_act_w > 1) || (num_blocks_act_w > 2));



    uint32_t act_tile_size = tt_metal::detail::TileSize(act_df);
    uint32_t tilized_act_tile_size = tt_metal::detail::TileSize(tilized_act_df);
    uint32_t weight_tile_size = tt_metal::detail::TileSize(weight_df);

    // Local L1 to store temp vars
    CircularBufferConfig cb_for_l1_array_config =
        CircularBufferConfig(32 * 2, {{cb_for_l1_array, tt::DataFormat::Float16_b}})
            .set_page_size(cb_for_l1_array, 32 * 2);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_for_l1_array_config);

    CircularBufferConfig cb_sharded_act_config =
            CircularBufferConfig(shard_shape[0] * shard_shape[1] * datum_size(act_df), {{sharded_act_cb, act_df}})
                .set_page_size(sharded_act_cb, shard_shape[1] * datum_size(act_df));
        cb_sharded_act_config.set_globally_allocated_address(*a.buffer());

    tt_metal::CreateCircularBuffer(program, all_cores, cb_sharded_act_config);

    CircularBufferConfig cb_act_config =
        CircularBufferConfig(act_block_num_tiles_split * tilized_act_tile_size, {{act_cb, tilized_act_df}})
            .set_page_size(act_cb, tilized_act_tile_size);
    auto cb_act = tt_metal::CreateCircularBuffer(program, all_cores, cb_act_config);
    log_debug(LogOp, "Act CB: {}, npages: {}, pagesize: {}", act_cb, act_block_num_tiles_split, tilized_act_tile_size);

    // Used for placing tilized activations
    CircularBufferConfig cb_src0_tilized_config =
        CircularBufferConfig(
            act_block_num_tiles * tilized_act_tile_size, {{tilize_mode_tilized_act_cb, tilized_act_df}})
            .set_page_size(tilize_mode_tilized_act_cb, tilized_act_tile_size);
    auto cb_src0_tilized = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_tilized_config);
    log_debug(LogOp, "Tilized Act CB: {}, npages: {}, pagesize: {}", tilize_mode_tilized_act_cb, act_block_num_tiles, tilized_act_tile_size);

    CircularBufferConfig cb_weight_config =
        CircularBufferConfig(weight_block_num_tiles * weight_tile_size, {{weight_cb, weight_df}})
            .set_page_size(weight_cb, weight_tile_size);
    auto cb_weight = tt_metal::CreateCircularBuffer(program, all_cores, cb_weight_config);
    log_debug(LogOp, "Weight CB: {}, npages: {}, pagesize: {}, ", weight_cb, weight_block_num_tiles, weight_tile_size);

    CircularBufferConfig cb_act_row_major_bfloat16_config =
                CircularBufferConfig(act_block_num_tiles_split * act_tile_size, {{act_cb_row_major_bfloat16, act_df}})
                    .set_page_size(act_cb_row_major_bfloat16, act_tile_size);
            auto cb_act_row_major_bfloat16 =
                tt_metal::CreateCircularBuffer(program, all_cores, cb_act_row_major_bfloat16_config);
    log_debug(LogOp, "Act Row Major CB: {}, npages: {}, pagesize: {}", act_cb_row_major_bfloat16, act_block_num_tiles_split, act_tile_size);

    CircularBufferConfig cb_for_reader_indices_config =
        CircularBufferConfig(out_block_h_datums * 2, {{cb_for_reader_indices, tt::DataFormat::Float16_b}})
            .set_page_size(cb_for_reader_indices, out_block_h_datums * 2);
    cb_for_reader_indices_config.set_globally_allocated_address(*conv_reader_indices.value().buffer());
    auto cb_for_reader_indices_id =
        tt_metal::CreateCircularBuffer(program, all_cores, cb_for_reader_indices_config);

   if (has_bias) {
        uint32_t bias_tile_size = tt_metal::detail::TileSize(bias_df);
        // bias input
        uint32_t bias_pagesize = bias_tile_size;
        CircularBufferConfig cb_bias_config = CircularBufferConfig(bias_ntiles * bias_pagesize, {{bias_cb, bias_df}})
                                                  .set_page_size(bias_cb, bias_pagesize);
        auto cb_bias = tt_metal::CreateCircularBuffer(program, all_cores, cb_bias_config);

        log_debug(LogOp, "Bias CB: {}, npages: {}, pagesize: {}", bias_cb, bias_ntiles, bias_pagesize);
    }

    tt::DataFormat interm0_df =
        packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b) : out_df;

    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);
    uint32_t interm0_single_tile_size = tt_metal::detail::TileSize(interm0_df);
    uint32_t num_output_tiles = per_core_out_matrix_height_ntiles*per_core_out_matrix_width_ntiles;


        // Share buffer if same data format
    CBHandle cb_output = 0;
    if (interm0_df == out_df) {
        // CoreRangeSet cores(std::set<CoreRange>({core}));
        std::map<uint8_t, tt::DataFormat> cb_output_data_format_spec = {
            {out0_cb, out_df}, {matmul_partials_cb, out_df}};
        CircularBufferConfig cb_matmul_partials_config =
            CircularBufferConfig(num_output_tiles * out_tile_size, cb_output_data_format_spec)
                .set_page_size(out0_cb, out_tile_size)
                .set_page_size(matmul_partials_cb, out_tile_size);
        if (output.is_sharded()) {
            cb_matmul_partials_config = cb_matmul_partials_config.set_globally_allocated_address(*output.buffer());
        }
        log_debug(LogOp, "Matmul Partials CB: {}, npages: {}, pagesize: {}", matmul_partials_cb, num_output_tiles, out_tile_size);
        cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_matmul_partials_config);
    } else {
        // Separate buffer if not same data format
        CircularBufferConfig cb_matmul_partials_config =
            CircularBufferConfig(num_output_tiles * interm0_single_tile_size, {{matmul_partials_cb, interm0_df}})
                .set_page_size(matmul_partials_cb, interm0_single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, all_cores, cb_matmul_partials_config);
        log_debug(LogOp, "Matmul Partials CB: {}, npages: {}, pagesize: {}", matmul_partials_cb, num_output_tiles, interm0_single_tile_size);

        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_output_tiles * out_tile_size, {{out0_cb, out_df}})
                .set_page_size(out0_cb, out_tile_size);
        if (output.is_sharded()) {
            cb_output_config = cb_output_config.set_globally_allocated_address(*output.buffer());
        }
        cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
    }


    auto act_kernel_id = CreateKernel(
        program,
        activation_kernel_path,
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = activation_kernel_compile_args
        }
    );

    auto weights_kernel_id = CreateKernel(
        program,
        weights_kernel_path,
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = weights_kernel_compile_args,
            .defines = writer_defines
    });

    auto compute_id = CreateKernel(
        program,
        compute_kernel_path,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args,
            .defines = compute_defines
    });

    auto full_core_grid = device->compute_with_storage_grid_size();
    std::vector<uint32_t> act_mcast_noc_y;
    std::vector<uint32_t> act_mcast_noc_x;

    for(uint32_t core_index = 0; core_index < full_core_grid.x; core_index++) {
        act_mcast_noc_x.push_back( device->worker_core_from_logical_core( CoreCoord(core_index, 0)).x );
    }

    for(uint32_t core_index = 0; core_index < full_core_grid.y; core_index++) {
        act_mcast_noc_y.push_back( device->worker_core_from_logical_core( CoreCoord(0,core_index) ).y );
    }

    uint32_t bias_base_address = 0;
    if(bias)
    {
        bias_base_address = bias.value().buffer()->address();
    }

    for(uint32_t core_index = 0; core_index < total_num_cores; core_index++) {
        uint32_t core_x = core_index % full_core_grid.x;
        uint32_t core_y = core_index / full_core_grid.x;
        std::vector<uint32_t> rt_args = {
            core_x,
            core_y,
            full_core_grid.x, //num_cores_x
        };

        //Mcast X Lookup table
        rt_args.insert(
            rt_args.end(),
            act_mcast_noc_x.begin(),
            act_mcast_noc_x.end()
        );

        //Mcast Y Lookup Table
        rt_args.insert(
            rt_args.end(),
            act_mcast_noc_y.begin(),
            act_mcast_noc_y.end()
        );

        SetRuntimeArgs(program, act_kernel_id, CoreCoord(core_x, core_y), rt_args);

        SetRuntimeArgs(program, weights_kernel_id, CoreCoord(core_x, core_y), {
            core_index * weight_block_w_ntiles,
            b.buffer()->address(),
            bias_base_address
        });
    }

    auto empty_callback = [](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            };
    return {
        .program = std::move(program), .override_runtime_arguments_callback = empty_callback
        };

}

}  // namespace tt_metal

}  // namespace tt
