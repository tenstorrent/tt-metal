// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/common/constants.hpp"

#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

using namespace tt::constants;

namespace conv_op_utils {
using namespace tt;
using namespace tt::tt_metal;

pair<uint32_t, uint32_t> compute_conv_output_face_shape(uint32_t conv_activation_h, uint32_t conv_activation_w, uint32_t filter_h, uint32_t filter_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w) {
    uint32_t conv_output_h = ((conv_activation_h - filter_h + (2 * pad_h)) / stride_h) + 1;
    uint32_t conv_output_w = ((conv_activation_w - filter_w + (2 * pad_w)) / stride_w) + 1;
    return {conv_output_h, conv_output_w};
}
pair<vector<uint32_t>, vector<uint32_t>> compute_conv_activation_as_mm_shape(tt::tt_metal::LegacyShape conv_activation_shape, vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, bool use_fast_reader) {
    uint32_t filter_h = (uint32_t) conv_params[0];
    uint32_t filter_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];
    auto [conv_output_h, conv_output_w] = compute_conv_output_face_shape(conv_activation_shape[1], conv_activation_shape[2], filter_h, filter_w, stride_h, stride_w, pad_h, pad_w);
    // pad height
    uint32_t num_rows = (uint32_t) conv_output_h*conv_output_w;
    uint32_t act_block_h_datums = act_block_h_ntiles * TILE_HEIGHT;
    uint32_t num_rows_padded = (uint32_t) (std::ceil((double) num_rows / (double) act_block_h_datums ) * act_block_h_datums);
    uint32_t num_cols = conv_activation_shape[3] * filter_h * filter_w;
    uint32_t act_block_w_datums = act_block_w_ntiles * TILE_WIDTH;
    uint32_t num_cols_padded = (uint32_t) (std::ceil((double) num_cols / (double) act_block_w_datums ) * act_block_w_datums);
    if(use_fast_reader) {
        assert(act_block_w_datums >= conv_activation_shape[3] * filter_w);
        num_cols_padded = act_block_w_datums * filter_h;
    }
    return {{1, num_rows_padded, num_cols_padded}, {1, num_rows, num_cols}};
}

}

namespace ttnn::operations::conv {
namespace conv2d {

using namespace tt;

const uint32_t act_cb                                 = CB::c_in0;
const uint32_t weight_cb                              = CB::c_in1;
const uint32_t bias_cb                                = CB::c_in2;
const uint32_t matmul_partials_cb                     = CB::c_intermed0;
const uint32_t tilize_mode_tilized_act_cb             = CB::c_intermed1;
const uint32_t untilize_mode_final_matmul_partials_cb = CB::c_intermed2;
const uint32_t untilize_mode_reblock_cb               = CB::c_intermed3;
const uint32_t out0_cb                                = CB::c_out0;


void create_CBs_for_fused_matmul_new_alloc(tt_metal::Program &program,
                                tt_metal::Device* device,
                                CoreRange core,
                                uint32_t num_cb0_tiles,
                                uint32_t num_cb1_tiles,
                                uint32_t num_cb0_tilized_tiles,
                                uint32_t num_output_tiles,
                                uint32_t num_reblock_cb_tiles,
                                uint32_t num_writer_output_tiles,
                                uint32_t num_bytes_for_df,
                                bool untilize_out,
                                uint32_t bias_ntiles = 0,
                                bool with_bias = false) {

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    // Invariants
    CircularBufferConfig cb_act_config = CircularBufferConfig(num_cb0_tiles * single_tile_size, {{act_cb, tt::DataFormat::Float16_b}})
		.set_page_size(act_cb, single_tile_size);
    auto cb_act = tt_metal::CreateCircularBuffer(program, core, cb_act_config);

    CircularBufferConfig cb_weight_config = CircularBufferConfig(num_cb1_tiles * single_tile_size, {{weight_cb, tt::DataFormat::Float16_b}})
		.set_page_size(weight_cb, single_tile_size);
    auto cb_weight = tt_metal::CreateCircularBuffer(program, core, cb_weight_config);

    // Used for placing tilized activations
    CircularBufferConfig cb_src0_tilized_config = CircularBufferConfig(num_cb0_tilized_tiles * single_tile_size, {{tilize_mode_tilized_act_cb, tt::DataFormat::Float16_b}})
		.set_page_size(tilize_mode_tilized_act_cb, single_tile_size);
    auto cb_src0_tilized = CreateCircularBuffer(program, core, cb_src0_tilized_config);

    if (untilize_out) {
        CircularBufferConfig cb_matmul_partials_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(matmul_partials_cb, single_tile_size);
        auto cb_matmul_partials = CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Shares same address space as matmul partials
        CircularBufferConfig cb_final_matmul_partials_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{untilize_mode_final_matmul_partials_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(untilize_mode_final_matmul_partials_cb, single_tile_size);
        auto cb_final_matmul_partials = CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        CircularBufferConfig cb_reblock_config = CircularBufferConfig(num_reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        auto cb_reblock = CreateCircularBuffer(program, core, cb_reblock_config);

        CircularBufferConfig cb_output_config = CircularBufferConfig(num_writer_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(out0_cb, single_tile_size);
        auto cb_output = CreateCircularBuffer(program, core, cb_output_config);
    } else {
        CoreRangeSet cores(std::set<CoreRange>({core}));
        std::map<uint8_t, tt::DataFormat> cb_output_data_format_spec = {
            {out0_cb, tt::DataFormat::Float16_b},
            {matmul_partials_cb, tt::DataFormat::Float16_b}
        };
        CircularBufferConfig cb_matmul_partials_config = CircularBufferConfig(num_output_tiles * single_tile_size, cb_output_data_format_spec)
		    .set_page_size(out0_cb, single_tile_size)
            .set_page_size(matmul_partials_cb, single_tile_size);
        auto cb_output = CreateCircularBuffer(program, core, cb_matmul_partials_config);
    }

    if (with_bias) {
        // bias input
        uint32_t bias_pagesize = single_tile_size;
        CircularBufferConfig cb_bias_config = CircularBufferConfig(bias_ntiles * bias_pagesize, {{bias_cb, tt::DataFormat::Float16_b}})
		    .set_page_size(bias_cb, bias_pagesize);
        auto cb_bias = CreateCircularBuffer(program, core, cb_bias_config);

        log_debug("BIAS CBs: {} {} {}", bias_cb, bias_ntiles, bias_pagesize);
    }
}

operation::ProgramWithCallbacks conv_as_large_bmm_single_core_(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias, vector<int> conv_params,
                                       uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
                                       uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, bool use_fast_reader, bool untilize_out, bool has_bias, bool fuse_relu, const MathFidelity math_fidelity, Tensor &output) {
    bool pass = true;
    tt_metal::Device *device = a.device();
    TT_ASSERT(a.get_layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    uint32_t act_batch_size = a.get_legacy_shape()[0];
    TT_ASSERT(act_batch_size == 1, "Only batch size 1 supported.");
    TT_ASSERT(output_channels <= b.get_legacy_shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    uint32_t num_bytes_of_df = 2; // 2 bytes for bfloat16
    // Compute the 2d matrix shape
    auto [act_matrix_shape, act_matrix_shape_unpadded] = conv_op_utils::compute_conv_activation_as_mm_shape(a.get_legacy_shape(), conv_params, act_block_h_ntiles, act_block_w_ntiles, use_fast_reader);
    assert(act_matrix_shape.size() == 3);
    assert(act_matrix_shape[0] == 1);
    uint32_t act_matrix_height = (uint32_t) act_matrix_shape[1];
    uint32_t act_matrix_width = (uint32_t) act_matrix_shape[2];

    // Tensor b has weights and it should be tiled layout after converting conv weights into weight matrix
    TT_ASSERT(b.get_layout() == Layout::TILE, "Conv weights should be in tiled layout");
    TT_ASSERT(b.get_legacy_shape()[0] == 1, "Conv weight matrix shape is invalid");
    TT_ASSERT(b.get_legacy_shape()[1] == 1, "Conv weight matrix shape is invalid");
    uint32_t weight_matrix_height = b.get_legacy_shape()[2];
    uint32_t weight_matrix_width = b.get_legacy_shape()[3];

    if (has_bias) {
        // Tensor bias is of shape {output_channels}
        TT_ASSERT(bias.has_value());
        TT_ASSERT(bias.value().buffer() != nullptr);
        auto bias_shape_without_padding = bias.value().get_legacy_shape().without_padding();
        TT_ASSERT(bias_shape_without_padding[0] == 1, "Bias should have batch == 1");
        TT_ASSERT(bias_shape_without_padding[1] == 1 && bias_shape_without_padding[2] == 1, "Bias should have H == W == 1");
        TT_ASSERT(bias_shape_without_padding[3] == output_channels, "Bias should have output_channels");
    }

    // Normal matrix shape check
    TT_ASSERT(act_matrix_width == weight_matrix_height, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_ASSERT(act_matrix_height % TILE_HEIGHT == 0, "Height of activation matrix needs to be divisible by 32");
    TT_ASSERT(act_matrix_width % TILE_WIDTH == 0, "Width of activation matrix needs to be divisible by 32");
    TT_ASSERT(weight_matrix_height % TILE_HEIGHT == 0, "Height of weight matrix needs to be divisible by 32");
    TT_ASSERT(weight_matrix_width % TILE_WIDTH == 0, "Width of weight matrix needs to be divisible by 32");

    // Device compatibility checks
    TT_ASSERT(a.storage_type() == StorageType::DEVICE &&
              b.storage_type() == StorageType::DEVICE &&
              "Operands to large matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to conv need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr && b.buffer() != nullptr, "Operands to conv need to be allocated in buffers on device!");
    if (has_bias) {
        TT_ASSERT(bias.value().storage_type() == StorageType::DEVICE, "Bias should be on device");
        TT_ASSERT(bias.value().device() == a.device(), "Bias should be on the same device as act tensor");
    }

    // Convert tensor dims to tile dims
    uint32_t act_matrix_height_ntiles = act_matrix_height / TILE_HEIGHT;
    uint32_t act_matrix_width_ntiles = act_matrix_width / TILE_WIDTH;
    uint32_t weight_matrix_height_ntiles = weight_matrix_height / TILE_HEIGHT;
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / TILE_WIDTH;

    assert(act_matrix_height_ntiles % act_block_h_ntiles == 0);
    assert(act_matrix_width_ntiles % act_block_w_ntiles == 0);
    assert(weight_matrix_width_ntiles % weight_block_w_ntiles == 0);

    uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    uint32_t num_blocks_act_w = act_matrix_width_ntiles / act_block_w_ntiles;
    uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

    // act block info
    uint32_t act_block_w_datums = act_matrix_width / num_blocks_act_w;
    uint32_t act_block_h_datums = act_matrix_height / num_blocks_act_h;

    // weight block info
    uint32_t weight_block_w_datums = weight_matrix_width / num_blocks_weight_w;
    assert(weight_block_w_ntiles % out_subblock_w_ntiles == 0);
    uint32_t weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
    uint32_t weight_block_h_ntiles = act_block_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles;

    uint32_t num_groups = num_blocks_act_h * num_blocks_act_w * num_blocks_weight_w;
    // writer of conv op partially removes padding on the width
    // it removes the padding done for block width but it doesn't remove padding done for tiled width
    uint32_t output_channels_padded_to_tile_width = round_up(output_channels, TILE_WIDTH);
    assert(output_channels_padded_to_tile_width <= weight_matrix_width);
    uint32_t output_width_num_tiles = output_channels_padded_to_tile_width / TILE_WIDTH;
    uint32_t num_blocks_output_w = (uint32_t) std::ceil((double) output_channels_padded_to_tile_width / (double) weight_block_w_datums);
    uint32_t last_block_width_datums = (output_channels_padded_to_tile_width % weight_block_w_datums == 0) ? weight_block_w_datums : (output_channels_padded_to_tile_width % weight_block_w_datums);
    assert(last_block_width_datums % TILE_WIDTH == 0);
    uint32_t output_row_size_bytes = output_channels_padded_to_tile_width * num_bytes_of_df;
    uint32_t last_block_row_size_bytes = last_block_width_datums * num_bytes_of_df;
    // sanity check
    assert(num_blocks_output_w == num_blocks_weight_w);

    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core_coord = {0, 0};      // TODO: avoid another var here. Find a way to use core range instead.
    CoreRange core({0, 0}, {0, 0});

    uint32_t single_tile_size = num_bytes_of_df * TILE_HEIGHT * TILE_WIDTH;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    uint32_t out_row_size = weight_matrix_width * num_bytes_of_df;
    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;
    TT_ASSERT(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    // act
    uint32_t act_dram_addr = src0_dram_buffer->address();
    auto act_dram_noc_xy = src0_dram_buffer->noc_coordinates();
    uint32_t act_noc_x = act_dram_noc_xy.x;
    uint32_t act_noc_y = act_dram_noc_xy.y;

    assert(act_matrix_width_ntiles % act_block_w_ntiles == 0);
    assert(act_block_h_ntiles % out_subblock_h_ntiles == 0);
    uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    // weight
    uint32_t weight_dram_addr = src1_dram_buffer->address();
    auto weight_dram_noc_xy = src1_dram_buffer->noc_coordinates();
    uint32_t weight_noc_x = weight_dram_noc_xy.x;
    uint32_t weight_noc_y = weight_dram_noc_xy.y;

    // bias
    Buffer *bias_buffer = nullptr;
    uint32_t bias_dram_addr = 0;
    uint32_t bias_ntiles = 0, bias_tile_nbytes = 0, bias_log2_of_pagesize = 0;
    if (has_bias) {
        bias_buffer = bias.value().buffer();
        bias_dram_addr = bias_buffer->address();
        bias_ntiles = bias.value().get_legacy_shape()[3] / constants::TILE_WIDTH;  // TODO: support non tile multiple sizes
        bias_tile_nbytes = single_tile_size;
        bias_log2_of_pagesize = (uint32_t) std::log2((float) bias_tile_nbytes);
    }

    // more args for reader
    uint32_t conv_act_size_h = a.get_legacy_shape()[1];
    uint32_t conv_act_size_w = a.get_legacy_shape()[2];
    uint32_t conv_act_size_c = a.get_legacy_shape()[3];
    uint32_t weight_size_h = (uint32_t) conv_params[0];
    uint32_t weight_size_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];
    uint32_t conv_output_size_h = ((conv_act_size_h - weight_size_h + (2 * pad_h)) / stride_h) + 1;
    uint32_t conv_output_size_w = ((conv_act_size_w - weight_size_w + (2 * pad_w)) / stride_w) + 1;
    std::map<string, string> reader_defines;
    if (use_fast_reader) {
        if(conv_act_size_c * weight_size_w != act_block_w_datums) {
            assert(act_block_w_datums > conv_act_size_c * weight_size_w);
            uint32_t conv_act_block_width_padding_bytes = (act_block_w_datums - (conv_act_size_c * weight_size_w)) * num_bytes_of_df;
            reader_defines["ACT_BLOCK_WIDTH_PADDING_BYTES"] = std::to_string(conv_act_block_width_padding_bytes);
        }
        if (conv_output_size_h * conv_output_size_w < act_block_h_datums * num_blocks_act_h) {
            reader_defines["ACT_BLOCK_HEIGHT_PADDING"] = "1";
        }
    }
    uint32_t output_height_padded_to_tile_height = round_up(conv_output_size_h*conv_output_size_w, TILE_HEIGHT);
    uint32_t output_height_num_tiles = output_height_padded_to_tile_height / TILE_HEIGHT;
    assert(output_height_num_tiles <= act_matrix_height_ntiles);

    uint32_t act_matrix_height_unpadded = conv_output_size_h * conv_output_size_w;
    uint32_t act_matrix_width_unpadded = conv_act_size_c * weight_size_h * weight_size_w;
    uint32_t src_dram_act_buffer_size_bytes = src0_dram_buffer->size();
    uint32_t src_dram_weight_buffer_size_bytes = src1_dram_buffer->size();
    uint32_t dst_l1_act_buffer_size_bytes = act_block_h_ntiles * act_block_w_ntiles * single_tile_size;
    uint32_t dst_l1_weight_buffer_size_bytes = weight_block_h_ntiles * weight_block_w_ntiles * single_tile_size;

    // more args for writer
    uint32_t out_block_row_size_bytes = weight_block_w_ntiles*TILE_WIDTH*num_bytes_of_df;
    uint32_t out_row_size_bytes = output_channels_padded_to_tile_width*num_bytes_of_df;
    uint32_t batch_size = 1;
    // output data format
    const auto out_df = datatype_to_dataformat_converter(a.get_dtype());
    // For debug
    {
        log_debug(tt::LogOp, "act_matrix_height_ntiles: {}", act_matrix_height_ntiles);
        log_debug(tt::LogOp, "act_matrix_width_ntiles: {}", act_matrix_width_ntiles);
        log_debug(tt::LogOp, "weight_matrix_width_ntiles: {}", weight_matrix_width_ntiles);
        log_debug(tt::LogOp, "num_blocks_act_h: {}", num_blocks_act_h);
        log_debug(tt::LogOp, "num_blocks_act_w: {}", num_blocks_act_w);
        log_debug(tt::LogOp, "num_blocks_weight_w: {}", num_blocks_weight_w);
        log_debug(tt::LogOp, "act_dram_addr: {}", act_dram_addr);
        log_debug(tt::LogOp, "act_block_h_ntiles: {}", act_block_h_ntiles);
        log_debug(tt::LogOp, "act_block_h_datums: {}", act_block_h_datums);
        log_debug(tt::LogOp, "act_block_w_ntiles: {}", act_block_w_ntiles);
        log_debug(tt::LogOp, "act_block_w_datums: {}", act_block_w_datums);
        log_debug(tt::LogOp, "act_num_subblocks: {}", act_num_subblocks);
        log_debug(tt::LogOp, "act_block_num_tiles: {}", act_block_num_tiles);
        log_debug(tt::LogOp, "act_subblock_h_ntiles: {}", act_subblock_h_ntiles);
        log_debug(tt::LogOp, "act_subblock_num_tiles: {}", act_subblock_num_tiles);
        log_debug(tt::LogOp, "weight_dram_addr: {}", weight_dram_addr);
        log_debug(tt::LogOp, "weight_num_subblocks: {}", weight_num_subblocks);
        log_debug(tt::LogOp, "weight_block_num_tiles: {}", weight_block_num_tiles);
        log_debug(tt::LogOp, "weight_block_w_ntiles: {}", weight_block_w_ntiles);
        log_debug(tt::LogOp, "weight_block_h_ntiles: {}", weight_block_h_ntiles);
        log_debug(tt::LogOp, "has_bias: {}", has_bias);
        log_debug(tt::LogOp, "bias_dram_addr: {}", bias_dram_addr);
        log_debug(tt::LogOp, "bias_ntiles: {}", bias_ntiles);
        log_debug(tt::LogOp, "out_dram_addr: {}", out_dram_addr);
        log_debug(tt::LogOp, "out_row_size: {}", out_row_size);
        log_debug(tt::LogOp, "out_subblock_h_ntiles: {}", out_subblock_h_ntiles);
        log_debug(tt::LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
        log_debug(tt::LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(tt::LogOp, "num_groups: {}", num_groups);
    }

    bool rn50_first_conv = (conv_act_size_h == 230 && conv_act_size_w == 231 &&
                            conv_output_size_h == 112 && conv_output_size_w == 112 &&
                            weight_size_h == 7 && weight_size_w == 8 &&
                            stride_h == 2 && stride_w == 2 &&
                            num_blocks_weight_w == 1);

    uint32_t num_weight_tiles_in_cb = weight_block_h_ntiles * weight_block_w_ntiles;
    if (rn50_first_conv) {
        num_weight_tiles_in_cb = weight_block_h_ntiles * weight_block_w_ntiles * num_blocks_weight_w * num_blocks_act_w;
    }
    create_CBs_for_fused_matmul_new_alloc(
        program,
        a.device(),
        core,
        act_block_h_ntiles * act_block_w_ntiles * 2, // row major act cb, double bufferred
        num_weight_tiles_in_cb, // tiled weight cb
        act_block_h_ntiles * act_block_w_ntiles, // tiled act cb
        act_block_h_ntiles * weight_block_w_ntiles, // math output cb
        weight_block_w_ntiles, // reblock cb
        act_block_h_ntiles * weight_block_w_ntiles * 2, // writer output cb, double bufferred
        num_bytes_of_df,
        untilize_out,
        bias_ntiles,
        has_bias);

    // define for bias
    std::map<string, string> all_defines;
    std::map<string, string> compute_defines;
    if (has_bias) {
        all_defines["FUSE_BIAS"] = "1";
        compute_defines["FUSE_BIAS"] = "1";
    }

    if (fuse_relu) {
        using ttnn::operations::unary::UnaryOpType;
        using ttnn::operations::unary::utils::get_defines;
        compute_defines.merge(get_defines(UnaryOpType::RELU, std::nullopt, "ACTIVATION", "i"));
        if (has_bias) {
            compute_defines["FUSE_BIAS"] = "1";
        }
    }

    string reader_kernel;
    vector<uint32_t> reader_rt_args;
    std::vector<uint32_t> reader_compile_time_args;
    string writer_kernel;
    string compute_kernel;
    if (use_fast_reader) {
        TT_ASSERT(!(conv_act_size_c & (conv_act_size_c - 1))); // channel depth power of 2 is supported only
        TT_ASSERT(!(out_row_size_bytes & (out_row_size_bytes - 1))); // output channels power of 2 is supported only
        if (pad_h == 0 && pad_w == 0) {
            if(rn50_first_conv) {
                reader_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_fast_resnet50_first_conv.cpp";
                compute_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/bmm_tilize_untilize_all_weights_in_l1_single_output_block_width_dim.cpp";
            } else {
                reader_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_fast_without_conv_padding.cpp";
                compute_kernel = "tt_eager/tt_dnn/kernels/compute/bmm_tilize_untilize.cpp";
            }
        } else {
            reader_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations_fast.cpp";
            compute_kernel = "tt_eager/tt_dnn/kernels/compute/bmm_tilize_untilize.cpp";
        }
        reader_compile_time_args = {(uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0),
                (uint32_t) stride_h, (uint32_t) stride_w, (uint32_t) conv_act_size_w, (uint32_t) conv_output_size_w,
                (uint32_t) conv_act_size_c * num_bytes_of_df, (uint32_t) std::log2(conv_act_size_c * num_bytes_of_df)};
    } else {
        reader_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_conv_activations.cpp";
        reader_compile_time_args = {(uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0)};
        compute_kernel = "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/bmm_tilize_untilize.cpp";
    }
    if (use_fast_reader && rn50_first_conv) {
        assert(pad_h == 0 && pad_w == 0);
        reader_rt_args = {
            act_dram_addr,
            conv_act_size_c,
            conv_output_size_w,
            weight_size_w,
            num_blocks_act_h,
            num_blocks_act_w,
            act_block_h_datums,
            act_block_num_tiles
        };
    } else {
        reader_rt_args = {
            // arguments for act
            act_dram_addr,
            act_noc_x,
            act_noc_y,

            conv_act_size_w,
            conv_act_size_h,
            conv_act_size_c,
            weight_size_h,
            weight_size_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            conv_output_size_h,
            conv_output_size_w,
            num_blocks_act_h,
            num_blocks_act_w,
            num_blocks_weight_w,
            num_groups,

            act_matrix_height_unpadded,
            act_matrix_width_unpadded,
            act_matrix_height,
            act_matrix_width,
            act_matrix_height_ntiles,
            act_matrix_width_ntiles,
            act_block_h_datums,
            act_block_w_datums,
            act_block_h_ntiles,
            act_block_w_ntiles,
            act_block_num_tiles,

            src_dram_act_buffer_size_bytes,
            dst_l1_act_buffer_size_bytes,
        };
    }

    vector<uint32_t> writer_rt_args;
    std::vector<uint32_t> writer_compile_time_args;
    if (untilize_out) {
        if (rn50_first_conv) {
            writer_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_and_reader_weights_resnet50_first_conv_untilize_out.cpp";
        } else if (use_fast_reader) {
            writer_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_unary_stick_8bank_blocks_reader_weight_tile_with_pow2_addr_gen_fast.cpp";
        } else {
            writer_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_unary_stick_layout_8bank_blocks_reader_weight_tile_layout.cpp";
        }
        writer_compile_time_args = {(uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0), out0_cb, weight_cb, (uint32_t) std::log2(out_row_size_bytes)};
        writer_rt_args = {
            out_dram_addr,
            weight_dram_addr,

            act_block_h_datums,
            out_block_row_size_bytes,
            1,
            num_blocks_act_h,
            num_blocks_weight_w,
            out_row_size_bytes,
            last_block_row_size_bytes,
            act_matrix_height_unpadded,

            num_blocks_act_w, // = number of blocks of weight in height dim
            weight_block_num_tiles,
            weight_block_h_ntiles,
            weight_block_w_ntiles,
            weight_matrix_width_ntiles, // weight_stride_h
            weight_matrix_width_ntiles * weight_block_h_ntiles, // weight_next_block_stride_h,
            weight_block_w_ntiles, // weight_next_block_stride_w

        };
    } else {
        assert(use_fast_reader); // tiled out not tested for generic conv
        if (rn50_first_conv) {
            writer_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_and_reader_weights_resnet50_first_conv_tiled_out.cpp";
        } else {
            writer_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_reader_conv_weights_tiled.cpp";
        }
        writer_compile_time_args = {
            (uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0),
            out0_cb,
            weight_cb,
            bias_cb,
            bias_log2_of_pagesize,
            bias_tile_nbytes,
            (uint32_t) (bias_buffer == nullptr ? 0 : (bias_buffer->buffer_type() == BufferType::DRAM ? 1 : 0))};
        writer_rt_args = {
            out_dram_addr,
            weight_dram_addr,

            output_width_num_tiles, // out_next_tile_stride_h
            1, // out_next_tile_stride_w
            out_subblock_h_ntiles * output_width_num_tiles, // out_next_subblock_stride_h
            out_subblock_w_ntiles, // out_next_subblock_stride_w
            act_block_h_ntiles * output_width_num_tiles, // out_next_block_stride_h
            weight_block_w_ntiles, // out_next_block_stride_w
            out_subblock_h_ntiles,
            out_subblock_w_ntiles,
            out_subblock_num_tiles,
            act_block_h_ntiles / out_subblock_h_ntiles, // out_num_subblocks_h
            weight_block_w_ntiles / out_subblock_w_ntiles,   // out_num_subblocks_w
            num_blocks_act_h, // out_num_blocks_h
            num_blocks_weight_w, // out_num_blocks_w
            act_block_h_ntiles, // out_block_height_num_tiles
            output_height_num_tiles, // out_height_num_tiles without block shape padding
            output_width_num_tiles, // out_width_num_tiles withoug block shape padding

            num_blocks_act_w, // = number of blocks of weight in height dim
            weight_block_num_tiles,
            weight_block_h_ntiles,
            weight_block_w_ntiles,
            weight_matrix_width_ntiles, // weight_stride_h
            weight_matrix_width_ntiles * weight_block_h_ntiles, // weight_next_block_stride_h,
            weight_block_w_ntiles, // weight_next_block_stride_w

            // bias
            bias_dram_addr,
            bias_ntiles
        };
    }
    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(a.get_dtype());
    auto reader_id = CreateKernel(
        program,
        reader_kernel,
        core,
        ReaderDataMovementConfig(
            reader_compile_time_args,
            reader_defines));

    auto writer_id = CreateKernel(
        program,
        writer_kernel,
        core,
        WriterDataMovementConfig(
            writer_compile_time_args,
            all_defines));

    vector<uint32_t> compute_kernel_args = {
        act_block_w_ntiles,
        act_num_subblocks,
        act_block_num_tiles,
        act_subblock_num_tiles,
        act_subblock_h_ntiles,

        weight_num_subblocks,
        weight_block_num_tiles,
        weight_block_w_ntiles,

        num_blocks_act_h,
        num_blocks_act_w,
        num_blocks_weight_w,

        out_subblock_h_ntiles,
        out_subblock_w_ntiles,
        out_subblock_num_tiles,

        true,
        untilize_out,

        bias_ntiles
    };

    auto compute = CreateKernel(
        program,
        compute_kernel,
        core,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    SetRuntimeArgs(
        program, reader_id, core,
        reader_rt_args
    );

    SetRuntimeArgs(
        program, writer_id, core,
        writer_rt_args
    );

    auto override_runtime_args_callback = [
        reader_kernel_id=reader_id,
        writer_kernel_id=writer_id,
        has_bias=has_bias
    ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        TT_ASSERT(input_buffers.size() == 3);
        TT_ASSERT(output_buffers.size() == 1);

        auto src_dram_buffer_a = input_buffers.at(0);
        auto src_dram_buffer_b = input_buffers.at(1);

        auto dst_dram_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer_a->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
            runtime_args[1] = src_dram_buffer_b->address();
            if (has_bias) {
                auto src_dram_buffer_c = input_buffers.at(2);
                TT_ASSERT(src_dram_buffer_c != nullptr);
                runtime_args[25] = src_dram_buffer_c->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};

}

// generates address map for reader kernel which reads from dram buffer (tiled layout) into l1 buffer
std::pair<vector<uint32_t>, vector<uint32_t>> generate_conv_weight_address_map(
                            const ttnn::Shape& weight_shape,
                            uint32_t weight_block_h_datums,
                            uint32_t weight_block_w_datums,
                            uint32_t num_blocks_act_h,
                            uint32_t num_blocks_weight_h,
                            uint32_t num_blocks_weight_w,
                            uint32_t num_bytes_df) {
    vector<uint32_t> address_map;
    vector<uint32_t> address_map_metadata;
    assert(weight_shape[0] == 1 && weight_shape[1] == 1);
    uint32_t matrix_height = weight_shape[2];
    uint32_t matrix_width = weight_shape[3];
    assert(matrix_height % weight_block_h_datums == 0);
    assert(matrix_width % weight_block_w_datums == 0);
    uint32_t src_dram_buffer_size_bytes = matrix_height * matrix_width * num_bytes_df;
    uint32_t dst_l1_buffer_size_bytes = weight_block_h_datums * weight_block_w_datums * num_bytes_df;
    uint32_t num_groups = num_blocks_act_h * num_blocks_weight_h * num_blocks_weight_w;
    assert(matrix_height % TILE_HEIGHT == 0);
    uint32_t matrix_height_ntiles = matrix_height / TILE_HEIGHT;
    assert(matrix_width % TILE_WIDTH == 0);
    uint32_t matrix_width_ntiles = matrix_width / TILE_WIDTH;
    assert(matrix_height_ntiles % num_blocks_weight_h == 0);
    uint32_t block_height_ntiles = matrix_height_ntiles / num_blocks_weight_h;
    assert(matrix_width_ntiles % num_blocks_weight_w == 0);
    uint32_t block_width_ntiles = matrix_width_ntiles / num_blocks_weight_w;
    uint32_t matrix_size_ntiles = matrix_height_ntiles * matrix_width_ntiles;
    assert(weight_block_h_datums % TILE_HEIGHT == 0);
    assert(weight_block_w_datums % TILE_WIDTH == 0);
    assert(block_height_ntiles == weight_block_h_datums / TILE_HEIGHT);
    assert(block_width_ntiles == weight_block_w_datums / TILE_WIDTH);
    address_map_metadata.push_back(num_groups);
    uint32_t address_map_current_group_dram_address_offset = 0;
    for(uint32_t group_idx = 0; group_idx < num_groups; group_idx++) {
        // Weight blocks are col major
        uint32_t block_idx_h = (uint32_t) (group_idx % num_blocks_weight_h);
        uint32_t block_idx_w = (uint32_t) (group_idx / num_blocks_weight_h) % (num_blocks_weight_w);
        uint32_t block_idx = (block_idx_w * num_blocks_weight_h) + block_idx_h;
        uint32_t start_block_tile_h_index = block_idx_h * block_height_ntiles;
        uint32_t start_block_tile_w_index = block_idx_w * block_width_ntiles;
        uint32_t single_tile_size_bytes = TILE_HEIGHT * TILE_WIDTH * num_bytes_df;
        uint32_t address_map_current_group_size = 0;
        // Weight tiles are in row major order within block
        for(uint32_t tile_h_index_in_block = 0; tile_h_index_in_block < block_height_ntiles; tile_h_index_in_block++) {
            for(uint32_t tile_w_index_in_block = 0; tile_w_index_in_block < block_width_ntiles; tile_w_index_in_block++) {
                uint32_t tile_index_h_in_matrix = tile_h_index_in_block + start_block_tile_h_index;
                uint32_t tile_index_w_in_matrix = tile_w_index_in_block + start_block_tile_w_index;
                // Weight tiles are in row major order in weight matrix in dram
                uint32_t tile_index_in_matrix = (tile_index_h_in_matrix * block_width_ntiles * num_blocks_weight_w) + tile_index_w_in_matrix;
                assert(tile_index_in_matrix < matrix_size_ntiles);
                // Weight tiles are in row major order in weight block in l1
                uint32_t tile_index_in_block = tile_h_index_in_block * block_width_ntiles + tile_w_index_in_block;
                uint32_t src_address_offset_dram = tile_index_in_matrix * single_tile_size_bytes;
                uint32_t read_size_bytes = single_tile_size_bytes;
                uint32_t dst_address_offset_l1 = tile_index_in_block * single_tile_size_bytes;
                uint32_t pad = 0;
                assert(read_size_bytes > 0);
                assert(pad == 0 || pad == 1);
                assert(src_address_offset_dram < src_dram_buffer_size_bytes);
                assert(dst_address_offset_l1 < dst_l1_buffer_size_bytes);
                address_map.push_back(src_address_offset_dram);
                address_map.push_back(dst_address_offset_l1);
                address_map.push_back(read_size_bytes);
                address_map.push_back(pad);
                address_map_current_group_size += 4;
            }
        }
        // DRAM reads should be 32B aligned
        assert(address_map_current_group_dram_address_offset%32 == 0);
        address_map_metadata.push_back(address_map_current_group_dram_address_offset);
        address_map_metadata.push_back(address_map_current_group_size);
        // Pad 0s in address map buffer to ensure each read address is 32B aligned (32/sizeof(uint32_t) == 8 elements)
        uint32_t address_map_current_group_size_padded = (uint32_t) (std::ceil((double) address_map_current_group_size / (double) 8) * 8);
        if(address_map_current_group_size_padded != address_map_current_group_size) {
            assert(address_map_current_group_size_padded > address_map_current_group_size);
            address_map.insert(address_map.end(), address_map_current_group_size_padded - address_map_current_group_size, 0);
        }
        // update next group's dram read address offset (in bytes)
        address_map_current_group_dram_address_offset += (address_map_current_group_size_padded*sizeof(uint32_t));
    }
    return make_pair(std::move(address_map), std::move(address_map_metadata));
}

std::pair<vector<uint32_t>, vector<uint32_t>> generate_conv_activation_address_map(
                            const ttnn::Shape& activation_shape,
                            const vector<int>& conv_params,
                            uint32_t act_block_h_datums,
                            uint32_t act_block_w_datums,
                            uint32_t weight_block_w_datums,
                            uint32_t num_blocks_act_h,
                            uint32_t num_blocks_act_w,
                            uint32_t num_blocks_weight_w,
                            uint32_t num_bytes_df) {
    vector<uint32_t> address_map;
    vector<uint32_t> address_map_metadata;
    uint32_t conv_input_y = activation_shape[1];
    uint32_t conv_input_x = activation_shape[2];
    uint32_t conv_input_z = activation_shape[3];
    uint32_t R = conv_params[0];
    uint32_t S = conv_params[1];
    uint32_t U = conv_params[2];
    uint32_t V = conv_params[3];
    uint32_t Pad_H = conv_params[4];
    uint32_t Pad_W = conv_params[5];
    uint32_t src_dram_buffer_size_bytes = conv_input_x * conv_input_y * conv_input_z * num_bytes_df;
    uint32_t dst_l1_buffer_size_bytes = act_block_h_datums * act_block_w_datums * num_bytes_df;
    int conv_output_h = ((conv_input_x - R + (2 * Pad_H)) / U) + 1;
    int conv_output_w = ((conv_input_y - S + (2 * Pad_W)) / V) + 1;
    uint32_t matrix_height_unpadded = conv_output_h * conv_output_w;
    uint32_t matrix_width_unpadded = conv_input_z * R * S;
    uint32_t matrix_height = (uint32_t) (std::ceil((double) matrix_height_unpadded / (double) act_block_h_datums ) * act_block_h_datums);
    uint32_t matrix_width = (uint32_t) (std::ceil((double) matrix_width_unpadded / (double) act_block_w_datums ) * act_block_w_datums);

    uint32_t num_groups = num_blocks_act_h * num_blocks_act_w * num_blocks_weight_w;
    uint32_t channel_stick_size = conv_input_z;
    uint32_t address_map_current_group_dram_address_offset = 0;
    address_map_metadata.push_back(num_groups);
    for(uint32_t group_idx = 0; group_idx < num_groups; group_idx++) {
        uint32_t block_idx_h = (uint32_t) (group_idx / num_blocks_act_w) / (num_blocks_weight_w);
        uint32_t block_idx_w = (uint32_t) (group_idx % num_blocks_act_w);
        uint32_t block_idx = (block_idx_h * num_blocks_act_w) + block_idx_w;
        uint32_t start_block_2d_index_h = block_idx_h * act_block_h_datums;
        uint32_t start_block_2d_index_w = block_idx_w * act_block_w_datums;
        uint32_t start_block_2d_index = (start_block_2d_index_h * act_block_w_datums * num_blocks_act_w) + start_block_2d_index_w;
        assert(start_block_2d_index_w < matrix_width_unpadded);
        uint32_t address_map_current_group_size = 0;
        for(uint32_t h_b = 0; h_b < act_block_h_datums; h_b++) {
            uint32_t h = start_block_2d_index_h + h_b;
            uint32_t dst_address_offset_l1 = h_b * act_block_w_datums * num_bytes_df;
            if (h >= matrix_height_unpadded) {
                // pad (block shape padding for height dim)
                uint32_t pad_size_bytes = act_block_w_datums * num_bytes_df;
                assert(dst_address_offset_l1 < dst_l1_buffer_size_bytes);
                address_map.push_back(0); // src address not used
                address_map.push_back(dst_address_offset_l1);
                address_map.push_back(pad_size_bytes);
                address_map.push_back(1); // pad = 1
                address_map_current_group_size += 4;
            }
            else {
                uint32_t w = start_block_2d_index_w;
                uint32_t end_block_2d_index_w = start_block_2d_index_w + act_block_w_datums - 1;
                assert(end_block_2d_index_w < matrix_width);
                while (w <= end_block_2d_index_w) {
                    uint32_t src_address_offset_dram = 0;
                    uint32_t read_size_bytes = 0;
                    uint32_t pad = 0;
                    if (w >= matrix_width_unpadded) {
                        // pad (block shape padding for width dim)
                        assert(end_block_2d_index_w == matrix_width-1);
                        read_size_bytes = (end_block_2d_index_w - w + 1) * num_bytes_df;
                        pad = 1;
                    }
                    else {
                        uint32_t channel_stick_offset = w % channel_stick_size;
                        uint32_t channel_stick_col_id = w / channel_stick_size;
                        uint32_t channel_stick_row_id = h;
                        assert(channel_stick_offset % (32/num_bytes_df) == 0); // DRAM read address must be aligned to 32 bytes
                        uint32_t channel_stick_row_id_x = channel_stick_row_id % conv_output_w;
                        uint32_t channel_stick_row_id_y = channel_stick_row_id / conv_output_w;
                        uint32_t act_tensor_start_x = channel_stick_row_id_x * V;
                        uint32_t act_tensor_start_y = channel_stick_row_id_y * U;
                        uint32_t act_tensor_padded_x = act_tensor_start_x + (channel_stick_col_id % S);
                        uint32_t act_tensor_padded_y = act_tensor_start_y + (channel_stick_col_id / S);
                        assert(w <= end_block_2d_index_w);
                        uint32_t read_size = std::min(channel_stick_size - channel_stick_offset, (end_block_2d_index_w+1)-w);
                        read_size_bytes = read_size * num_bytes_df;
                        if(act_tensor_padded_x < Pad_W || act_tensor_padded_x >= (Pad_W + conv_input_x) || act_tensor_padded_y < Pad_H || act_tensor_padded_y >= (Pad_H + conv_input_y)) {
                            // pad (conv padding)
                            pad = 1;
                        }
                        else {
                            uint32_t act_tensor_x = act_tensor_padded_x - Pad_W;
                            uint32_t act_tensor_y = act_tensor_padded_y - Pad_H;
                            assert(act_tensor_x < conv_input_x && act_tensor_x >= 0 && act_tensor_y < conv_input_y && act_tensor_y >= 0);
                            uint32_t act_tensor_channel_id = act_tensor_y * conv_input_x + act_tensor_x;
                            src_address_offset_dram = ((act_tensor_channel_id * channel_stick_size) + channel_stick_offset) * num_bytes_df;
                            assert(src_address_offset_dram % 32 == 0); // DRAM read address must be aligned to 32 bytes
                        }
                    }
                    assert(read_size_bytes > 0);
                    assert(pad == 0 || pad == 1);
                    assert(src_address_offset_dram < src_dram_buffer_size_bytes);
                    assert(dst_address_offset_l1 < dst_l1_buffer_size_bytes);
                    address_map.push_back(src_address_offset_dram);
                    address_map.push_back(dst_address_offset_l1);
                    address_map.push_back(read_size_bytes);
                    address_map.push_back(pad);
                    address_map_current_group_size += 4;
                    dst_address_offset_l1 += read_size_bytes;
                    w += (read_size_bytes/num_bytes_df);
                    assert(w <= end_block_2d_index_w+1);
                }
            }
        }
        // DRAM reads should be 32B aligned
        assert(address_map_current_group_dram_address_offset%32 == 0);
        address_map_metadata.push_back(address_map_current_group_dram_address_offset);
        address_map_metadata.push_back(address_map_current_group_size);
        // Pad 0s in address map buffer to ensure each read address is 32B aligned (32/sizeof(uint32_t) == 8 elements)
        uint32_t address_map_current_group_size_padded = (uint32_t) (std::ceil((double) address_map_current_group_size / (double) 8) * 8);
        if(address_map_current_group_size_padded != address_map_current_group_size) {
            assert(address_map_current_group_size_padded > address_map_current_group_size);
            address_map.insert(address_map.end(), address_map_current_group_size_padded - address_map_current_group_size, 0);
        }
        // update next group's dram read address offset (in bytes)
        address_map_current_group_dram_address_offset += (address_map_current_group_size_padded*sizeof(uint32_t));
    }
    return make_pair(std::move(address_map), std::move(address_map_metadata));
}

std::pair<vector<uint32_t>, vector<uint32_t>> populate_address_map_vectors_for_reader_kernel(vector<uint32_t> address_map_raw) {
    // This function is called twice i.e., for activation and weight address maps
    // "address_map_raw" is the DTX address map vector returned from DTX "conv_transform" function.
    // "address_map_raw" contains metadata along with the address map data for all groups
    // To keep the reader kernel simple, the metadata is separated into a different buffer
    // So two buffers are created -
    // First buffer is in DRAM containing the address map for all groups
    //      This DRAM buffer is big and is streamed into L1 scratchpad
    // Second buffer contains the metadata and is copied to L1 from host
    // It contains number of groups in its first index, followed by group info for each group -
    //      1. dram read address offset of address map group in dram buffer (in bytes)
    //      2. size of address map group in dram buffer (in datums, not bytes)
    // TODO (nshanker), support for streaming the second buffer from dram if it does not fit in L1
    vector<uint32_t> address_map; // will be in dram
    vector<uint32_t> address_map_metadata; // will be in l1

    uint32_t num_address_map_fields_per_transfer = 4; // TODO (nshanker): remove hardcoded 4 and get this value from output of DTX
    uint32_t num_dtx_groups = address_map_raw[0];
    address_map_metadata.push_back(address_map_raw[0]);
    uint32_t address_map_raw_index = 1;
    uint32_t current_group_dram_address_offset = 0;
    for(uint32_t g = 0; g < num_dtx_groups; g++) {
        // insert group's dram read address (in bytes) in metadata buffer
        // Separate reads are issued for each "address map group"
        // DRAM reads should be 32B aligned
        assert(current_group_dram_address_offset%32 == 0);
        address_map_metadata.push_back(current_group_dram_address_offset);
        // insert group size (datums, not in bytes) into metadata buffer
        uint32_t current_group_size = address_map_raw[address_map_raw_index];
        address_map_metadata.push_back(current_group_size);
        address_map_raw_index += 1;
        // insert address map for this group into the address map buffer
        auto address_map_raw_current_group_start = address_map_raw.begin() + address_map_raw_index;
        address_map.insert(address_map.end(),
                                address_map_raw_current_group_start,
                                address_map_raw_current_group_start + current_group_size);
        address_map_raw_index += current_group_size;
        // Pad 0s in address map buffer to ensure each read address is 32B aligned (32/sizeof(uint32_t) == 8 elements)
        uint32_t current_group_size_padded = (uint32_t) (std::ceil((double) current_group_size / (double) 8) * 8);
        if(current_group_size_padded != current_group_size) {
            assert(current_group_size_padded > current_group_size);
            address_map.insert(address_map.end(), current_group_size_padded - current_group_size, 0);
        }
        // update next group's dram read address offset (in bytes)
        current_group_dram_address_offset += (current_group_size_padded*sizeof(uint32_t));
    }
    return make_pair(std::move(address_map), std::move(address_map_metadata));
}

operation::ProgramWithCallbacks conv_as_large_bmm_with_address_map_single_core_(const Tensor& a, const Tensor &b, vector<int> conv_params,
                                       uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
                                       uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, bool untilize_out, Tensor &output) {
    bool pass = true;
    assert(untilize_out == true);
    tt_metal::Device *device = a.device();
    TT_ASSERT(a.get_layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_ASSERT(a.get_legacy_shape()[0] == 1, "Only batch size 1 supported.");
    TT_ASSERT(output_channels <= b.get_legacy_shape()[3], "Invalid weight shape. Incorrect weight tensor.");

    uint32_t num_bytes_of_df = 2; // 2 bytes for bfloat16
    // Compute the 2d matrix shape
    auto [matrix_shape, matrix_shape_unpadded] = conv_op_utils::compute_conv_activation_as_mm_shape(a.get_legacy_shape(), conv_params, act_block_h_ntiles, act_block_w_ntiles, false);
    assert(matrix_shape.size() == 3);
    assert(matrix_shape[0] == 1);
    uint32_t num_rows = (uint32_t) matrix_shape[1];
    uint32_t num_cols = (uint32_t) matrix_shape[2];

    // More Checks
    uint32_t Ba = 1;
    uint32_t Ca = 1;
    auto Ha = num_rows;
    auto Wa = num_cols;
    uint32_t Bb = b.get_legacy_shape()[0];
    uint32_t Cb = b.get_legacy_shape()[1];
    uint32_t Hb = b.get_legacy_shape()[2];
    uint32_t Wb = b.get_legacy_shape()[3];
    // Normal matrix shape checks
    TT_ASSERT(Ba == 1, "So far, large matmul op has only been tested for batch one.");
    TT_ASSERT(Ba == Bb, "Batch dimension needs to match");
    TT_ASSERT(Ca == Cb, "Channel dimension needs to match");
    TT_ASSERT(Wa == Hb, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_ASSERT(Ha % TILE_HEIGHT == 0, "Height of tensor a needs to be divisible by 32");
    TT_ASSERT(Wa % TILE_WIDTH == 0, "Width of tensor a needs to be divisible by 32");
    TT_ASSERT(Hb % TILE_HEIGHT == 0, "Height of tensor b needs to be divisible by 32");
    TT_ASSERT(Wb % TILE_WIDTH == 0, "Width of tensor b needs to be divisible by 32");

    // Device compatibility checks
    TT_ASSERT(a.storage_type() == StorageType::DEVICE and b.storage_type() == StorageType::DEVICE, "Operands to large matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to large matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to large matmul need to be allocated in buffers on device!");
    // Convert tensor dims to tile dims
    uint32_t B   = Ba;
    uint32_t Hat = Ha / TILE_HEIGHT;
    uint32_t Wat = Wa / TILE_WIDTH;
    uint32_t Wbt = Wb / TILE_WIDTH;
    log_debug(tt::LogOp, "Hat(MM Activation H in tiles): {}", Hat);
    log_debug(tt::LogOp, "Wat(MM Activation W (MM Weight H) in tiles): {}", Wat);
    log_debug(tt::LogOp, "Wbt(MM Weight W in tiles): {}", Wbt);

    assert(Hat % act_block_h_ntiles == 0);
    assert(Wat % act_block_w_ntiles == 0);
    assert(Wbt % weight_block_w_ntiles == 0);

    uint32_t num_blocks_act_h = Hat / act_block_h_ntiles;
    uint32_t num_blocks_act_w = Wat / act_block_w_ntiles;
    uint32_t num_blocks_weight_w = Wbt / weight_block_w_ntiles;

    // act block info
    uint32_t act_block_w_datums = Wa / num_blocks_act_w;
    uint32_t act_block_h_datums = Ha / num_blocks_act_h;

    // weight block info
    uint32_t weight_block_w_datums = Wb / num_blocks_weight_w;
    assert(weight_block_w_ntiles % out_subblock_w_ntiles == 0);
    uint32_t weight_num_subblocks = weight_block_w_ntiles / out_subblock_w_ntiles;
    uint32_t weight_block_h_ntiles = act_block_w_ntiles;
    uint32_t weight_block_num_tiles = weight_block_w_ntiles * weight_block_h_ntiles;
    // writer of conv op partially removes padding on the width
    // it removes the padding done for block width but it doesn't remove padding done for tiled width
    uint32_t output_channels_padded_to_tile_width = round_up(output_channels, TILE_WIDTH);
    assert(output_channels_padded_to_tile_width <= Wb);
    uint32_t num_blocks_output_w = (uint32_t) std::ceil((double) output_channels_padded_to_tile_width / (double) weight_block_w_datums);
    uint32_t last_block_width_datums = (output_channels_padded_to_tile_width % weight_block_w_datums == 0) ? weight_block_w_datums : (output_channels_padded_to_tile_width % weight_block_w_datums);
    assert(last_block_width_datums % TILE_WIDTH == 0);
    uint32_t output_row_size_bytes = output_channels_padded_to_tile_width * num_bytes_of_df;
    uint32_t last_block_row_size_bytes = last_block_width_datums * num_bytes_of_df;
    // sanity check
    assert(num_blocks_output_w == num_blocks_weight_w);

    // DTX conv activation transform data access pattern
    auto [act_address_map, act_address_map_metadata] = generate_conv_activation_address_map(ttnn::Shape(a.get_legacy_shape()), conv_params, act_block_h_datums, act_block_w_datums, weight_block_w_datums,
                                                            num_blocks_act_h, num_blocks_act_w, num_blocks_weight_w, num_bytes_of_df);

    auto [weight_address_map, weight_address_map_metadata] = generate_conv_weight_address_map(ttnn::Shape(b.get_legacy_shape()), act_block_w_datums, weight_block_w_datums,
                                                                num_blocks_act_h, num_blocks_act_w, num_blocks_weight_w, num_bytes_of_df);

    // sanity check
    uint32_t num_dtx_groups = act_address_map_metadata[0];
    assert(weight_address_map_metadata[0] == num_dtx_groups);

    // debug prints
    int detailed_debug = 1;
    if(detailed_debug > 0) {
        log_debug(tt::LogOp, "Printing activation and weight address maps.");
        log_debug(tt::LogOp, "DTX groups: {}", num_dtx_groups);
        uint32_t act_metadata_index = 1;
        uint32_t weight_metadata_index = 1;
        uint32_t act_addr_map_index = 0;
        uint32_t weight_addr_map_index = 0;
        for(uint32_t g = 0; g < num_dtx_groups; g++) {
            log_debug(tt::LogOp, "  DTX group: {}", g);
            uint32_t act_current_group_address = act_address_map_metadata[act_metadata_index];
            act_metadata_index += 1;
            uint32_t act_current_group_size = act_address_map_metadata[act_metadata_index];
            act_metadata_index += 1;
            log_debug(tt::LogOp, "      act_current_group_address: {}", act_current_group_address);
            log_debug(tt::LogOp, "      act_current_group_size: {}", act_current_group_size);
            if(detailed_debug > 1) {
                uint32_t act_current_group_index = act_current_group_address/sizeof(uint32_t);
                for(uint32_t i = act_current_group_index; i < act_current_group_index + act_current_group_size; i+=4) {
                    log_debug(tt::LogOp, "          act_addr_map[0]: {}", act_address_map[i]);
                    log_debug(tt::LogOp, "          act_addr_map[1]: {}", act_address_map[i+1]);
                    log_debug(tt::LogOp, "          act_addr_map[2]: {}", act_address_map[i+2]);
                    log_debug(tt::LogOp, "          act_addr_map[3]: {}", act_address_map[i+3]);
                }
            }
            uint32_t weight_current_group_address = weight_address_map_metadata[weight_metadata_index];
            weight_metadata_index += 1;
            uint32_t weight_current_group_size = weight_address_map_metadata[weight_metadata_index];
            weight_metadata_index += 1;
            log_debug(tt::LogOp, "      weight_current_group_address: {}", weight_current_group_address);
            log_debug(tt::LogOp, "      weight_current_group_size: {}", weight_current_group_size);
            if(detailed_debug > 1) {
                uint32_t weight_current_group_index = weight_current_group_address/sizeof(uint32_t);
                for(uint32_t i = weight_current_group_index; i < weight_current_group_index + weight_current_group_size; i+=4) {
                    log_debug(tt::LogOp, "          weight_addr_map[0]: {}", weight_address_map[i]);
                    log_debug(tt::LogOp, "          weight_addr_map[1]: {}", weight_address_map[i+1]);
                    log_debug(tt::LogOp, "          weight_addr_map[2]: {}", weight_address_map[i+2]);
                    log_debug(tt::LogOp, "          weight_addr_map[3]: {}", weight_address_map[i+3]);
                }
            }
        }
    }

    uint32_t dram_bank_id = 0;
    auto act_address_map_buffer_size_in_dram = act_address_map.size() * sizeof(uint32_t);
    tt_metal::InterleavedBufferConfig act_config{
                    .device= device,
                    .size = act_address_map_buffer_size_in_dram,
                    .page_size = act_address_map_buffer_size_in_dram,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    auto weight_address_map_buffer_size_in_dram = weight_address_map.size() * sizeof(uint32_t);
    tt_metal::InterleavedBufferConfig weight_config{
                    .device= device,
                    .size = weight_address_map_buffer_size_in_dram,
                    .page_size = weight_address_map_buffer_size_in_dram,
                    .buffer_type = tt_metal::BufferType::DRAM
        };


    auto act_address_map_dram_buffer = CreateBuffer(act_config);
    auto weight_address_map_dram_buffer = CreateBuffer(weight_config);
    uint32_t act_address_map_dram_addr = act_address_map_dram_buffer->address();
    // DRAM to L1 writes should 32B aligned
    assert(act_address_map_dram_addr%32 == 0);
    auto act_address_map_dram_noc_xy = act_address_map_dram_buffer->noc_coordinates();
    uint32_t act_address_map_dram_noc_x = act_address_map_dram_noc_xy.x;
    uint32_t act_address_map_dram_noc_y = act_address_map_dram_noc_xy.y;
    uint32_t weight_address_map_dram_addr = weight_address_map_dram_buffer->address();
    // DRAM to L1 writes should 32B aligned
    assert(weight_address_map_dram_addr%32 == 0);
    auto weight_address_map_dram_noc_xy = weight_address_map_dram_buffer->noc_coordinates();
    uint32_t weight_address_map_dram_noc_x = weight_address_map_dram_noc_xy.x;
    uint32_t weight_address_map_dram_noc_y = weight_address_map_dram_noc_xy.y;

    // Write address maps to DRAM
    detail::WriteToDeviceDRAMChannel(device, dram_bank_id, act_address_map_dram_addr, act_address_map);
    detail::WriteToDeviceDRAMChannel(device, dram_bank_id, weight_address_map_dram_addr, weight_address_map);

    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core_coord = {0, 0};      // TODO: avoid another var here. Find a way to use core range instead.
    CoreRange core({0, 0}, {0, 0});

    uint32_t single_tile_size = num_bytes_of_df * TILE_HEIGHT * TILE_WIDTH;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // L1 buffers
    // Create scratchpad buffer in L1 to stream in dtx address map from dram
    // One scratchpad buffer is used for both activation and weight address maps
    uint32_t num_address_map_fields_per_transfer = 4; // TODO: (nshanker): remove hardcoded 4 and get this value from output of DTX
    // Scratchpad buffer size must be a multiple of 32B to ensure DRAM->L1 addresses align 32B
    auto scratch_pad_for_address_map_in_l1_b0_size_bytes = 32;
    // Scratchpad buffer size must also be a multiple of address map fields per transfer. We need all address map fields for a transfer in scratchpad.
    assert(scratch_pad_for_address_map_in_l1_b0_size_bytes % (num_address_map_fields_per_transfer*sizeof(uint32_t)) == 0);

    tt_metal::InterleavedBufferConfig scratchpad_l1_config{
                    .device= device,
                    .size = (uint64_t)scratch_pad_for_address_map_in_l1_b0_size_bytes,
                    .page_size = (uint64_t)scratch_pad_for_address_map_in_l1_b0_size_bytes,
                    .buffer_type = tt_metal::BufferType::L1
        };

    auto scratch_pad_for_address_map_l1_buffer = CreateBuffer(scratchpad_l1_config);
    uint32_t scratch_pad_for_address_map_l1_address = scratch_pad_for_address_map_l1_buffer->address();
    // DRAM to L1 writes should 32B aligned
    assert(scratch_pad_for_address_map_l1_address%32 == 0);
    // Create address map metadata buffers in L1
    // Metadata vectors are copied to L1 buffers from host before calling detail::LaunchProgram
    auto act_address_map_metadata_l1_b0_size = act_address_map_metadata.size() * sizeof(uint32_t);

    tt_metal::InterleavedBufferConfig act_l1_config{
                    .device= device,
                    .size = (uint64_t)act_address_map_metadata_l1_b0_size,
                    .page_size = (uint64_t)act_address_map_metadata_l1_b0_size,
                    .buffer_type = tt_metal::BufferType::L1
        };
    auto act_address_map_metadata_l1_buffer = CreateBuffer(act_l1_config);
    uint32_t act_address_map_metadata_l1_address = act_address_map_metadata_l1_buffer->address();
    auto weight_address_map_metadata_l1_b0_size = weight_address_map_metadata.size() * sizeof(uint32_t);

    tt_metal::InterleavedBufferConfig weight_l1_config{
                    .device= device,
                    .size = (uint64_t)weight_address_map_metadata_l1_b0_size,
                    .page_size = (uint64_t)weight_address_map_metadata_l1_b0_size,
                    .buffer_type = tt_metal::BufferType::L1
        };


    auto weight_address_map_metadata_l1_buffer = CreateBuffer(weight_l1_config);
    uint32_t weight_address_map_metadata_l1_address = weight_address_map_metadata_l1_buffer->address();

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    uint32_t out_row_size = Wb * num_bytes_of_df;
    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;

    TT_ASSERT(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    // act
    uint32_t act_dram_addr = src0_dram_buffer->address();
    auto act_dram_noc_xy = src0_dram_buffer->noc_coordinates();
    uint32_t act_noc_x = act_dram_noc_xy.x;
    uint32_t act_noc_y = act_dram_noc_xy.y;

    assert(Wat % act_block_w_ntiles == 0);
    assert(act_block_h_ntiles % out_subblock_h_ntiles == 0);
    uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    // weight
    uint32_t weight_dram_addr = src1_dram_buffer->address();
    auto weight_dram_noc_xy = src1_dram_buffer->noc_coordinates();
    uint32_t weight_noc_x = weight_dram_noc_xy.x;
    uint32_t weight_noc_y = weight_dram_noc_xy.y;

    // output data format
    const auto out_df = datatype_to_dataformat_converter(a.get_dtype());
    // For debug
    {
        log_debug(tt::LogOp, "Hat (activation height in tiles): {}", Hat);
        log_debug(tt::LogOp, "Wat (activation width in tiles): {}", Wat);
        log_debug(tt::LogOp, "Wbt (weight width in tiles): {}", Wbt);
        log_debug(tt::LogOp, "num_blocks_act_h: {}", num_blocks_act_h);
        log_debug(tt::LogOp, "num_blocks_act_w: {}", num_blocks_act_w);
        log_debug(tt::LogOp, "num_blocks_weight_w: {}", num_blocks_weight_w);
        log_debug(tt::LogOp, "act_dram_addr: {}", act_dram_addr);
        log_debug(tt::LogOp, "act_block_h_ntiles: {}", act_block_h_ntiles);
        log_debug(tt::LogOp, "act_block_h_datums: {}", act_block_h_datums);
        log_debug(tt::LogOp, "act_block_w_ntiles: {}", act_block_w_ntiles);
        log_debug(tt::LogOp, "act_block_w_datums: {}", act_block_w_datums);
        log_debug(tt::LogOp, "act_num_subblocks: {}", act_num_subblocks);
        log_debug(tt::LogOp, "act_block_num_tiles: {}", act_block_num_tiles);
        log_debug(tt::LogOp, "act_address_map_dram_addr: {}", act_address_map_dram_addr);
        log_debug(tt::LogOp, "act_address_map_metadata_l1_address: {}", act_address_map_metadata_l1_address);
        log_debug(tt::LogOp, "act_subblock_h_ntiles: {}", act_subblock_h_ntiles);
        log_debug(tt::LogOp, "act_subblock_num_tiles: {}", act_subblock_num_tiles);
        log_debug(tt::LogOp, "weight_dram_addr: {}", weight_dram_addr);
        log_debug(tt::LogOp, "weight_num_subblocks: {}", weight_num_subblocks);
        log_debug(tt::LogOp, "weight_block_num_tiles: {}", weight_block_num_tiles);
        log_debug(tt::LogOp, "weight_address_map_dram_addr: {}", weight_address_map_dram_addr);
        log_debug(tt::LogOp, "weight_address_map_metadata_l1_address: {}", weight_address_map_metadata_l1_address);
        log_debug(tt::LogOp, "weight_block_w_ntiles: {}", weight_block_w_ntiles);
        log_debug(tt::LogOp, "weight_block_h_ntiles: {}", weight_block_h_ntiles);
        log_debug(tt::LogOp, "out_dram_addr: {}", out_dram_addr);
        log_debug(tt::LogOp, "out_row_size: {}", out_row_size);
        log_debug(tt::LogOp, "out_subblock_h_ntiles: {}", out_subblock_h_ntiles);
        log_debug(tt::LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
        log_debug(tt::LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(tt::LogOp, "num_dtx_groups: {}", num_dtx_groups);
        log_debug(tt::LogOp, "scratch_pad_for_address_map_l1_address: {}", scratch_pad_for_address_map_l1_address);
    }

    create_CBs_for_fused_matmul_new_alloc(
        program,
        a.device(),
        core,
        act_block_h_ntiles * act_block_w_ntiles, // row major act cb
        weight_block_h_ntiles * weight_block_w_ntiles, // tiled weight cb
        act_block_h_ntiles * act_block_w_ntiles, // tiled act cb
        act_block_h_ntiles * weight_block_w_ntiles, // math output cb
        weight_block_w_ntiles, // reblock cb
        act_block_h_ntiles * weight_block_w_ntiles, // writer output cb
        num_bytes_of_df,
        untilize_out);

    string reader_kernel;
    vector<uint32_t> reader_rt_args;
    reader_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/reader_binary_dtx.cpp";
    reader_rt_args = {
        // arguments for act
        act_dram_addr,
        act_noc_x,
        act_noc_y,
        act_address_map_dram_addr,
        act_address_map_dram_noc_x,
        act_address_map_dram_noc_y,
        act_address_map_metadata_l1_address,
        act_block_num_tiles,

        // arguments for weight
        weight_dram_addr,
        weight_noc_x,
        weight_noc_y,
        weight_address_map_dram_addr,
        weight_address_map_dram_noc_x,
        weight_address_map_dram_noc_y,
        weight_address_map_metadata_l1_address,
        weight_block_num_tiles,

        scratch_pad_for_address_map_l1_address,
    };

    string writer_kernel;
    vector<uint32_t> writer_rt_args;
    if (untilize_out) {
        writer_kernel = "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp";
        writer_rt_args = {
            out_dram_addr,
            act_block_h_datums,
            weight_block_w_ntiles*TILE_WIDTH*num_bytes_of_df,
            1,
            num_blocks_act_h,
            num_blocks_weight_w,
            output_channels_padded_to_tile_width*num_bytes_of_df,
            last_block_row_size_bytes,
            matrix_shape_unpadded[1],
            0,
            0
        };
    } else {
        assert(false && "Tiled output unsupported");
        writer_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_matmul_tile_layout.cpp";
        writer_rt_args = {
            out_dram_addr,
            0,
            1,
            Wbt,
            out_subblock_w_ntiles,
            out_subblock_h_ntiles * Wbt,

            out_subblock_w_ntiles,
            out_subblock_h_ntiles,
            out_subblock_w_ntiles * out_subblock_h_ntiles,
            Wbt / out_subblock_w_ntiles,
            Hat / out_subblock_h_ntiles
        };
    }
    auto reader_id = tt_metal::CreateKernel(
        program,
        reader_kernel,
        core,
        tt_metal::ReaderDataMovementConfig{});
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t) (src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0)};
    auto writer_id = tt_metal::CreateKernel(
        program,
        writer_kernel,
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_kernel_args = {
        act_block_w_ntiles,
        act_num_subblocks,
        act_block_num_tiles,
        act_subblock_num_tiles,
        act_subblock_h_ntiles,

        weight_num_subblocks,
        weight_block_num_tiles,
        weight_block_w_ntiles,

        num_blocks_act_h,
        num_blocks_act_w,
        num_blocks_weight_w,

        out_subblock_h_ntiles,
        out_subblock_w_ntiles,
        out_subblock_num_tiles,

        true,
        untilize_out
    };

    auto eltwise_binary_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/bmm_tilize_untilize.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    tt_metal::SetRuntimeArgs(
        program, reader_id, core,
        reader_rt_args
    );

    tt_metal::SetRuntimeArgs(
        program, writer_id, core,
        writer_rt_args
    );

    tt_metal::detail::WriteToDeviceL1(device, core_coord, act_address_map_metadata_l1_address, act_address_map_metadata);
    tt_metal::detail::WriteToDeviceL1(device, core_coord, weight_address_map_metadata_l1_address, weight_address_map_metadata);

     auto override_runtime_args_callback = [
        reader_kernel_id=reader_id,
        writer_kernel_id=writer_id
    ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer_a = input_buffers.at(0);
        auto src_dram_buffer_b = input_buffers.at(1);

        auto dst_dram_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer_a->address();
            runtime_args[1] = src_dram_buffer_a->noc_coordinates().x;
            runtime_args[2] = src_dram_buffer_a->noc_coordinates().y;
            runtime_args[8] = src_dram_buffer_b->address();
            runtime_args[9] = src_dram_buffer_b->noc_coordinates().x;
            runtime_args[10] = src_dram_buffer_b->noc_coordinates().y;
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

inline Tensor conv_(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias, const vector<int> conv_params,
                    uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
                    uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels,
                    bool use_address_map, bool use_fast_reader, bool untilize_out, bool has_bias = false, bool fuse_relu = false, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    TT_ASSERT(b.get_layout() == Layout::TILE); // Weights should already be formatted
    auto padded_a_shape = ttnn::Shape(std::vector<uint32_t>{a.get_legacy_shape()[0], a.get_legacy_shape()[1], a.get_legacy_shape()[2], round_up(a.get_legacy_shape()[3], 16)});
    ttnn::operations::experimental::auto_format::FormatParams input_a_format_params = {.pad_shape=padded_a_shape.value, .pad_value=0.0, .target_layout=Layout::ROW_MAJOR};
    ttnn::operations::experimental::auto_format::FormatParams input_b_format_params = {.pad_shape=b.get_legacy_shape(), .pad_value=0.0, .target_layout=Layout::TILE};
    ttnn::operations::experimental::auto_format::FormatParams input_bias_format_params = {};
    if (has_bias) {
        input_bias_format_params = {.pad_shape=bias.value().get_legacy_shape(), .pad_value=0, .target_layout=Layout::TILE};
    }
    auto output_layout = untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    return operation::run_without_autoformat(
        Conv(act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, conv_params, output_channels, use_address_map, use_fast_reader, untilize_out, has_bias, fuse_relu, math_fidelity),
        {a, b},
        {bias}).at(0);
}

Tensor conv(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, bool has_bias) {
    return conv_(a, b, bias, conv_params, act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, output_channels, false, false, true, has_bias);
}


operation::ProgramWithCallbacks conv_single_core(const Tensor& a, const Tensor &b, std::optional<const Tensor> bias, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, bool use_fast_reader, bool untilize_out, bool has_bias, bool fuse_relu, const MathFidelity math_fidelity, Tensor &output) {
    return conv_as_large_bmm_single_core_(a, b, bias, conv_params, act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, output_channels, use_fast_reader, untilize_out, has_bias, fuse_relu, math_fidelity, output);
}

operation::ProgramWithCallbacks conv_with_address_map_single_core(const Tensor& a, const Tensor &b, const vector<int> conv_params, uint32_t act_block_h_ntiles, uint32_t act_block_w_ntiles, uint32_t weight_block_w_ntiles,
             uint32_t out_subblock_h_ntiles, uint32_t out_subblock_w_ntiles, uint32_t output_channels, bool untilize_out, Tensor &output) {
    return conv_as_large_bmm_with_address_map_single_core_(a, b, conv_params, act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, output_channels, untilize_out, output);
}

void Conv::validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    // TODO: ...
}

std::vector<tt_metal::LegacyShape> Conv::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    uint32_t conv_activation_h = input_tensor_a.get_legacy_shape()[1];
    uint32_t conv_activation_w = input_tensor_a.get_legacy_shape()[2];
    // TODO: clean up here
    uint32_t filter_h = (uint32_t) conv_params[0];
    uint32_t filter_w = (uint32_t) conv_params[1];
    uint32_t stride_h = (uint32_t) conv_params[2];
    uint32_t stride_w = (uint32_t) conv_params[3];
    uint32_t pad_h = (uint32_t) conv_params[4];
    uint32_t pad_w = (uint32_t) conv_params[5];
    auto [conv_output_h, conv_output_w] = conv_op_utils::compute_conv_output_face_shape(conv_activation_h, conv_activation_w, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w);

    if (untilize_out) {
        // TODO: Update batch size below
        // RM output has unpadded output height and padded output width to 32.
        // pad the output channels to TILE_WIDTH as conv writer kernel does not remove padding for tile
        // TODO (nshanker): specify padding explicitly here with "Padding" object and add unit test
        auto output_channels = round_up(this->output_channels, TILE_WIDTH);
        auto output_tensor_shape = ttnn::Shape(std::vector<uint32_t>{1, conv_output_h, conv_output_w, output_channels});
        return {output_tensor_shape.value};
    } else {
        // Tiled output shape is padded shape. Padded to tile shape.
        auto shape_w = conv_output_h*conv_output_w;
        auto shape_c = output_channels;
        auto padded_shape_w = round_up(shape_w, TILE_HEIGHT);
        auto padded_shape_c = round_up(this->output_channels, TILE_WIDTH);
        auto output_padding = Padding({{0, 0}, {0, 0}, {0, (padded_shape_w - shape_w)}, {0, (padded_shape_c - shape_c)}}, Padding::PadValue::Any);
        auto output_tensor_shape = ttnn::Shape(tt::tt_metal::LegacyShape({1, 1, padded_shape_w, padded_shape_c}, output_padding));
        return {output_tensor_shape.value};
    }
}

std::vector<Tensor> Conv::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_layout = this->untilize_out ? Layout::ROW_MAJOR : Layout::TILE;
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), output_layout, input_tensor.memory_config());
}

operation::ProgramWithCallbacks Conv::create_program(const std::vector<Tensor>& input_tensors,
                                                     const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                                     std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& input_tensor_bias = optional_input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if(use_address_map) {
        return {conv_with_address_map_single_core(input_tensor_a, input_tensor_b, conv_params, act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, output_channels, untilize_out, output_tensor)};
    } else {
        return {conv_single_core(input_tensor_a, input_tensor_b, input_tensor_bias, conv_params, act_block_h_ntiles, act_block_w_ntiles, weight_block_w_ntiles, out_subblock_h_ntiles, out_subblock_w_ntiles, output_channels, use_fast_reader, untilize_out, has_bias, fuse_relu, math_fidelity, output_tensor)};
    }
}

}  // namespace tt_metal

}  // namespace tt
