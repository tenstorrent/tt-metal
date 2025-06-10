// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/circular_buffer.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/compute_throttle_utils.hpp"

using namespace tt::constants;

namespace ttnn::operations::conv {
namespace conv2d {

using namespace tt;

tt::tt_metal::operation::ProgramWithCallbacks multi_core_optimized_conv_width_sharded_v2_impl(
    tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const ttnn::Shape& ashape,
    std::optional<const Tensor> bias,
    const std::optional<const Tensor>& conv_reader_indices,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
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
    bool enable_subblock_padding);

// TODO: Add namespace for utilities?
std::tuple<tt::tt_metal::CBHandle, tt::tt_metal::CBHandle, tt::tt_metal::CBHandle> create_CBs_for_sharded_input_v2(
    tt_metal::Program& program,
    const Tensor& input,
    CoreRange core,
    uint32_t num_cb0_tiles,
    uint32_t num_cb0_second_reader_tiles,
    uint32_t num_cb1_tiles,
    uint32_t num_cb0_tilized_tiles,
    uint32_t num_output_tiles,
    uint32_t num_reblock_cb_tiles,
    uint32_t num_writer_output_tiles,
    bool untilize_out,
    tt::DataFormat act_df,
    tt::DataFormat weight_df,
    tt::DataFormat tilized_act_df,
    tt::DataFormat out_df,
    tt::DataFormat bias_df,
    bool weight_width_sliced,
    const Tensor& output,
    uint32_t bias_ntiles,
    bool with_bias,
    bool split_reader,
    bool fp32_dest_acc_en,
    bool packer_l1_acc_en,
    CBIndices& cb_indices) {
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;

    tt::DataFormat interm0_df =
        packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b) : out_df;

    uint32_t act_tile_size = tt_metal::detail::TileSize(act_df);
    uint32_t weight_tile_size = tt_metal::detail::TileSize(weight_df);
    uint32_t tilized_act_tile_size = tt_metal::detail::TileSize(tilized_act_df);
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);
    uint32_t interm0_single_tile_size = tt_metal::detail::TileSize(interm0_df);

    CBHandle cb_sharded_act = 0;
    CBHandle cb_output = 0;
    CBHandle cb_matmul_partials = 0;
    if (input.memory_config().is_sharded()) {
        uint32_t num_bytes_for_df = datum_size(act_df);
        auto shard_shape = input.shard_spec().value().shape;
        // 2D-sys-conv already has uint16_t indicies, TODO: do the same for 1D-sys-conv
        TT_FATAL(
            shard_shape[0] <= (1 << 16), "Shard height must be less than 2^16, read pattern indicies are uint16_t");
        // incoming data is the input cb instead of raw l1/dram addr
        std::tie(cb_indices.sharded_act_cb, cb_sharded_act) = tt::tt_metal::create_cb(
            cb_indices.get_next_cb_index(),
            program,
            core,
            shard_shape[1] * num_bytes_for_df,
            shard_shape[0],
            act_df,
            input.buffer());

        if (weight_width_sliced) {
            // For 2D convs, each core creates and tilizes full input matrix then mcasts round robin style
            // Each core receives input into act_cb, so won't need a separate cb to receive
            // However, we need a separate cb to push ROW_MAJOR BFLOAT16 data for tilizing and configure act cb to be
            // output df

            // num_cb0_tiles is double buffered
            cb_indices.act_cb = cb_indices.get_next_cb_index();
            tt::tt_metal::create_cb(
                cb_indices.act_cb, program, core, tilized_act_tile_size, num_cb0_tiles, tilized_act_df);
            log_debug(
                LogOp, "Act CB: {}, npages: {}, pagesize: {}", cb_indices.act_cb, num_cb0_tiles, tilized_act_tile_size);

            // num_cb0_tilized_tiles is single buffered
            cb_indices.act_cb_row_major_bfloat16 = cb_indices.get_next_cb_index();
            tt::tt_metal::create_cb(
                cb_indices.act_cb_row_major_bfloat16, program, core, act_tile_size, num_cb0_tilized_tiles, act_df);
            log_debug(
                LogOp,
                "Act CB Row Major BFLOAT16: {}, npages: {}, pagesize: {}",
                cb_indices.act_cb_row_major_bfloat16,
                num_cb0_tilized_tiles,
                act_tile_size);
        } else {
            // For 1D convs, locally create act matrix in act_cb, which is always ROW_MAJOR BFLOAT16
            // Then, tilize input in compute

            // Extra cb for second reader if we split act reads across two RISCs
            // In this case, the regular reader only does first half of reads along output block h
            if (split_reader) {
                cb_indices.act_cb_second_reader = cb_indices.get_next_cb_index();
                tt::tt_metal::create_cb(
                    cb_indices.act_cb_second_reader, program, core, act_tile_size, num_cb0_second_reader_tiles, act_df);
                log_debug(
                    LogOp,
                    "Act CB Second Reader: {}, npages: {}, pagesize: {}",
                    cb_indices.act_cb_second_reader,
                    num_cb0_second_reader_tiles,
                    act_tile_size);
            }
            cb_indices.act_cb = cb_indices.get_next_cb_index();
            tt::tt_metal::create_cb(cb_indices.act_cb, program, core, act_tile_size, num_cb0_tiles, act_df);
            log_debug(LogOp, "Act CB: {}, npages: {}, pagesize: {}", cb_indices.act_cb, num_cb0_tiles, act_tile_size);
        }
    } else {
        TT_THROW("Input must be sharded!");
    }

    tt::tt_metal::create_cb(cb_indices.weight_cb, program, core, weight_tile_size, num_cb1_tiles, weight_df);
    log_debug(LogOp, "Weight CB: {}, npages: {}, pagesize: {}", cb_indices.weight_cb, num_cb1_tiles, weight_tile_size);

    // Used for placing tilized activations
    tt::tt_metal::create_cb(
        cb_indices.tilize_mode_tilized_act_cb,
        program,
        core,
        tilized_act_tile_size,
        num_cb0_tilized_tiles,
        tilized_act_df);
    log_debug(
        LogOp,
        "Tilized Act CB: {}, npages: {}, pagesize: {}",
        cb_indices.tilize_mode_tilized_act_cb,
        num_cb0_tilized_tiles,
        tilized_act_tile_size);

    if (untilize_out) {
        auto output_shard_shape = output.shard_spec().value().shape;
        std::tie(cb_indices.matmul_partials_cb, cb_matmul_partials) = tt::tt_metal::create_cb(
            cb_indices.get_next_cb_index(), program, core, interm0_single_tile_size, num_output_tiles, interm0_df);
        log_debug(
            LogOp,
            "Matmul Partials CB: {}, npages: {}, pagesize: {}",
            cb_indices.matmul_partials_cb,
            num_output_tiles,
            interm0_single_tile_size);

        bool need_unpad_after_untilize =
            output_shard_shape[1] * output_shard_shape[0] < num_writer_output_tiles * TILE_HW;

        auto shard_shape = output.shard_spec().value().shape;
        uint32_t aligned_output_stick_nbytes = out_tile_size;
        uint32_t aligned_output_num_pages = num_writer_output_tiles;
        std::tie(cb_indices.out0_cb, cb_output) = tt::tt_metal::create_cb(
            cb_indices.get_next_cb_index(),
            program,
            core,
            aligned_output_stick_nbytes,
            aligned_output_num_pages,
            out_df,
            output.buffer());
    } else {
        // Share buffer if same data format
        if (interm0_df == out_df) {
            cb_indices.matmul_partials_cb = cb_indices.get_next_cb_index();
            cb_indices.out0_cb = cb_indices.get_next_cb_index();
            auto cb_tuple = tt::tt_metal::create_cb(
                {cb_indices.matmul_partials_cb, cb_indices.out0_cb},
                program,
                core,
                out_tile_size,
                num_output_tiles,
                out_df,
                output.is_sharded() ? output.buffer() : nullptr);

            cb_output = cb_matmul_partials = std::get<1>(cb_tuple);

            if (!output.is_sharded()) {
                log_debug(
                    LogOp,
                    "Matmul Partials CB: {}, npages: {}, pagesize: {}",
                    cb_indices.matmul_partials_cb,
                    num_output_tiles,
                    out_tile_size);
            }
        } else {
            // Separate buffer if not same data format
            std::tie(cb_indices.matmul_partials_cb, cb_matmul_partials) = tt::tt_metal::create_cb(
                cb_indices.get_next_cb_index(), program, core, interm0_single_tile_size, num_output_tiles, interm0_df);
            log_debug(
                LogOp,
                "Matmul Partials CB: {}, npages: {}, pagesize: {}",
                cb_indices.matmul_partials_cb,
                num_output_tiles,
                interm0_single_tile_size);

            std::tie(cb_indices.out0_cb, cb_output) = tt::tt_metal::create_cb(
                cb_indices.get_next_cb_index(),
                program,
                core,
                out_tile_size,
                num_output_tiles,
                out_df,
                output.is_sharded() ? output.buffer() : nullptr);
        }
    }

    if (with_bias) {
        uint32_t bias_tile_size = tt_metal::detail::TileSize(bias_df);
        // bias input
        uint32_t bias_pagesize = bias_tile_size;
        cb_indices.bias_cb = cb_indices.get_next_cb_index();
        tt::tt_metal::create_cb(cb_indices.bias_cb, program, core, bias_pagesize, bias_ntiles, bias_df);
        log_debug(LogOp, "Bias CB: {}, npages: {}, pagesize: {}", cb_indices.bias_cb, bias_ntiles, bias_pagesize);
    }

    return {cb_sharded_act, cb_output, cb_matmul_partials};
}

// TODO: Add namespace for utilities?
std::tuple<tt::tt_metal::CBHandle, tt::tt_metal::CBHandle, tt::tt_metal::CBHandle>
create_CBs_for_depthwise_sharded_input(
    tt_metal::Program& program,
    const Tensor& input,
    CoreRange core,
    uint32_t num_cb0_tiles,
    uint32_t num_cb1_tiles,
    uint32_t num_cb0_tilized_tiles,
    uint32_t num_output_tiles,
    uint32_t num_reblock_cb_tiles,
    uint32_t num_writer_output_tiles,
    bool untilize_out,
    tt::DataFormat act_df,
    tt::DataFormat weight_df,
    tt::DataFormat tilized_act_df,
    tt::DataFormat out_df,
    tt::DataFormat bias_df,
    bool weight_width_sliced,
    const Tensor& output,
    uint32_t bias_ntiles,
    bool with_bias,
    bool split_reader,
    bool fp32_dest_acc_en,
    bool packer_l1_acc_en,
    CBIndices& cb_indices) {
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;

    tt::DataFormat interm0_df =
        packer_l1_acc_en ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b) : out_df;

    uint32_t act_tile_size = tt_metal::detail::TileSize(act_df);
    uint32_t weight_tile_size = tt_metal::detail::TileSize(weight_df);
    uint32_t tilized_act_tile_size = tt_metal::detail::TileSize(tilized_act_df);
    uint32_t out_tile_size = tt_metal::detail::TileSize(out_df);
    uint32_t interm0_single_tile_size = tt_metal::detail::TileSize(interm0_df);

    CBHandle cb_sharded_act = 0;
    if (input.memory_config().is_sharded()) {
        uint32_t num_bytes_for_df = datum_size(act_df);
        auto shard_shape = input.shard_spec().value().shape;
        // 2D-sys-conv already has uint16_t indicies, TODO: do the same for 1D-sys-conv
        TT_FATAL(
            shard_shape[0] <= (1 << 16), "Shard height must be less than 2^16, read pattern indicies are uint16_t");
        // incoming data is the input cb instead of raw l1/dram addr
        std::tie(cb_indices.sharded_act_cb, cb_sharded_act) = tt::tt_metal::create_cb(
            cb_indices.get_next_cb_index(),
            program,
            core,
            shard_shape[1] * num_bytes_for_df,
            shard_shape[0],
            act_df,
            input.buffer());

        // For 1D convs, locally create act matrix in act_cb, which is always ROW_MAJOR BFLOAT16
        // Then, tilize input in compute
        cb_indices.act_cb = cb_indices.get_next_cb_index();
        tt::tt_metal::create_cb(cb_indices.act_cb, program, core, act_tile_size, num_cb0_tiles, act_df);
        log_debug(LogOp, "Act CB: {}, npages: {}, pagesize: {}", cb_indices.act_cb, num_cb0_tiles, act_tile_size);

    } else {
        TT_THROW("Input must be sharded!");
    }

    tt::tt_metal::create_cb(cb_indices.weight_cb, program, core, weight_tile_size, num_cb1_tiles, weight_df);
    log_debug(LogOp, "Weight CB: {}, npages: {}, pagesize: {}", cb_indices.weight_cb, num_cb1_tiles, weight_tile_size);

    // Used for placing tilized activations
    tt::tt_metal::create_cb(
        cb_indices.tilize_mode_tilized_act_cb,
        program,
        core,
        tilized_act_tile_size,
        num_cb0_tilized_tiles,
        tilized_act_df);
    log_debug(
        LogOp,
        "Act Tilized CB: {}, npages: {}, pagesize: {}",
        cb_indices.tilize_mode_tilized_act_cb,
        num_cb0_tilized_tiles,
        tilized_act_tile_size);

    CBHandle cb_output = 0;
    // Share buffer if same data format
    CoreRangeSet cores(std::set<CoreRange>({core}));

    // breakdown above as separate CBs
    CBHandle cb_matmul_partials = 0;

    std::tie(cb_indices.matmul_partials_cb, cb_matmul_partials) =
        tt::tt_metal::create_cb(cb_indices.get_next_cb_index(), program, core, out_tile_size, 1, out_df);
    log_debug(
        LogOp, "Matmul Partials CB: {}, npages: {}, pagesize: {}", cb_indices.matmul_partials_cb, 1, out_tile_size);

    cb_indices.temp_sum_cb = cb_indices.get_next_cb_index();
    tt::tt_metal::create_cb(cb_indices.temp_sum_cb, program, core, out_tile_size, 1, out_df);
    log_debug(LogOp, "Temp Sum CB: {}, npages: {}, pagesize: {}", cb_indices.temp_sum_cb, 1, out_tile_size);

    std::tie(cb_indices.out0_cb, cb_output) = tt::tt_metal::create_cb(
        cb_indices.get_next_cb_index(),
        program,
        cores,
        out_tile_size,
        num_output_tiles,
        out_df,
        output.is_sharded() ? output.buffer() : nullptr);

    if (!output.is_sharded()) {
        log_debug(
            LogOp, "Output CB: {}, npages: {}, pagesize: {}", cb_indices.out0_cb, num_output_tiles, out_tile_size);
    }

    return {cb_sharded_act, cb_output, cb_matmul_partials};
}

tt::tt_metal::operation::ProgramWithCallbacks multi_core_optimized_conv_sharded_v2_impl(
    tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const ttnn::Shape& ashape,
    std::optional<const Tensor> bias,
    const std::optional<const Tensor>& conv_reader_indices,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
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
    using tt::tt_metal::CBHandle;
    using tt::tt_metal::CircularBuffer;
    using tt::tt_metal::CircularBufferConfig;
    CBIndices cb_indices = CBIndices();
    // Non-optional circular buffer indicies
    cb_indices.weight_cb = cb_indices.get_next_cb_index();
    cb_indices.tilize_mode_tilized_act_cb = cb_indices.get_next_cb_index();

    bool pass = true;
    tt_metal::IDevice* device = a.device();
    TT_FATAL(a.layout() == Layout::ROW_MAJOR, "Conv activation should be in row major layout");
    TT_FATAL(a.memory_config().is_sharded(), "Conv activation must be sharded.");
    TT_FATAL(output_channels <= b.padded_shape()[3], "Invalid weight shape. Incorrect weight tensor.");
    uint32_t act_block_h_ntiles = block_config.act_block_h_ntiles;
    uint32_t act_block_w_ntiles = block_config.act_block_w_ntiles;
    uint32_t weight_block_w_ntiles = parallelization_config.per_core_out_matrix_width_ntile;
    uint32_t out_block_h_ntiles = parallelization_config.per_core_out_matrix_height_ntile;
    uint32_t out_subblock_h_ntiles = block_config.out_subblock_h_ntiles;
    uint32_t out_subblock_w_ntiles = block_config.out_subblock_w_ntiles;

    auto conv_reader_indices_storage = conv_reader_indices.value().device_storage();

    tt::DataFormat act_df = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat weight_df = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output.dtype());
    tt::DataFormat bias_df =
        has_bias ? tt_metal::datatype_to_dataformat_converter(bias.value().dtype()) : tt::DataFormat::Float16_b;
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
            act_block_h_ntiles_padded = out_subblock_h_ntiles_padded * num_subblock_h;
        }
    }

    // TT_FATAL(out_block_h_ntiles == act_block_h_ntiles); // TODO: fix output block sizing
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
    uint32_t weight_matrix_height_ntiles = weight_matrix_height / TILE_HEIGHT;
    uint32_t weight_matrix_width_ntiles = weight_matrix_width / TILE_WIDTH;

    // Partitions conv inner dim into blocks to support sharding along this dim
    // TODO: Only 2D convs with sharded input use this, but we can uplift to support generically
    // TODO: Only updated variables which is affected, but there may be more that needs to account for this
    // TODO: Loop naming in reader, writer, and compute kernels could also be cleaned up
    // TODO: Can conv_act_c_blocks be same as num_blocks_act_w?
    auto a_shard_spec = a.shard_spec().value();
    auto shard_shape = a.shard_spec().value().shape;

    // parallelization config
    const auto& p_config = parallelization_config;
    uint32_t num_cores_x = p_config.grid_size.x;
    uint32_t num_cores_y = p_config.grid_size.y;
    uint32_t total_num_cores = num_cores_x * num_cores_y;

    uint32_t per_core_out_matrix_width_ntiles = parallelization_config.per_core_out_matrix_width_ntile;
    uint32_t per_core_out_matrix_height_ntiles = parallelization_config.per_core_out_matrix_height_ntile;
    const bool block_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    const bool height_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;

    // weight_width_sliced determines is 1d-sysarr-conv or 2d-sysarr-conv
    bool weight_width_sliced = per_core_out_matrix_width_ntiles < weight_matrix_width_ntiles;
    uint32_t conv_act_c_blocks = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    if (height_sharded) {
        TT_FATAL(conv_act_c_blocks == 1, "conv_act_c_blocks == 1 in HS, got {}", conv_act_c_blocks);
    }
    uint32_t input_channels_padded = 0;
    if (weight_width_sliced) {
        conv_act_c_blocks = a_shard_spec.orientation == ShardOrientation::ROW_MAJOR ? num_cores_x : num_cores_y;
        if (transpose_mcast) {
            TT_FATAL(conv_act_c_blocks == num_cores_y, "Expected conv_act_c_blocks to be equal to height of grid");
            input_channels_padded = shard_shape[1] * num_cores_y;
        } else {
            TT_FATAL(conv_act_c_blocks == num_cores_x, "Expected conv_act_c_blocks to be equal to width of grid");
            input_channels_padded = shard_shape[1] * num_cores_x;
        }
    } else {
        input_channels_padded = shard_shape[1];
    }
    ttnn::Shape ashape_with_channels_padded({ashape[0], ashape[1], ashape[2], input_channels_padded});
    uint32_t conv_act_size_h = ashape_with_channels_padded[1];
    uint32_t conv_act_size_w = ashape_with_channels_padded[2];
    uint32_t conv_act_size_c = ashape_with_channels_padded[3];
    uint32_t filter_h = (uint32_t)sliding_window_config.window_hw.first;   // filter_h
    uint32_t filter_w = (uint32_t)sliding_window_config.window_hw.second;  // filter_W
    uint32_t stride_h = (uint32_t)sliding_window_config.stride_hw.first;
    uint32_t stride_w = (uint32_t)sliding_window_config.stride_hw.second;
    uint32_t pad_h = (uint32_t)sliding_window_config.get_pad_h();
    uint32_t pad_w = (uint32_t)sliding_window_config.get_pad_w();
    uint32_t dilation_h = (uint32_t)sliding_window_config.dilation_hw.first;
    uint32_t dilation_w = (uint32_t)sliding_window_config.dilation_hw.second;

    if (sliding_window_config.is_transpose) {
        auto input_shape = sliding_window_config.get_transposed_full_input_shape();
        conv_act_size_h = input_shape[1];
        conv_act_size_w = input_shape[2];
        pad_h = 0;
        pad_w = 0;
    }

    uint32_t input_width = ashape[2];
    uint32_t input_channels = ashape[3];
    const bool is_conv1d = is_1d_conv(filter_w, input_width);
    const bool is_conv_1d_depthwise_conv =
        is_1d_deptwise_conv(groups, input_channels, output_channels, filter_w, input_width, has_bias);
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
            round_up((input_channels_padded / conv_act_c_blocks) * filter_w * filter_h, TILE_WIDTH) * conv_act_c_blocks;
    }
    uint32_t act_matrix_height_unpadded = (uint32_t)act_matrix_shape_unpadded[1];
    uint32_t act_matrix_width_unpadded = (uint32_t)act_matrix_shape_unpadded[2];

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
    if (!is_conv_1d_depthwise_conv) {
        TT_FATAL(
            act_matrix_width == weight_matrix_height,
            "The width of tensor a {} needs to match the height of tensor b {}",
            act_matrix_width,
            weight_matrix_height);
    }
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

    uint32_t num_blocks_act_h = act_matrix_height_ntiles / act_block_h_ntiles;
    uint32_t num_blocks_out_h = act_matrix_height_ntiles / out_block_h_ntiles;
    uint32_t num_blocks_act_w = a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ? 1 : filter_h;
    uint32_t num_blocks_weight_w = weight_matrix_width_ntiles / weight_block_w_ntiles;

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
    uint32_t act_block_h_datums_split = act_block_h_nsubblocks_split * out_subblock_h_ntiles * TILE_HEIGHT;
    uint32_t act_block_h_datums_split_last = act_block_h_nsubblocks_split_last * out_subblock_h_ntiles * TILE_HEIGHT;

    uint32_t act_block_num_tiles_split = act_block_h_nsubblocks_split * out_subblock_h_ntiles * act_block_w_ntiles;
    uint32_t act_block_num_tiles_split_last =
        act_block_h_nsubblocks_split_last * out_subblock_h_ntiles * act_block_w_ntiles;

    log_debug(LogOp, "act_block_h_nsubblocks_split: {}", act_block_h_nsubblocks_split);
    log_debug(LogOp, "act_block_h_nsubblocks_split_last: {}", act_block_h_nsubblocks_split_last);
    log_debug(LogOp, "act_block_h_datums_split: {}", act_block_h_datums_split);
    log_debug(LogOp, "act_block_h_datums_split_last: {}", act_block_h_datums_split_last);
    log_debug(LogOp, "act_block_num_tiles_split: {}", act_block_num_tiles_split);
    log_debug(LogOp, "act_block_num_tiles_split_last: {}", act_block_num_tiles_split_last);

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

    uint32_t num_groups = num_blocks_act_h * num_blocks_act_w * num_blocks_weight_w;
    // writer of conv op partially removes padding on the width
    // it removes the padding done for block width but it doesn't remove padding done for tiled width
    uint32_t output_channels_padded_to_tile_width = round_up(output_channels, TILE_WIDTH);
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
        last_block_width_datums % TILE_WIDTH == 0,
        "last_block_width_datums {} should be divisible by TILE_WIDTH",
        last_block_width_datums);

    uint32_t out_block_h_datums = out_block_h_ntiles * TILE_HEIGHT;

    TT_FATAL(output.is_sharded(), "Output buffer must be sharded!");

    // out
    uint32_t out_subblock_num_tiles = out_subblock_h_ntiles * out_subblock_w_ntiles;
    TT_FATAL(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    TT_FATAL(
        act_block_h_ntiles % out_subblock_h_ntiles == 0,
        "act_block_h_ntiles {} should be divisible by out_subblock_h_ntiles {}",
        act_block_h_ntiles,
        out_subblock_h_ntiles);
    // TT_FATAL(out_block_h_ntiles % out_subblock_h_ntiles == 0);
    uint32_t act_num_subblocks = act_block_h_ntiles / out_subblock_h_ntiles;
    uint32_t act_block_num_tiles = act_block_h_ntiles * act_block_w_ntiles;
    uint32_t act_subblock_h_ntiles = out_subblock_h_ntiles;
    uint32_t act_subblock_num_tiles = act_subblock_h_ntiles * act_block_w_ntiles;

    // weight
    const uint32_t weight_dram_addr = b.buffer()->address();

    // bias
    tt_metal::Buffer* bias_buffer = nullptr;
    uint32_t bias_dram_addr = 0;
    uint32_t bias_ntiles = 0;
    if (has_bias) {
        bias_buffer = bias.value().buffer();
        bias_dram_addr = bias_buffer->address();
        bias_ntiles = bias.value().padded_shape()[3] / constants::TILE_WIDTH;  // TODO: support non tile multiple sizes
    }

    std::map<string, string> reader_defines;

    if (act_matrix_height_unpadded < act_matrix_height) {
        reader_defines["ACT_BLOCK_HEIGHT_PADDING"] = "1";
    }

    if (conv_act_c_blocks > 1) {
        reader_defines["ACT_W_OUTER_BLOCKS"] = "1";
    }

    uint32_t output_height_padded_to_tile_height = round_up(act_matrix_height_unpadded, TILE_HEIGHT);
    uint32_t output_height_num_tiles = output_height_padded_to_tile_height / TILE_HEIGHT;
    TT_FATAL(
        output_height_num_tiles <= act_matrix_height_ntiles,
        "output_height_num_tiles {} should be less than or equal to act_matrix_height_ntiles {}",
        output_height_num_tiles,
        act_matrix_height_ntiles);

    uint32_t dst_l1_act_buffer_size_bytes =
        out_block_h_ntiles * act_block_w_ntiles * tt::tt_metal::detail::TileSize(act_df);
    uint32_t dst_l1_weight_buffer_size_bytes =
        weight_block_h_ntiles * weight_block_w_ntiles * tt::tt_metal::detail::TileSize(weight_df);

    // log info for debugging opts
    {
        log_debug(LogOp, "grid_size: {}", p_config.grid_size);
        log_debug(LogOp, "packer_l1: {}", packer_l1_acc);
        log_debug(LogOp, "enable_split_reader: {}", enable_split_reader);
        log_debug(LogOp, "enable_act_double_buffer: {}", enable_act_double_buffer);
        log_debug(LogOp, "enable block padding: {}", (per_core_out_matrix_height_ntiles % act_block_h_ntiles != 0));
        log_debug(LogOp, "enable subblock padding: {}", enable_subblock_padding);
        log_debug(LogOp, "per_core_out_matrix_height_ntiles: {}", per_core_out_matrix_height_ntiles);
        log_debug(LogOp, "act_block_h_ntiles_padded: {}", act_block_h_ntiles_padded);
        log_debug(LogOp, "act_block_w_ntiles: {}", act_block_w_ntiles);
        log_debug(LogOp, "weight_block_w_ntiles: {}", weight_block_w_ntiles);
        log_debug(LogOp, "out_subblock_h_ntiles_padded: {}", out_subblock_h_ntiles_padded);
        log_debug(LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
        log_debug(LogOp, "filter_h: {}", filter_h);
        log_debug(LogOp, "filter_w: {}", filter_w);
        log_debug(LogOp, "dilation_h: {}", dilation_h);
        log_debug(LogOp, "dilation_w: {}", dilation_w);
        log_debug(LogOp, "stride_h: {}", stride_h);
        log_debug(LogOp, "stride_w: {}", stride_w);
        log_debug(LogOp, "pad_h: {}", pad_h);
        log_debug(LogOp, "pad_w: {}", pad_w);
    }

    // For debug
    {
        log_debug(LogOp, "multi_core_optimized_conv_sharded_v2_");
        log_debug(LogOp, "conv_act_size_h: {}", conv_act_size_h);
        log_debug(LogOp, "conv_act_size_w: {}", conv_act_size_w);
        log_debug(LogOp, "conv_act_c_blocks: {}", conv_act_c_blocks);
        log_debug(LogOp, "act_matrix_height: {}", act_matrix_height);
        log_debug(LogOp, "act_matrix_width: {}", act_matrix_width);
        log_debug(LogOp, "act_matrix_height_unpadded: {}", act_matrix_height_unpadded);
        log_debug(LogOp, "act_matrix_width_unpadded: {}", act_matrix_width_unpadded);
        log_debug(LogOp, "act_matrix_height_ntiles: {}", act_matrix_height_ntiles);
        log_debug(LogOp, "act_matrix_width_ntiles: {}", act_matrix_width_ntiles);
        log_debug(LogOp, "weight_matrix_width_ntiles: {}", weight_matrix_width_ntiles);
        log_debug(LogOp, "per_core_out_matrix_height_ntiles: {}", per_core_out_matrix_height_ntiles);
        log_debug(LogOp, "per_core_out_matrix_width_ntiles: {}", per_core_out_matrix_width_ntiles);
        log_debug(LogOp, "num_blocks_act_h: {}", num_blocks_act_h);
        log_debug(LogOp, "num_blocks_act_w: {}", num_blocks_act_w);
        log_debug(LogOp, "num_blocks_weight_w: {}", num_blocks_weight_w);
        log_debug(LogOp, "num_blocks_out_h: {}", num_blocks_out_h);
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
        log_debug(LogOp, "has_bias: {}", has_bias);
        log_debug(LogOp, "bias_dram_addr: {}", bias_dram_addr);
        log_debug(LogOp, "bias_ntiles: {}", bias_ntiles);
        log_debug(LogOp, "out_subblock_h_ntiles: {}", out_subblock_h_ntiles);
        log_debug(LogOp, "out_subblock_w_ntiles: {}", out_subblock_w_ntiles);
        log_debug(LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(LogOp, "num_groups: {}", num_groups);
        log_debug(LogOp, "math_fidelity: {}", math_fidelity);
        log_debug(LogOp, "math_approx_mode: {}", math_approx_mode);
        log_debug(LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);
        log_debug(LogOp, "packer_l1_acc: {}", packer_l1_acc);
    }

    uint32_t window_outer;
    uint32_t window_inner;

    if (weight_width_sliced) {
        window_outer = 1;
        window_inner = filter_h;
    } else {
        window_outer = num_blocks_act_w;
        window_inner = filter_h * filter_w / num_blocks_act_w;
    }

    reader_defines["WINDOW_INNER"] = std::to_string(window_inner);
    log_debug(LogOp, "window_outer: {}, window_inner: {}", window_outer, window_inner);

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
    if (not weight_width_sliced) {
        TT_FATAL(
            num_blocks_weight_w_per_core == num_blocks_weight_w,
            "num_blocks_weight_w_per_core {} should be equal to num_blocks_weight_w {}",
            num_blocks_weight_w_per_core,
            num_blocks_weight_w);
    }
    uint32_t num_weight_slices_width = weight_matrix_width_ntiles / per_core_out_matrix_width_ntiles;
    uint32_t total_num_cores_per_weight_slice = 0;
    uint32_t total_num_cores_per_act_slice = 0;  // only used when (BLOCK_SHARDING && !transpose_mcast)
    if (weight_width_sliced) {
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
            log_debug(LogOp, "total_num_cores_per_act_slice: {}", total_num_cores_per_act_slice);
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
        (per_core_out_matrix_height_ntiles - (num_blocks_act_h_per_core - 1) * act_block_h_ntiles) * TILE_HEIGHT;

    log_debug(LogOp, "total_num_cores_per_weight_slice: {}", total_num_cores_per_weight_slice);
    log_debug(LogOp, "num_blocks_act_h_per_core: {}", num_blocks_act_h_per_core);
    log_debug(LogOp, "num_blocks_out_h_per_core: {}", num_blocks_out_h_per_core);

    TT_FATAL(act_matrix_height_ntiles % per_core_out_matrix_height_ntiles == 0, "Error");
    uint32_t total_active_num_cores_per_weight_slice = act_matrix_height_ntiles / per_core_out_matrix_height_ntiles;
    TT_FATAL(total_active_num_cores_per_weight_slice <= total_num_cores_per_weight_slice, "Error");
    uint32_t total_noop_cores = total_num_cores_per_weight_slice - total_active_num_cores_per_weight_slice;
    uint32_t total_active_num_cores = total_active_num_cores_per_weight_slice * num_weight_slices_width;
    if (weight_width_sliced) {
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
    if (weight_width_sliced) {
        // 2D mcast
        if (transpose_mcast) {
            mcast_sender_cores = CoreRange(top_left_core, CoreCoord(0, num_cores_y - 1));
            mcast_receiver_cores = CoreRange(CoreCoord(1, 0), bottom_right_core);
        } else {
            mcast_sender_cores = CoreRange(top_left_core, CoreCoord(num_cores_x - 1, 0));
            mcast_receiver_cores = CoreRange(CoreCoord(0, 1), bottom_right_core);
        }
        weights_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
        weights_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
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
            weights_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
            weights_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
        }
    }

    bool read_window_in_inner_loop = false;
    uint32_t num_weight_cb_tiles = weight_block_h_ntiles * weight_block_w_ntiles / conv_act_c_blocks;
    bool fully_buffer_weights = false;
    uint32_t num_act_cb_tiles = act_block_h_ntiles * act_block_w_ntiles / conv_act_c_blocks;

    if (block_sharded) {
        num_act_cb_tiles = act_block_h_ntiles * act_block_w_ntiles;
        num_weight_cb_tiles = weight_block_h_ntiles * weight_block_w_ntiles;
    }
    uint32_t num_act_cb_second_reader_tiles = 0;
    // TODO: This flag should be set in kernel logic but need this for create_CB
    if (weight_width_sliced) {
        // If conv_act_c_blocks > 1 and we have 2D conv with sharded input, we always read entire filter_h x filter_w
        // window before pushing in reader/writer
        // TODO: Generalize this to not make this assumption
        read_window_in_inner_loop = true;
        if (!block_sharded) {
            num_weight_cb_tiles *= filter_h * filter_w;
            num_act_cb_tiles *= filter_h * filter_w;
        }
    } else if (num_blocks_act_h_per_core > 1) {
        fully_buffer_weights = true;
    }
    uint32_t num_cb0_tilized_tiles = num_act_cb_tiles;

    if (fully_buffer_weights) {
        num_weight_cb_tiles *= window_outer;
    } else if (enable_weights_double_buffer) {
        num_weight_cb_tiles = num_weight_cb_tiles * 2;
    }

    if (enable_split_reader) {
        if (enable_act_double_buffer) {
            num_act_cb_tiles = act_block_num_tiles_split;
            num_act_cb_second_reader_tiles = act_block_num_tiles_split_last;
            num_act_cb_tiles = num_act_cb_tiles * 2;                              // double buffered
            num_act_cb_second_reader_tiles = num_act_cb_second_reader_tiles * 2;  // double buffered
        } else {
            num_act_cb_tiles = act_block_num_tiles_split;
            num_act_cb_second_reader_tiles = act_block_num_tiles_split_last;
        }
    } else {
        if (enable_act_double_buffer) {
            num_act_cb_tiles = num_act_cb_tiles * 2;
        }
    }
    uint32_t out_block_h_ntiles_padded = num_blocks_act_h_per_core * act_block_h_ntiles;
    uint32_t writer_output_block_num_tiles = out_block_h_ntiles_padded * weight_block_w_ntiles;
    uint32_t output_block_num_tiles =
        enable_subblock_padding ? (act_block_h_ntiles_padded * weight_block_w_ntiles) : writer_output_block_num_tiles;

    uint32_t aligned_output_num_pages = writer_output_block_num_tiles;

    std::vector<uint32_t> reader_rt_args;
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_compile_time_args;

    uint32_t conv_act_c_read_bytes = conv_act_size_c * a.element_size() / conv_act_c_blocks;
    uint32_t act_block_w_extra_align_bytes =
        block_sharded ? (round_up(a_shard_spec.shape[1] * filter_h * filter_w, TILE_WIDTH) -
                         (a_shard_spec.shape[1] * filter_h * filter_w)) *
                            a.element_size()
                      : (round_up(a_shard_spec.shape[1] * filter_w, TILE_WIDTH) - (a_shard_spec.shape[1] * filter_w)) *
                            a.element_size();
    const uint32_t act_block_w_extra_align_scalars = act_block_w_extra_align_bytes / a.element_size();
    // When using block float format, we must handle cases where the data doesn't align to 16-scalar boundaries.
    // If act_block_w_extra_align_bytes contains a number of scalars that isn't a multiple of 16,
    // we need to zero out the temporary circular buffers used during the tiling process.
    // Failing to do this could allow residual junk data in L1 memory to corrupt valid input data.
    const bool needs_act_block_zero_out =
        act_block_w_extra_align_scalars % 16 != 0 && tt_metal::is_block_float(output.dtype());

    uint32_t in0_block_w = act_block_w_ntiles / conv_act_c_blocks;
    uint32_t in0_block_num_tiles = act_block_num_tiles / conv_act_c_blocks;
    uint32_t in1_block_num_tiles = weight_block_num_tiles / conv_act_c_blocks;
    uint32_t in0_num_blocks_w =
        num_blocks_act_w * conv_act_c_blocks;  // Fold outer c_block loop together with weight_block_num_tiles = 9

    uint32_t tilized_act_tile_size = tt_metal::detail::TileSize(tilized_act_df);

    // Only enable packer l1 accumulation when there are in0_num_blocks_w > 2, otherwise
    // unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
    // does a spill and reload, so need more than 2 blocks to use l1 acc for packer
    // For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
    bool packer_l1_acc_en = determine_packer_l1_acc(packer_l1_acc, has_bias, in0_num_blocks_w);

    std::tuple<CBHandle, CBHandle, CBHandle> input_output_cbs = {0, 0, 0};
    if (is_conv_1d_depthwise_conv) {
        input_output_cbs = create_CBs_for_depthwise_sharded_input(
            program,
            a,
            all_cores,
            num_act_cb_tiles,               // row major act cb
            num_weight_cb_tiles,            // tiled weight cb
            num_cb0_tilized_tiles,          // tiled act cb
            writer_output_block_num_tiles,  // math output cb
            weight_block_w_ntiles,          // reblock cb
            writer_output_block_num_tiles,  // writer output cb, double bufferred
            untilize_out,
            act_df,
            weight_df,
            tilized_act_df,
            out_df,
            bias_df,
            weight_width_sliced,
            output,
            bias_ntiles_per_core,
            has_bias,
            enable_split_reader,
            fp32_dest_acc_en,
            packer_l1_acc_en,
            cb_indices);
    } else {
        // TODO: Moving this function call to after kernel logic causes pcc fails
        // There are additional CBs and semaphores created in 2D conv in kernel logic,
        // so does order of create_cb calls matter?
        input_output_cbs = create_CBs_for_sharded_input_v2(
            program,
            a,
            all_cores,
            num_act_cb_tiles,                // row major act cb
            num_act_cb_second_reader_tiles,  // row major act cb second reader
            num_weight_cb_tiles,             // tiled weight cb
            num_cb0_tilized_tiles,           // tiled act cb
            output_block_num_tiles,          // math output cb
            weight_block_w_ntiles,           // reblock cb
            writer_output_block_num_tiles,   // writer output cb, double bufferred
            untilize_out,
            act_df,
            weight_df,
            tilized_act_df,
            out_df,
            bias_df,
            weight_width_sliced,
            output,
            bias_ntiles_per_core,
            has_bias,
            enable_split_reader,
            fp32_dest_acc_en,
            packer_l1_acc_en,
            cb_indices);
    }
    CBHandle cb_sharded_act = std::get<0>(input_output_cbs);
    CBHandle cb_output = std::get<1>(input_output_cbs);
    CBHandle cb_matmul_partials = std::get<2>(input_output_cbs);
    CircularBufferConfig cb_config_output = GetCircularBufferConfig(program, cb_output);
    CircularBufferConfig cb_config_matmul_partials = GetCircularBufferConfig(program, cb_matmul_partials);
    bool partials_cb_uses_output = false;
    if (cb_config_matmul_partials.globally_allocated_address().has_value() &&
        cb_config_output.globally_allocated_address().has_value()) {
        partials_cb_uses_output = cb_config_matmul_partials.globally_allocated_address().value() ==
                                  cb_config_output.globally_allocated_address().value();
    }

    string reader_kernel;
    string compute_kernel;
    string writer_mcast_sender_kernel;
    string writer_mcast_receiver_kernel;
    bool tilize_in0 = true;

    compute_kernel = "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_bmm_tilize_col_major_out_blocks.cpp";
    // Input should always be sharded in this conv; always use reader kernel for input shard with halo and padding
    if (filter_h >= 1 and filter_w >= 1) {
        if (!is_conv_1d_depthwise_conv and weight_width_sliced) {
            // Block sharded conv
            TT_FATAL(read_window_in_inner_loop == true, "read_window_in_inner_loop should be true for this conv");
            reader_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp";
            writer_mcast_sender_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp";
            writer_mcast_receiver_kernel =
                "ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/"
                "writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp";
            act_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
            act_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

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

            // For 2D convs, pre-tilize input and round robin self-mcast tilized act matrix to other cores
            tilize_in0 = false;
        } else if (is_conv_1d_depthwise_conv) {
            // 1D Depthwise Conv (height sharded)
            TT_FATAL(act_block_w_datums == round_up(conv_act_size_c * filter_w, TILE_WIDTH), "Error");

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
            TT_FATAL(act_block_w_datums == round_up(conv_act_size_c * filter_w, TILE_WIDTH), "Error");

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
        // Local L1 to store array for reader indices
        // All convs use packed uint16 indices, so each entry can be 2B (not 4)
        CBHandle cb_for_reader_indices_id = 0;
        std::tie(cb_indices.cb_for_reader_indices, cb_for_reader_indices_id) = tt::tt_metal::create_cb(
            cb_indices.get_next_cb_index(),
            program,
            all_cores,
            out_block_h_datums * 2,
            1,
            tt::DataFormat::Float16_b,
            conv_reader_indices_storage.get_buffer());

        // Local L1 to store temp vars
        cb_indices.cb_for_l1_array = cb_indices.get_next_cb_index();
        tt::tt_metal::create_cb(
            cb_indices.cb_for_l1_array, program, all_cores, l1_scratchpad_CB_size, 1, tt::DataFormat::Float16_b);
    } else {
        TT_THROW("Sharded input not supported for this conv yet!");
    }

    if (read_window_in_inner_loop) {
        const uint32_t window_size = filter_h * filter_w;
        in0_block_w *= window_size;
        in0_block_num_tiles *= window_size;
        in1_block_num_tiles *= window_size;
        in0_num_blocks_w /= window_size;
    }
    uint32_t reader_arg_act_block_h_datums = (enable_split_reader ? act_block_h_datums_split : act_block_h_datums);
    TT_FATAL(reader_arg_act_block_h_datums % 2 == 0, "2 Indices are packed in one uint32_t word.");
    if (block_sharded) {
        in0_block_num_tiles = act_block_num_tiles;
        in1_block_num_tiles = weight_block_num_tiles;
        in0_block_w = act_block_w_ntiles;
        in0_num_blocks_w = 1 * conv_act_c_blocks;
    }
    reader_compile_time_args = {
        (uint32_t)dilation_h,
        (uint32_t)dilation_w,
        (uint32_t)conv_act_c_read_bytes,
        (uint32_t)window_outer,
        (uint32_t)window_inner,
        (uint32_t)reader_arg_act_block_h_datums,
        (uint32_t)(enable_split_reader ? act_block_num_tiles_split / conv_act_c_blocks
                                       : act_block_num_tiles / conv_act_c_blocks),
        (uint32_t)filter_h,
        (uint32_t)filter_w,
        (uint32_t)conv_act_size_w + (pad_w),
        (uint32_t)act_block_w_extra_align_bytes,                          // only used for 1d systolic variant
        (uint32_t)num_blocks_act_h_per_core,                              // act_num_blocks_h
        (uint32_t)in0_block_num_tiles,                                    // act_block_num_tiles
        (uint32_t)conv_act_c_blocks,                                      // act_w_num_outer
        (uint32_t)(transpose_mcast ? num_cores_y - 1 : num_cores_x - 1),  // act_mcast_num_dests
        (uint32_t)(transpose_mcast ? num_cores_y - 1 : num_cores_x - 1),  // act_mcast_num_cores
        (uint32_t)act_mcast_sender_semaphore_id,
        (uint32_t)act_mcast_receiver_semaphore_id,
        (uint32_t)in0_block_num_tiles * tilized_act_tile_size,  // act_mcast_sender_size_bytes
        (uint32_t)(transpose_mcast ? 1 : 0),
        (uint32_t)act_block_h_datums_last_block,
        (uint32_t)act_block_h_datums_split_last,
        (uint32_t)needs_act_block_zero_out,  // zero_out_act_cb
        (uint32_t)cb_indices.act_cb,
        (uint32_t)cb_indices.sharded_act_cb,
        (uint32_t)cb_indices.cb_for_reader_indices,
        (uint32_t)cb_indices.tilize_mode_tilized_act_cb,
        (uint32_t)cb_indices.act_cb_row_major_bfloat16,
        (uint32_t)cb_indices.cb_for_l1_array};

    // define for bias
    std::map<string, string> writer_defines;
    std::map<string, string> writer_mcast_sender_defines;
    std::map<string, string> compute_defines;
    if (total_num_cores == 1) {
        writer_mcast_sender_defines["SKIP_MCAST"] = "1";
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

    if (!tilize_in0) {
        compute_defines["PRE_TILIZE"] = "1";
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
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(device->arch(), total_num_cores, compute_defines);

    for (auto elem : compute_defines) {
        log_debug(LogOp, "compute_defines: {} = {}", elem.first, elem.second);
    }

    writer_compile_time_args = {
        cb_indices.weight_cb,
        cb_indices.bias_cb,
        (uint32_t)(bias_buffer == nullptr ? 0 : (bias_buffer->buffer_type() == BufferType::DRAM ? 1 : 0)),
        cb_indices.act_cb_second_reader,
        cb_indices.sharded_act_cb,
        cb_indices.cb_for_reader_indices,
        num_blocks_act_w,
        in1_block_num_tiles,
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
            (uint32_t)act_block_h_datums_split_last,
            (uint32_t)act_block_num_tiles_split_last / conv_act_c_blocks,
            (uint32_t)conv_act_c_read_bytes,
            (uint32_t)filter_w,                       // weight_size_w
            (uint32_t)(conv_act_size_w + pad_w),      // conv_act_size_w_padded
            (uint32_t)act_block_w_extra_align_bytes,  // only used for 1d systolic variant
            (uint32_t)act_block_h_datums_split,       // only used for 1d systolic variant
            (uint32_t)act_block_h_datums_last_block,
            (uint32_t)needs_act_block_zero_out,
            (uint32_t)dilation_h,
            (uint32_t)dilation_w};
        writer_compile_time_args.insert(
            writer_compile_time_args.end(), split_reader_args.begin(), split_reader_args.end());
    } else {
        std::vector<uint32_t> split_reader_args(11, 0);
        writer_compile_time_args.insert(
            writer_compile_time_args.end(), split_reader_args.begin(), split_reader_args.end());
    }

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        act_num_subblocks,
        act_block_num_tiles,
        act_subblock_num_tiles,
        act_subblock_h_ntiles,

        weight_num_subblocks,
        in1_block_num_tiles,
        weight_block_w_ntiles,

        num_blocks_act_h_per_core,
        in0_num_blocks_w,
        num_blocks_weight_w_per_core,

        out_subblock_h_ntiles_padded,
        out_subblock_w_ntiles,
        out_subblock_num_tiles,

        tilize_in0,
        untilize_out,

        bias_ntiles_per_core,

        cb_indices.bias_cb,
        cb_indices.act_cb,
        cb_indices.weight_cb,
        cb_indices.act_cb_row_major_bfloat16,
        cb_indices.act_cb_second_reader,
        cb_indices.matmul_partials_cb,
        cb_indices.tilize_mode_tilized_act_cb,

        cb_indices.out0_cb,
        cb_indices.temp_sum_cb,
        partials_cb_uses_output};

    const tt::tt_metal::NOC writer_mcast_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());
    ;
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
        if (weight_width_sliced && transpose_mcast || !weight_width_sliced) {
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

        if (weight_width_sliced) {
            auto shard_shape = a.shard_spec().value().shape;
            uint32_t tilized_act_tile_size = tt_metal::detail::TileSize(tilized_act_df);

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

        if (weight_width_sliced) {
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

            std::optional<tt_metal::Buffer*> src_buffer_c = std::nullopt;
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
    tt_metal::Program program = tt_metal::CreateProgram();

    ttnn::operations::sliding_window::ParallelConfig parallel_config;
    parallel_config.grid = a.shard_spec().value().grid;
    parallel_config.shard_scheme = a.memory_config().memory_layout();
    parallel_config.shard_orientation = a.shard_spec().value().orientation;

    // create conv config tensors
    auto pad_metadata = ttnn::operations::sliding_window::generate_pad_metadata(sliding_window_config);
    auto op_trace_metadata = ttnn::operations::sliding_window::generate_op_trace_metadata(sliding_window_config);
    auto shard_boundaries =
        ttnn::operations::sliding_window::generate_shard_boundaries(sliding_window_config, op_trace_metadata);
    auto conv_sharded_input_top_left_indices = ttnn::operations::sliding_window::generate_sliding_window_op_config(
        op_trace_metadata, shard_boundaries, true, true);
    // create sharded ttnn config tensors
    DataType indices_tt_dtype = DataType::UINT16;
    // For 2d convs, each core in a column or row share the same specs
    CoreCoord grid_size = parallel_config.grid.bounding_box().grid_size();

    bool is_block_sharded = a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
    auto conv_reader_indices_tensor = ttnn::operations::sliding_window::construct_on_host_config_tensor(
        conv_sharded_input_top_left_indices, parallel_config);
    conv_reader_indices_tensor = ttnn::operations::sliding_window::move_config_tensor_to_device(
        conv_reader_indices_tensor, parallel_config, is_block_sharded, a.device());

    if (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        return multi_core_optimized_conv_width_sharded_v2_impl(
            program,
            a,
            b,
            ttnn::Shape(input_tensor_shape),
            bias,
            conv_reader_indices_tensor,
            sliding_window_config,
            output_channels,
            groups,
            untilize_out,
            bias.has_value(),
            fused_activation,
            parallelization_config,
            block_config,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            output,
            compute_kernel_config.value(),
            enable_act_double_buffer,
            enable_weights_double_buffer,
            enable_split_reader,
            enable_subblock_padding);
    }
    return multi_core_optimized_conv_sharded_v2_impl(
        program,
        a,
        b,
        ttnn::Shape(input_tensor_shape),
        bias,
        conv_reader_indices_tensor,
        sliding_window_config,
        output_channels,
        groups,
        untilize_out,
        bias.has_value(),
        fused_activation,
        parallelization_config,
        block_config,
        parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
        output,
        compute_kernel_config.value(),
        enable_act_double_buffer,
        enable_weights_double_buffer,
        enable_split_reader,
        enable_subblock_padding);
}
}  // namespace conv2d

}  // namespace ttnn::operations::conv
