// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/buffer_types.hpp"

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace ttnn::prim {

namespace reuse_mcast_optimized_helpers {

MatmulMultiCoreReuseMcast2DProgramFactory::cached_program_t create_program_mcast_in0_in1(
    tt::tt_metal::Program& program,
    tt::tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    bool transpose_a,
    bool transpose_b,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t in0_block_w,
    uint32_t in0_last_ktile_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    tt::tt_metal::Buffer* in0_buffer,
    tt::tt_metal::Buffer* in1_buffer,
    tt::tt_metal::Buffer* bias_buffer,
    tt::tt_metal::Buffer* out_buffer,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
    using namespace tt;
    using tt::tt_metal::TensorMemoryLayout;

    // currently only support transpose of the full tile
    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    bool fuse_op = fused_op_signaler.has_value();

    TensorMemoryLayout in0_memory_layout = in0_buffer->buffer_layout();

    uint32_t num_blocks = K / in0_block_w;

    // Only enable packer l1 accumulation when there are num_blocks > 2, otherwise
    // unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
    // does a spill and reload, so need more than 2 blocks to use l1 acc for packer
    // For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
    bool packer_l1_acc_en = packer_l1_acc && (((bias_buffer != nullptr) && num_blocks > 1) || (num_blocks > 2));

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    const bool in0_block_sharded = in0_memory_layout == TensorMemoryLayout::BLOCK_SHARDED;
    const bool in0_height_sharded = in0_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool in0_is_sharded = in0_block_sharded || in0_height_sharded;
    const bool in1_is_sharded = in1_buffer->buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const bool output_is_sharded = out_buffer->buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = out_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles *= operations::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_shard_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    if (output_is_sharded) {
        out_CB_tiles = out_shard_tiles;
    }
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t interm0_CB_size = interm0_CB_tiles * interm0_single_tile_size;

    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    uint32_t in0_shard_height_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_width();
        in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_height();
        in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    }

    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
    uint32_t num_cores_with_work_c = num_blocks_x;
    uint32_t num_cores_with_work_r = num_blocks_y;
    if (transpose_mcast) {
        std::swap(num_cores_with_work_c, num_cores_with_work_r);
    }

    CoreRange all_cores_with_work(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_with_work_c - 1, (std::size_t)start_core_y + num_cores_with_work_r - 1});

    ////////////////////////////////////////////////////////////////////////////
    //                      IN0 SHARDED SENDER/RECEIVER
    ////////////////////////////////////////////////////////////////////////////
    uint32_t num_cores_c = num_cores_with_work_c;
    uint32_t num_cores_r = num_cores_with_work_r;
    uint32_t in0_mcast_receiver_grid_diff_coord_start;
    uint32_t in0_mcast_receiver_grid_diff_coord_end;
    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    uint32_t in0_sender_num_cores_along_width = 0;

    // Only used for in0 block sharded
    std::optional<CoreRange> in0_mcast_cores_without_work_and_not_in_receiver_grid;
    if (in0_block_sharded) {
        CoreCoord in0_shard_grid = in0_buffer->shard_spec().grid().bounding_box().grid_size();
        // in0 shard grid already accounts for transpose_mcast
        // ie. If transpose_mcast, in0 width is along y
        in0_sender_num_cores_along_width = transpose_mcast ? in0_shard_grid.y : in0_shard_grid.x;
        if (in0_sender_num_cores_along_width > num_blocks_x) {
            in0_mcast_cores_without_work_and_not_in_receiver_grid =
                transpose_mcast ? CoreRange(
                                      {(std::size_t)start_core_x, (std::size_t)start_core_y + num_blocks_x},
                                      {(std::size_t)start_core_x + num_blocks_y - 1,
                                       (std::size_t)start_core_y + in0_sender_num_cores_along_width - 1})
                                : CoreRange(
                                      {(std::size_t)start_core_x + num_blocks_x, (std::size_t)start_core_y},
                                      {(std::size_t)start_core_x + in0_sender_num_cores_along_width - 1,
                                       (std::size_t)start_core_y + num_blocks_y - 1});
        }

        if (transpose_mcast) {
            in0_mcast_receiver_grid_diff_coord_start = device->worker_core_from_logical_core({0, start_core_y}).y;
            in0_mcast_receiver_grid_diff_coord_end =
                device->worker_core_from_logical_core({0, start_core_y + num_blocks_x - 1}).y;
            in0_mcast_noc_y.reserve(in0_sender_num_cores_along_width);
            for (uint32_t core_idx_y = 0; core_idx_y < in0_sender_num_cores_along_width; ++core_idx_y) {
                in0_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
            }
        } else {
            in0_mcast_receiver_grid_diff_coord_start = device->worker_core_from_logical_core({start_core_x, 0}).x;
            in0_mcast_receiver_grid_diff_coord_end =
                device->worker_core_from_logical_core({start_core_x + num_blocks_x - 1, 0}).x;
            in0_mcast_noc_x.reserve(in0_sender_num_cores_along_width);
            for (uint32_t core_idx_x = 0; core_idx_x < in0_sender_num_cores_along_width; ++core_idx_x) {
                in0_mcast_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
            }
        }

        // Set in0 sender/receiver cores to be maximum
        if (transpose_mcast) {
            num_cores_r = std::max(num_cores_r, in0_sender_num_cores_along_width);
        } else {
            num_cores_c = std::max(num_cores_c, in0_sender_num_cores_along_width);
        }
    }

    // Used for setting up CBs and semaphores by both in0 interleaved or sharded
    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
    const auto& cores = grid_to_cores(all_cores.start_coord, all_cores.end_coord, true);
    //////////////////////////////////////////////////////////////////////////////////////////
    //       IN0 SENDER (interleaved only) and IN1 SENDER (both interleaved and sharded)
    //////////////////////////////////////////////////////////////////////////////////////////
    // Left column
    CoreRange in0_sender_interleaved(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_with_work_r - 1});

    // Top row
    CoreRange in1_sender(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_with_work_c - 1, (std::size_t)start_core_y});

    // Left column except corner
    std::optional<CoreRange> in0_sender_in1_receiver;
    if (num_cores_with_work_r > 1) {
        in0_sender_in1_receiver = {
            {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
            {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_with_work_r - 1}};
    }

    // Top row except corner
    std::optional<CoreRange> in0_receiver_in1_sender;
    if (num_cores_with_work_c > 1) {
        in0_receiver_in1_sender = {
            {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
            {(std::size_t)start_core_x + num_cores_with_work_c - 1, (std::size_t)start_core_y}};
    }

    if (transpose_mcast) {
        std::swap(in0_sender_interleaved, in1_sender);
        std::swap(in0_sender_in1_receiver, in0_receiver_in1_sender);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    //       IN0 RECEIVER (interleaved only) and IN1 RECEIVER (both interleaved and sharded)
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Not exactly half-half; this seems to get slightly better perf for fused qkv and selfout
    // TODO: Experiment with different splits?
    bool split_half = num_cores_with_work_c > 2 && num_cores_with_work_r > 1 && !in0_is_sharded;
    uint32_t half_core = split_half ? (num_cores_with_work_c) / 2 : num_cores_with_work_c - 1;

    std::optional<CoreRange> in0_receiver_in1_receiver_left_half;
    if (num_cores_with_work_c > 1 and num_cores_with_work_r > 1) {
        in0_receiver_in1_receiver_left_half = {
            {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
            {(std::size_t)start_core_x + half_core, (std::size_t)start_core_y + num_cores_with_work_r - 1}};
    }

    std::set<CoreRange> in0_receiver_interleaved_set;
    std::set<CoreRange> in1_receiver_set;
    if (in0_receiver_in1_sender.has_value()) {
        in0_receiver_interleaved_set.insert(in0_receiver_in1_sender.value());
    }
    if (in0_sender_in1_receiver.has_value()) {
        in1_receiver_set.insert(in0_sender_in1_receiver.value());
    }
    if (in0_receiver_in1_receiver_left_half.has_value()) {
        in0_receiver_interleaved_set.insert(in0_receiver_in1_receiver_left_half.value());
        in1_receiver_set.insert(in0_receiver_in1_receiver_left_half.value());
    }
    CoreRangeSet in0_receiver_interleaved(in0_receiver_interleaved_set);
    CoreRangeSet in1_receiver(in1_receiver_set);

    std::optional<CoreRange> in0_receiver_in1_receiver_interleaved_other_cores;
    if (split_half) {
        in0_receiver_in1_receiver_interleaved_other_cores = {
            {(std::size_t)start_core_x + half_core + 1, (std::size_t)start_core_y + 1},
            {(std::size_t)start_core_x + num_cores_with_work_c - 1,
             (std::size_t)start_core_y + num_cores_with_work_r - 1}};
    }

    // Mcast args
    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    TT_FATAL(
        out_block_h % out_subblock_h == 0 and out_block_h >= out_subblock_h,
        "out_block_h must be multiple of out_subblock_h");
    TT_FATAL(
        out_block_w % out_subblock_w == 0 and out_block_w >= out_subblock_w,
        "out_block_w must be multiple of out_subblock_w");

    std::vector<uint32_t> in0_sender_compile_time_args;

    uint32_t num_dram_banks = 0;
    uint32_t per_core_N_storage = 0;
    if (in1_is_sharded and in1_is_dram) {
        num_dram_banks = device->num_dram_channels();
        per_core_N_storage = (N + num_dram_banks - 1) / num_dram_banks;
    }

    const auto in0_tensor_stride_w = transpose_a ? M : 1;
    const auto in0_tensor_stride_h = transpose_a ? 1 : K;
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;
    const auto in0_tensor_start_tile_id_stride = per_core_M * in0_tensor_stride_h;

    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
    const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;
    const auto in1_tensor_start_tile_id_stride = per_core_N * in1_tensor_stride_w;

    if (in0_block_sharded) {
        uint32_t num_x = in0_sender_num_cores_along_width;
        uint32_t num_y = 1;
        if (transpose_mcast) {
            std::swap(num_x, num_y);
        }

        in0_sender_compile_time_args = {
            (std::uint32_t)1,  // core_has_output_block_work
            (std::uint32_t)1,  // core_in_in0_receiver_mcast_grid

            (std::uint32_t)in0_block_num_tiles,                         // in0_block_num_tiles
            (std::uint32_t)in0_block_num_tiles * in0_single_tile_size,  // in0_block_size_bytes
            (std::uint32_t)in0_last_ktile_w,

            // in0/in1 common args
            (std::uint32_t)num_blocks,  // num_blocks
            (std::uint32_t)out_num_blocks_x,
            (std::uint32_t)out_num_blocks_y,
            // in0 mcast args
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            (std::uint32_t)num_blocks_x,  // in0_mcast_num_dests
            (std::uint32_t)num_blocks_x,  // in0_mcast_num_cores
            (std::uint32_t)num_x,
            (std::uint32_t)num_y,
            (std::uint32_t)transpose_mcast,
            (std::uint32_t)in0_shard_width_in_tiles,
            (std::uint32_t)in0_shard_height_in_tiles,
            (std::uint32_t)in0_block_w,
            (std::uint32_t)in0_block_h,
            // batch args
            (std::uint32_t)B  // batch
        };
    } else {
        in0_sender_compile_time_args = {
            // in0 tensor args
            (std::uint32_t)in0_tensor_stride_w,
            (std::uint32_t)in0_tensor_stride_h,
            (std::uint32_t)in0_tensor_next_block_stride,
            (std::uint32_t)in0_tensor_next_h_dim_block_stride,
            // in0 block args
            (std::uint32_t)in0_block_w,          // in0_block_w
            (std::uint32_t)in0_block_h,          // in0_block_h
            (std::uint32_t)in0_block_num_tiles,  // in0_block_num_tiles
            (std::uint32_t)in0_last_ktile_w,

            (std::uint32_t)false,                      // extract_shard_sub_blocks (not used for interleaved)
            (std::uint32_t)in0_shard_width_in_tiles,   // shard_width_in_tiles (not used for interleaved)
            (std::uint32_t)in0_shard_height_in_tiles,  // shard_height_in_tiles (not used for interleaved)
            // in0/in1 common args
            (std::uint32_t)num_blocks,  // num_blocks
            (std::uint32_t)out_num_blocks_x,
            (std::uint32_t)out_num_blocks_y,
            // in0 mcast args
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            (std::uint32_t)(num_blocks_x - 1),  // in0_mcast_num_dests
            (std::uint32_t)(num_blocks_x - 1),  // in0_mcast_num_cores
            // batch args
            (std::uint32_t)M * K,  // MtKt
            (std::uint32_t)B,      // batch

            // sparsity args
            (std::uint32_t)0,      // batchB
            (std::uint32_t)0,      // sparsity_pagesize (placeholder since sparsity not used in this case)
            (std::uint32_t)true,   // bcast_A
            (std::uint32_t)false,  // get_batch_from_reader
        };
    }
    in0_sender_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_all_gather()));
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        // READER
        // in1 tensor args
        (std::uint32_t)in1_tensor_stride_w,
        (std::uint32_t)in1_tensor_stride_h,
        (std::uint32_t)in1_tensor_next_block_stride,
        (std::uint32_t)in1_tensor_next_w_dim_block_stride,
        // in1 block args
        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,                // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,  // num_blocks
        (std::uint32_t)out_num_blocks_x,
        (std::uint32_t)out_num_blocks_y,
        // in1 mcast args
        (std::uint32_t)in1_mcast_sender_semaphore_id,
        (std::uint32_t)in1_mcast_receiver_semaphore_id,
        (std::uint32_t)(num_blocks_y - 1),  // in1_mcast_num_dests
        (std::uint32_t)(num_blocks_y - 1),  // in1_mcast_num_cores
        // batch args
        (std::uint32_t)K * N,        // KtNt
        (std::uint32_t)B,            // batch
        (std::uint32_t)bcast_batch,  // bcast_B
        // sparsity args
        (std::uint32_t)0,  // batchB
        (std::uint32_t)0,  // sparsity_pagesize (placeholder since sparsity not used in this case)

        // WRITER
        // out tensor args
        (std::uint32_t)1,                   // out_tensor_stride_w
        (std::uint32_t)N,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_block_w,         // out_tensor_next_w_dim_block_stride
        (std::uint32_t)out_block_h * N,     // out_tensor_next_h_dim_block_stride
        // out subblock args
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        // batch args
        (std::uint32_t)M * N  // MtNt
    };
    if (bias_buffer != nullptr) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);  // in3_tensor_stride_w
    } else {
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }

    in1_sender_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_all_gather()));
    in1_sender_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_reduce_scatter()));

    // Append TensorAccessorArgs
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(in1_sender_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in1_sender_writer_compile_time_args);  // placeholder for sparsity
    tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(in1_sender_writer_compile_time_args);
    if (bias_buffer != nullptr) {
        tt::tt_metal::TensorAccessorArgs(*bias_buffer).append_to(in1_sender_writer_compile_time_args);
    }

    if (in1_is_sharded and in1_is_dram) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)per_core_N_storage * in0_block_w);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)per_core_N_storage * in1_single_tile_size);
    }
    std::vector<uint32_t> in0_receiver_compile_time_args = {
        // in0 block args
        (std::uint32_t)in0_block_w * in0_block_h,  // in0_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,  // num_blocks
        (std::uint32_t)out_num_blocks_x,
        (std::uint32_t)out_num_blocks_y,
        // in0 mcast args
        (std::uint32_t)in0_mcast_sender_semaphore_id,
        (std::uint32_t)in0_mcast_receiver_semaphore_id,
        // batch args
        (std::uint32_t)B,     // batch
        (std::uint32_t)false  // get_batch_from_reader
    };
    std::vector<uint32_t> in1_receiver_writer_compile_time_args = {
        // READER
        // in1 block args
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,  // num_blocks
        (std::uint32_t)out_num_blocks_x,
        (std::uint32_t)out_num_blocks_y,
        // in1 mcast args
        (std::uint32_t)in1_mcast_sender_semaphore_id,
        (std::uint32_t)in1_mcast_receiver_semaphore_id,
        // batch args
        (std::uint32_t)B,  // batch

        // WRITER
        // out tensor args
        (std::uint32_t)1,                   // out_tensor_stride_w
        (std::uint32_t)N,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_block_w,         // out_tensor_next_w_dim_block_stride
        (std::uint32_t)out_block_h * N,     // out_tensor_next_h_dim_block_stride
        // out subblock args
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        // batch args
        (std::uint32_t)M * N  // MtNt
    };
    if (bias_buffer != nullptr) {
        in1_receiver_writer_compile_time_args.push_back((std::uint32_t)in1_block_w);
    } else {
        in1_receiver_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }
    in1_receiver_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_reduce_scatter()));
    tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(in1_receiver_writer_compile_time_args);

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_sharded_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_interleaved_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_receiver_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_receiver_writer_other_noc_setup_defines;
    if (bias_buffer != nullptr) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_other_noc_setup_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            using ttnn::operations::unary::utils::get_defines;
            mm_kernel_defines.merge(get_defines(
                fused_activation.value().op_type,
                fused_activation.value().params,
                "ACTIVATION",
                "i",
                tt_metal::dataformat_to_datatype_converter(output_data_format)));
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    if (in1_transpose_tile) {
        mm_kernel_defines["IN1_TRANSPOSE_TILE"] = "1";
    }

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), cores.size(), mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), cores.size(), mm_kernel_defines, throttle_level);

    if (in0_receiver_interleaved.num_cores() == 0) {
        mm_kernel_in0_sender_interleaved_defines["SKIP_MCAST"] = "1";
    }
    if (in0_height_sharded) {
        mm_kernel_in0_sender_interleaved_defines["IN0_SHARDED"] = "1";
    }

    if (in1_receiver.num_cores() == 0) {
        mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";
    }
    if (in1_is_sharded) {
        if (in1_is_dram) {
            mm_kernel_in1_sender_writer_defines["IN1_DRAM_SHARDED"] = "1";
        } else {
            mm_kernel_in1_sender_writer_defines["IN1_SHARDED"] = "1";
        }
    }

    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
        mm_kernel_in1_receiver_writer_defines["OUT_SHARDED"] = "1";
        mm_kernel_in1_receiver_writer_other_noc_setup_defines["OUT_SHARDED"] = "1";
    }

    // Intermediate CB read
    /*
    Blackhole architecture alignment issue workaround for tiny tiles:

    Problem: When reading tiny tiles from DRAM to circular buffers (CB), address alignment
    issues occur. DRAM tile addresses are 64-byte aligned within each block, but L1 CB
    addresses are not necessarily aligned due to non-64-byte-aligned page sizes.

    Example scenario:
    - Two consecutive 544-byte tiles (16x32 tile of dtype bfloat8_b) stored on different DRAM banks
    - CB configured with size=2 to hold both tiles

    Result:
    - Tile 0: DRAM Bank 0, Address 64    → CB L1 Address 0   (64-byte aligned ✓)
    - Tile 1: DRAM Bank 1, Address 64    → CB L1 Address 544 (not 64-byte aligned ✗)

    Solution: Use an intermediate single-tile CB as a staging area. Read each tile into
    the intermediate CB first, then copy to the destination CB. This ensures proper
    alignment at the cost of additional memory bandwidth overhead.

    Note: This workaround should only be used for this specific alignment issue case.
    */
    bool in0_needs_intermediate_cb_read = false;
    bool in1_needs_intermediate_cb_read = false;
    if (device->arch() == tt::ARCH::BLACKHOLE) {
        in0_needs_intermediate_cb_read = ((in0_single_tile_size % 64) != 0);
        if (in0_needs_intermediate_cb_read) {
            mm_kernel_in0_sender_interleaved_defines["INTERMEDIATE_CB_READ"] = "1";
        }
        in1_needs_intermediate_cb_read = ((in1_single_tile_size % 64) != 0);
        if (in1_needs_intermediate_cb_read) {
            mm_kernel_in1_sender_writer_defines["INTERMEDIATE_CB_READ"] = "1";
        }
    }

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    tt_metal::NOC in0_split_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    tt_metal::NOC in1_split_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    tt::tt_metal::KernelHandle mm_kernel_in0_sender_id = 0;
    tt::tt_metal::KernelHandle mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id = 0;
    if (in0_block_sharded) {
        mm_kernel_in0_sender_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",
            all_cores_with_work,  // in0_mcast_cores_with_work_and_in_receiver_grid
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_sender_compile_time_args,
                .defines = mm_kernel_in0_sender_sharded_defines});
        if (in0_mcast_cores_without_work_and_not_in_receiver_grid.has_value()) {
            in0_sender_compile_time_args[0] = 0;  // core_has_output_block_work
            in0_sender_compile_time_args[1] = 0;  // core_in_in0_receiver_mcast_grid
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",
                in0_mcast_cores_without_work_and_not_in_receiver_grid.value(),
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = in0_noc,
                    .compile_args = in0_sender_compile_time_args,
                    .defines = mm_kernel_in0_sender_sharded_defines});
        }
    } else {
        if (fuse_op) {
            if (fused_op_signaler->is_all_gather()) {
                // Create semaphores
                fused_op_signaler->init_fused_op(program, device, in0_sender_interleaved);
            } else if (fused_op_signaler->is_reduce_scatter()) {
                fused_op_signaler->init_fused_op(program, device, all_cores, cores);
            } else {
                TT_FATAL(false, "Fused operation must be either all_gather or reduce_scatter.");
            }
        }

        mm_kernel_in0_sender_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
            in0_sender_interleaved,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_sender_compile_time_args,
                .defines = mm_kernel_in0_sender_interleaved_defines});
    }

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        in1_sender,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_kernel_in1_sender_writer_defines});

    tt::tt_metal::KernelHandle mm_kernel_in1_receiver_writer_id = 0;
    if (in1_receiver.num_cores() > 0) {
        mm_kernel_in1_receiver_writer_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
            /* in0_sender_in1_receiver, // If not using half-half noc setup */
            in1_receiver,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_noc,
                .compile_args = in1_receiver_writer_compile_time_args,
                .defines = mm_kernel_in1_receiver_writer_defines});
    }

    tt::tt_metal::KernelHandle mm_kernel_in0_receiver_id = 0;
    if (!in0_block_sharded and in0_receiver_interleaved.num_cores() > 0) {
        mm_kernel_in0_receiver_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
            /* in0_receiver_in1_sender, // If not using half-half noc setup */
            in0_receiver_interleaved,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_receiver_compile_time_args});
    }

    tt::tt_metal::KernelHandle mm_kernel_in1_receiver_writer_other_noc_setup_id = mm_kernel_in1_receiver_writer_id;
    tt::tt_metal::KernelHandle mm_kernel_in0_receiver_other_noc_setup_id = mm_kernel_in0_receiver_id;

    if (in0_receiver_in1_receiver_interleaved_other_cores.has_value()) {
        mm_kernel_in1_receiver_writer_other_noc_setup_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
            in0_receiver_in1_receiver_interleaved_other_cores.value(),
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_split_noc,
                .compile_args = in1_receiver_writer_compile_time_args,
                .defines = mm_kernel_in1_receiver_writer_other_noc_setup_defines});

        mm_kernel_in0_receiver_other_noc_setup_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
            in0_receiver_in1_receiver_interleaved_other_cores.value(),
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_split_noc,
                .compile_args = in0_receiver_compile_time_args});
    }

    // Compute kernel compile time args

    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,  // num_blocks
        out_num_blocks_x,
        out_num_blocks_y,

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch,
        out_block_tiles,         // out_block_num_tiles

        untilize_out,  // untilize_out
        false,         // get_batch_from_reader
        in0_transpose_tile,
    };

    // Create compute kernel
    // bool fp32_dest_acc_en = true;
    // Gelu currently has better accuracy when run in approx mode
    // bool math_approx_mode = false;
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores_with_work,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    if (in0_height_sharded) {
        src0_cb_config.set_globally_allocated_address(*in0_buffer);
    }
    tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src0_cb_index,
        in0_single_tile_size,
        in0_CB_size / in0_single_tile_size,
        in0_CB_size);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    if (in1_is_sharded and not in1_is_dram) {
        src1_cb_config.set_globally_allocated_address(*in1_buffer);
    }
    tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        in1_CB_size / in1_single_tile_size,
        in1_CB_size);

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CBHandle cb_src2 = 0;
    if (in0_block_sharded) {
        tt_metal::CircularBufferConfig src2_cb_config =
            tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
                .set_page_size(src2_cb_index, in0_single_tile_size)
                .set_globally_allocated_address(*in0_buffer)
                .set_tile_dims(src2_cb_index, in0_tile);
        cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, src2_cb_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src2_cb_index,
            in0_single_tile_size,
            in2_CB_size / in0_single_tile_size,
            in2_CB_size);

        // Local L1 to store temp vars
        uint32_t l1_cb_index = tt::CBIndex::c_6;
        tt::tt_metal::CircularBufferConfig cb_for_l1_array_config =
            tt::tt_metal::CircularBufferConfig(32 * 2, {{l1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(l1_cb_index, 32 * 2);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_for_l1_array_config);
    }

    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});

    if (do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
        (untilize_out && (in1_num_subblocks > 1))) {
        // output
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format},
        };
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile);
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{
            {interm0_cb_index, interm0_data_format},
        };
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                .set_tile_dims(interm0_cb_index, output_tile);

        tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), interm0_cb_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            interm0_cb_index,
            interm0_single_tile_size,
            interm0_CB_size / interm0_single_tile_size,
            interm0_CB_size);
    } else {
        // share buffer
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_page_size(interm0_cb_index, interm0_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_tile_dims(interm0_cb_index, output_tile);
    }

    if (output_is_sharded) {
        output_cb_config = output_cb_config.set_globally_allocated_address(*out_buffer);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), output_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        output_cb_index,
        output_single_tile_size,
        out_CB_size / output_single_tile_size,
        out_CB_size);

    // CB for bias
    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = tt::CBIndex::c_3;
        tt_metal::CircularBufferConfig cb_src3_config =
            tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
                .set_page_size(src3_cb_index, bias_single_tile_size)
                .set_tile_dims(src3_cb_index, bias_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_src3_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src3_cb_index,
            bias_single_tile_size,
            in3_CB_size / bias_single_tile_size,
            in3_CB_size);
    }
    // Intermediate CB read
    if (in1_needs_intermediate_cb_read) {
        uint32_t in1_intermediate_cb_index = tt::CBIndex::c_9;
        tt_metal::CircularBufferConfig cb_in1_intermediate_config =
            tt_metal::CircularBufferConfig(in1_single_tile_size, {{in1_intermediate_cb_index, in1_data_format}})
                .set_page_size(in1_intermediate_cb_index, in1_single_tile_size)
                .set_tile_dims(in1_intermediate_cb_index, in1_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_in1_intermediate_config);
    }
    if (in0_needs_intermediate_cb_read) {
        uint32_t in0_intermediate_cb_index = tt::CBIndex::c_8;
        tt_metal::CircularBufferConfig cb_in0_intermediate_config =
            tt_metal::CircularBufferConfig(in0_single_tile_size, {{in0_intermediate_cb_index, in0_data_format}})
                .set_page_size(in0_intermediate_cb_index, in0_single_tile_size)
                .set_tile_dims(in0_intermediate_cb_index, in0_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_in0_intermediate_config);
    }

    if (in0_transpose_tile) {
        uint32_t in0_transpose_cb_index = tt::CBIndex::c_10;
        auto in0_transpose_cb_config =
            tt_metal::CircularBufferConfig(in0_CB_size, {{in0_transpose_cb_index, in0_data_format}})
                .set_page_size(in0_transpose_cb_index, in0_single_tile_size)
                .set_tile_dims(in0_transpose_cb_index, in0_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, in0_transpose_cb_config);
    }

    // Parameters for last row, col, or block
    uint32_t last_per_core_M = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_per_core_N = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_out_block_h = last_per_core_M % out_block_h == 0 ? out_block_h : last_per_core_M % out_block_h;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_out_num_blocks_h = ((last_per_core_M - 1) / out_block_h) + 1;
    uint32_t last_out_num_blocks_w = ((last_per_core_N - 1) / out_block_w) + 1;
    uint32_t last_block_num_nonzero_subblocks_h = ((last_out_block_h - 1) / out_subblock_h) + 1;
    uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;
    uint32_t last_subblock_of_last_block_h =
        last_out_block_h % out_subblock_h == 0 ? out_subblock_h : last_out_block_h % out_subblock_h;
    uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);
    uint32_t last_block_padded_block_tiles_h_skip =
        (out_block_h / out_subblock_h - last_block_num_nonzero_subblocks_h) * (out_block_w * out_subblock_h);

    if (in0_block_sharded) {
        if (in0_noc == tt::tt_metal::NOC::NOC_1) {
            std::swap(in0_mcast_receiver_grid_diff_coord_start, in0_mcast_receiver_grid_diff_coord_end);
        }
    }

    // dram sharded weights stride params
    uint32_t worker_core_stride = 0;   // stride in the worker core
    uint32_t storage_core_stride = 0;  // stride in the dram bank
    uint32_t curr_worker_core = 0;     // current worker core
    uint32_t curr_storage_core = 0;    // current read dram bank
    uint32_t vc = 0;

    uint32_t in0_end_idx = num_blocks_y - 1;
    uint32_t in1_end_idx = num_blocks_x - 1;
    const auto& in0_sender_interleaved_cores = grid_to_cores(
        in0_sender_interleaved.start_coord, in0_sender_interleaved.end_coord, true);  // Only used for interleaved in0
    const auto& in1_sender_cores = grid_to_cores(in1_sender.start_coord, in1_sender.end_coord, true);
    const auto& in1_receiver_cores = corerange_to_cores(in1_receiver, std::nullopt, true);
    std::vector<CoreCoord> in1_receiver_other_cores;
    if (in0_receiver_in1_receiver_interleaved_other_cores.has_value()) {
        in1_receiver_other_cores = grid_to_cores(
            in0_receiver_in1_receiver_interleaved_other_cores.value().start_coord,
            in0_receiver_in1_receiver_interleaved_other_cores.value().end_coord,
            true);
    }

    for (const auto& core : cores) {
        CoreCoord left_core = {(std::size_t)start_core_x, (std::size_t)core.y};
        CoreCoord left_core_plus_one = {(std::size_t)start_core_x + 1, (std::size_t)core.y};
        CoreCoord right_core = {(std::size_t)start_core_x + num_cores_with_work_c - 1, (std::size_t)core.y};
        CoreCoord top_core = {(std::size_t)core.x, (std::size_t)start_core_y};
        CoreCoord top_core_plus_one = {(std::size_t)core.x, (std::size_t)start_core_y + 1};
        CoreCoord bottom_core = {(std::size_t)core.x, (std::size_t)start_core_y + num_cores_with_work_r - 1};

        auto left_core_physical = device->worker_core_from_logical_core(left_core);
        auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
        auto right_core_physical = device->worker_core_from_logical_core(right_core);
        auto top_core_physical = device->worker_core_from_logical_core(top_core);
        auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
        auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);
        uint32_t in0_idx = core.y - start_core_y;
        uint32_t in1_idx = core.x - start_core_x;

        auto in0_mcast_sender = left_core_physical;
        auto in1_mcast_sender = top_core_physical;

        // Assuming in0 is NOC0
        auto in0_mcast_start = left_core_plus_one_physical;
        auto in0_mcast_end = right_core_physical;
        if (in0_noc == tt::tt_metal::NOC::NOC_1) {
            std::swap(in0_mcast_start, in0_mcast_end);
        }

        // Assuming in1 is NOC1
        auto in1_mcast_start = bottom_core_physical;
        auto in1_mcast_end = top_core_plus_one_physical;
        if (in1_noc == tt::tt_metal::NOC::NOC_0) {
            std::swap(in1_mcast_start, in1_mcast_end);
        }

        if (transpose_mcast) {
            std::swap(in0_idx, in1_idx);
            std::swap(in0_mcast_sender, in1_mcast_sender);
            std::swap(in0_mcast_start, in1_mcast_end);
            std::swap(in0_mcast_end, in1_mcast_start);
        }

        // in0 sender
        if (in0_block_sharded) {
            uint32_t in0_mcast_receiver_grid_same_coord;
            std::vector<uint32_t> mm_in0_sender_args;
            if (transpose_mcast) {
                in0_mcast_receiver_grid_same_coord = device->worker_core_from_logical_core(core).x;
                mm_in0_sender_args.push_back(core.y);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_diff_coord_start);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_diff_coord_end);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
                mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());
            } else {
                in0_mcast_receiver_grid_same_coord = device->worker_core_from_logical_core(core).y;
                mm_in0_sender_args.push_back(core.x);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_diff_coord_start);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_diff_coord_end);
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
                mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
                mm_in0_sender_args.push_back(in0_mcast_receiver_grid_same_coord);
            }
            if (in1_idx < num_blocks_x) {
                tt_metal::SetRuntimeArgs(
                    program, mm_kernel_in0_sender_id, core, mm_in0_sender_args);  // RISCV_0_default
            } else {
                tt_metal::SetRuntimeArgs(
                    program,
                    mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id,
                    core,
                    mm_in0_sender_args);  // RISCV_0_default
            }
        } else if (in1_idx == 0) {
            std::vector<uint32_t> mm_in0_sender_args = {
                // in0 tensor args
                (std::uint32_t)in0_buffer->address(),
                (std::uint32_t)in0_tensor_start_tile_id_stride * in0_idx,  // in0_tensor_start_tile_id
                // in0 mcast args
                (std::uint32_t)in0_mcast_start.x,  // in0_mcast_dest_noc_start_x
                (std::uint32_t)in0_mcast_start.y,  // in0_mcast_dest_noc_start_y
                (std::uint32_t)in0_mcast_end.x,    // in0_mcast_dest_noc_end_x
                (std::uint32_t)in0_mcast_end.y,    // in0_mcast_dest_noc_end_y
            };
            if (in0_idx == in0_end_idx) {
                // padding args (READER)
                mm_in0_sender_args.push_back(last_out_block_h);  // last_out_block_h
            } else {
                mm_in0_sender_args.push_back(out_block_h);
            }

            // sparsity args
            mm_in0_sender_args.push_back(0);  // sparsity_addr

            if (fuse_op && fused_op_signaler->is_all_gather()) {
                fused_op_signaler->push_matmul_fused_op_rt_args(mm_in0_sender_args, false);
            }

            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args);  // RISCV_0_default

            // in0 receiver
        } else {
            std::vector<uint32_t> mm_in0_receiver_args = {
                // in0 mcast args
                (std::uint32_t)in0_mcast_sender.x,  // in0_mcast_sender_noc_x
                (std::uint32_t)in0_mcast_sender.y   // in0_mcast_sender_noc_y
            };
            // left half
            if (core.x <= half_core || (!transpose_mcast and core.y == start_core_y)) {
                tt_metal::SetRuntimeArgs(program, mm_kernel_in0_receiver_id, core, mm_in0_receiver_args);
            }
            // right half
            else {
                tt_metal::SetRuntimeArgs(
                    program, mm_kernel_in0_receiver_other_noc_setup_id, core, mm_in0_receiver_args);
            }
        }

        if (in0_idx < num_blocks_y and in1_idx < num_blocks_x) {
            // in1 sender
            if (in0_idx == 0) {
                std::vector<uint32_t> mm_in1_sender_writer_args = {
                    // READER
                    // in1 tensor args
                    (std::uint32_t)in1_buffer->address(),
                    (std::uint32_t)in1_tensor_start_tile_id_stride * in1_idx,  // in1_tensor_start_tile_id
                    // in1 mcast args
                    (std::uint32_t)in1_mcast_start.x,  // in1_mcast_dest_noc_start_x
                    (std::uint32_t)in1_mcast_start.y,  // in1_mcast_dest_noc_start_y
                    (std::uint32_t)in1_mcast_end.x,    // in1_mcast_dest_noc_end_x
                    (std::uint32_t)in1_mcast_end.y,    // in1_mcast_dest_noc_end_y

                    // sparsity args
                    (std::uint32_t)0,  // sparsity_addr

                    // WRITER
                    // out tensor args
                    (std::uint32_t)out_buffer->address(),
                    ((std::uint32_t)in1_idx * per_core_N) + (in0_idx * per_core_M * N)  // out_tensor_start_tile_id
                };

                if (in1_idx == in1_end_idx) {  // right cores when no transpose_mcast
                    // padding args (READER)
                    mm_in1_sender_writer_args.push_back(last_out_block_w);

                    // padding args (WRITER)
                    mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_sender_writer_args.push_back(out_subblock_h);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_sender_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_sender_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_sender_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else {
                    // padding args (READER)
                    mm_in1_sender_writer_args.push_back(out_block_w);

                    // padding args (WRITER)
                    mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_sender_writer_args.push_back(out_subblock_h);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(out_subblock_w);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(0);
                }

                mm_in1_sender_writer_args.push_back(bias_buffer ? (std::uint32_t)bias_buffer->address() : 0);
                mm_in1_sender_writer_args.push_back(
                    bias_buffer ? (std::uint32_t)per_core_N * in1_idx : 0);  // in1_tensor_start_tile_id
                if (!output_is_sharded) {
                    if (in1_idx == in1_end_idx) {  // right cores when no transpose_mcast
                        mm_in1_sender_writer_args.push_back(last_out_num_blocks_w);
                    } else {
                        mm_in1_sender_writer_args.push_back(out_num_blocks_x);
                    }
                }

                if (in1_is_sharded and in1_is_dram) {  // in1 is dram sharded
                    uint32_t num_iter_index = mm_in1_sender_writer_args.size() + 1;
                    vc = vc == 3 ? 0 : vc + 1;
                    mm_in1_sender_writer_args.push_back(vc);

                    uint32_t num_iter = 0;  // iterate how many banks, till fill the current worker block

                    if (curr_storage_core < num_dram_banks) {
                        num_iter++;

                        worker_core_stride = per_core_N_storage - storage_core_stride;

                        mm_in1_sender_writer_args.push_back(
                            storage_core_stride * in1_single_tile_size);  // dram_tensor_start_offset
                        mm_in1_sender_writer_args.push_back(
                            worker_core_stride * in1_single_tile_size);          // per_core_N_dram_bytes
                        mm_in1_sender_writer_args.push_back(curr_storage_core);  // current_dram_bank_id

                        log_debug(
                            tt::LogOp,
                            "curr worker core: {} read {} tiles from dram bank: {}, start from index: {}",
                            curr_worker_core,
                            worker_core_stride,
                            curr_storage_core,
                            storage_core_stride);

                        curr_storage_core += (storage_core_stride + worker_core_stride) / per_core_N_storage;
                        storage_core_stride = (storage_core_stride + worker_core_stride) % per_core_N_storage;

                        uint32_t curr_worker_core_old = curr_worker_core;
                        if (worker_core_stride >= per_core_N) {
                            curr_worker_core += 1;
                        }

                        while (curr_worker_core <= curr_worker_core_old and curr_storage_core < num_dram_banks) {
                            num_iter++;

                            uint32_t stride = worker_core_stride + per_core_N_storage;
                            stride = std::min(stride, per_core_N);

                            mm_in1_sender_writer_args.push_back(
                                (stride - worker_core_stride) * in1_single_tile_size);  // per_core_N_dram_bytes
                            mm_in1_sender_writer_args.push_back(curr_storage_core);     // current_dram_bank_id

                            log_debug(
                                tt::LogOp,
                                "curr worker core: {} read {} tiles from dram bank: {}, start from index: {}",
                                curr_worker_core,
                                (stride - worker_core_stride),
                                curr_storage_core,
                                storage_core_stride);

                            if (stride >= per_core_N) {
                                curr_worker_core += 1;
                            }
                            storage_core_stride = (stride - worker_core_stride) % per_core_N_storage;
                            curr_storage_core += (stride - worker_core_stride) / per_core_N_storage;
                            worker_core_stride = stride;
                        }
                    }
                    mm_in1_sender_writer_args.insert(mm_in1_sender_writer_args.begin() + num_iter_index, num_iter);
                }
                if (fuse_op) {
                    if (fused_op_signaler->is_all_gather()) {
                        fused_op_signaler->push_matmul_fused_op_rt_args(mm_in1_sender_writer_args, true);
                    } else if (fused_op_signaler->is_reduce_scatter()) {
                        fused_op_signaler->push_matmul_fused_op_rt_args(mm_in1_sender_writer_args, in0_idx, in1_idx);
                    } else {
                        TT_FATAL(false, "Fused operation must be either all_gather or reduce_scatter.");
                    }
                }
                tt_metal::SetRuntimeArgs(
                    program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);  // RISCV_1_default

                // in1 receiver
            } else {
                std::vector<uint32_t> mm_in1_receiver_writer_args = {
                    // READER
                    // in1 mcast args
                    (std::uint32_t)in1_mcast_sender.x,  // in1_mcast_sender_noc_x
                    (std::uint32_t)in1_mcast_sender.y,  // in1_mcast_sender_noc_y

                    // WRITER
                    // out tensor args
                    (std::uint32_t)out_buffer->address(),                               // out_tensor_addr
                    ((std::uint32_t)in1_idx * per_core_N) + (in0_idx * per_core_M * N)  // out_tensor_start_tile_id
                };

                if (in1_idx == in1_end_idx and in0_idx == in0_end_idx) {  // bottom-right core when no transpose_mcast
                    // padding args (WRITER)
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else if (in0_idx == in0_end_idx) {  // bottom cores except bottom-right when no transpose_mcast
                    // padding args (WRITER)
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(0);
                } else if (in1_idx == in1_end_idx) {  // right cores except bottom when no transpose_mcast
                    // padding args (WRITER)
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else {
                    // padding args (WRITER)
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_block_h / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_block_w / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(0);
                }
                if (!output_is_sharded) {
                    if (in1_idx == in1_end_idx and
                        in0_idx == in0_end_idx) {  // bottom-right core when no transpose_mcast
                        mm_in1_receiver_writer_args.push_back(last_out_num_blocks_h);
                        mm_in1_receiver_writer_args.push_back(last_out_num_blocks_w);
                    } else if (in0_idx == in0_end_idx) {  // bottom cores except bottom-right when no transpose_mcast
                        mm_in1_receiver_writer_args.push_back(last_out_num_blocks_h);
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_x);
                    } else if (in1_idx == in1_end_idx) {  // right cores except bottom when no transpose_mcast
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_y);
                        mm_in1_receiver_writer_args.push_back(last_out_num_blocks_w);
                    } else {
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_y);
                        mm_in1_receiver_writer_args.push_back(out_num_blocks_x);
                    }
                }

                if (fuse_op && fused_op_signaler->is_reduce_scatter()) {
                    fused_op_signaler->push_matmul_fused_op_rt_args(mm_in1_receiver_writer_args, in0_idx, in1_idx);
                }

                // left half
                if (core.x <= half_core || (transpose_mcast and core.y == start_core_y)) {
                    tt_metal::SetRuntimeArgs(
                        program, mm_kernel_in1_receiver_writer_id, core, mm_in1_receiver_writer_args);
                }
                // right half
                else {
                    tt_metal::SetRuntimeArgs(
                        program, mm_kernel_in1_receiver_writer_other_noc_setup_id, core, mm_in1_receiver_writer_args);
                }
            }
        }
    }

    return {
        std::move(program),
        {mm_kernel_in0_sender_id,
         in0_sender_interleaved_cores,
         mm_kernel_in1_sender_writer_id,
         in1_sender_cores,
         mm_kernel_in1_receiver_writer_id,
         in1_receiver_cores,
         mm_kernel_in1_receiver_writer_other_noc_setup_id,
         in1_receiver_other_cores,
         cb_src2,
         cb_output,
         num_cores_with_work_r,
         num_cores_with_work_c,
         start_core_x,
         start_core_y,
         transpose_mcast,
         cores}};
}

void override_runtime_arguments_impl(
    const MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t& shared_variables,
    tt::tt_metal::Program& program,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& output_tensors) {
    auto mm_kernel_in0_sender_id = shared_variables.mm_kernel_in0_sender_id;
    auto in0_sender_interleaved_cores = shared_variables.in0_sender_interleaved_cores;
    auto mm_kernel_in1_sender_writer_id = shared_variables.mm_kernel_in1_sender_writer_id;
    auto in1_sender_cores = shared_variables.in1_sender_cores;
    auto mm_kernel_in1_receiver_writer_id = shared_variables.mm_kernel_in1_receiver_writer_id;
    auto in1_receiver_cores = shared_variables.in1_receiver_cores;
    auto mm_kernel_in1_receiver_writer_other_noc_setup_id =
        shared_variables.mm_kernel_in1_receiver_writer_other_noc_setup_id;
    auto in1_receiver_other_cores = shared_variables.in1_receiver_other_cores;
    auto cb_src2 = shared_variables.cb_src2;
    auto cb_output = shared_variables.cb_output;
    auto cores = shared_variables.cores;

    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;

    TT_FATAL(
        input_tensors.size() + optional_input_tensors.size() == 3,
        "Total number of input tensors (required + optional) must be 3, but got {} + {} = {}",
        input_tensors.size(),
        optional_input_tensors.size(),
        input_tensors.size() + optional_input_tensors.size());
    TT_FATAL(output_tensors.size() == 1, "Number of output tensors must be 1, but got {}", output_tensors.size());

    auto* src_buffer_a = input_tensors.at(0).buffer();
    auto* src_buffer_b = input_tensors.at(1).buffer();
    const auto& bias_tensor = optional_input_tensors.at(0);

    auto* dst_buffer = output_tensors.at(0).buffer();

    bool src0_sharded = input_tensors[0].memory_config().is_sharded();
    bool out_sharded = output_tensors[0].memory_config().is_sharded();

    std::optional<tt::tt_metal::Buffer*> bias_buffer;
    if (bias_tensor.has_value()) {
        bias_buffer = bias_tensor.value().buffer();
    }

    // in0 sender
    if (src0_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_src2, *src_buffer_a);
    } else {
        auto& reader_sender_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in0_sender_id);
        for (const auto& core : in0_sender_interleaved_cores) {
            auto& reader_runtime_args = reader_sender_runtime_args_by_core[core.x][core.y];
            reader_runtime_args[0] = src_buffer_a->address();
        }
    }

    // in1 sender
    auto& sender_writer_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in1_sender_writer_id);
    for (const auto& core : in1_sender_cores) {
        auto& writer_runtime_args = sender_writer_runtime_args_by_core[core.x][core.y];
        writer_runtime_args[0] = src_buffer_b->address();
        writer_runtime_args[7] = dst_buffer->address();
        if (bias_tensor.has_value()) {
            writer_runtime_args[18] = (*bias_buffer)->address();
        }
    }

    // in1 receiver
    auto& receiver_writer_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in1_receiver_writer_id);
    for (const auto& core : in1_receiver_cores) {
        auto& writer_runtime_args = receiver_writer_runtime_args_by_core[core.x][core.y];
        writer_runtime_args[2] = dst_buffer->address();
    }
    if (mm_kernel_in1_receiver_writer_id != mm_kernel_in1_receiver_writer_other_noc_setup_id) {
        auto& receiver_writer_runtime_args_by_core =
            GetRuntimeArgs(program, mm_kernel_in1_receiver_writer_other_noc_setup_id);
        for (const auto& core : in1_receiver_other_cores) {
            auto& writer_runtime_args = receiver_writer_runtime_args_by_core[core.x][core.y];
            writer_runtime_args[2] = dst_buffer->address();
        }
    }

    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    }
}
}  // namespace reuse_mcast_optimized_helpers

static MatmulMultiCoreReuseMcast2DProgramFactory::cached_program_t matmul_multi_core_reuse_mcast_2d_optimized_(
    tt::tt_metal::Program& program,
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
    using namespace tt;
    using namespace operations::matmul::utilities;

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& bias = tensor_args.optional_input_tensors.at(0);
    const auto& output = tensor_return_value.at(0);

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    TT_FATAL(operation_attributes.program_config.has_value(), "Error: program_config field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();
    bool transpose_a = operation_attributes.transpose_a;
    bool transpose_b = operation_attributes.transpose_b;

    auto program_config = std::get<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(
        operation_attributes.program_config.value());

    auto fuse_batch = program_config.fuse_batch;
    auto in0_block_w = program_config.in0_block_w;
    auto compute_with_storage_grid_size = program_config.compute_with_storage_grid_size;
    auto out_subblock_h = program_config.out_subblock_h;
    auto out_subblock_w = program_config.out_subblock_w;
    auto out_block_h = program_config.out_block_h;
    auto out_block_w = program_config.out_block_w;
    auto per_core_M = program_config.per_core_M;
    auto per_core_N = program_config.per_core_N;
    auto transpose_mcast = program_config.transpose_mcast;

    TT_FATAL(
        operation_attributes.compute_kernel_config.has_value(),
        "Error: compute_kernel_config field should have been populated");
    auto compute_kernel_config = operation_attributes.compute_kernel_config.value();
    auto untilize_out = operation_attributes.untilize_out;
    // auto fused_op_signaler = std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>();

    const auto& a_shape_padded = get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& b_shape_padded = get_matmul_tensor_padded_shape(b, transpose_b);
    const auto in0_tile = get_matmul_tile(a, transpose_a);
    const auto in1_tile = get_matmul_tile(b, transpose_b);

    // cannot use the output tensor tile directly as that might be changed by user override
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());          // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());          // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());  // output

    const auto& a_shape_logical = get_matmul_tensor_logical_shape(a, transpose_a);
    const auto in0_last_ktile_w = a_shape_logical[-1] % in0_tile.get_width();

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;  // bias; doesn't matter if bias=nullptr
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(
            c.storage_type() == StorageType::DEVICE,
            "Bias tensor must be on device, got storage type: {}",
            c.storage_type());
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

        bias_buffer = c.buffer();

        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    tt_metal::IDevice* device = a.device();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();
    TT_FATAL(
        in0_buffer->size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        in0_buffer->size(),
        in0_single_tile_size);
    TT_FATAL(
        in1_buffer->size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        in1_buffer->size(),
        in1_single_tile_size);

    TT_FATAL(
        a_shape_padded[-1] == b_shape_padded[-2],
        "Dimension K (A.shape[-1] = {}, B.shape[-2] = {}) must match for matmul",
        a_shape_padded[-1],
        b_shape_padded[-2]);
    TT_FATAL(
        a_shape_padded[-2] % in0_tile.get_height() == 0,
        "A.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        a_shape_padded[-2],
        in0_tile.get_height());
    TT_FATAL(
        a_shape_padded[-1] % in0_tile.get_width() == 0,
        "A.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        a_shape_padded[-1],
        in0_tile.get_width());
    TT_FATAL(
        b_shape_padded[-2] % in1_tile.get_height() == 0,
        "B.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        b_shape_padded[-2],
        in1_tile.get_height());
    TT_FATAL(
        b_shape_padded[-1] % in1_tile.get_width() == 0,
        "B.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        b_shape_padded[-1],
        in1_tile.get_width());

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    const auto B = fuse_batch ? 1 : get_batch_size(a_shape_padded);
    const auto Mt = get_M_dim(a_shape_padded, in0_tile, fuse_batch);
    const auto Kt = get_K_dim(a_shape_padded, in0_tile);
    const auto Nt = get_N_dim(b_shape_padded, in1_tile);

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

    // This should allocate a DRAM buffer on the device
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    uint32_t num_blocks_y = ((Mt - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((Nt - 1) / per_core_N) + 1;
    if (transpose_mcast) {
        std::swap(num_blocks_x, num_blocks_y);
    }

    // TODO: Max used grid can actually exceed mcast receiver grid if in0 is sharded
    // TODO: Move these validates to op validate and properly check for this
    TT_FATAL(
        num_blocks_x <= num_cores_x,
        "Num output blocks along x ({}) must be smaller than or equal to the number of columns in compute grid ({})!",
        num_blocks_x,
        num_cores_x);
    TT_FATAL(
        num_blocks_y <= num_cores_y,
        "Num output blocks along y ({}) must be smaller than or equal to the number of rows in compute grid ({})!",
        num_blocks_y,
        num_cores_y);

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer* out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    return reuse_mcast_optimized_helpers::create_program_mcast_in0_in1(
        program,
        device,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        B,
        Mt,
        Nt,
        Kt,
        bcast_batch,
        transpose_a,
        transpose_b,
        ttnn::get_throttle_level(compute_kernel_config),
        in0_block_w,
        in0_last_ktile_w,
        out_subblock_h,
        out_subblock_w,
        out_block_h,
        out_block_w,
        per_core_M,
        per_core_N,
        transpose_mcast,
        program_config.fused_activation,
        in0_buffer,
        in1_buffer,
        bias_buffer,
        out_buffer,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->tensor_spec().tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        untilize_out,
        fused_op_signaler);
}

MatmulMultiCoreReuseMcast2DProgramFactory::cached_program_t MatmulMultiCoreReuseMcast2DProgramFactory::create(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::Program program{};
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> fused_op_signaler = std::nullopt;

    return matmul_multi_core_reuse_mcast_2d_optimized_(
        program, operation_attributes, tensor_args, tensor_return_value, fused_op_signaler);
}

void MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::MatmulParams& /*operation_attributes*/,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    reuse_mcast_optimized_helpers::override_runtime_arguments_impl(
        cached_program.shared_variables, cached_program.program, tensor_args, tensor_return_value);
}

void MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const shared_variables_t& shared_variables,
    const ttnn::prim::MatmulParams& /*operation_attributes*/,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    reuse_mcast_optimized_helpers::override_runtime_arguments_impl(
        shared_variables, program, tensor_args, tensor_return_value);
}

MatmulMeshWorkloadMultiCoreReuseMcast2DProgramFactory::cached_mesh_workload_t
MatmulMeshWorkloadMultiCoreReuseMcast2DProgramFactory::create_mesh_workload(
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange mesh_coord_range{mesh_coord, mesh_coord};
            auto single_device_program =
                MatmulMultiCoreReuseMcast2DProgramFactory::create(attributes, tensor_args, tensor_return_value);
            shared_variables[mesh_coord_range] = single_device_program.shared_variables;
            workload.add_program(mesh_coord_range, std::move(single_device_program.program));
        }
    }
    return {std::move(workload), std::move(shared_variables)};
}

void MatmulMeshWorkloadMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        auto cached_program_proxy = MatmulMultiCoreReuseMcast2DProgramFactory::cached_program_t::proxy(
            program, cached_workload.shared_variables.at(mesh_coord_range));
        MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments(
            cached_program_proxy, attributes, tensor_args, tensor_return_value);
    }
}

MatmulMultiCoreReuseMcast2DProgramFactory::cached_program_t matmul_multi_core_reuse_mcast_2d_optimized_helper(
    tt::tt_metal::Program& program, /* Take programa as input by reference */
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
    auto attributes = ttnn::prim::MatmulParams{.program_config = program_config, .bcast_batch = broadcast_batch};
    attributes.compute_kernel_config = compute_kernel_config;
    attributes.untilize_out = untilize_out;

    auto output_tensors = std::vector<ttnn::Tensor>{output_tensor};
    return matmul_multi_core_reuse_mcast_2d_optimized_(
        program, attributes, ttnn::prim::MatmulInputs{{a, b}, {bias}, {}}, output_tensors, fused_op_signaler);
}

}  // namespace ttnn::prim
