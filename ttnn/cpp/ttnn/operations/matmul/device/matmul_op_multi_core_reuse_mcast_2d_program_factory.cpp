// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "hostdevcommon/common_values.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

using namespace tt::constants;
using namespace tt;
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace reuse_mcast_optimized_helpers {
using namespace tt::constants;
using namespace tt;
using namespace tt_metal;

operation::ProgramWithCallbacks create_program_mcast_in0_in1(
    tt_metal::Program& program,
    tt_metal::Device* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    tt_metal::Buffer* bias_buffer,
    tt_metal::Buffer* out_buffer,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> &fused_op_signaler) {

    bool fuse_op = fused_op_signaler.has_value();

    TensorMemoryLayout in0_memory_layout = in0_buffer->buffer_layout();
    // tt_metal::Program program{};

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

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t bias_single_tile_size = tt_metal::detail::TileSize(bias_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);
    uint32_t interm0_single_tile_size = tt_metal::detail::TileSize(interm0_data_format);

    const bool in0_block_sharded = in0_memory_layout == TensorMemoryLayout::BLOCK_SHARDED;
    const bool in0_height_sharded = in0_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool in0_is_sharded = in0_block_sharded || in0_height_sharded;
    const bool in1_is_sharded = in1_buffer->buffer_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const bool output_is_sharded = out_buffer->buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles = in0_CB_tiles * 2;  // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles = in1_CB_tiles * 2;  // double buffer
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    uint32_t in0_shard_height_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / TILE_WIDTH;
        in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / TILE_HEIGHT;
        in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    }

    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = per_core_N;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
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

    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool in3_is_dram = true;
    if (bias_buffer != nullptr) {
        in3_is_dram = bias_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    }
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    std::vector<uint32_t> in0_sender_compile_time_args;

    uint32_t num_dram_banks = 0;
    uint32_t per_core_N_storage = 0;
    if (in1_is_sharded and in1_is_dram) {
        num_dram_banks = device->num_dram_channels();
        per_core_N_storage = (N + num_dram_banks - 1) / num_dram_banks;
    }

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
            // in0/in1 common args
            (std::uint32_t)num_blocks,  // num_blocks
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
            // batch args
            (std::uint32_t)B  // batch
        };
    } else {
        in0_sender_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t)in0_is_dram,

            // in0 tensor args
            (std::uint32_t)1,            // in0_tensor_stride_w
            (std::uint32_t)K,            // in0_tensor_stride_h
            (std::uint32_t)in0_block_w,  // in0_tensor_next_block_stride
            // in0 block args
            (std::uint32_t)in0_block_w,          // in0_block_w
            (std::uint32_t)per_core_M,           // in0_block_h
            (std::uint32_t)in0_block_num_tiles,  // in0_block_num_tiles
            (std::uint32_t) false,               // extract_shard_sub_blocks (not used for interleaved)
            (std::uint32_t)0,                    // shard_width_in_tiles (not used for interleaved)
            (std::uint32_t)0,                    // shard_height_in_tiles (not used for interleaved)
            // in0/in1 common args
            (std::uint32_t)num_blocks,  // num_blocks
            // in0 mcast args
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            (std::uint32_t)(num_blocks_x - 1),  // in0_mcast_num_dests
            (std::uint32_t)(num_blocks_x - 1),  // in0_mcast_num_cores
            // batch args
            (std::uint32_t)M * K,  // MtKt
            (std::uint32_t)B       // batch
        };
    }
    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)in1_is_dram,
        (std::uint32_t)out_is_dram,

        // READER
        // in1 tensor args
        (std::uint32_t)1,                // in1_tensor_stride_w
        (std::uint32_t)N,                // in1_tensor_stride_h
        (std::uint32_t)in0_block_w * N,  // in1_tensor_next_block_stride
        // in1 block args
        (std::uint32_t)per_core_N,                // in1_block_w
        (std::uint32_t)in0_block_w,               // in1_block_h
        (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,  // num_blocks
        // in1 mcast args
        (std::uint32_t)in1_mcast_sender_semaphore_id,
        (std::uint32_t)in1_mcast_receiver_semaphore_id,
        (std::uint32_t)(num_blocks_y - 1),  // in1_mcast_num_dests
        (std::uint32_t)(num_blocks_y - 1),  // in1_mcast_num_cores
        // batch args
        (std::uint32_t)K * N,        // KtNt
        (std::uint32_t)B,            // batch
        (std::uint32_t)bcast_batch,  // bcast_B

        // WRITER
        // out tensor args
        (std::uint32_t)1,                   // out_tensor_stride_w
        (std::uint32_t)N,                   // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,      // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * N,  // out_tensor_next_subblock_stride_h
        // out subblock args
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        // batch args
        (std::uint32_t)M * N  // MtNt
    };
    if (bias_buffer != nullptr) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)in3_is_dram);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
    } else {
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }
    if (in1_is_sharded and in1_is_dram) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)per_core_N_storage * in0_block_w);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)per_core_N_storage * in1_single_tile_size);
    }
    std::vector<uint32_t> in0_receiver_compile_time_args = {
        // in0 block args
        (std::uint32_t)in0_block_w * per_core_M,  // in0_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,  // num_blocks
        // in0 mcast args
        (std::uint32_t)in0_mcast_sender_semaphore_id,
        (std::uint32_t)in0_mcast_receiver_semaphore_id,
        // batch args
        (std::uint32_t)B  // batch
    };
    std::vector<uint32_t> in1_receiver_writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)out_is_dram,

        // READER
        // in1 block args
        (std::uint32_t)per_core_N * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,  // num_blocks
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
        // out subblock args
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        // batch args
        (std::uint32_t)M * N  // MtNt
    };
    if (bias_buffer != nullptr) {
        in1_receiver_writer_compile_time_args.push_back((std::uint32_t)per_core_N);
    }

    std::map<string, string> mm_kernel_defines;
    std::map<string, string> mm_kernel_in0_sender_sharded_defines;
    std::map<string, string> mm_kernel_in0_sender_interleaved_defines;
    std::map<string, string> mm_kernel_in1_sender_writer_defines;
    std::map<string, string> mm_kernel_in1_receiver_writer_defines;
    std::map<string, string> mm_kernel_in1_receiver_writer_other_noc_setup_defines;
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
            mm_kernel_defines.merge(
                get_defines(fused_activation.value().op_type, fused_activation.value().params, "ACTIVATION", "i"));
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }

    bmm_op_utils::add_stagger_defines_if_needed(device->arch(), cores.size(), mm_kernel_defines);

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

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = detail::GetPreferredNOCForDRAMRead(device->arch());
    tt_metal::NOC in0_split_noc = detail::GetPreferredNOCForDRAMRead(device->arch());
    tt_metal::NOC in1_split_noc = detail::GetPreferredNOCForDRAMWrite(device->arch());

    KernelHandle mm_kernel_in0_sender_id = 0;
    KernelHandle mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id = 0;
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
            // Create semaphores
            fused_op_signaler->init_fused_op(program, device, in0_sender_interleaved);
            fused_op_signaler->emit_matmul_fused_op_ct_args(in0_sender_compile_time_args);

            // Switch the offesets for the weights
            fused_op_signaler->output_page_offset = fused_op_signaler->tensor_slice_shape_width * fused_op_signaler->weight_tensor_width;
            fused_op_signaler->emit_matmul_fused_op_ct_args(in1_sender_writer_compile_time_args);
        } else {
            // Push false for fuse_op param
            in0_sender_compile_time_args.push_back(static_cast<bool>(false));
            in1_sender_writer_compile_time_args.push_back(static_cast<bool>(false));
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

    KernelHandle mm_kernel_in1_receiver_writer_id = 0;
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

    KernelHandle mm_kernel_in0_receiver_id = 0;
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

    KernelHandle mm_kernel_in1_receiver_writer_other_noc_setup_id = mm_kernel_in1_receiver_writer_id;
    KernelHandle mm_kernel_in0_receiver_other_noc_setup_id = mm_kernel_in0_receiver_id;

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

    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,  // num_blocks

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch,
        out_block_tiles,         // out_block_num_tiles

        untilize_out};

    // Create compute kernel
    // bool fp32_dest_acc_en = true;
    // Gelu currently has better accuracy when run in approx mode
    // bool math_approx_mode = false;
    auto mm_kernel = tt_metal::CreateKernel(
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
    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size);
    if (in0_height_sharded) {
        src0_cb_config.set_globally_allocated_address(*in0_buffer);
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src0_cb_index,
        in0_single_tile_size,
        in0_CB_size / in0_single_tile_size,
        in0_CB_size);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size);
    if (in1_is_sharded and not in1_is_dram) {
        src1_cb_config.set_globally_allocated_address(*in1_buffer);
    }
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        in1_CB_size / in1_single_tile_size,
        in1_CB_size);

    uint32_t src2_cb_index = 2;
    CBHandle cb_src2 = 0;
    if (in0_block_sharded) {
        tt_metal::CircularBufferConfig src2_cb_config =
            tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
                .set_page_size(src2_cb_index, in0_single_tile_size)
                .set_globally_allocated_address(*in0_buffer);
        cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, src2_cb_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src2_cb_index,
            in0_single_tile_size,
            in2_CB_size / in0_single_tile_size,
            in2_CB_size);

        // Local L1 to store temp vars
        uint32_t l1_cb_index = 5;
        CircularBufferConfig cb_for_l1_array_config =
            CircularBufferConfig(32 * 2, {{l1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(l1_cb_index, 32 * 2);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_for_l1_array_config);
    }

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    tt_metal::CircularBufferConfig interm0_cb_config =
        tt_metal::CircularBufferConfig(0, {{interm0_cb_index, interm0_data_format}});
    tt_metal::CircularBufferConfig output_cb_config =
        tt_metal::CircularBufferConfig(0, {{output_cb_index, output_data_format}});

    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        // output
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format},
        };
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size);
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{
            {interm0_cb_index, interm0_data_format},
        };
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                                .set_page_size(interm0_cb_index, interm0_single_tile_size);

        auto cb_interm0 = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), interm0_cb_config);
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
                               .set_page_size(interm0_cb_index, interm0_single_tile_size);
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
        uint32_t src3_cb_index = 3;
        tt_metal::CircularBufferConfig cb_src3_config =
            tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
                .set_page_size(src3_cb_index, bias_single_tile_size);
        auto cb_src3 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src3_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src3_cb_index,
            bias_single_tile_size,
            in3_CB_size / bias_single_tile_size,
            in3_CB_size);
    }

    // Parameters for last row, col, or block
    uint32_t last_block_h = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_block_w = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_block_num_nonzero_subblocks_h = (last_block_h - 1) / out_subblock_h + 1;
    uint32_t last_block_num_nonzero_subblocks_w = (last_block_w - 1) / out_subblock_w + 1;
    uint32_t last_subblock_of_last_block_h =
        last_block_h % out_subblock_h == 0 ? out_subblock_h : last_block_h % out_subblock_h;
    uint32_t last_subblock_of_last_block_w =
        last_block_w % out_subblock_w == 0 ? out_subblock_w : last_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (per_core_N / out_subblock_w - last_block_num_nonzero_subblocks_w);
    uint32_t last_block_padded_block_tiles_h_skip =
        (per_core_M / out_subblock_h - last_block_num_nonzero_subblocks_h) * (per_core_N * out_subblock_h);

    if (in0_block_sharded) {
        if (in0_noc == NOC::NOC_1) {
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
    const auto& in0_sender_interleaved_cores =
        grid_to_cores(in0_sender_interleaved.start_coord, in0_sender_interleaved.end_coord, true);  // Only used for interleaved in0
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
        if (in0_noc == NOC::NOC_1) {
            std::swap(in0_mcast_start, in0_mcast_end);
        }

        // Assuming in1 is NOC1
        auto in1_mcast_start = bottom_core_physical;
        auto in1_mcast_end = top_core_plus_one_physical;
        if (in1_noc == NOC::NOC_0) {
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
                (std::uint32_t)K * per_core_M * in0_idx,  // in0_tensor_start_tile_id
                // in0 mcast args
                (std::uint32_t)in0_mcast_start.x,  // in0_mcast_dest_noc_start_x
                (std::uint32_t)in0_mcast_start.y,  // in0_mcast_dest_noc_start_y
                (std::uint32_t)in0_mcast_end.x,    // in0_mcast_dest_noc_end_x
                (std::uint32_t)in0_mcast_end.y,    // in0_mcast_dest_noc_end_y
            };
            if (in0_idx == in0_end_idx) {
                // padding args (READER)
                mm_in0_sender_args.push_back(last_block_h);  // last_block_h
            } else {
                mm_in0_sender_args.push_back(per_core_M);
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
                    (std::uint32_t)per_core_N * in1_idx,  // in1_tensor_start_tile_id
                    // in1 mcast args
                    (std::uint32_t)in1_mcast_start.x,  // in1_mcast_dest_noc_start_x
                    (std::uint32_t)in1_mcast_start.y,  // in1_mcast_dest_noc_start_y
                    (std::uint32_t)in1_mcast_end.x,    // in1_mcast_dest_noc_end_x
                    (std::uint32_t)in1_mcast_end.y,    // in1_mcast_dest_noc_end_y

                    // WRITER
                    // out tensor args
                    (std::uint32_t)out_buffer->address(),
                    (std::uint32_t)in1_idx * per_core_N + in0_idx * per_core_M * N  // out_tensor_start_tile_id
                };

                if (in1_idx == in1_end_idx) {
                    // padding args (READER)
                    mm_in1_sender_writer_args.push_back(last_block_w);

                    // padding args (WRITER)
                    mm_in1_sender_writer_args.push_back(per_core_M / out_subblock_h);
                    mm_in1_sender_writer_args.push_back(out_subblock_h);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_sender_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_sender_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_sender_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else {
                    // padding args (READER)
                    mm_in1_sender_writer_args.push_back(per_core_N);

                    // padding args (WRITER)
                    mm_in1_sender_writer_args.push_back(per_core_M / out_subblock_h);
                    mm_in1_sender_writer_args.push_back(out_subblock_h);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(per_core_N / out_subblock_w);
                    mm_in1_sender_writer_args.push_back(out_subblock_w);
                    mm_in1_sender_writer_args.push_back(0);
                    mm_in1_sender_writer_args.push_back(0);
                }

                if (bias_buffer != nullptr) {
                    mm_in1_sender_writer_args.push_back((std::uint32_t)bias_buffer->address());
                    mm_in1_sender_writer_args.push_back(
                        (std::uint32_t)per_core_N * in1_idx);  // in1_tensor_start_tile_id
                } else {
                    mm_in1_sender_writer_args.push_back(0);  // Placeholder; not used
                    mm_in1_sender_writer_args.push_back(0);  // Placeholder; not used
                }

                if (in1_is_sharded and in1_is_dram) {  // in1 is dram sharded
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
                            if (stride >= per_core_N) {
                                stride = per_core_N;
                            }

                            mm_in1_sender_writer_args.push_back(
                                (stride - worker_core_stride) * in1_single_tile_size);  // per_core_N_dram_bytes
                            mm_in1_sender_writer_args.push_back(curr_storage_core);     // current_dram_bank_id

                            log_debug(
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
                    mm_in1_sender_writer_args.insert(mm_in1_sender_writer_args.begin() + 19, num_iter);
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
                    (std::uint32_t)out_buffer->address(),                           // out_tensor_addr
                    (std::uint32_t)in1_idx * per_core_N + in0_idx * per_core_M * N  // out_tensor_start_tile_id
                };

                if (in1_idx == in1_end_idx and in0_idx == in0_end_idx) {
                    // padding args (WRITER)
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else if (in0_idx == in0_end_idx) {
                    // padding args (WRITER)
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_h);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                    mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(0);
                } else if (in1_idx == in1_end_idx) {
                    // padding args (WRITER)
                    mm_in1_receiver_writer_args.push_back(per_core_M / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                    mm_in1_receiver_writer_args.push_back(last_subblock_of_last_block_w);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                    mm_in1_receiver_writer_args.push_back(last_block_padded_block_tiles_w_skip);
                } else {
                    // padding args (WRITER)
                    mm_in1_receiver_writer_args.push_back(per_core_M / out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(out_subblock_h);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(per_core_N / out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(out_subblock_w);
                    mm_in1_receiver_writer_args.push_back(0);
                    mm_in1_receiver_writer_args.push_back(0);
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

    auto override_runtime_arguments_callback =
        [mm_kernel_in0_sender_id,
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
         cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            TT_ASSERT(input_tensors.size() + optional_input_tensors.size() == 3);
            TT_ASSERT(output_tensors.size() == 1);

            auto src_buffer_a = input_tensors.at(0).buffer();
            auto src_buffer_b = input_tensors.at(1).buffer();
            auto bias_tensor = optional_input_tensors.at(0);

            auto dst_buffer = output_tensors.at(0).buffer();

            bool src0_sharded = input_tensors[0].memory_config().is_sharded();
            bool out_sharded = output_tensors[0].memory_config().is_sharded();

            std::optional<Buffer*> bias_buffer;
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
                writer_runtime_args[6] = dst_buffer->address();
                if (bias_tensor.has_value()) {
                    writer_runtime_args[16] = (*bias_buffer)->address();
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
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace reuse_mcast_optimized_helpers

namespace ttnn {

namespace operations {

namespace matmul {

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized_(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor> bias,
    Tensor& output,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> &fused_op_signaler) {
    const auto &ashape = a.get_legacy_shape(), bshape = b.get_legacy_shape();

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());          // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());          // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());  // output

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;  // bias; doesn't matter if bias=nullptr
    if (bias.has_value()) {
        auto& c = bias.value();
        TT_FATAL(c.storage_type() == StorageType::DEVICE);
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

        bias_buffer = c.buffer();

        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.get_dtype());
    }

    tt_metal::Device* device = a.device();

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();
    TT_FATAL(in0_buffer->size() % in0_single_tile_size == 0);
    TT_FATAL(in1_buffer->size() % in1_single_tile_size == 0);

    TT_FATAL(
        ashape[-1] == bshape[-2] &&
        "Dimension K (A.shape[-1] and B.shape[-2]) must match for A and B in bmm_op");  // A.K == B.K
    TT_FATAL(ashape[-2] % TILE_HEIGHT == 0);
    TT_FATAL(ashape[-1] % TILE_WIDTH == 0);
    TT_FATAL(bshape[-2] % TILE_HEIGHT == 0);
    TT_FATAL(bshape[-1] % TILE_WIDTH == 0);

    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;
    bool packer_l1_acc;

    std::visit(
        [&](auto&& compute_kernel_config) {
            using T = std::decay_t<decltype(compute_kernel_config)>;
            if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
                TT_FATAL(device->arch() == ARCH::GRAYSKULL, "kernel config is not for graykull");
                math_fidelity = compute_kernel_config.math_fidelity;
                math_approx_mode = compute_kernel_config.math_approx_mode;
                fp32_dest_acc_en = false;
                packer_l1_acc = false;
            } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
                TT_FATAL(ttnn::device::is_wormhole_or_blackhole(device->arch()), "kernel config is not for wormhole_b0 or blackhole");
                math_fidelity = compute_kernel_config.math_fidelity;
                math_approx_mode = compute_kernel_config.math_approx_mode;
                fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;
                packer_l1_acc = compute_kernel_config.packer_l1_acc;
            } else {
                TT_FATAL("arch not supported");
            }
        },
        compute_kernel_config);

    if (fp32_dest_acc_en) {
        TT_FATAL(
            out_subblock_h * out_subblock_w <= 4 &&
            "Total number of tiles in a subblock must be less than 4 when in fp32_dest_acc mode");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;

    if (fuse_batch) {
        Mt = B * Mt;
        B = 1;
    }
    TT_FATAL(Kt % in0_block_w == 0);

    // This should allocate a DRAM buffer on the device
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    uint32_t num_blocks_y = (Mt - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (Nt - 1) / per_core_N + 1;
    if (transpose_mcast) {
        std::swap(num_blocks_x, num_blocks_y);
    }

    // TODO: Max used grid can actually exceed mcast receiver grid if in0 is sharded
    // TODO: Move these validates to op validate and properly check for this
    TT_FATAL(
        num_blocks_x <= num_cores_x,
        "Num output blocks along x {} must be smaller than or equal to the number of columns in compute grid {}!",
        num_blocks_x,
        num_cores_x);
    TT_FATAL(
        num_blocks_y <= num_cores_y,
        "Num output blocks along y {} must be smaller than or equal to the number of rows in compute grid {}!",
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
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        transpose_mcast,
        fused_activation,
        in0_buffer,
        in1_buffer,
        bias_buffer,
        out_buffer,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        untilize_out,
        fused_op_signaler);
}

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor> bias,
    Tensor& output_tensor,
    bool broadcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out) {

    tt_metal::Program program{}; /* Create a program */
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> empty_fused_op_signaler;

    return matmul_multi_core_reuse_mcast_2d_optimized_(
        program,
        a,
        b,
        bias,
        output_tensor,
        broadcast_batch,
        compute_with_storage_grid_size,
        compute_kernel_config,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        fuse_batch,
        transpose_mcast,
        fused_activation,
        untilize_out,
        empty_fused_op_signaler);
}

operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_2d_optimized_helper(
    tt_metal::Program& program, /* Take programa as input by reference */
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor> bias,
    Tensor& output_tensor,
    bool broadcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> &fused_op_signaler) {
    return matmul_multi_core_reuse_mcast_2d_optimized_(
        program,
        a,
        b,
        bias,
        output_tensor,
        broadcast_batch,
        compute_with_storage_grid_size,
        compute_kernel_config,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        fuse_batch,
        transpose_mcast,
        fused_activation,
        untilize_out,
        fused_op_signaler);
}

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn
