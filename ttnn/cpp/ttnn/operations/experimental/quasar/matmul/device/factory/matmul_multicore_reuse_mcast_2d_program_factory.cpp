// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/utilities/matmul_utilities.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include "tt-metalium/buffer_types.hpp"

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/experimental/quasar/matmul/shared_with_host/activation_type.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::MeshTensor;
using tt::tt_metal::ProgramDescriptor;

using namespace tt;

namespace ttnn::prim::qsr {

namespace reuse_mcast_optimized_helpers {

// Legacy ProgramDescriptor builder for the active create_descriptor path. No longer called after the
// create_program_artifacts port (the Metal 2.0 builder in the anonymous namespace below replaces it).
// [[maybe_unused]] suppresses -Wunused-function pending removal in a follow-up.
[[maybe_unused]] static ProgramDescriptor create_program_mcast_in0_in1_descriptor(
    tt::tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    bool dst_full_sync_en,
    uint32_t B,
    uint32_t M,
    uint32_t M_per_batch,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    bool transpose_a,
    bool transpose_b,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t in0_block_w,
    uint32_t in0_last_ktile_w,
    uint32_t in0_last_ktile_h,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    const tt::tt_metal::MeshTensor& in0_tensor,
    const tt::tt_metal::MeshTensor& in1_tensor,
    ttsl::optional_reference<const tt::tt_metal::MeshTensor> bias_tensor,
    const tt::tt_metal::MeshTensor& out_tensor,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    bool row_broadcast_bias = true,
    CoreCoord sub_device_start_core = {0, 0}) {
    using namespace tt;
    using tt::tt_metal::TensorMemoryLayout;

    ttsl::optional_reference<const tt_metal::MeshTensor> bias_mesh;
    if (bias_tensor.has_value()) {
        bias_mesh = *bias_tensor;
    }

    // currently only support transpose of the full tile
    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    bool fuse_op = fused_op_signaler.has_value();

    TensorMemoryLayout in0_memory_layout = in0_tensor.memory_config().memory_layout();

    uint32_t num_blocks = K / in0_block_w;

    // Only enable packer l1 accumulation when there are num_blocks > 2, otherwise
    // unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
    // does a spill and reload, so need more than 2 blocks to use l1 acc for packer
    // For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
    bool packer_l1_acc_en = packer_l1_acc && (((bias_mesh.has_value()) && num_blocks > 1) || (num_blocks > 2));

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
    const bool in1_is_width_sharded = in1_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const bool in1_is_height_sharded = in1_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool in1_is_sharded = in1_is_width_sharded || in1_is_height_sharded;
    const bool output_is_sharded = out_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    // Tiles whose size is not a multiple of the DRAM alignment (e.g. bfp8 32x16 = 544B on
    // Blackhole's 64B alignment) are padded to it in DRAM. The interleaved reader copies tiles at
    // the padded stride, so the in0/in1/bias CBs must hold pages at the aligned stride and the
    // reader/unpacker walk tiles at the same stride. No-op when already aligned (all bf16 tiles,
    // 32-wide bfp8, Wormhole). Replaces the staging-CB workaround. Sharded CBs are backed by the
    // tensor buffer and keep their natural page size.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size =
        in1_is_sharded ? in1_single_tile_size : tt::align(in1_single_tile_size, dram_alignment);

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
        in0_CB_tiles *= operations::experimental::quasar::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_aligned_tile_size;
    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles *= operations::experimental::quasar::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_aligned_tile_size;

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
        in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_width();
        in0_shard_height_in_tiles = in0_tensor.shard_spec()->shape[0] / in0_tile.get_height();
        in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    }

    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    uint32_t start_core_x = sub_device_start_core.x;
    uint32_t start_core_y = sub_device_start_core.y;

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
        CoreCoord in0_shard_grid = in0_tensor.shard_spec()->grid.bounding_box().grid_size();
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
            in0_mcast_receiver_grid_diff_coord_start =
                device->worker_core_from_logical_core({start_core_x, start_core_y}).y;
            in0_mcast_receiver_grid_diff_coord_end =
                device->worker_core_from_logical_core({start_core_x, start_core_y + num_blocks_x - 1}).y;
            in0_mcast_noc_y.reserve(in0_sender_num_cores_along_width);
            for (uint32_t core_idx_y = 0; core_idx_y < in0_sender_num_cores_along_width; ++core_idx_y) {
                in0_mcast_noc_y.push_back(
                    device->worker_core_from_logical_core({start_core_x, start_core_y + core_idx_y}).y);
            }
        } else {
            in0_mcast_receiver_grid_diff_coord_start =
                device->worker_core_from_logical_core({start_core_x, start_core_y}).x;
            in0_mcast_receiver_grid_diff_coord_end =
                device->worker_core_from_logical_core({start_core_x + num_blocks_x - 1, start_core_y}).x;
            in0_mcast_noc_x.reserve(in0_sender_num_cores_along_width);
            for (uint32_t core_idx_x = 0; core_idx_x < in0_sender_num_cores_along_width; ++core_idx_x) {
                in0_mcast_noc_x.push_back(
                    device->worker_core_from_logical_core({start_core_x + core_idx_x, start_core_y}).x);
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

    // Mcast args — semaphore IDs assigned sequentially (0, 1, 2, 3)
    uint32_t in0_mcast_sender_semaphore_id = 0;
    uint32_t in0_mcast_receiver_semaphore_id = 1;
    uint32_t in1_mcast_sender_semaphore_id = 2;
    uint32_t in1_mcast_receiver_semaphore_id = 3;

    bool in1_is_dram = in1_tensor.mesh_buffer().device_local_config().buffer_type == tt_metal::BufferType::DRAM;

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
    uint32_t batches_per_bank = 0;
    if (in1_is_sharded and in1_is_dram) {
        num_dram_banks = device->num_dram_channels();
        if (in1_is_width_sharded) {
            per_core_N_storage = (N + num_dram_banks - 1) / num_dram_banks;
        } else {
            // Height sharded: batches are distributed across DRAM banks
            uint32_t in1_shard_height_in_tiles = in1_tensor.shard_spec()->shape[0] / in1_tile.get_height();
            batches_per_bank = in1_shard_height_in_tiles / K;
        }
    }

    const auto [in0_tensor_stride_w, in0_tensor_stride_h] =
        operations::experimental::quasar::matmul::utilities::get_in0_transpose_strides(M, M_per_batch, transpose_a, K);
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
            (std::uint32_t)in0_last_ktile_h,

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
            (std::uint32_t)in0_last_ktile_h,

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
            (std::uint32_t)B,      // batch
            (std::uint32_t)false,  // reuse_in0_in_CB

            // sparsity args
            (std::uint32_t)0,      // batchB
            (std::uint32_t)0,      // sparsity_pagesize (placeholder since sparsity not used in this case)
            (std::uint32_t)true,   // bcast_A
            (std::uint32_t)false,  // get_batch_from_reader
        };
    }
    in0_sender_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_all_gather()));
    tt::tt_metal::TensorAccessorArgs(in0_tensor).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity
    in0_sender_compile_time_args.push_back((std::uint32_t)0);  // num_batch_compute (unused, sparsity disabled)

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
    if (bias_mesh.has_value()) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);  // in3_tensor_stride_w
    } else {
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }

    in1_sender_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_all_gather()));
    in1_sender_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_reduce_scatter()));

    // Append TensorAccessorArgs
    tt::tt_metal::TensorAccessorArgs(in1_tensor).append_to(in1_sender_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in1_sender_writer_compile_time_args);  // placeholder for sparsity
    tt::tt_metal::TensorAccessorArgs(out_tensor).append_to(in1_sender_writer_compile_time_args);
    if (bias_mesh.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_mesh).append_to(in1_sender_writer_compile_time_args);
    }

    if (in1_is_sharded and in1_is_dram) {
        if (in1_is_width_sharded) {
            in1_sender_writer_compile_time_args.push_back((std::uint32_t)per_core_N_storage * in0_block_w);
            in1_sender_writer_compile_time_args.push_back((std::uint32_t)per_core_N_storage * in1_single_tile_size);
        } else {
            // Height sharded: pass tiles per batch and batches per bank
            in1_sender_writer_compile_time_args.push_back((std::uint32_t)(K * N));  // KtNt per batch (tiles)
            in1_sender_writer_compile_time_args.push_back((std::uint32_t)batches_per_bank);
        }
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
    if (bias_mesh.has_value()) {
        in1_receiver_writer_compile_time_args.push_back((std::uint32_t)in1_block_w);
    } else {
        in1_receiver_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }
    in1_receiver_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_reduce_scatter()));
    tt::tt_metal::TensorAccessorArgs(out_tensor).append_to(in1_receiver_writer_compile_time_args);

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_sharded_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_interleaved_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_receiver_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_receiver_writer_other_noc_setup_defines;
    if (bias_mesh.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_other_noc_setup_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines["SFPU_ACTIVATION"] = "1";
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
            if (in1_is_width_sharded) {
                mm_kernel_in1_sender_writer_defines["IN1_DRAM_WIDTH_SHARDED"] = "1";
            } else {
                mm_kernel_in1_sender_writer_defines["IN1_DRAM_HEIGHT_SHARDED"] = "1";
            }
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

    // Helper to convert std::map defines to KernelDescriptor::Defines (vector of pairs)
    auto map_to_defines = [](const std::map<std::string, std::string>& m) -> KernelDescriptor::Defines {
        KernelDescriptor::Defines result;
        result.reserve(m.size());
        for (const auto& [k, v] : m) {
            result.emplace_back(k, v);
        }
        return result;
    };

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    tt_metal::NOC in0_split_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());
    tt_metal::NOC in1_split_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    ////////////////////////////////////////////////////////////////////////////
    //                      Build Kernel Descriptors
    ////////////////////////////////////////////////////////////////////////////
    // We build kernel descriptors as local variables, populate their runtime_args
    // in the per-core loop, then push them all to desc.kernels at the end.
    // The order they are pushed determines the kernel handle index.

    // Kernel index tracking:
    // Index 0: mm_kernel_in0_sender (block sharded or interleaved)
    // Index 1: mm_kernel_in0_mcast_cores_without_work (only if in0_block_sharded && extra cores exist)
    //   OR: (not created if not needed)
    // Then: mm_kernel_in1_sender_writer
    // Then: mm_kernel_in1_receiver_writer (if in1_receiver.num_cores() > 0)
    // Then: mm_kernel_in0_receiver (if !in0_block_sharded && in0_receiver_interleaved.num_cores() > 0)
    // Then: mm_kernel_in1_receiver_writer_other_noc_setup (if split_half other cores exist)
    // Then: mm_kernel_in0_receiver_other_noc_setup (if split_half other cores exist)
    // Then: compute kernel

    KernelDescriptor in0_sender_kernel_desc;
    KernelDescriptor in0_mcast_no_work_kernel_desc;
    bool has_in0_mcast_no_work_kernel = false;
    KernelDescriptor in1_sender_writer_kernel_desc;
    KernelDescriptor in1_receiver_writer_kernel_desc;
    bool has_in1_receiver_writer_kernel = false;
    KernelDescriptor in0_receiver_kernel_desc;
    bool has_in0_receiver_kernel = false;
    KernelDescriptor in1_receiver_writer_other_kernel_desc;
    bool has_in1_receiver_writer_other_kernel = false;
    KernelDescriptor in0_receiver_other_kernel_desc;
    bool has_in0_receiver_other_kernel = false;

    if (in0_block_sharded) {
        in0_sender_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp";
        in0_sender_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        in0_sender_kernel_desc.core_ranges = CoreRangeSet(all_cores_with_work);
        in0_sender_kernel_desc.compile_time_args = in0_sender_compile_time_args;
        in0_sender_kernel_desc.defines = map_to_defines(mm_kernel_in0_sender_sharded_defines);
        in0_sender_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in0_sharded", tt::CBIndex::c_2},
            {"cb_l1_array", tt::CBIndex::c_6},
        };
        in0_sender_kernel_desc.config =
            DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};

        if (in0_mcast_cores_without_work_and_not_in_receiver_grid.has_value()) {
            has_in0_mcast_no_work_kernel = true;
            auto no_work_ct_args = in0_sender_compile_time_args;
            no_work_ct_args[0] = 0;  // core_has_output_block_work
            no_work_ct_args[1] = 0;  // core_in_in0_receiver_mcast_grid
            in0_mcast_no_work_kernel_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp";
            in0_mcast_no_work_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            in0_mcast_no_work_kernel_desc.core_ranges =
                CoreRangeSet(in0_mcast_cores_without_work_and_not_in_receiver_grid.value());
            in0_mcast_no_work_kernel_desc.compile_time_args = no_work_ct_args;
            in0_mcast_no_work_kernel_desc.defines = map_to_defines(mm_kernel_in0_sender_sharded_defines);
            in0_mcast_no_work_kernel_desc.named_compile_time_args = {
                {"cb_in0", tt::CBIndex::c_0},
                {"cb_in0_sharded", tt::CBIndex::c_2},
                {"cb_l1_array", tt::CBIndex::c_6},
            };
            in0_mcast_no_work_kernel_desc.config =
                DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};
        }
    } else {
        // NOTE: fused_op_signaler init_fused_op() calls are NOT translated to the descriptor
        // because they modify the Program directly. These are handled when the Program is
        // constructed from the descriptor in create_program_mcast_in0_in1().

        in0_sender_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_sender_padding.cpp";
        in0_sender_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        in0_sender_kernel_desc.core_ranges = CoreRangeSet(in0_sender_interleaved);
        in0_sender_kernel_desc.compile_time_args = in0_sender_compile_time_args;
        in0_sender_kernel_desc.defines = map_to_defines(mm_kernel_in0_sender_interleaved_defines);
        in0_sender_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in0_sharded", tt::CBIndex::c_2},
            {"cb_sparsity", tt::CBIndex::c_6},
        };
        in0_sender_kernel_desc.config =
            DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};
    }

    in1_sender_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_writer_padding.cpp";
    in1_sender_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    in1_sender_writer_kernel_desc.core_ranges = CoreRangeSet(in1_sender);
    in1_sender_writer_kernel_desc.compile_time_args = in1_sender_writer_compile_time_args;
    in1_sender_writer_kernel_desc.defines = map_to_defines(mm_kernel_in1_sender_writer_defines);
    in1_sender_writer_kernel_desc.named_compile_time_args = {
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_sparsity", tt::CBIndex::c_7},
    };
    in1_sender_writer_kernel_desc.config =
        DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc};

    if (in1_receiver.num_cores() > 0) {
        has_in1_receiver_writer_kernel = true;
        in1_receiver_writer_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp";
        in1_receiver_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        in1_receiver_writer_kernel_desc.core_ranges = in1_receiver;
        in1_receiver_writer_kernel_desc.compile_time_args = in1_receiver_writer_compile_time_args;
        in1_receiver_writer_kernel_desc.defines = map_to_defines(mm_kernel_in1_receiver_writer_defines);
        in1_receiver_writer_kernel_desc.named_compile_time_args = {
            {"cb_in1", tt::CBIndex::c_1},
            {"cb_bias", tt::CBIndex::c_3},
            {"cb_out", tt::CBIndex::c_4},
        };
        in1_receiver_writer_kernel_desc.config =
            DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc};
    }

    if (!in0_block_sharded and in0_receiver_interleaved.num_cores() > 0) {
        has_in0_receiver_kernel = true;
        in0_receiver_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_receiver.cpp";
        in0_receiver_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        in0_receiver_kernel_desc.core_ranges = in0_receiver_interleaved;
        in0_receiver_kernel_desc.compile_time_args = in0_receiver_compile_time_args;
        in0_receiver_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
        };
        in0_receiver_kernel_desc.config =
            DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc};
    }

    if (in0_receiver_in1_receiver_interleaved_other_cores.has_value()) {
        has_in1_receiver_writer_other_kernel = true;
        in1_receiver_writer_other_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp";
        in1_receiver_writer_other_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        in1_receiver_writer_other_kernel_desc.core_ranges =
            CoreRangeSet(in0_receiver_in1_receiver_interleaved_other_cores.value());
        in1_receiver_writer_other_kernel_desc.compile_time_args = in1_receiver_writer_compile_time_args;
        in1_receiver_writer_other_kernel_desc.defines =
            map_to_defines(mm_kernel_in1_receiver_writer_other_noc_setup_defines);
        in1_receiver_writer_other_kernel_desc.named_compile_time_args = {
            {"cb_in1", tt::CBIndex::c_1},
            {"cb_bias", tt::CBIndex::c_3},
            {"cb_out", tt::CBIndex::c_4},
        };
        in1_receiver_writer_other_kernel_desc.config =
            DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_split_noc};

        has_in0_receiver_other_kernel = true;
        in0_receiver_other_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_receiver.cpp";
        in0_receiver_other_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        in0_receiver_other_kernel_desc.core_ranges =
            CoreRangeSet(in0_receiver_in1_receiver_interleaved_other_cores.value());
        in0_receiver_other_kernel_desc.compile_time_args = in0_receiver_compile_time_args;
        in0_receiver_other_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
        };
        in0_receiver_other_kernel_desc.config =
            DataMovementConfigDescriptor{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_split_noc};
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
    if (bias_mesh.has_value()) {
        compute_kernel_args.push_back(row_broadcast_bias ? 1u : 0u);
    }

    // Create compute kernel descriptor
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/compute/"
        "bmm_large_block_zm_fused_bias_activation.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = CoreRangeSet(all_cores_with_work);
    compute_kernel_desc.compile_time_args = compute_kernel_args;
    compute_kernel_desc.defines = map_to_defines(mm_kernel_defines);
    {
        KernelDescriptor::NamedCompileTimeArgs named_compile_args = {
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in1", tt::CBIndex::c_1},
            {"cb_bias", tt::CBIndex::c_3},
            {"cb_out", tt::CBIndex::c_4},
            {"cb_intermed0", tt::CBIndex::c_5},
            {"cb_in0_transposed", tt::CBIndex::c_10},
            {"bias_ntiles", in1_per_core_w},
        };
        if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
            using ttnn::operations::experimental::quasar::matmul::utilities::get_activation_params;
            const auto params = get_activation_params(fused_activation.value());
            named_compile_args.push_back({"activation_type", static_cast<uint32_t>(params.type)});
            named_compile_args.push_back({"activation_param0", params.param0});
            named_compile_args.push_back({"activation_param1", params.param1});
            named_compile_args.push_back({"activation_param2", params.param2});
        }
        compute_kernel_desc.named_compile_time_args = std::move(named_compile_args);
    }
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    tt::tt_metal::TileDescriptor in0_tile_desc{in0_tile};
    tt::tt_metal::TileDescriptor in1_tile_desc{in1_tile};
    tt::tt_metal::TileDescriptor bias_tile_desc{bias_tile};
    tt::tt_metal::TileDescriptor output_tile_desc{output_tile};

    // CB 0: in0
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in0_CB_size;
        cb_desc.core_ranges = CoreRangeSet(all_cores);
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = in0_data_format,
            .page_size = in0_aligned_tile_size,
            .tile = in0_tile_desc});
        if (in0_height_sharded) {
            cb_desc.tensor = &in0_tensor;
        }
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 1: in1
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in1_CB_size;
        cb_desc.core_ranges = CoreRangeSet(all_cores);
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = in1_data_format,
            .page_size = in1_aligned_tile_size,
            .tile = in1_tile_desc});
        if (in1_is_sharded and not in1_is_dram) {
            cb_desc.tensor = &in1_tensor;
        }
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 2: in0 sharded (only for block sharded)
    if (in0_block_sharded) {
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = in2_CB_size;
            cb_desc.core_ranges = CoreRangeSet(all_cores);
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_2,
                .data_format = in0_data_format,
                .page_size = in0_single_tile_size,
                .tile = in0_tile_desc});
            cb_desc.tensor = &in0_tensor;
            desc.cbs.push_back(std::move(cb_desc));
        }

        // Local L1 to store temp vars
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = 32 * 2;
            cb_desc.core_ranges = CoreRangeSet(all_cores);
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_6, .data_format = tt::DataFormat::Float16_b, .page_size = 32 * 2});
            desc.cbs.push_back(std::move(cb_desc));
        }
    }

    // CB 4 and CB 5: output and intermediate
    if (do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
        (untilize_out && (in1_num_subblocks > 1))) {
        // Separate output and intermediate CBs
        // output
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = out_CB_size;
            cb_desc.core_ranges = CoreRangeSet({all_cores});
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_4,
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = output_tile_desc});
            if (output_is_sharded) {
                cb_desc.tensor = &out_tensor;
            }
            desc.cbs.push_back(std::move(cb_desc));
        }
        // interm0
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = interm0_CB_size;
            cb_desc.core_ranges = CoreRangeSet({all_cores});
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_5,
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = output_tile_desc});
            desc.cbs.push_back(std::move(cb_desc));
        }
    } else {
        // share buffer
        CBDescriptor cb_desc;
        cb_desc.total_size = out_CB_size;
        cb_desc.core_ranges = CoreRangeSet({all_cores});
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_4,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_5,
            .data_format = interm0_data_format,
            .page_size = interm0_single_tile_size,
            .tile = output_tile_desc});
        if (output_is_sharded) {
            cb_desc.tensor = &out_tensor;
        }
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB for bias
    if (bias_mesh.has_value()) {
        CBDescriptor cb_desc;
        cb_desc.total_size = in3_CB_size;
        cb_desc.core_ranges = CoreRangeSet(all_cores);
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_3,
            .data_format = bias_data_format,
            .page_size = bias_single_tile_size,
            .tile = bias_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // Intermediate CB read

    if (in0_transpose_tile) {
        CBDescriptor cb_desc;
        cb_desc.total_size = in0_CB_size;
        cb_desc.core_ranges = CoreRangeSet(all_cores);
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_10,
            .data_format = in0_data_format,
            .page_size = in0_aligned_tile_size,
            .tile = in0_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Semaphore Descriptors
    ////////////////////////////////////////////////////////////////////////////
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = in0_mcast_sender_semaphore_id, .core_ranges = CoreRangeSet(all_cores), .initial_value = INVALID});
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = in0_mcast_receiver_semaphore_id, .core_ranges = CoreRangeSet(all_cores), .initial_value = INVALID});
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = in1_mcast_sender_semaphore_id, .core_ranges = CoreRangeSet(all_cores), .initial_value = INVALID});
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = in1_mcast_receiver_semaphore_id, .core_ranges = CoreRangeSet(all_cores), .initial_value = INVALID});

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime Args (per-core loop)
    ////////////////////////////////////////////////////////////////////////////
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
                in0_sender_kernel_desc.runtime_args.emplace_back(core, mm_in0_sender_args);
            } else {
                in0_mcast_no_work_kernel_desc.runtime_args.emplace_back(core, mm_in0_sender_args);
            }
        } else if (in1_idx == 0) {
            std::vector<uint32_t> mm_in0_sender_args = {
                // in0 tensor args
                (std::uint32_t)in0_tensor.address(),
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

            {
                std::vector<std::variant<uint32_t, std::reference_wrapper<const tt::tt_metal::MeshTensor>>> in0_args(
                    mm_in0_sender_args.begin(), mm_in0_sender_args.end());
                in0_args[0] = in0_tensor;
                in0_sender_kernel_desc.emplace_runtime_args(core, in0_args);
            }

            // in0 receiver
        } else {
            std::vector<uint32_t> mm_in0_receiver_args = {
                // in0 mcast args
                (std::uint32_t)in0_mcast_sender.x,  // in0_mcast_sender_noc_x
                (std::uint32_t)in0_mcast_sender.y   // in0_mcast_sender_noc_y
            };
            // left half
            if ((core.x - start_core_x) <= half_core || (!transpose_mcast and core.y == start_core_y)) {
                in0_receiver_kernel_desc.runtime_args.emplace_back(core, mm_in0_receiver_args);
            }
            // right half
            else {
                in0_receiver_other_kernel_desc.runtime_args.emplace_back(core, mm_in0_receiver_args);
            }
        }

        if (in0_idx < num_blocks_y and in1_idx < num_blocks_x) {
            // in1 sender
            if (in0_idx == 0) {
                std::vector<uint32_t> mm_in1_sender_writer_args = {
                    // READER
                    // in1 tensor args
                    (std::uint32_t)in1_tensor.address(),
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
                    (std::uint32_t)out_tensor.address(),
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

                mm_in1_sender_writer_args.push_back(
                    bias_mesh.has_value() ? (std::uint32_t)bias_mesh->address() : 0);  // smuggled-rta-ok
                mm_in1_sender_writer_args.push_back(
                    bias_mesh.has_value() ? (std::uint32_t)per_core_N * in1_idx : 0);  // in1_tensor_start_tile_id
                if (!output_is_sharded) {
                    if (in1_idx == in1_end_idx) {  // right cores when no transpose_mcast
                        mm_in1_sender_writer_args.push_back(last_out_num_blocks_w);
                    } else {
                        mm_in1_sender_writer_args.push_back(out_num_blocks_x);
                    }
                }

                if (in1_is_sharded and in1_is_dram) {  // in1 is dram sharded
                    if (in1_is_width_sharded) {
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
                    } else {
                        // Height sharded: no additional runtime args needed
                        // (bank/offset computed from compile-time args + batch index)
                    }
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
                {
                    std::vector<std::variant<uint32_t, std::reference_wrapper<const tt::tt_metal::MeshTensor>>>
                        in1_sender_variant(mm_in1_sender_writer_args.begin(), mm_in1_sender_writer_args.end());
                    in1_sender_variant[0] = in1_tensor;
                    in1_sender_variant[7] = out_tensor;
                    if (bias_mesh.has_value()) {
                        in1_sender_variant[18] = *bias_mesh;
                    }
                    in1_sender_writer_kernel_desc.emplace_runtime_args(core, in1_sender_variant);
                }

                // in1 receiver
            } else {
                std::vector<uint32_t> mm_in1_receiver_writer_args = {
                    // READER
                    // in1 mcast args
                    (std::uint32_t)in1_mcast_sender.x,  // in1_mcast_sender_noc_x
                    (std::uint32_t)in1_mcast_sender.y,  // in1_mcast_sender_noc_y

                    // WRITER
                    // out tensor args
                    (std::uint32_t)out_tensor.address(),                                // out_tensor_addr
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

                {
                    std::vector<std::variant<uint32_t, std::reference_wrapper<const tt::tt_metal::MeshTensor>>>
                        in1_recv_variant(mm_in1_receiver_writer_args.begin(), mm_in1_receiver_writer_args.end());
                    in1_recv_variant[2] = out_tensor;
                    // left half
                    if ((core.x - start_core_x) <= half_core || (transpose_mcast and core.y == start_core_y)) {
                        in1_receiver_writer_kernel_desc.emplace_runtime_args(core, in1_recv_variant);
                    }
                    // right half
                    else {
                        in1_receiver_writer_other_kernel_desc.emplace_runtime_args(core, in1_recv_variant);
                    }
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Push Kernels to Descriptor
    ////////////////////////////////////////////////////////////////////////////
    // Order matters — determines kernel handle indices
    desc.kernels.push_back(std::move(in0_sender_kernel_desc));
    if (has_in0_mcast_no_work_kernel) {
        desc.kernels.push_back(std::move(in0_mcast_no_work_kernel_desc));
    }
    desc.kernels.push_back(std::move(in1_sender_writer_kernel_desc));
    if (has_in1_receiver_writer_kernel) {
        desc.kernels.push_back(std::move(in1_receiver_writer_kernel_desc));
    }
    if (has_in0_receiver_kernel) {
        desc.kernels.push_back(std::move(in0_receiver_kernel_desc));
    }
    if (has_in1_receiver_writer_other_kernel) {
        desc.kernels.push_back(std::move(in1_receiver_writer_other_kernel_desc));
    }
    if (has_in0_receiver_other_kernel) {
        desc.kernels.push_back(std::move(in0_receiver_other_kernel_desc));
    }
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

ttnn::device_operation::CachedProgram<MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t>
create_program_mcast_in0_in1(
    tt::tt_metal::Program& program,
    tt::tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    bool dst_full_sync_en,
    uint32_t B,
    uint32_t M,
    uint32_t M_per_batch,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    bool transpose_a,
    bool transpose_b,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t in0_block_w,
    uint32_t in0_last_ktile_w,
    uint32_t in0_last_ktile_h,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    const tt::tt_metal::MeshTensor& in0_tensor,
    const tt::tt_metal::MeshTensor& in1_tensor,
    ttsl::optional_reference<const tt::tt_metal::MeshTensor> bias_tensor,
    const tt::tt_metal::MeshTensor& out_tensor,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    bool row_broadcast_bias = true,
    CoreCoord sub_device_start_core = {0, 0}) {
    using namespace tt;
    using tt::tt_metal::TensorMemoryLayout;

    ttsl::optional_reference<const tt_metal::MeshTensor> bias_mesh;
    if (bias_tensor.has_value()) {
        bias_mesh = *bias_tensor;
    }

    // currently only support transpose of the full tile
    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    bool fuse_op = fused_op_signaler.has_value();

    TensorMemoryLayout in0_memory_layout = in0_tensor.memory_config().memory_layout();

    uint32_t num_blocks = K / in0_block_w;

    // Only enable packer l1 accumulation when there are num_blocks > 2, otherwise
    // unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
    // does a spill and reload, so need more than 2 blocks to use l1 acc for packer
    // For bias, last iteration of l1 acc remains in intermediate buffer, does not spill and reload
    bool packer_l1_acc_en = packer_l1_acc && (((bias_mesh.has_value()) && num_blocks > 1) || (num_blocks > 2));

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
    const bool in1_is_width_sharded = in1_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const bool in1_is_height_sharded = in1_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool in1_is_sharded = in1_is_width_sharded || in1_is_height_sharded;
    const bool output_is_sharded = out_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    TT_FATAL(
        !(output_is_sharded && B > 1),
        "Block-sharded output is incompatible with batch > 1 (B={}). The output CB is backed by the shard buffer "
        "which only holds per_core_M * per_core_N = {} tiles, but the kernel would produce B * per_core_M * per_core_N "
        "= {} tiles without draining. Use fuse_batch=True.",
        B,
        per_core_M * per_core_N,
        B * per_core_M * per_core_N);
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
        in0_CB_tiles *= operations::experimental::quasar::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    // Tiles whose size is not a multiple of the DRAM alignment (e.g. bfp8 32x16 = 544B on
    // Blackhole's 64B alignment) are padded to it in DRAM. The interleaved reader copies tiles at
    // the padded stride, so the in0/in1/bias CBs must hold pages at the aligned stride and the
    // reader/unpacker walk tiles at the same stride. No-op when already aligned (all bf16 tiles,
    // 32-wide bfp8, Wormhole). Replaces the staging-CB workaround. Sharded CBs are backed by the
    // tensor buffer and keep their natural page size.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size =
        in1_is_sharded ? in1_single_tile_size : tt::align(in1_single_tile_size, dram_alignment);
    uint32_t in0_CB_size = in0_CB_tiles * in0_aligned_tile_size;
    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles *= operations::experimental::quasar::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_aligned_tile_size;

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
        in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_width();
        in0_shard_height_in_tiles = in0_tensor.shard_spec()->shape[0] / in0_tile.get_height();
        in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    }

    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    uint32_t start_core_x = sub_device_start_core.x;
    uint32_t start_core_y = sub_device_start_core.y;

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
        CoreCoord in0_shard_grid = in0_tensor.shard_spec()->grid.bounding_box().grid_size();
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
            in0_mcast_receiver_grid_diff_coord_start =
                device->worker_core_from_logical_core({start_core_x, start_core_y}).y;
            in0_mcast_receiver_grid_diff_coord_end =
                device->worker_core_from_logical_core({start_core_x, start_core_y + num_blocks_x - 1}).y;
            in0_mcast_noc_y.reserve(in0_sender_num_cores_along_width);
            for (uint32_t core_idx_y = 0; core_idx_y < in0_sender_num_cores_along_width; ++core_idx_y) {
                in0_mcast_noc_y.push_back(
                    device->worker_core_from_logical_core({start_core_x, start_core_y + core_idx_y}).y);
            }
        } else {
            in0_mcast_receiver_grid_diff_coord_start =
                device->worker_core_from_logical_core({start_core_x, start_core_y}).x;
            in0_mcast_receiver_grid_diff_coord_end =
                device->worker_core_from_logical_core({start_core_x + num_blocks_x - 1, start_core_y}).x;
            in0_mcast_noc_x.reserve(in0_sender_num_cores_along_width);
            for (uint32_t core_idx_x = 0; core_idx_x < in0_sender_num_cores_along_width; ++core_idx_x) {
                in0_mcast_noc_x.push_back(
                    device->worker_core_from_logical_core({start_core_x + core_idx_x, start_core_y}).x);
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

    bool in1_is_dram = in1_tensor.mesh_buffer().device_local_config().buffer_type == tt_metal::BufferType::DRAM;

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
    uint32_t batches_per_bank = 0;
    if (in1_is_sharded and in1_is_dram) {
        num_dram_banks = device->num_dram_channels();
        if (in1_is_width_sharded) {
            per_core_N_storage = (N + num_dram_banks - 1) / num_dram_banks;
        } else {
            // Height sharded: batches are distributed across DRAM banks
            uint32_t in1_shard_height_in_tiles = in1_tensor.shard_spec()->shape[0] / in1_tile.get_height();
            batches_per_bank = in1_shard_height_in_tiles / K;
        }
    }

    const auto [in0_tensor_stride_w, in0_tensor_stride_h] =
        operations::experimental::quasar::matmul::utilities::get_in0_transpose_strides(M, M_per_batch, transpose_a, K);
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
            (std::uint32_t)in0_last_ktile_h,

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
            (std::uint32_t)in0_last_ktile_h,

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
            (std::uint32_t)B,      // batch
            (std::uint32_t)false,  // reuse_in0_in_CB

            // sparsity args
            (std::uint32_t)0,      // batchB
            (std::uint32_t)0,      // sparsity_pagesize (placeholder since sparsity not used in this case)
            (std::uint32_t)true,   // bcast_A
            (std::uint32_t)false,  // get_batch_from_reader
        };
    }
    in0_sender_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_all_gather()));
    tt::tt_metal::TensorAccessorArgs(in0_tensor).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity
    in0_sender_compile_time_args.push_back((std::uint32_t)0);  // num_batch_compute (unused, sparsity disabled)

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
    if (bias_mesh.has_value()) {
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);  // in3_tensor_stride_w
    } else {
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }

    in1_sender_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_all_gather()));
    in1_sender_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_reduce_scatter()));

    // Append TensorAccessorArgs
    tt::tt_metal::TensorAccessorArgs(in1_tensor).append_to(in1_sender_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in1_sender_writer_compile_time_args);  // placeholder for sparsity
    tt::tt_metal::TensorAccessorArgs(out_tensor).append_to(in1_sender_writer_compile_time_args);
    if (bias_mesh.has_value()) {
        tt::tt_metal::TensorAccessorArgs(*bias_mesh).append_to(in1_sender_writer_compile_time_args);
    }

    if (in1_is_sharded and in1_is_dram) {
        if (in1_is_width_sharded) {
            in1_sender_writer_compile_time_args.push_back((std::uint32_t)per_core_N_storage * in0_block_w);
            in1_sender_writer_compile_time_args.push_back((std::uint32_t)per_core_N_storage * in1_single_tile_size);
        } else {
            // Height sharded: pass tiles per batch and batches per bank
            in1_sender_writer_compile_time_args.push_back((std::uint32_t)(K * N));  // KtNt per batch (tiles)
            in1_sender_writer_compile_time_args.push_back((std::uint32_t)batches_per_bank);
        }
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
    if (bias_mesh.has_value()) {
        in1_receiver_writer_compile_time_args.push_back((std::uint32_t)in1_block_w);
    } else {
        in1_receiver_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }
    in1_receiver_writer_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_reduce_scatter()));
    tt::tt_metal::TensorAccessorArgs(out_tensor).append_to(in1_receiver_writer_compile_time_args);

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_sharded_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_interleaved_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_receiver_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_receiver_writer_other_noc_setup_defines;
    if (bias_mesh.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_other_noc_setup_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines["SFPU_ACTIVATION"] = "1";
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
            if (in1_is_width_sharded) {
                mm_kernel_in1_sender_writer_defines["IN1_DRAM_WIDTH_SHARDED"] = "1";
            } else {
                mm_kernel_in1_sender_writer_defines["IN1_DRAM_HEIGHT_SHARDED"] = "1";
            }
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
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",
            all_cores_with_work,  // in0_mcast_cores_with_work_and_in_receiver_grid
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_sender_compile_time_args,
                .defines = mm_kernel_in0_sender_sharded_defines,
                .named_compile_args = {
                    {"cb_in0", tt::CBIndex::c_0},
                    {"cb_in0_sharded", tt::CBIndex::c_2},
                    {"cb_l1_array", tt::CBIndex::c_6},
                }});
        if (in0_mcast_cores_without_work_and_not_in_receiver_grid.has_value()) {
            in0_sender_compile_time_args[0] = 0;  // core_has_output_block_work
            in0_sender_compile_time_args[1] = 0;  // core_in_in0_receiver_mcast_grid
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",
                in0_mcast_cores_without_work_and_not_in_receiver_grid.value(),
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = in0_noc,
                    .compile_args = in0_sender_compile_time_args,
                    .defines = mm_kernel_in0_sender_sharded_defines,
                    .named_compile_args = {
                        {"cb_in0", tt::CBIndex::c_0},
                        {"cb_in0_sharded", tt::CBIndex::c_2},
                        {"cb_l1_array", tt::CBIndex::c_6},
                    }});
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
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_sender_padding.cpp",
            in0_sender_interleaved,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_sender_compile_time_args,
                .defines = mm_kernel_in0_sender_interleaved_defines,
                .named_compile_args = {
                    {"cb_in0", tt::CBIndex::c_0},
                    {"cb_in0_sharded", tt::CBIndex::c_2},
                    {"cb_sparsity", tt::CBIndex::c_6},
                }});
    }

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        in1_sender,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_kernel_in1_sender_writer_defines,
            .named_compile_args = {
                {"cb_in1", tt::CBIndex::c_1},
                {"cb_bias", tt::CBIndex::c_3},
                {"cb_out", tt::CBIndex::c_4},
                {"cb_sparsity", tt::CBIndex::c_7},
            }});

    tt::tt_metal::KernelHandle mm_kernel_in1_receiver_writer_id = 0;
    if (in1_receiver.num_cores() > 0) {
        mm_kernel_in1_receiver_writer_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
            /* in0_sender_in1_receiver, // If not using half-half noc setup */
            in1_receiver,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_noc,
                .compile_args = in1_receiver_writer_compile_time_args,
                .defines = mm_kernel_in1_receiver_writer_defines,
                .named_compile_args = {
                    {"cb_in1", tt::CBIndex::c_1},
                    {"cb_bias", tt::CBIndex::c_3},
                    {"cb_out", tt::CBIndex::c_4},
                }});
    }

    tt::tt_metal::KernelHandle mm_kernel_in0_receiver_id = 0;
    if (!in0_block_sharded and in0_receiver_interleaved.num_cores() > 0) {
        mm_kernel_in0_receiver_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_receiver.cpp",
            /* in0_receiver_in1_sender, // If not using half-half noc setup */
            in0_receiver_interleaved,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_receiver_compile_time_args,
                .named_compile_args = {
                    {"cb_in0", tt::CBIndex::c_0},
                }});
    }

    tt::tt_metal::KernelHandle mm_kernel_in1_receiver_writer_other_noc_setup_id = mm_kernel_in1_receiver_writer_id;
    tt::tt_metal::KernelHandle mm_kernel_in0_receiver_other_noc_setup_id = mm_kernel_in0_receiver_id;

    if (in0_receiver_in1_receiver_interleaved_other_cores.has_value()) {
        mm_kernel_in1_receiver_writer_other_noc_setup_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
            in0_receiver_in1_receiver_interleaved_other_cores.value(),
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_split_noc,
                .compile_args = in1_receiver_writer_compile_time_args,
                .defines = mm_kernel_in1_receiver_writer_other_noc_setup_defines,
                .named_compile_args = {
                    {"cb_in1", tt::CBIndex::c_1},
                    {"cb_bias", tt::CBIndex::c_3},
                    {"cb_out", tt::CBIndex::c_4},
                }});

        mm_kernel_in0_receiver_other_noc_setup_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in0_receiver.cpp",
            in0_receiver_in1_receiver_interleaved_other_cores.value(),
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_split_noc,
                .compile_args = in0_receiver_compile_time_args,
                .named_compile_args = {
                    {"cb_in0", tt::CBIndex::c_0},
                }});
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
    if (bias_mesh.has_value()) {
        compute_kernel_args.push_back(row_broadcast_bias ? 1u : 0u);
    }

    std::unordered_map<std::string, uint32_t> compute_named_compile_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_intermed0", tt::CBIndex::c_5},
        {"cb_in0_transposed", tt::CBIndex::c_10},
        {"bias_ntiles", in1_per_core_w},
    };

    if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
        using ttnn::operations::experimental::quasar::matmul::utilities::get_activation_params;
        const auto& activation = fused_activation.value();
        const auto params = get_activation_params(activation);
        compute_named_compile_args["activation_type"] = static_cast<uint32_t>(params.type);
        compute_named_compile_args["activation_param0"] = params.param0;
        compute_named_compile_args["activation_param1"] = params.param1;
        compute_named_compile_args["activation_param2"] = params.param2;
    }

    // Create compute kernel
    // bool fp32_dest_acc_en = true;
    // Gelu currently has better accuracy when run in approx mode
    // bool math_approx_mode = false;
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/compute/"
        "bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores_with_work,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines,
            .named_compile_args = compute_named_compile_args});

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_aligned_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    if (in0_height_sharded) {
        src0_cb_config.set_globally_allocated_address(in0_tensor);
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
            .set_page_size(src1_cb_index, in1_aligned_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    if (in1_is_sharded and not in1_is_dram) {
        src1_cb_config.set_globally_allocated_address(in1_tensor);
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
                .set_globally_allocated_address(in0_tensor)
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
        output_cb_config = output_cb_config.set_globally_allocated_address(out_tensor);
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
    if (bias_mesh.has_value()) {
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

    if (in0_transpose_tile) {
        uint32_t in0_transpose_cb_index = tt::CBIndex::c_10;
        auto in0_transpose_cb_config =
            tt_metal::CircularBufferConfig(in0_CB_size, {{in0_transpose_cb_index, in0_data_format}})
                .set_page_size(in0_transpose_cb_index, in0_aligned_tile_size)
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
                (std::uint32_t)in0_tensor.address(),
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
            if ((core.x - start_core_x) <= half_core || (!transpose_mcast and core.y == start_core_y)) {
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
                    (std::uint32_t)in1_tensor.address(),
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
                    (std::uint32_t)out_tensor.address(),
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

                mm_in1_sender_writer_args.push_back(
                    bias_mesh.has_value() ? (std::uint32_t)bias_mesh->address() : 0);  // smuggled-rta-ok
                mm_in1_sender_writer_args.push_back(
                    bias_mesh.has_value() ? (std::uint32_t)per_core_N * in1_idx : 0);  // in1_tensor_start_tile_id
                if (!output_is_sharded) {
                    if (in1_idx == in1_end_idx) {  // right cores when no transpose_mcast
                        mm_in1_sender_writer_args.push_back(last_out_num_blocks_w);
                    } else {
                        mm_in1_sender_writer_args.push_back(out_num_blocks_x);
                    }
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
                if (in1_is_sharded and in1_is_dram) {  // in1 is dram sharded
                    if (in1_is_width_sharded) {
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
                    } else {
                        // Height sharded: no additional runtime args needed
                        // (bank/offset computed from compile-time args + batch index)
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
                    (std::uint32_t)out_tensor.address(),                                // out_tensor_addr
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
                if ((core.x - start_core_x) <= half_core || (transpose_mcast and core.y == start_core_y)) {
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
    const ttnn::prim::qsr::MatmulInputs& tensor_args,
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

    const auto& in0 = input_tensors.at(0).mesh_tensor();
    const auto& in1 = input_tensors.at(1).mesh_tensor();
    const auto& bias_tensor = optional_input_tensors.at(0);

    const auto& out = output_tensors.at(0).mesh_tensor();

    bool src0_sharded = input_tensors[0].memory_config().is_sharded();
    bool out_sharded = output_tensors[0].memory_config().is_sharded();

    ttsl::optional_reference<const tt::tt_metal::MeshTensor> bias_mesh;
    if (bias_tensor.has_value()) {
        bias_mesh = bias_tensor.value().mesh_tensor();
    }

    // in0 sender
    if (src0_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_src2, in0);
    } else {
        auto& reader_sender_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in0_sender_id);
        for (const auto& core : in0_sender_interleaved_cores) {
            auto& reader_runtime_args = reader_sender_runtime_args_by_core[core.x][core.y];
            reader_runtime_args[0] = in0.address();
        }
    }

    // in1 sender
    auto& sender_writer_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in1_sender_writer_id);
    for (const auto& core : in1_sender_cores) {
        auto& writer_runtime_args = sender_writer_runtime_args_by_core[core.x][core.y];
        writer_runtime_args[0] = in1.address();
        writer_runtime_args[7] = out.address();
        if (bias_tensor.has_value()) {
            writer_runtime_args[18] = bias_mesh->address();
        }
    }

    // in1 receiver
    auto& receiver_writer_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in1_receiver_writer_id);
    for (const auto& core : in1_receiver_cores) {
        auto& writer_runtime_args = receiver_writer_runtime_args_by_core[core.x][core.y];
        writer_runtime_args[2] = out.address();
    }
    if (mm_kernel_in1_receiver_writer_id != mm_kernel_in1_receiver_writer_other_noc_setup_id) {
        auto& receiver_writer_runtime_args_by_core =
            GetRuntimeArgs(program, mm_kernel_in1_receiver_writer_other_noc_setup_id);
        for (const auto& core : in1_receiver_other_cores) {
            auto& writer_runtime_args = receiver_writer_runtime_args_by_core[core.x][core.y];
            writer_runtime_args[2] = out.address();
        }
    }

    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_output, out);
    }
}
}  // namespace reuse_mcast_optimized_helpers

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// ===========================================================================================
// Metal 2.0 (ProgramArtifacts) port of the resnet50 mcast_2d path.
//
// This builder mirrors reuse_mcast_optimized_helpers::create_program_mcast_in0_in1_descriptor (the
// active ProgramDescriptor path) but emits a ProgramSpec + ProgramRunArgs. Both in0 and in1 are
// mcast simultaneously (the distinguishing feature of the 2D factory). The kernel sources point at
// the op-local _metal2 forks shared with mcast_1d. gather_in0 / global_cb / fused-op are not
// reachable here (resnet50). mcast_2d uses no global_cb (simpler than 1d).
//
// Resource-name constants are RO_-prefixed to avoid anonymous-namespace duplicate-symbol collisions
// under unity builds; the StrongType string *values* are the plain accessor names the kernels expect.
// ===========================================================================================

namespace m2 = tt::tt_metal::experimental;

namespace CMAKE_UNIQUE_NAMESPACE {
// Create a generation-agnostic data movement hardware config: Gen1 (WH/BH) takes the given
// processor & NOC; Gen2 (Quasar) uses the default config.
m2::DataMovementHardwareConfig make_datamovement_hardware_config(
    tt::ARCH arch, tt::tt_metal::DataMovementProcessor processor, tt::tt_metal::NOC noc) {
    if (arch == tt::ARCH::QUASAR) {
        return m2::DataMovementGen2Config{};
    }
    return m2::DataMovementGen1Config{.processor = processor, .noc = noc};
}
}  // namespace CMAKE_UNIQUE_NAMESPACE

const m2::DFBSpecName RO_IN0_DFB{"cb_in0"};
const m2::DFBSpecName RO_IN1_DFB{"cb_in1"};
const m2::DFBSpecName RO_BIAS_DFB{"cb_bias"};
const m2::DFBSpecName RO_OUT_DFB{"cb_out"};
const m2::DFBSpecName RO_INTERM0_DFB{"cb_intermed0"};
const m2::DFBSpecName RO_SPARSITY_DFB{"cb_sparsity"};
// Per-kernel sparsity scratch for the in1 sender writer (separate DFB from the in0 sender's
// so their self-loops do not put two producers on overlapping cores). Inert (batchB == 0).
const m2::DFBSpecName RO_SPARSITY_IN1_DFB{"cb_sparsity_in1"};
const m2::DFBSpecName RO_IN0_TRANSPOSE_DFB{"cb_in0_transposed"};

const m2::TensorParamName RO_IN0_TENSOR{"in0"};
const m2::TensorParamName RO_IN1_TENSOR{"in1"};
const m2::TensorParamName RO_OUT_TENSOR{"out"};
const m2::TensorParamName RO_BIAS_TENSOR{"bias"};
const m2::TensorParamName RO_SPARSITY_TENSOR{"sparsity"};

const m2::SemaphoreSpecName RO_IN0_SENDER_SEM{"in0_sender"};
const m2::SemaphoreSpecName RO_IN0_RECEIVER_SEM{"in0_receiver"};
const m2::SemaphoreSpecName RO_IN1_SENDER_SEM{"in1_sender"};
const m2::SemaphoreSpecName RO_IN1_RECEIVER_SEM{"in1_receiver"};

const m2::KernelSpecName RO_IN0_SENDER_KERNEL{"in0_sender"};
const m2::KernelSpecName RO_IN0_NO_WORK_KERNEL{"in0_no_work"};
const m2::KernelSpecName RO_IN0_RECEIVER_KERNEL{"in0_receiver"};
const m2::KernelSpecName RO_IN0_RECEIVER_OTHER_KERNEL{"in0_receiver_other"};
const m2::KernelSpecName RO_IN1_SENDER_WRITER_KERNEL{"in1_sender_writer"};
const m2::KernelSpecName RO_IN1_RECEIVER_WRITER_KERNEL{"in1_receiver_writer"};
const m2::KernelSpecName RO_IN1_RECEIVER_WRITER_OTHER_KERNEL{"in1_receiver_writer_other"};
const m2::KernelSpecName RO_COMPUTE_KERNEL{"compute"};

constexpr const char* IN0_SENDER_PADDING_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
    "reader_bmm_tile_layout_in0_sender_padding_metal2.cpp";
constexpr const char* IN0_SENDER_BLOCK_SHARDED_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
    "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded_metal2.cpp";
constexpr const char* IN0_RECEIVER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
    "reader_bmm_tile_layout_in0_receiver_metal2.cpp";
constexpr const char* IN1_SENDER_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
    "reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp";
constexpr const char* IN1_RECEIVER_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
    "reader_bmm_tile_layout_in1_receiver_writer_padding_metal2.cpp";
constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/compute/"
    "bmm_large_block_zm_fused_bias_activation_metal2.cpp";

// std::map<string,string> defines -> Metal 2.0 Defines (Table<string,string>).
m2::KernelSpec::CompilerOptions::Defines to_m2_defines(const std::map<std::string, std::string>& m) {
    m2::KernelSpec::CompilerOptions::Defines result;
    for (const auto& [k, v] : m) {
        result.insert({k, v});
    }
    return result;
}

// Build the compute KernelSpec. Mirrors reuse_optimized / mcast_1d: in0_transpose handled via the
// IN0_TRANSPOSE_TILE_PATH define + the self-loop transposed DFB bindings (NOT a CTA).
m2::KernelSpec make_compute_kernel(
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_per_core_w,
    uint32_t num_blocks,
    uint32_t out_num_blocks_x,
    uint32_t out_num_blocks_y,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    uint32_t batch,
    uint32_t out_block_tiles,
    bool untilize_out,
    bool in0_transpose_tile,
    bool has_bias,
    uint32_t bias_ntiles,
    bool row_broadcast_bias,
    const std::optional<UnaryWithParam>& fused_activation,
    const std::map<std::string, std::string>& mm_kernel_defines,
    const m2::ComputeHardwareConfig& compute_hw_config) {
    std::vector<m2::DFBBinding> dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = RO_IN0_DFB, .accessor_name = "cb_in0", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = RO_IN1_DFB, .accessor_name = "cb_in1", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        m2::DFBBinding{
            .dfb_spec_name = RO_OUT_DFB, .accessor_name = "cb_out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = RO_INTERM0_DFB,
            .accessor_name = "cb_intermed0",
            .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = RO_INTERM0_DFB,
            .accessor_name = "cb_intermed0",
            .endpoint_type = m2::DFBEndpointType::CONSUMER},
    };
    if (has_bias) {
        dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = RO_BIAS_DFB, .accessor_name = "cb_bias", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    if (in0_transpose_tile) {
        dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = RO_IN0_TRANSPOSE_DFB,
            .accessor_name = "cb_in0_transposed",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = RO_IN0_TRANSPOSE_DFB,
            .accessor_name = "cb_in0_transposed",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }

    m2::KernelSpec::CompileTimeArgs cta = {
        {"in0_block_w", in0_block_w},
        {"in0_num_subblocks", in0_num_subblocks},
        {"in0_block_num_tiles", in0_block_num_tiles},
        {"in0_subblock_num_tiles", in0_subblock_num_tiles},
        {"in1_num_subblocks", in1_num_subblocks},
        {"in1_block_num_tiles", in1_block_num_tiles},
        {"in1_block_w", in1_per_core_w},
        {"num_blocks_inner_dim", num_blocks},
        {"num_blocks_w_dim", out_num_blocks_x},
        {"num_blocks_h_dim", out_num_blocks_y},
        {"out_subblock_h", out_subblock_h},
        {"out_subblock_w", out_subblock_w},
        {"out_subblock_num_tiles", out_subblock_num_tiles},
        {"batch", batch},
        {"out_block_num_tiles", out_block_tiles},
        {"untilize_out", (uint32_t)untilize_out},
        {"get_batch_from_reader", 0u},
        {"bias_ntiles", bias_ntiles},
    };
    if (has_bias) {
        cta.insert({"row_broadcast_bias", row_broadcast_bias ? 1u : 0u});
    }
    if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
        using ttnn::operations::experimental::quasar::matmul::utilities::get_activation_params;
        const auto params = get_activation_params(fused_activation.value());
        cta.insert({"activation_type", static_cast<uint32_t>(params.type)});
        cta.insert({"activation_param0", params.param0});
        cta.insert({"activation_param1", params.param1});
        cta.insert({"activation_param2", params.param2});
    }

    return m2::KernelSpec{
        .unique_id = RO_COMPUTE_KERNEL,
        .source = std::filesystem::path(COMPUTE_KERNEL_PATH),
        .compiler_options = {.defines = to_m2_defines(mm_kernel_defines)},
        .dfb_bindings = std::move(dfb_bindings),
        .compile_time_args = std::move(cta),
        .hw_config = compute_hw_config,
    };
}

// ---------------------------------------------------------------------------------------------------
// mcast_in0_in1 (ProgramArtifacts). Mirrors create_program_mcast_in0_in1_descriptor.
// ---------------------------------------------------------------------------------------------------
ttnn::device_operation::ProgramArtifacts create_program_mcast_in0_in1_artifacts(
    const tt::tt_metal::Tensor& a,
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    uint32_t B,
    uint32_t M,
    uint32_t M_per_batch,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    bool transpose_a,
    bool transpose_b,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t in0_block_w,
    uint32_t in0_last_ktile_w,
    uint32_t in0_last_ktile_h,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool transpose_mcast,
    std::optional<UnaryWithParam> fused_activation,
    const MeshTensor& in0_tensor,
    const MeshTensor& in1_tensor,
    ttsl::optional_reference<const MeshTensor> bias_tensor,
    const MeshTensor& out_tensor,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    bool row_broadcast_bias,
    CoreCoord sub_device_start_core) {
    using namespace tt;
    using tt::tt_metal::TensorMemoryLayout;

    // gather_in0 / fused-op are not reachable here (resnet50). Keep the fuse_op flag wired but false.
    const bool fuse_op = false;
    (void)a;

    // currently only support transpose of the full tile
    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    TensorMemoryLayout in0_memory_layout = in0_tensor.memory_config().memory_layout();

    uint32_t num_blocks = K / in0_block_w;

    bool packer_l1_acc_en = packer_l1_acc && (((bias_tensor.has_value()) && num_blocks > 1) || (num_blocks > 2));

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
    const bool in1_is_width_sharded = in1_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    const bool in1_is_height_sharded = in1_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool in1_is_sharded = in1_is_width_sharded || in1_is_height_sharded;
    const bool output_is_sharded = out_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size =
        in1_is_sharded ? in1_single_tile_size : tt::align(in1_single_tile_size, dram_alignment);

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
        in0_CB_tiles *= operations::experimental::quasar::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles *= operations::experimental::quasar::matmul::utilities::MCAST_INPUT_BUFFERING_DEPTH;
    }

    uint32_t out_block_tiles = out_block_h * out_block_w;
    uint32_t out_shard_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    if (output_is_sharded) {
        out_CB_tiles = out_shard_tiles;
    }
    uint32_t interm0_CB_tiles = out_block_tiles;  // No double buffer

    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    uint32_t in0_shard_height_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_width();
        in0_shard_height_in_tiles = in0_tensor.shard_spec()->shape[0] / in0_tile.get_height();
        in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    }
    [[maybe_unused]] uint32_t in2_CB_tiles = in2_block_tiles;

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer

    uint32_t start_core_x = sub_device_start_core.x;
    uint32_t start_core_y = sub_device_start_core.y;

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
    uint32_t in0_mcast_receiver_grid_diff_coord_start = 0;
    uint32_t in0_mcast_receiver_grid_diff_coord_end = 0;
    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    uint32_t in0_sender_num_cores_along_width = 0;

    // Only used for in0 block sharded
    std::optional<CoreRange> in0_mcast_cores_without_work_and_not_in_receiver_grid;
    if (in0_block_sharded) {
        CoreCoord in0_shard_grid = in0_tensor.shard_spec()->grid.bounding_box().grid_size();
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
            in0_mcast_receiver_grid_diff_coord_start =
                device->worker_core_from_logical_core({start_core_x, start_core_y}).y;
            in0_mcast_receiver_grid_diff_coord_end =
                device->worker_core_from_logical_core({start_core_x, start_core_y + num_blocks_x - 1}).y;
            in0_mcast_noc_y.reserve(in0_sender_num_cores_along_width);
            for (uint32_t core_idx_y = 0; core_idx_y < in0_sender_num_cores_along_width; ++core_idx_y) {
                in0_mcast_noc_y.push_back(
                    device->worker_core_from_logical_core({start_core_x, start_core_y + core_idx_y}).y);
            }
        } else {
            in0_mcast_receiver_grid_diff_coord_start =
                device->worker_core_from_logical_core({start_core_x, start_core_y}).x;
            in0_mcast_receiver_grid_diff_coord_end =
                device->worker_core_from_logical_core({start_core_x + num_blocks_x - 1, start_core_y}).x;
            in0_mcast_noc_x.reserve(in0_sender_num_cores_along_width);
            for (uint32_t core_idx_x = 0; core_idx_x < in0_sender_num_cores_along_width; ++core_idx_x) {
                in0_mcast_noc_x.push_back(
                    device->worker_core_from_logical_core({start_core_x + core_idx_x, start_core_y}).x);
            }
        }

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
    CoreRangeSet all_cores_set(all_cores);
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

    bool in1_is_dram = in1_tensor.mesh_buffer().device_local_config().buffer_type == tt_metal::BufferType::DRAM;

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    TT_FATAL(
        out_block_h % out_subblock_h == 0 and out_block_h >= out_subblock_h,
        "out_block_h must be multiple of out_subblock_h");
    TT_FATAL(
        out_block_w % out_subblock_w == 0 and out_block_w >= out_subblock_w,
        "out_block_w must be multiple of out_subblock_w");

    // NOTE: the in1 DRAM-sharded reader paths are not served by the _metal2 in1_sender_writer fork
    // (those configs stay on the legacy original factory), so the dram-bank stride bookkeeping the
    // legacy descriptor computed here is intentionally omitted from this Metal 2.0 builder.

    const auto [in0_tensor_stride_w, in0_tensor_stride_h] =
        operations::experimental::quasar::matmul::utilities::get_in0_transpose_strides(M, M_per_batch, transpose_a, K);
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in0_tensor_next_h_dim_block_stride = in0_block_h * in0_tensor_stride_h;
    const auto in0_tensor_start_tile_id_stride = per_core_M * in0_tensor_stride_h;

    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;
    const auto in1_tensor_next_w_dim_block_stride = in1_block_w * in1_tensor_stride_w;
    const auto in1_tensor_start_tile_id_stride = per_core_N * in1_tensor_stride_w;

    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    // ---- Mcast semaphore numbering for the block-sharded sender (num_x/num_y CTAs) ----
    uint32_t num_x_bs = in0_sender_num_cores_along_width;
    uint32_t num_y_bs = 1;
    if (transpose_mcast) {
        std::swap(num_x_bs, num_y_bs);
    }

    // ---- Defines ----
    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_sharded_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_interleaved_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_receiver_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_receiver_writer_other_noc_setup_defines;
    if (bias_tensor.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_other_noc_setup_defines["FUSE_BIAS"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines["SFPU_ACTIVATION"] = "1";
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
    if (in0_transpose_tile) {
        mm_kernel_defines["IN0_TRANSPOSE_TILE_PATH"] = "1";
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
            if (in1_is_width_sharded) {
                mm_kernel_in1_sender_writer_defines["IN1_DRAM_WIDTH_SHARDED"] = "1";
            } else {
                mm_kernel_in1_sender_writer_defines["IN1_DRAM_HEIGHT_SHARDED"] = "1";
            }
        } else {
            mm_kernel_in1_sender_writer_defines["IN1_SHARDED"] = "1";
        }
    }
    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
        mm_kernel_in1_receiver_writer_defines["OUT_SHARDED"] = "1";
        mm_kernel_in1_receiver_writer_other_noc_setup_defines["OUT_SHARDED"] = "1";
    }

    // ---- Tensor parameters ----
    // Sparsity TensorParameter is inert (batchB == 0); aliases the in0 spec so tensor::sparsity binds.
    m2::Group<m2::TensorParameter> tensor_parameters = {
        m2::TensorParameter{.unique_id = RO_IN0_TENSOR, .spec = in0_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = RO_IN1_TENSOR, .spec = in1_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = RO_OUT_TENSOR, .spec = out_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = RO_SPARSITY_TENSOR, .spec = in0_tensor.tensor_spec()},
    };
    if (bias_tensor.has_value()) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = RO_BIAS_TENSOR, .spec = bias_tensor->tensor_spec()});
    }

    // ---- Dataflow buffers ----
    m2::Group<m2::DataflowBufferSpec> dataflow_buffers;
    {
        m2::DataflowBufferSpec in0_dfb{
            .unique_id = RO_IN0_DFB,
            .entry_size = in0_aligned_tile_size,
            .num_entries = in0_CB_tiles,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        };
        if (in0_height_sharded) {
            in0_dfb.borrowed_from = RO_IN0_TENSOR;
        }
        dataflow_buffers.push_back(std::move(in0_dfb));
    }
    // Sparsity scratch DFBs are a single-kernel DMA-landing self-loop (PRODUCER+CONSUMER on one DM
    // kernel), which the Metal 2.0 DM-kernel self-loop validator rejects. These non-sparse mcast
    // factories never enable sparsity (batchB is hardcoded 0; the sparse path is a separate factory),
    // so the sparsity DFBs/bindings — and the kernels' SPARSITY-gated cb_sparsity usage — are simply
    // absent here. When the sparse matmul is ported to Metal 2.0, flip sparsity_enabled, define
    // SPARSITY on the sender kernels, and replace the self-loop with a scratchpad/LocalTensorAccessor.
    const bool sparsity_enabled = false;
    if (sparsity_enabled && !in0_block_sharded) {
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = RO_SPARSITY_DFB,
            .entry_size = in0_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        });
    }
    if (sparsity_enabled) {
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = RO_SPARSITY_IN1_DFB,
            .entry_size = in0_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        });
    }
    {
        m2::DataflowBufferSpec in1_dfb{
            .unique_id = RO_IN1_DFB,
            .entry_size = in1_aligned_tile_size,
            .num_entries = in1_CB_tiles,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = in1_tile,
        };
        if (in1_is_sharded and not in1_is_dram) {
            in1_dfb.borrowed_from = RO_IN1_TENSOR;
        }
        dataflow_buffers.push_back(std::move(in1_dfb));
    }
    // Block-sharded in0: the resident shard is read by L1 base address from a local TensorAccessor
    // over the in0 tensor in the sender kernel (no borrowed self-loop CB, which Metal 2.0 forbids on
    // DM kernels), and the legacy c_6 l1 scratch (cb_l1_array) is inert (no kernel reads it back), so
    // neither needs a DataflowBuffer here.

    // out / intermed0: separate or aliased (shared memory) — predicate matches the legacy CB-sharing.
    const bool separate_out_interm = do_not_inplace_interm0_out_CB || (interm0_data_format != output_data_format) ||
                                     (untilize_out && (in1_num_subblocks > 1));
    {
        m2::DataflowBufferSpec out_dfb{
            .unique_id = RO_OUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = out_CB_tiles,
            .data_format_metadata = output_data_format,
            .tile_format_metadata = output_tile,
        };
        if (output_is_sharded) {
            out_dfb.borrowed_from = RO_OUT_TENSOR;
        }
        m2::DataflowBufferSpec interm0_dfb{
            .unique_id = RO_INTERM0_DFB,
            .entry_size = interm0_single_tile_size,
            .num_entries = interm0_CB_tiles,
            .data_format_metadata = interm0_data_format,
            .tile_format_metadata = output_tile,
        };
        if (!separate_out_interm) {
            out_dfb.advanced_options.alias_with = {RO_INTERM0_DFB};
            interm0_dfb.advanced_options.alias_with = {RO_OUT_DFB};
            if (output_is_sharded) {
                interm0_dfb.borrowed_from = RO_OUT_TENSOR;
            }
        }
        dataflow_buffers.push_back(std::move(out_dfb));
        dataflow_buffers.push_back(std::move(interm0_dfb));
    }
    if (bias_tensor.has_value()) {
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = RO_BIAS_DFB,
            .entry_size = bias_single_tile_size,
            .num_entries = in3_CB_tiles,
            .data_format_metadata = bias_data_format,
            .tile_format_metadata = bias_tile,
        });
    }
    if (in0_transpose_tile) {
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = RO_IN0_TRANSPOSE_DFB,
            .entry_size = in0_aligned_tile_size,
            .num_entries = in0_CB_tiles,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        });
    }

    // ---- Semaphores (in0 sender/receiver + in1 sender/receiver). Default init; kernels set
    // VALID/INVALID. ----
    m2::Group<m2::SemaphoreSpec> semaphores = {
        m2::SemaphoreSpec{.unique_id = RO_IN0_SENDER_SEM, .target_nodes = all_cores_set},
        m2::SemaphoreSpec{.unique_id = RO_IN0_RECEIVER_SEM, .target_nodes = all_cores_set},
        m2::SemaphoreSpec{.unique_id = RO_IN1_SENDER_SEM, .target_nodes = all_cores_set},
        m2::SemaphoreSpec{.unique_id = RO_IN1_RECEIVER_SEM, .target_nodes = all_cores_set},
    };

    m2::ComputeHardwareConfig compute_hw_config = ttnn::to_compute_hardware_config(
        device->arch(),
        ttnn::ComputeKernelConfig{
            .math_fidelity = math_fidelity,
            .math_approx_mode = math_approx_mode,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = false});

    // in1 reader uses the optimized reader noc; in0 the dram-write noc. (The legacy split-half
    // _other receivers ran on the opposite NOC for perf; the Metal 2.0 host API selects the NOC via
    // the reader/writer Gen1 config, so the _other kernels are functionally identical here — see report.)
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    m2::Group<m2::KernelSpec> kernels;

    // ---- in0 sender CTAs (named). Block-sharded vs interleaved layouts mapped to the metal2 fork's
    // fixed named CTA set. ----
    auto make_in0_sender_cta = [&](uint32_t core_has_output_block_work,
                                   uint32_t core_in_in0_receiver_mcast_grid) -> m2::KernelSpec::CompileTimeArgs {
        if (in0_block_sharded) {
            return {
                {"core_has_output_block_work", core_has_output_block_work},
                {"core_in_in0_receiver_mcast_grid", core_in_in0_receiver_mcast_grid},
                {"in0_block_num_tiles", in0_block_num_tiles},
                {"in0_block_size_bytes", (uint32_t)(in0_block_num_tiles * in0_single_tile_size)},
                {"in0_last_ktile_w", (uint32_t)in0_last_ktile_w},
                {"in0_last_ktile_h", (uint32_t)in0_last_ktile_h},
                {"num_blocks_inner_dim", num_blocks},
                {"num_blocks_w_dim", out_num_blocks_x},
                {"num_blocks_h_dim", out_num_blocks_y},
                {"in0_mcast_num_dests", num_blocks_x},
                {"in0_mcast_num_cores", num_blocks_x},
                {"num_x", num_x_bs},
                {"num_y", num_y_bs},
                {"transpose_mcast", (uint32_t)transpose_mcast},
                {"shard_width_in_tiles", in0_shard_width_in_tiles},
                {"shard_height_in_tiles", in0_shard_height_in_tiles},
                {"in0_block_w", in0_block_w},
                {"in0_block_h", in0_block_h},
                {"batch", B},
                {"fuse_op", (uint32_t)(fuse_op)},
            };
        }
        return {
            {"in0_tensor_stride_w", (uint32_t)in0_tensor_stride_w},
            {"in0_tensor_stride_h", (uint32_t)in0_tensor_stride_h},
            {"in0_tensor_next_inner_dim_block_stride", (uint32_t)in0_tensor_next_block_stride},
            {"in0_tensor_next_h_dim_block_stride", (uint32_t)in0_tensor_next_h_dim_block_stride},
            {"in0_block_w", in0_block_w},
            {"in0_block_h", in0_block_h},
            {"in0_block_num_tiles", in0_block_num_tiles},
            {"in0_last_ktile_w", (uint32_t)in0_last_ktile_w},
            {"in0_last_ktile_h", (uint32_t)in0_last_ktile_h},
            {"extract_shard_sub_blocks", 0u},
            {"shard_width_in_tiles", in0_shard_width_in_tiles},
            {"shard_height_in_tiles", in0_shard_height_in_tiles},
            {"num_blocks_inner_dim", num_blocks},
            {"num_blocks_w_dim", out_num_blocks_x},
            {"num_blocks_h_dim", out_num_blocks_y},
            {"in0_mcast_num_dests", num_blocks_x - 1},
            {"in0_mcast_num_cores", num_blocks_x - 1},
            {"MtKt", M * K},
            {"in0_B", B},
            {"in1_B", B},
            {"in0_reuse_in_CB", 0u},
            {"batchB", 0u},
            {"sparsity_pagesize", 0u},
            {"bcast_A", 1u},
            {"get_batch_from_reader", 0u},
            {"fuse_op", (uint32_t)(fuse_op)},
            {"num_batch_compute", 0u},
        };
    };

    // in0 sender DFB bindings (interleaved vs block-sharded).
    auto in0_dfb_bindings = [&]() -> std::vector<m2::DFBBinding> {
        std::vector<m2::DFBBinding> b = {
            m2::DFBBinding{
                .dfb_spec_name = RO_IN0_DFB, .accessor_name = "cb_in0", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        };
        // Block-sharded in0 reads its resident shard via tensor::in0 in the sender kernel, so it
        // needs no extra DFB bindings here (only cb_in0 above, the mcast staging buffer).
        if (sparsity_enabled) {
            // Sparsity scratch self-loop — gated off (see sparsity_enabled note above; kernels
            // build without SPARSITY so they do not reference dfb::cb_sparsity).
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = RO_SPARSITY_DFB,
                .accessor_name = "cb_sparsity",
                .endpoint_type = m2::DFBEndpointType::PRODUCER});
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = RO_SPARSITY_DFB,
                .accessor_name = "cb_sparsity",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        return b;
    };
    auto in0_tensor_bindings = [&]() -> std::vector<m2::TensorBinding> {
        std::vector<m2::TensorBinding> tb = {
            m2::TensorBinding{.tensor_parameter_name = RO_IN0_TENSOR, .accessor_name = "in0"},
        };
        if (!in0_block_sharded) {
            tb.push_back(m2::TensorBinding{.tensor_parameter_name = RO_SPARSITY_TENSOR, .accessor_name = "sparsity"});
        }
        return tb;
    };

    const auto& in0_sender_defines =
        in0_block_sharded ? mm_kernel_in0_sender_sharded_defines : mm_kernel_in0_sender_interleaved_defines;
    const char* in0_sender_source =
        in0_block_sharded ? IN0_SENDER_BLOCK_SHARDED_KERNEL_PATH : IN0_SENDER_PADDING_KERNEL_PATH;

    // in0 sender RTA schema (interleaved vs block-sharded). The block-sharded fork reads sender_id +
    // 4 mcast dest coords + noc varargs.
    m2::Group<std::string> in0_sender_rta_names =
        in0_block_sharded ? m2::Group<std::string>{
                                "sender_id",
                                "in0_mcast_dest_noc_start_x",
                                "in0_mcast_dest_noc_start_y",
                                "in0_mcast_dest_noc_end_x",
                                "in0_mcast_dest_noc_end_y",
                            }
                          : m2::Group<std::string>{
                                "in0_tensor_start_tile_id",
                                "in0_mcast_dest_noc_start_x",
                                "in0_mcast_dest_noc_start_y",
                                "in0_mcast_dest_noc_end_x",
                                "in0_mcast_dest_noc_end_y",
                                "last_block_h",
                                "sparsity_addr",
                            };

    // in0 sender (work cores). Interleaved: left column. Block-sharded: all cores with work.
    {
        m2::KernelSpec ks{
            .unique_id = RO_IN0_SENDER_KERNEL,
            .source = std::filesystem::path(in0_sender_source),
            .compiler_options = {.defines = to_m2_defines(in0_sender_defines)},
            .dfb_bindings = in0_dfb_bindings(),
            .semaphore_bindings =
                {
                    m2::SemaphoreBinding{.semaphore_spec_name = RO_IN0_SENDER_SEM, .accessor_name = "in0_sender"},
                    m2::SemaphoreBinding{.semaphore_spec_name = RO_IN0_RECEIVER_SEM, .accessor_name = "in0_receiver"},
                },
            .tensor_bindings = in0_tensor_bindings(),
            .compile_time_args = make_in0_sender_cta(1, 1),
            .runtime_arg_schema = {.runtime_arg_names = in0_sender_rta_names},
            // Pin RISCV_1 + in0_noc (legacy parity): the in0 row-mcast dest rectangle is swapped for
            // in0_noc below, so the mcast must issue on in0_noc or it inverts and only the sender's
            // own column receives in0 (degenerate 2-corner delivery -> partial-grid hang).
            .hw_config = CMAKE_UNIQUE_NAMESPACE::make_datamovement_hardware_config(
                device->arch(), tt::tt_metal::DataMovementProcessor::RISCV_1, in0_noc),
        };
        // Block-sharded in0 sender reads num_x + num_y per-core mcast-coord varargs (in0_mcast_noc_x/y);
        // declare the count so the framework allocates the vararg slots (else get_vararg is OOB).
        if (in0_block_sharded) {
            ks.advanced_options.num_runtime_varargs = num_x_bs + num_y_bs;
        }
        kernels.push_back(std::move(ks));
    }

    // in0 sender no-work variant (block-sharded only, extra sender cores not in the work grid).
    const bool has_in0_no_work = in0_block_sharded && in0_mcast_cores_without_work_and_not_in_receiver_grid.has_value();
    if (has_in0_no_work) {
        auto no_work_cta = make_in0_sender_cta(0, 0);
        m2::KernelSpec no_work_ks{
            .unique_id = RO_IN0_NO_WORK_KERNEL,
            .source = std::filesystem::path(IN0_SENDER_BLOCK_SHARDED_KERNEL_PATH),
            .compiler_options = {.defines = to_m2_defines(in0_sender_defines)},
            .dfb_bindings = in0_dfb_bindings(),
            .semaphore_bindings =
                {
                    m2::SemaphoreBinding{.semaphore_spec_name = RO_IN0_SENDER_SEM, .accessor_name = "in0_sender"},
                    m2::SemaphoreBinding{.semaphore_spec_name = RO_IN0_RECEIVER_SEM, .accessor_name = "in0_receiver"},
                },
            .tensor_bindings = in0_tensor_bindings(),
            .compile_time_args = std::move(no_work_cta),
            .runtime_arg_schema = {.runtime_arg_names = in0_sender_rta_names},
            // Pin RISCV_1 + in0_noc (legacy parity) to match the in0-mcast rectangle geometry.
            .hw_config = CMAKE_UNIQUE_NAMESPACE::make_datamovement_hardware_config(
                device->arch(), tt::tt_metal::DataMovementProcessor::RISCV_1, in0_noc),
        };
        // Same varargs (in0_mcast_noc_x/y) as the work in0 sender (block-sharded only path).
        no_work_ks.advanced_options.num_runtime_varargs = num_x_bs + num_y_bs;
        kernels.push_back(std::move(no_work_ks));
    }

    // in0 receiver (interleaved path only). Left half + (split_half) right half (different NOC).
    const bool has_in0_receiver = !in0_block_sharded && in0_receiver_interleaved.num_cores() > 0;
    auto make_in0_receiver_kernel = [&](const m2::KernelSpecName& id) -> m2::KernelSpec {
        return m2::KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path(IN0_RECEIVER_KERNEL_PATH),
            .dfb_bindings =
                {
                    m2::DFBBinding{
                        .dfb_spec_name = RO_IN0_DFB,
                        .accessor_name = "cb_in0",
                        .endpoint_type = m2::DFBEndpointType::PRODUCER},
                },
            .semaphore_bindings =
                {
                    m2::SemaphoreBinding{.semaphore_spec_name = RO_IN0_SENDER_SEM, .accessor_name = "in0_sender"},
                    m2::SemaphoreBinding{.semaphore_spec_name = RO_IN0_RECEIVER_SEM, .accessor_name = "in0_receiver"},
                },
            .compile_time_args =
                {
                    {"in0_block_num_tiles", in0_block_w * in0_block_h},
                    {"num_blocks_inner_dim", num_blocks},
                    {"num_blocks_w_dim", out_num_blocks_x},
                    {"num_blocks_h_dim", out_num_blocks_y},
                    {"batch", B},
                    {"get_batch_from_reader", 0u},
                },
            .runtime_arg_schema = {.runtime_arg_names = {"in0_mcast_sender_noc_x", "in0_mcast_sender_noc_y"}},
            // Pin RISCV_1 + in0_noc (legacy parity). The m2 path computes a single in0-mcast geometry
            // on in0_noc for both the main and _other receivers, so both must use in0_noc.
            .hw_config = CMAKE_UNIQUE_NAMESPACE::make_datamovement_hardware_config(
                device->arch(), tt::tt_metal::DataMovementProcessor::RISCV_1, in0_noc),
        };
    };
    if (has_in0_receiver) {
        kernels.push_back(make_in0_receiver_kernel(RO_IN0_RECEIVER_KERNEL));
    }
    const bool has_in0_receiver_other =
        !in0_block_sharded && in0_receiver_in1_receiver_interleaved_other_cores.has_value();
    if (has_in0_receiver_other) {
        kernels.push_back(make_in0_receiver_kernel(RO_IN0_RECEIVER_OTHER_KERNEL));
    }

    // ---- in1 sender writer CTAs (named). ----
    auto make_in1_sender_writer_cta = [&]() -> m2::KernelSpec::CompileTimeArgs {
        m2::KernelSpec::CompileTimeArgs cta = {
            {"in1_tensor_stride_w", (uint32_t)in1_tensor_stride_w},
            {"in1_tensor_stride_h", (uint32_t)in1_tensor_stride_h},
            {"in1_tensor_next_block_stride", (uint32_t)in1_tensor_next_block_stride},
            {"in1_tensor_next_w_dim_block_stride", (uint32_t)in1_tensor_next_w_dim_block_stride},
            {"in1_block_w", in1_block_w},
            {"in1_block_h", in0_block_w},
            {"in1_block_num_tiles", in1_block_w * in0_block_w},
            {"num_blocks_inner_dim", num_blocks},
            {"num_blocks_w_dim", out_num_blocks_x},
            {"num_blocks_h_dim", out_num_blocks_y},
            {"in1_mcast_num_dests", num_blocks_y - 1},
            {"in1_mcast_num_cores", num_blocks_y - 1},
            {"KtNt", K * N},
            {"batch", B},
            {"bcast_B", (uint32_t)bcast_batch},
            {"batchB", 0u},
            {"sparsity_pagesize", 0u},
            {"out_tensor_stride_w", 1u},
            {"out_tensor_stride_h", N},
            {"out_tensor_next_subblock_stride_w", out_subblock_w},
            {"out_tensor_next_subblock_stride_h", out_subblock_h * N},
            {"out_tensor_next_w_dim_block_stride", out_block_w},
            {"out_tensor_next_h_dim_block_stride", out_block_h * N},
            {"out_subblock_w", out_subblock_w},
            {"out_subblock_h", out_subblock_h},
            {"out_subblock_tile_count", out_subblock_w * out_subblock_h},
            {"MtNt", M * N},
            {"in3_tensor_stride_w", bias_tensor.has_value() ? 1u : 0u},
            {"fuse_op_all_gather", 0u},
            {"fuse_op_reduce_scatter", 0u},
        };
        return cta;
    };

    // in1 sender writer (top row).
    {
        std::vector<m2::DFBBinding> b = {
            m2::DFBBinding{
                .dfb_spec_name = RO_IN1_DFB, .accessor_name = "cb_in1", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = RO_OUT_DFB, .accessor_name = "cb_out", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        };
        if (sparsity_enabled) {
            // Sparsity scratch self-loop — gated off (see sparsity_enabled note above; the kernel
            // builds without SPARSITY so it does not reference dfb::cb_sparsity).
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = RO_SPARSITY_IN1_DFB,
                .accessor_name = "cb_sparsity",
                .endpoint_type = m2::DFBEndpointType::PRODUCER});
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = RO_SPARSITY_IN1_DFB,
                .accessor_name = "cb_sparsity",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        std::vector<m2::TensorBinding> tb = {
            m2::TensorBinding{.tensor_parameter_name = RO_IN1_TENSOR, .accessor_name = "in1"},
            m2::TensorBinding{.tensor_parameter_name = RO_OUT_TENSOR, .accessor_name = "out"},
            m2::TensorBinding{.tensor_parameter_name = RO_SPARSITY_TENSOR, .accessor_name = "sparsity"},
        };
        if (bias_tensor.has_value()) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = RO_BIAS_DFB,
                .accessor_name = "cb_bias",
                .endpoint_type = m2::DFBEndpointType::PRODUCER});
            tb.push_back(m2::TensorBinding{.tensor_parameter_name = RO_BIAS_TENSOR, .accessor_name = "bias"});
        }
        // Named RTAs. The legacy per-core padding-arg block (last-col vs interior) is modeled as fixed
        // named slots, padded to the max. The !output_is_sharded tail adds last_num_blocks_w_dim.
        m2::Group<std::string> rta_names = {
            "in1_tensor_start_tile_id",
            "in1_mcast_dest_noc_start_x",
            "in1_mcast_dest_noc_start_y",
            "in1_mcast_dest_noc_end_x",
            "in1_mcast_dest_noc_end_y",
            "sparsity_addr",
            "out_tensor_start_tile_id",
            "last_block_w",
            "out_num_nonzero_subblocks_h",
            "out_last_subblock_h",
            "padded_block_tiles_h_skip",
            "out_num_nonzero_subblocks_w",
            "out_last_num_nonzero_subblocks_w",
            "out_last_subblock_w",
            "padded_subblock_tiles_addr_skip",
            "padded_block_tiles_w_skip",
        };
        if (bias_tensor.has_value()) {
            rta_names.push_back("in3_tensor_start_tile_id");
        }
        if (!output_is_sharded) {
            rta_names.push_back("last_num_blocks_w_dim");
        }
        kernels.push_back(m2::KernelSpec{
            .unique_id = RO_IN1_SENDER_WRITER_KERNEL,
            .source = std::filesystem::path(IN1_SENDER_WRITER_KERNEL_PATH),
            .compiler_options = {.defines = to_m2_defines(mm_kernel_in1_sender_writer_defines)},
            .dfb_bindings = std::move(b),
            .semaphore_bindings =
                {
                    m2::SemaphoreBinding{.semaphore_spec_name = RO_IN1_SENDER_SEM, .accessor_name = "in1_sender"},
                    m2::SemaphoreBinding{.semaphore_spec_name = RO_IN1_RECEIVER_SEM, .accessor_name = "in1_receiver"},
                },
            .tensor_bindings = std::move(tb),
            .compile_time_args = make_in1_sender_writer_cta(),
            .runtime_arg_schema = {.runtime_arg_names = std::move(rta_names)},
            // Pin RISCV_0 + in1_noc (legacy parity): the in1 column-mcast dest rectangle is swapped
            // for in1_noc below, so the mcast must issue on in1_noc or it inverts and degenerates.
            .hw_config = CMAKE_UNIQUE_NAMESPACE::make_datamovement_hardware_config(
                device->arch(), tt::tt_metal::DataMovementProcessor::RISCV_0, in1_noc),
        });
    }

    // ---- in1 receiver writer CTAs (named). ----
    auto make_in1_receiver_writer_cta = [&]() -> m2::KernelSpec::CompileTimeArgs {
        return {
            {"in1_block_num_tiles", in1_block_w * in0_block_w},
            {"num_blocks_inner_dim", num_blocks},
            {"num_blocks_w_dim", out_num_blocks_x},
            {"num_blocks_h_dim", out_num_blocks_y},
            {"batch", B},
            {"out_tensor_stride_w", 1u},
            {"out_tensor_stride_h", N},
            {"out_tensor_next_subblock_stride_w", out_subblock_w},
            {"out_tensor_next_subblock_stride_h", out_subblock_h * N},
            {"out_tensor_next_w_dim_block_stride", out_block_w},
            {"out_tensor_next_h_dim_block_stride", out_block_h * N},
            {"out_subblock_w", out_subblock_w},
            {"out_subblock_h", out_subblock_h},
            {"out_subblock_tile_count", out_subblock_w * out_subblock_h},
            {"MtNt", M * N},
            {"in3_block_w", bias_tensor.has_value() ? in1_block_w : 0u},
            {"fuse_op_reduce_scatter", 0u},
        };
    };
    auto make_in1_receiver_writer_kernel = [&](const m2::KernelSpecName& id,
                                               const std::map<std::string, std::string>& defines) -> m2::KernelSpec {
        std::vector<m2::DFBBinding> b = {
            m2::DFBBinding{
                .dfb_spec_name = RO_IN1_DFB, .accessor_name = "cb_in1", .endpoint_type = m2::DFBEndpointType::PRODUCER},
            m2::DFBBinding{
                .dfb_spec_name = RO_OUT_DFB, .accessor_name = "cb_out", .endpoint_type = m2::DFBEndpointType::CONSUMER},
        };
        if (bias_tensor.has_value()) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = RO_BIAS_DFB,
                .accessor_name = "cb_bias",
                .endpoint_type = m2::DFBEndpointType::PRODUCER});
        }
        m2::Group<std::string> rta_names = {
            "in1_mcast_sender_noc_x",
            "in1_mcast_sender_noc_y",
            "out_tensor_start_tile_id",
            "out_num_nonzero_subblocks_h",
            "out_last_num_nonzero_subblocks_h",
            "out_last_subblock_h",
            "padded_block_tiles_h_skip",
            "out_num_nonzero_subblocks_w",
            "out_last_num_nonzero_subblocks_w",
            "out_last_subblock_w",
            "padded_subblock_tiles_addr_skip",
            "padded_block_tiles_w_skip",
        };
        if (!output_is_sharded) {
            rta_names.push_back("last_num_blocks_h_dim");
            rta_names.push_back("last_num_blocks_w_dim");
        }
        return m2::KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path(IN1_RECEIVER_WRITER_KERNEL_PATH),
            .compiler_options = {.defines = to_m2_defines(defines)},
            .dfb_bindings = std::move(b),
            .semaphore_bindings =
                {
                    m2::SemaphoreBinding{.semaphore_spec_name = RO_IN1_SENDER_SEM, .accessor_name = "in1_sender"},
                    m2::SemaphoreBinding{.semaphore_spec_name = RO_IN1_RECEIVER_SEM, .accessor_name = "in1_receiver"},
                },
            .tensor_bindings =
                {
                    m2::TensorBinding{.tensor_parameter_name = RO_OUT_TENSOR, .accessor_name = "out"},
                },
            .compile_time_args = make_in1_receiver_writer_cta(),
            .runtime_arg_schema = {.runtime_arg_names = std::move(rta_names)},
            // Pin RISCV_0 + in1_noc (legacy parity). The m2 path computes a single in1-mcast geometry
            // on in1_noc for both the main and _other receivers, so both must use in1_noc to match
            // the sender's multicast rectangle and semaphore signaling.
            .hw_config = CMAKE_UNIQUE_NAMESPACE::make_datamovement_hardware_config(
                device->arch(), tt::tt_metal::DataMovementProcessor::RISCV_0, in1_noc),
        };
    };
    const bool has_in1_receiver = in1_receiver.num_cores() > 0;
    if (has_in1_receiver) {
        kernels.push_back(
            make_in1_receiver_writer_kernel(RO_IN1_RECEIVER_WRITER_KERNEL, mm_kernel_in1_receiver_writer_defines));
    }
    const bool has_in1_receiver_other = in0_receiver_in1_receiver_interleaved_other_cores.has_value();
    if (has_in1_receiver_other) {
        kernels.push_back(make_in1_receiver_writer_kernel(
            RO_IN1_RECEIVER_WRITER_OTHER_KERNEL, mm_kernel_in1_receiver_writer_other_noc_setup_defines));
    }

    // compute (on all_cores_with_work).
    kernels.push_back(make_compute_kernel(
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        out_num_blocks_x,
        out_num_blocks_y,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        B,
        out_block_tiles,
        untilize_out,
        in0_transpose_tile,
        bias_tensor.has_value(),
        in1_per_core_w,
        row_broadcast_bias,
        fused_activation,
        mm_kernel_defines,
        compute_hw_config));

    // ---- Work units. WorkUnitSpec target_nodes MUST be DISJOINT (program_spec.cpp); each group lists
    // ALL kernels that run on those cores (a kernel's placement is the union of the groups listing it),
    // and compute runs on every work core so it appears in every group. This mirrors mcast_1d -- the
    // earlier per-kernel work units overlapped (e.g. the corner is both in0 and in1 sender). ----
    m2::Group<m2::WorkUnitSpec> work_units;
    if (in0_block_sharded) {
        // in0_sender runs on every work core; partition by in1 role: top row (senders) vs the rest
        // (receivers). in1_sender (top row) and in1_receiver (rows below) partition all_cores_with_work.
        work_units.push_back(m2::WorkUnitSpec{
            .name = "wu_top_row",
            .kernels = {RO_IN0_SENDER_KERNEL, RO_IN1_SENDER_WRITER_KERNEL, RO_COMPUTE_KERNEL},
            .target_nodes = CoreRangeSet(in1_sender),
        });
        if (has_in1_receiver) {
            work_units.push_back(m2::WorkUnitSpec{
                .name = "wu_rest",
                .kernels = {RO_IN0_SENDER_KERNEL, RO_IN1_RECEIVER_WRITER_KERNEL, RO_COMPUTE_KERNEL},
                .target_nodes = in1_receiver,
            });
        }
        if (has_in0_no_work) {
            work_units.push_back(m2::WorkUnitSpec{
                .name = "wu_in0_no_work",
                .kernels = {RO_IN0_NO_WORK_KERNEL},
                .target_nodes = CoreRangeSet(in0_mcast_cores_without_work_and_not_in_receiver_grid.value()),
            });
        }
    } else {
        // Interleaved: disjoint partition by (in0 role, in1 role). The corner (start_core) is both the
        // in0 sender (left column) and in1 sender (top row); the remaining left column / top row /
        // interior groups each pair the right in0+in1 receiver/sender kernels. compute is in every group.
        const CoreRange corner_core(
            {(std::size_t)start_core_x, (std::size_t)start_core_y},
            {(std::size_t)start_core_x, (std::size_t)start_core_y});
        work_units.push_back(m2::WorkUnitSpec{
            .name = "wu_corner",
            .kernels = {RO_IN0_SENDER_KERNEL, RO_IN1_SENDER_WRITER_KERNEL, RO_COMPUTE_KERNEL},
            .target_nodes = CoreRangeSet(corner_core),
        });
        if (in0_sender_in1_receiver.has_value()) {
            work_units.push_back(m2::WorkUnitSpec{
                .name = "wu_left_col",
                .kernels = {RO_IN0_SENDER_KERNEL, RO_IN1_RECEIVER_WRITER_KERNEL, RO_COMPUTE_KERNEL},
                .target_nodes = CoreRangeSet(in0_sender_in1_receiver.value()),
            });
        }
        if (in0_receiver_in1_sender.has_value()) {
            work_units.push_back(m2::WorkUnitSpec{
                .name = "wu_top_row",
                .kernels = {RO_IN0_RECEIVER_KERNEL, RO_IN1_SENDER_WRITER_KERNEL, RO_COMPUTE_KERNEL},
                .target_nodes = CoreRangeSet(in0_receiver_in1_sender.value()),
            });
        }
        if (in0_receiver_in1_receiver_left_half.has_value()) {
            work_units.push_back(m2::WorkUnitSpec{
                .name = "wu_interior",
                .kernels = {RO_IN0_RECEIVER_KERNEL, RO_IN1_RECEIVER_WRITER_KERNEL, RO_COMPUTE_KERNEL},
                .target_nodes = CoreRangeSet(in0_receiver_in1_receiver_left_half.value()),
            });
        }
        if (in0_receiver_in1_receiver_interleaved_other_cores.has_value()) {
            work_units.push_back(m2::WorkUnitSpec{
                .name = "wu_interior_other",
                .kernels = {RO_IN0_RECEIVER_OTHER_KERNEL, RO_IN1_RECEIVER_WRITER_OTHER_KERNEL, RO_COMPUTE_KERNEL},
                .target_nodes = CoreRangeSet(in0_receiver_in1_receiver_interleaved_other_cores.value()),
            });
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime Args (per-core loop)
    ////////////////////////////////////////////////////////////////////////////
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

    uint32_t in0_end_idx = num_blocks_y - 1;
    uint32_t in1_end_idx = num_blocks_x - 1;

    // For block-sharded the noc_x/noc_y arrays are runtime varargs. The kernel reads num_x + num_y
    // entries (in0_mcast_noc_x[num_x] then in0_mcast_noc_y[num_y]); the mcast-dimension table holds W
    // coords and the off-axis dimension contributes its single same-coord, so the total is
    // num_x_bs + num_y_bs (matches the per-core vararg vector built below).
    const uint32_t in0_sharded_num_varargs = in0_block_sharded ? (num_x_bs + num_y_bs) : 0;

    m2::ProgramRunArgs run_args;
    m2::KernelRunArgs in0_sender_run_args{.kernel = RO_IN0_SENDER_KERNEL};
    m2::KernelRunArgs in0_no_work_run_args{.kernel = RO_IN0_NO_WORK_KERNEL};
    m2::KernelRunArgs in0_receiver_run_args{.kernel = RO_IN0_RECEIVER_KERNEL};
    m2::KernelRunArgs in0_receiver_other_run_args{.kernel = RO_IN0_RECEIVER_OTHER_KERNEL};
    m2::KernelRunArgs in1_sender_writer_run_args{.kernel = RO_IN1_SENDER_WRITER_KERNEL};
    m2::KernelRunArgs in1_receiver_writer_run_args{.kernel = RO_IN1_RECEIVER_WRITER_KERNEL};
    m2::KernelRunArgs in1_receiver_writer_other_run_args{.kernel = RO_IN1_RECEIVER_WRITER_OTHER_KERNEL};

    // ---- [DEBUG #47797] one-time summary of the in0 block-sharded mcast geometry. ----
    log_info(
        LogOp,
        "[in0bs-host] in0_block_sharded={} transpose_mcast={} in0_noc={} start_core=({},{}) "
        "num_blocks_x={} num_blocks_y={} num_blocks(inner)={} num_x_bs={} num_y_bs={} "
        "in0_sender_num_cores_along_width={} in0_shard_width_in_tiles={} in0_block_w={} "
        "diff_coord[start={} end={}]",
        in0_block_sharded,
        transpose_mcast,
        static_cast<int>(in0_noc),
        start_core_x,
        start_core_y,
        num_blocks_x,
        num_blocks_y,
        num_blocks,
        num_x_bs,
        num_y_bs,
        in0_sender_num_cores_along_width,
        in0_shard_width_in_tiles,
        in0_block_w,
        in0_mcast_receiver_grid_diff_coord_start,
        in0_mcast_receiver_grid_diff_coord_end);

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

        // in0 sender / receiver
        if (in0_block_sharded) {
            m2::AdvancedKernelRunArgs::Varargs v;
            uint32_t in0_mcast_receiver_grid_same_coord;
            m2::KernelRunArgs::RuntimeArgValues leading;
            if (transpose_mcast) {
                in0_mcast_receiver_grid_same_coord = device->worker_core_from_logical_core(core).x;
                leading = {
                    {"sender_id", core.y},
                    {"in0_mcast_dest_noc_start_x", in0_mcast_receiver_grid_same_coord},
                    {"in0_mcast_dest_noc_start_y", in0_mcast_receiver_grid_diff_coord_start},
                    {"in0_mcast_dest_noc_end_x", in0_mcast_receiver_grid_same_coord},
                    {"in0_mcast_dest_noc_end_y", in0_mcast_receiver_grid_diff_coord_end},
                };
                // Kernel reads in0_mcast_noc_x[num_x] then in0_mcast_noc_y[num_y]. For transpose the y
                // table holds the W mcast coords; the x side is num_x_bs copies of this core's physical
                // X (same_coord). Total == num_x_bs + num_y_bs == declared num_runtime_varargs.
                v.reserve(num_x_bs + in0_mcast_noc_y.size());
                for (uint32_t i = 0; i < num_x_bs; ++i) {
                    v.push_back(in0_mcast_receiver_grid_same_coord);
                }
                for (auto y : in0_mcast_noc_y) {
                    v.push_back(y);
                }
            } else {
                in0_mcast_receiver_grid_same_coord = device->worker_core_from_logical_core(core).y;
                leading = {
                    {"sender_id", core.x},
                    {"in0_mcast_dest_noc_start_x", in0_mcast_receiver_grid_diff_coord_start},
                    {"in0_mcast_dest_noc_start_y", in0_mcast_receiver_grid_same_coord},
                    {"in0_mcast_dest_noc_end_x", in0_mcast_receiver_grid_diff_coord_end},
                    {"in0_mcast_dest_noc_end_y", in0_mcast_receiver_grid_same_coord},
                };
                // Kernel reads in0_mcast_noc_x[num_x] then in0_mcast_noc_y[num_y]. For non-transpose the
                // x table holds the W mcast coords; the y side is num_y_bs copies of this core's physical
                // Y (same_coord). Total == num_x_bs + num_y_bs == declared num_runtime_varargs.
                v.reserve(in0_mcast_noc_x.size() + num_y_bs);
                for (auto x : in0_mcast_noc_x) {
                    v.push_back(x);
                }
                for (uint32_t i = 0; i < num_y_bs; ++i) {
                    v.push_back(in0_mcast_receiver_grid_same_coord);
                }
            }
            // [DEBUG #47797] Per-core in0 sender RTAs as actually emitted. `sender_id` here MUST
            // match the device "[in0bs-dev] sid=" for the same core; for the leftmost work column
            // (the block-0 senders) sender_id must be 0. `kernel` shows which variant the core got
            // (SENDER cores participate in the handshake; NO_WORK cores are outside the receiver grid).
            // Interpret the dest rectangle with transpose_mcast from the one-time summary: non-transpose
            // mcasts along x = [diff_start..diff_end] at y = same; transpose mcasts along y similarly.
            const uint32_t dbg_sender_id =
                transpose_mcast ? static_cast<uint32_t>(core.y) : static_cast<uint32_t>(core.x);
            log_info(
                LogOp,
                "[in0bs-host] core=({},{}) in0_idx={} in1_idx={} kernel={} sender_id={} "
                "diff_coord[{}..{}] same_coord={} nvarargs={} vararg_first={} vararg_last={}",
                core.x,
                core.y,
                in0_idx,
                in1_idx,
                (in1_idx < num_blocks_x) ? "SENDER" : (has_in0_no_work ? "NO_WORK" : "DROPPED"),
                dbg_sender_id,
                in0_mcast_receiver_grid_diff_coord_start,
                in0_mcast_receiver_grid_diff_coord_end,
                in0_mcast_receiver_grid_same_coord,
                v.size(),
                v.empty() ? 0u : v.front(),
                v.empty() ? 0u : v.back());

            if (in1_idx < num_blocks_x) {
                in0_sender_run_args.runtime_arg_values.push_back(
                    m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = leading});
                in0_sender_run_args.advanced_options.runtime_varargs.emplace(core, v);
            } else if (has_in0_no_work) {
                in0_no_work_run_args.runtime_arg_values.push_back(
                    m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = leading});
                in0_no_work_run_args.advanced_options.runtime_varargs.emplace(core, v);
            }
        } else if (in1_idx == 0) {
            // in0 interleaved sender (left column).
            m2::KernelRunArgs::RuntimeArgValues args = {
                {"in0_tensor_start_tile_id", (uint32_t)in0_tensor_start_tile_id_stride * in0_idx},
                {"in0_mcast_dest_noc_start_x", (uint32_t)in0_mcast_start.x},
                {"in0_mcast_dest_noc_start_y", (uint32_t)in0_mcast_start.y},
                {"in0_mcast_dest_noc_end_x", (uint32_t)in0_mcast_end.x},
                {"in0_mcast_dest_noc_end_y", (uint32_t)in0_mcast_end.y},
                {"last_block_h", in0_idx == in0_end_idx ? last_out_block_h : out_block_h},
                {"sparsity_addr", 0u},
            };
            in0_sender_run_args.runtime_arg_values.push_back(
                m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = std::move(args)});
        } else {
            // in0 interleaved receiver.
            m2::KernelRunArgs::RuntimeArgValues args = {
                {"in0_mcast_sender_noc_x", (uint32_t)in0_mcast_sender.x},
                {"in0_mcast_sender_noc_y", (uint32_t)in0_mcast_sender.y},
            };
            if ((core.x - start_core_x) <= half_core || (!transpose_mcast and core.y == start_core_y)) {
                in0_receiver_run_args.runtime_arg_values.push_back(
                    m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = std::move(args)});
            } else {
                in0_receiver_other_run_args.runtime_arg_values.push_back(
                    m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = std::move(args)});
            }
        }

        if (in0_idx < num_blocks_y and in1_idx < num_blocks_x) {
            // in1 sender (top row).
            if (in0_idx == 0) {
                m2::KernelRunArgs::RuntimeArgValues args = {
                    {"in1_tensor_start_tile_id", (uint32_t)in1_tensor_start_tile_id_stride * in1_idx},
                    {"in1_mcast_dest_noc_start_x", (uint32_t)in1_mcast_start.x},
                    {"in1_mcast_dest_noc_start_y", (uint32_t)in1_mcast_start.y},
                    {"in1_mcast_dest_noc_end_x", (uint32_t)in1_mcast_end.x},
                    {"in1_mcast_dest_noc_end_y", (uint32_t)in1_mcast_end.y},
                    {"sparsity_addr", 0u},
                    {"out_tensor_start_tile_id", ((uint32_t)in1_idx * per_core_N) + (in0_idx * per_core_M * N)},
                };
                if (in1_idx == in1_end_idx) {
                    args.insert({"last_block_w", last_out_block_w});
                    args.insert({"out_num_nonzero_subblocks_h", out_block_h / out_subblock_h});
                    args.insert({"out_last_subblock_h", out_subblock_h});
                    args.insert({"padded_block_tiles_h_skip", 0u});
                    args.insert({"out_num_nonzero_subblocks_w", out_block_w / out_subblock_w});
                    args.insert({"out_last_num_nonzero_subblocks_w", last_block_num_nonzero_subblocks_w});
                    args.insert({"out_last_subblock_w", last_subblock_of_last_block_w});
                    args.insert({"padded_subblock_tiles_addr_skip", last_block_padded_subblock_tiles_addr_skip});
                    args.insert({"padded_block_tiles_w_skip", last_block_padded_block_tiles_w_skip});
                } else {
                    args.insert({"last_block_w", out_block_w});
                    args.insert({"out_num_nonzero_subblocks_h", out_block_h / out_subblock_h});
                    args.insert({"out_last_subblock_h", out_subblock_h});
                    args.insert({"padded_block_tiles_h_skip", 0u});
                    args.insert({"out_num_nonzero_subblocks_w", out_block_w / out_subblock_w});
                    args.insert({"out_last_num_nonzero_subblocks_w", out_block_w / out_subblock_w});
                    args.insert({"out_last_subblock_w", out_subblock_w});
                    args.insert({"padded_subblock_tiles_addr_skip", 0u});
                    args.insert({"padded_block_tiles_w_skip", 0u});
                }
                if (bias_tensor.has_value()) {
                    args.insert({"in3_tensor_start_tile_id", (uint32_t)per_core_N * in1_idx});
                }
                if (!output_is_sharded) {
                    args.insert(
                        {"last_num_blocks_w_dim", in1_idx == in1_end_idx ? last_out_num_blocks_w : out_num_blocks_x});
                }
                in1_sender_writer_run_args.runtime_arg_values.push_back(
                    m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = std::move(args)});
            } else {
                // in1 receiver.
                m2::KernelRunArgs::RuntimeArgValues args = {
                    {"in1_mcast_sender_noc_x", (uint32_t)in1_mcast_sender.x},
                    {"in1_mcast_sender_noc_y", (uint32_t)in1_mcast_sender.y},
                    {"out_tensor_start_tile_id", ((uint32_t)in1_idx * per_core_N) + (in0_idx * per_core_M * N)},
                };
                if (in1_idx == in1_end_idx and in0_idx == in0_end_idx) {
                    args.insert({"out_num_nonzero_subblocks_h", out_block_h / out_subblock_h});
                    args.insert({"out_last_num_nonzero_subblocks_h", last_block_num_nonzero_subblocks_h});
                    args.insert({"out_last_subblock_h", last_subblock_of_last_block_h});
                    args.insert({"padded_block_tiles_h_skip", last_block_padded_block_tiles_h_skip});
                    args.insert({"out_num_nonzero_subblocks_w", out_block_w / out_subblock_w});
                    args.insert({"out_last_num_nonzero_subblocks_w", last_block_num_nonzero_subblocks_w});
                    args.insert({"out_last_subblock_w", last_subblock_of_last_block_w});
                    args.insert({"padded_subblock_tiles_addr_skip", last_block_padded_subblock_tiles_addr_skip});
                    args.insert({"padded_block_tiles_w_skip", last_block_padded_block_tiles_w_skip});
                } else if (in0_idx == in0_end_idx) {
                    args.insert({"out_num_nonzero_subblocks_h", out_block_h / out_subblock_h});
                    args.insert({"out_last_num_nonzero_subblocks_h", last_block_num_nonzero_subblocks_h});
                    args.insert({"out_last_subblock_h", last_subblock_of_last_block_h});
                    args.insert({"padded_block_tiles_h_skip", last_block_padded_block_tiles_h_skip});
                    args.insert({"out_num_nonzero_subblocks_w", out_block_w / out_subblock_w});
                    args.insert({"out_last_num_nonzero_subblocks_w", out_block_w / out_subblock_w});
                    args.insert({"out_last_subblock_w", out_subblock_w});
                    args.insert({"padded_subblock_tiles_addr_skip", 0u});
                    args.insert({"padded_block_tiles_w_skip", 0u});
                } else if (in1_idx == in1_end_idx) {
                    args.insert({"out_num_nonzero_subblocks_h", out_block_h / out_subblock_h});
                    args.insert({"out_last_num_nonzero_subblocks_h", out_block_h / out_subblock_h});
                    args.insert({"out_last_subblock_h", out_subblock_h});
                    args.insert({"padded_block_tiles_h_skip", 0u});
                    args.insert({"out_num_nonzero_subblocks_w", out_block_w / out_subblock_w});
                    args.insert({"out_last_num_nonzero_subblocks_w", last_block_num_nonzero_subblocks_w});
                    args.insert({"out_last_subblock_w", last_subblock_of_last_block_w});
                    args.insert({"padded_subblock_tiles_addr_skip", last_block_padded_subblock_tiles_addr_skip});
                    args.insert({"padded_block_tiles_w_skip", last_block_padded_block_tiles_w_skip});
                } else {
                    args.insert({"out_num_nonzero_subblocks_h", out_block_h / out_subblock_h});
                    args.insert({"out_last_num_nonzero_subblocks_h", out_block_h / out_subblock_h});
                    args.insert({"out_last_subblock_h", out_subblock_h});
                    args.insert({"padded_block_tiles_h_skip", 0u});
                    args.insert({"out_num_nonzero_subblocks_w", out_block_w / out_subblock_w});
                    args.insert({"out_last_num_nonzero_subblocks_w", out_block_w / out_subblock_w});
                    args.insert({"out_last_subblock_w", out_subblock_w});
                    args.insert({"padded_subblock_tiles_addr_skip", 0u});
                    args.insert({"padded_block_tiles_w_skip", 0u});
                }
                if (!output_is_sharded) {
                    if (in1_idx == in1_end_idx and in0_idx == in0_end_idx) {
                        args.insert({"last_num_blocks_h_dim", last_out_num_blocks_h});
                        args.insert({"last_num_blocks_w_dim", last_out_num_blocks_w});
                    } else if (in0_idx == in0_end_idx) {
                        args.insert({"last_num_blocks_h_dim", last_out_num_blocks_h});
                        args.insert({"last_num_blocks_w_dim", out_num_blocks_x});
                    } else if (in1_idx == in1_end_idx) {
                        args.insert({"last_num_blocks_h_dim", out_num_blocks_y});
                        args.insert({"last_num_blocks_w_dim", last_out_num_blocks_w});
                    } else {
                        args.insert({"last_num_blocks_h_dim", out_num_blocks_y});
                        args.insert({"last_num_blocks_w_dim", out_num_blocks_x});
                    }
                }
                if ((core.x - start_core_x) <= half_core || (transpose_mcast and core.y == start_core_y)) {
                    in1_receiver_writer_run_args.runtime_arg_values.push_back(
                        m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = std::move(args)});
                } else {
                    in1_receiver_writer_other_run_args.runtime_arg_values.push_back(
                        m2::KernelRunArgs::NodeRuntimeArgs{.node = core, .args = std::move(args)});
                }
            }
        }
    }

    if (in0_block_sharded) {
        for (auto& k : kernels) {
            if (k.unique_id == RO_IN0_SENDER_KERNEL || k.unique_id == RO_IN0_NO_WORK_KERNEL) {
                k.advanced_options.num_runtime_varargs = in0_sharded_num_varargs;
            }
        }
    }

    run_args.kernel_run_args.push_back(std::move(in0_sender_run_args));
    if (has_in0_no_work) {
        run_args.kernel_run_args.push_back(std::move(in0_no_work_run_args));
    }
    if (has_in0_receiver) {
        run_args.kernel_run_args.push_back(std::move(in0_receiver_run_args));
    }
    if (has_in0_receiver_other) {
        run_args.kernel_run_args.push_back(std::move(in0_receiver_other_run_args));
    }
    run_args.kernel_run_args.push_back(std::move(in1_sender_writer_run_args));
    if (has_in1_receiver) {
        run_args.kernel_run_args.push_back(std::move(in1_receiver_writer_run_args));
    }
    if (has_in1_receiver_other) {
        run_args.kernel_run_args.push_back(std::move(in1_receiver_writer_other_run_args));
    }

    run_args.tensor_args.emplace(RO_IN0_TENSOR, in0_tensor);
    run_args.tensor_args.emplace(RO_IN1_TENSOR, in1_tensor);
    run_args.tensor_args.emplace(RO_OUT_TENSOR, out_tensor);
    run_args.tensor_args.emplace(RO_SPARSITY_TENSOR, in0_tensor);  // inert alias
    if (bias_tensor.has_value()) {
        run_args.tensor_args.emplace(RO_BIAS_TENSOR, *bias_tensor);
    }

    m2::ProgramSpec spec{
        .name = "matmul_multicore_reuse_mcast_2d",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

static ttnn::device_operation::CachedProgram<MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t>
matmul_multi_core_reuse_mcast_2d_optimized_(
    tt::tt_metal::Program& program,
    const ttnn::prim::qsr::MatmulParams& operation_attributes,
    const ttnn::prim::qsr::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
    using namespace tt;
    using namespace operations::experimental::quasar::matmul::utilities;

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& bias = tensor_args.optional_input_tensors.at(0);
    const auto& output = tensor_return_value.at(0).mesh_tensor();

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    TT_FATAL(operation_attributes.program_config.has_value(), "Error: program_config field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();
    bool transpose_a = operation_attributes.transpose_a;
    bool transpose_b = operation_attributes.transpose_b;

    auto program_config =
        std::get<operations::experimental::quasar::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(
            operation_attributes.program_config.value());

    if (!program_config.allowed_worker_cores.has_value()) {
        log_warning(
            tt::LogOp,
            "matmul_multi_core_reuse_mcast_2d_optimized_helper: program_config.allowed_worker_cores not populated; "
            "auto-populating from compute_with_storage_grid_size. Callers that bypass ttnn::prim::qsr::matmul() (e.g. "
            "CCL fused ops) should invoke ttnn::operations::experimental::quasar::matmul::normalize_program_config() "
            "on the program "
            "config first. This will become a hard error in a future release.");
        program_config.allowed_worker_cores = CoreRangeSet(CoreRange(
            CoreCoord(0, 0),
            CoreCoord(
                program_config.compute_with_storage_grid_size.x - 1,
                program_config.compute_with_storage_grid_size.y - 1)));
    }

    auto fuse_batch = program_config.fuse_batch;
    auto in0_block_w = program_config.in0_block_w;
    auto compute_with_storage_grid_size = program_config.allowed_worker_cores.value().bounding_box().grid_size();
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
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());  // in0
    const auto& in1_tensor = b.mesh_tensor();
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(in1_tensor.dtype());  // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());   // output

    const auto& a_shape_logical = get_matmul_tensor_logical_shape(a, transpose_a);
    // When transpose_a is true, the K dimension maps to the row dimension of the raw tile,
    // which is already zero-padded during tile layout conversion. pad_last_ktile operates on
    // columns, so applying it would incorrectly zero valid data that becomes output rows
    // after the compute kernel transposes the tile.
    const auto in0_last_ktile_w = transpose_a ? 0 : a_shape_logical[-1] % in0_tile.get_width();
    const auto in0_last_ktile_h = transpose_a ? a_shape_logical[-1] % in0_tile.get_width() : 0;
    TT_FATAL(
        in0_last_ktile_w == 0 || in0_last_ktile_h == 0,
        "At most one of in0_last_ktile_w ({}) and in0_last_ktile_h ({}) can be non-zero",
        in0_last_ktile_w,
        in0_last_ktile_h);

    ttsl::optional_reference<const tt_metal::MeshTensor> bias_mesh;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;  // bias; doesn't matter if bias=nullptr
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(
            c.storage_type() == StorageType::DEVICE,
            "Bias tensor must be on device, got storage type: {}",
            c.storage_type());
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.is_allocated(), "Operands to matmul need to be allocated in buffers on device!");

        bias_mesh = c.mesh_tensor();

        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    tt_metal::IDevice* device = a.device();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    const auto& in0_tensor = a.mesh_tensor();
    TT_FATAL(
        in0_tensor.mesh_buffer().device_local_size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        in0_tensor.mesh_buffer().device_local_size(),
        in0_single_tile_size);
    TT_FATAL(
        in1_tensor.mesh_buffer().device_local_size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        in1_tensor.mesh_buffer().device_local_size(),
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
    const auto Mt_per_batch = get_M_dim(a_shape_padded, in0_tile, false);
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
    //                      Sub-device start core
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord sub_device_start_core = {0, 0};
    if (operation_attributes.sub_device_id.has_value()) {
        auto sub_device_cores = device->worker_cores(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.sub_device_id.value());
        auto bbox = sub_device_cores.bounding_box();
        sub_device_start_core = bbox.start_coord;
    }

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
        dst_full_sync_en,
        B,
        Mt,
        Mt_per_batch,
        Nt,
        Kt,
        bcast_batch,
        transpose_a,
        transpose_b,
        ttnn::get_throttle_level(compute_kernel_config),
        in0_block_w,
        in0_last_ktile_w,
        in0_last_ktile_h,
        out_subblock_h,
        out_subblock_w,
        out_block_h,
        out_block_w,
        per_core_M,
        per_core_N,
        transpose_mcast,
        program_config.fused_activation,
        in0_tensor,
        in1_tensor,
        bias_mesh,
        output,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->tensor_spec().tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        untilize_out,
        fused_op_signaler,
        fused_matmul_bias_row_broadcastable(bias),
        sub_device_start_core);
}

void MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const shared_variables_t& shared_variables,
    const ttnn::prim::qsr::MatmulParams& /*operation_attributes*/,
    const ttnn::prim::qsr::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    reuse_mcast_optimized_helpers::override_runtime_arguments_impl(
        shared_variables, program, tensor_args, tensor_return_value);
}

ttnn::device_operation::ProgramArtifacts MatmulMultiCoreReuseMcast2DProgramFactory::create_program_artifacts(
    const ttnn::prim::qsr::MatmulParams& operation_attributes,
    const ttnn::prim::qsr::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    using namespace tt;
    using namespace operations::experimental::quasar::matmul::utilities;

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& bias = tensor_args.optional_input_tensors.at(0);
    const auto& output = tensor_return_value.at(0).mesh_tensor();

    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been populated");
    TT_FATAL(operation_attributes.program_config.has_value(), "Error: program_config field should have been populated");
    bool bcast_batch = operation_attributes.bcast_batch.value();
    bool transpose_a = operation_attributes.transpose_a;
    bool transpose_b = operation_attributes.transpose_b;

    auto program_config =
        std::get<operations::experimental::quasar::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(
            operation_attributes.program_config.value());

    auto fuse_batch = program_config.fuse_batch;
    auto in0_block_w = program_config.in0_block_w;
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

    const auto& a_shape_padded = get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& b_shape_padded = get_matmul_tensor_padded_shape(b, transpose_b);
    const auto in0_tile = get_matmul_tile(a, transpose_a);
    const auto in1_tile = get_matmul_tile(b, transpose_b);

    // cannot use the output tensor tile directly as that might be changed by user override
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});

    // CB dataformats
    const auto& in0_tensor = a.mesh_tensor();
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(in0_tensor.dtype());
    const auto& in1_tensor = b.mesh_tensor();
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(in1_tensor.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    const auto& a_shape_logical = get_matmul_tensor_logical_shape(a, transpose_a);
    const auto in0_last_ktile_w = transpose_a ? 0 : a_shape_logical[-1] % in0_tile.get_width();
    const auto in0_last_ktile_h = transpose_a ? a_shape_logical[-1] % in0_tile.get_width() : 0;
    TT_FATAL(
        in0_last_ktile_w == 0 || in0_last_ktile_h == 0,
        "At most one of in0_last_ktile_w ({}) and in0_last_ktile_h ({}) can be non-zero",
        in0_last_ktile_w,
        in0_last_ktile_h);

    ttsl::optional_reference<const tt_metal::MeshTensor> bias_mesh;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    if (bias.has_value()) {
        const auto& c = bias.value();
        bias_mesh = c.mesh_tensor();
        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    tt_metal::IDevice* device = &in0_tensor.mutable_device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    const auto B = fuse_batch ? 1 : get_batch_size(a_shape_padded);
    const auto Mt = get_M_dim(a_shape_padded, in0_tile, fuse_batch);
    const auto Mt_per_batch = get_M_dim(a_shape_padded, in0_tile, false);
    const auto Kt = get_K_dim(a_shape_padded, in0_tile);
    const auto Nt = get_N_dim(b_shape_padded, in1_tile);

    // When a sub-device is present use its bounding-box start; otherwise fall
    // back to allowed_worker_cores start so non-(0,0) placements work correctly.
    CoreCoord sub_device_start_core = program_config.allowed_worker_cores.has_value()
                                          ? program_config.allowed_worker_cores.value().bounding_box().start_coord
                                          : CoreCoord{0, 0};
    if (operation_attributes.sub_device_id.has_value()) {
        auto sub_device_cores = device->worker_cores(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.sub_device_id.value());
        auto bbox = sub_device_cores.bounding_box();
        sub_device_start_core = bbox.start_coord;
    }

    (void)dst_full_sync_en;
    return CMAKE_UNIQUE_NAMESPACE::create_program_mcast_in0_in1_artifacts(
        a,
        device,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        B,
        Mt,
        Mt_per_batch,
        Nt,
        Kt,
        bcast_batch,
        transpose_a,
        transpose_b,
        ttnn::get_throttle_level(compute_kernel_config),
        in0_block_w,
        in0_last_ktile_w,
        in0_last_ktile_h,
        out_subblock_h,
        out_subblock_w,
        out_block_h,
        out_block_w,
        per_core_M,
        per_core_N,
        transpose_mcast,
        program_config.fused_activation,
        in0_tensor,
        in1_tensor,
        bias_mesh,
        output,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->tensor_spec().tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        untilize_out,
        fused_matmul_bias_row_broadcastable(bias),
        sub_device_start_core);
}

ttnn::device_operation::CachedProgram<MatmulMultiCoreReuseMcast2DProgramFactory::shared_variables_t>
matmul_multi_core_reuse_mcast_2d_optimized_helper(
    tt::tt_metal::Program& program, /* Take programa as input by reference */
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::experimental::quasar::matmul::MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
    auto attributes = ttnn::prim::qsr::MatmulParams{.program_config = program_config, .bcast_batch = broadcast_batch};
    attributes.compute_kernel_config = compute_kernel_config;
    attributes.untilize_out = untilize_out;

    auto output_tensors = std::vector<ttnn::Tensor>{output_tensor};
    return matmul_multi_core_reuse_mcast_2d_optimized_(
        program, attributes, ttnn::prim::qsr::MatmulInputs{{a, b}, {bias}, {}}, output_tensors, fused_op_signaler);
}

}  // namespace ttnn::prim::qsr
