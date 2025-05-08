// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt;
using namespace tt::constants;
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace reuse_mcast_1d_optimized_helpers {

uint32_t get_preferred_noc(
    const ttnn::CoreCoord src,
    const ttnn::CoreCoord dst,
    const tt_metal::IDevice* device,
    const bool use_dedicated_noc = false) {
    /*
        NOC0: Preferred +x -> +y
        NOC1: Preferred -y -> -x
    */

    uint32_t src_x = src.x, src_y = src.y;
    uint32_t dst_x = dst.x, dst_y = dst.y;

    uint32_t MAX_X = device->grid_size().x;
    uint32_t MAX_Y = device->grid_size().y;

    // Get the wrapped distances
    uint32_t dist_right = src_x <= dst_x ? dst_x - src_x : MAX_X - src_x + dst_x;
    uint32_t dist_left = src_x < dst_x ? src_x + MAX_X - dst_x : src_x - dst_x;

    uint32_t dist_bottom = src_y <= dst_y ? dst_y - src_y : MAX_Y - src_y + dst_y;
    uint32_t dist_top = src_y < dst_y ? src_y + MAX_Y - dst_y : src_y - dst_y;

    uint32_t dist_noc_0 = dist_right + dist_bottom;
    uint32_t dist_noc_1 = dist_top + dist_left;

    uint32_t noc = dist_noc_0 < dist_noc_1 ? 0 : 1;

    // Debug print if needed
    // std::cout << "src: (" << src_x << ", " << src_y << "), dst: (" << dst_x << ", " << dst_y << "), noc: " << noc <<
    // std::endl;

    return use_dedicated_noc ? 1 : noc;
}

tt::tt_metal::operation::ProgramWithCallbacks create_program_mcast_in0(
    tt_metal::Program& program,
    const tt::tt_metal::Tensor& a,
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    CoreCoord compute_with_storage_grid_size,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    tt_metal::Buffer* bias_buffer,
    tt_metal::Buffer* out_buffer,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool in0_is_sharded,
    bool in1_is_sharded,
    bool bias_is_sharded,
    bool output_is_sharded,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler) {
    using tt::tt_metal::num_cores_to_corerangeset;

    // currently only support transpose of the full tile
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    bool fuse_op = fused_op_signaler.has_value();

    uint32_t num_blocks = K / in0_block_w;
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles *= ttnn::operations::matmul::MCAST_INPUT_BUFFERING_DEPTH;
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    uint32_t in0_shard_height_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
        in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_tile_shape()[0];
        in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    }
    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles *= ttnn::operations::matmul::MCAST_INPUT_BUFFERING_DEPTH;
    }
    if (in1_is_sharded) {
        uint32_t in1_shard_height_in_tiles = in1_buffer->shard_spec().shape()[0] / in1_tile.get_tile_shape()[0];
        in1_CB_tiles = per_core_N * in1_shard_height_in_tiles;
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

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    CoreCoord start_core = {0, 0};
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = compute_with_storage_grid_size.x;

    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores_with_work = num_blocks_total;

    uint32_t in0_sender_num_cores = in0_is_sharded ? a.shard_spec().value().grid.num_cores() : 1;
    uint32_t num_cores = in0_is_sharded ? std::max(num_cores_with_work, in0_sender_num_cores) : num_cores_with_work;

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        num_cores_to_corerangeset(start_core, num_cores, compute_with_storage_grid_size, row_major);

    CoreRangeSet in0_mcast_sender_cores =
        num_cores_to_corerangeset(in0_sender_num_cores, compute_with_storage_grid_size, row_major);
    CoreCoord in0_mcast_sender_cores_grid = in0_mcast_sender_cores.bounding_box().grid_size();

    CoreRangeSet all_cores_with_work =
        num_cores_to_corerangeset(num_cores_with_work, compute_with_storage_grid_size, row_major);
    CoreRange in0_mcast_receiver_cores_bounding_box = all_cores_with_work.bounding_box();
    uint32_t in0_mcast_receiver_num_cores = in0_mcast_receiver_cores_bounding_box.size();  // always mcast to full grid
    uint32_t in0_mcast_receiver_num_dests = std::min(
        in0_mcast_receiver_num_cores,
        num_cores);  // should always be number of cores in receiver grid up to number of active cores

    CoreRangeSet in0_mcast_cores_with_work_and_in_receiver_grid;
    CoreRangeSet in0_mcast_cores_without_work_and_in_receiver_grid;
    CoreRangeSet in0_mcast_cores_without_work_and_not_in_receiver_grid;
    CoreRangeSet in0_mcast_receivers;
    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;
    if (in0_is_sharded) {
        in0_mcast_cores_with_work_and_in_receiver_grid = all_cores_with_work;

        if (in0_mcast_receiver_num_dests > num_cores_with_work) {
            const uint32_t in0_mcast_cores_without_work_and_in_receiver_grid_num_cores =
                in0_mcast_receiver_num_dests - num_cores_with_work;
            uint32_t core_idx_x = num_cores_with_work % num_cores_c;
            uint32_t core_idx_y = num_cores_with_work / num_cores_c;
            CoreCoord start_core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            in0_mcast_cores_without_work_and_in_receiver_grid = num_cores_to_corerangeset(
                start_core,
                in0_mcast_cores_without_work_and_in_receiver_grid_num_cores,
                compute_with_storage_grid_size,
                row_major);
        }

        if (in0_sender_num_cores > in0_mcast_receiver_num_dests) {
            const uint32_t in0_mcast_cores_without_work_and_not_in_receiver_grid_num_cores =
                in0_sender_num_cores - in0_mcast_receiver_num_dests;
            uint32_t core_idx_x = in0_mcast_receiver_num_dests % num_cores_c;
            uint32_t core_idx_y = in0_mcast_receiver_num_dests / num_cores_c;
            CoreCoord start_core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            in0_mcast_cores_without_work_and_not_in_receiver_grid = num_cores_to_corerangeset(
                start_core,
                in0_mcast_cores_without_work_and_not_in_receiver_grid_num_cores,
                compute_with_storage_grid_size,
                row_major);
        }

        in0_mcast_noc_x.reserve(in0_mcast_sender_cores_grid.x);
        in0_mcast_noc_y.reserve(in0_mcast_sender_cores_grid.y);
        for (uint32_t core_idx_x = 0; core_idx_x < in0_mcast_sender_cores_grid.x; ++core_idx_x) {
            in0_mcast_noc_x.push_back(device->worker_core_from_logical_core({core_idx_x, 0}).x);
        }
        for (uint32_t core_idx_y = 0; core_idx_y < in0_mcast_sender_cores_grid.y; ++core_idx_y) {
            in0_mcast_noc_y.push_back(device->worker_core_from_logical_core({0, core_idx_y}).y);
        }
    } else {
        in0_mcast_cores_with_work_and_in_receiver_grid = CoreRangeSet({CoreRange(start_core, start_core)});
        if (in0_mcast_receiver_num_cores > 1) {
            auto receiver_start_core = start_core.x != (compute_with_storage_grid_size.x - 1)
                                           ? CoreCoord{start_core.x + 1, start_core.y}
                                           : CoreCoord{start_core.x, start_core.y + 1};
            in0_mcast_receivers = num_cores_to_corerangeset(
                receiver_start_core, num_cores - 1, compute_with_storage_grid_size, row_major);
        }
    }

    // Mcast args
    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    CoreCoord top_left_core = in0_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in0_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool in3_is_dram = true;
    if (bias_buffer != nullptr) {
        in3_is_dram = bias_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    }
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM;

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    std::vector<uint32_t> in0_sender_compile_time_args;
    if (in0_is_sharded) {
        in0_sender_compile_time_args = {
            (std::uint32_t)1,  // core_has_output_block_work
            (std::uint32_t)1,  // core_in_in0_receiver_mcast_grid

            (std::uint32_t)in0_block_num_tiles,                         // in0_block_num_tiles
            (std::uint32_t)in0_block_num_tiles * in0_single_tile_size,  // in0_block_size_bytes
            // in0/in1 common args
            (std::uint32_t)num_blocks,        // num_blocks
            (std::uint32_t)out_num_blocks_x,  // num_blocks_x
            (std::uint32_t)out_num_blocks_y,  // num_blocks_y
            // in0 mcast args
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_num_dests,  // in0_mcast_num_dests
            (std::uint32_t)in0_mcast_receiver_num_cores,  // in0_mcast_num_cores
            (std::uint32_t)(in0_mcast_sender_cores_grid.x),
            (std::uint32_t)(in0_mcast_sender_cores_grid.y),
            (std::uint32_t)(false),
            (std::uint32_t)(in0_shard_width_in_tiles),
            (std::uint32_t)(in0_shard_height_in_tiles),
            (std::uint32_t)(in0_block_w),
            (std::uint32_t)in0_block_h,  // in0_block_h

            // batch args
            (std::uint32_t)B  // batch
        };
    } else {
        in0_sender_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t)in0_is_dram,

            // in0 tensor args
            (std::uint32_t)1,                // in0_tensor_stride_w
            (std::uint32_t)K,                // in0_tensor_stride_h
            (std::uint32_t)in0_block_w,      // in0_tensor_next_block_stride
            (std::uint32_t)K * in0_block_h,  // in0_tensor_next_h_dim_block_stride
            // in0 block args
            (std::uint32_t)in0_block_w,          // in0_block_w
            (std::uint32_t)in0_block_h,          // in0_block_h
            (std::uint32_t)in0_block_num_tiles,  // in0_block_num_tiles
            (std::uint32_t)false,                // extract_shard_sub_blocks (not used for interleaved)
            (std::uint32_t)0,                    // shard_width_in_tiles (not used for interleaved)
            (std::uint32_t)0,                    // shard_height_in_tiles (not used for interleaved)
            // in0/in1 common args
            (std::uint32_t)num_blocks,        // num_blocks
            (std::uint32_t)out_num_blocks_x,  // num_blocks_x
            (std::uint32_t)out_num_blocks_y,  // num_blocks_y
            // in0 mcast args
            (std::uint32_t)in0_mcast_sender_semaphore_id,
            (std::uint32_t)in0_mcast_receiver_semaphore_id,
            (std::uint32_t)num_cores - 1,                     // in0_mcast_num_dests
            (std::uint32_t)in0_mcast_receiver_num_cores - 1,  // in0_mcast_num_cores
            // batch args
            (std::uint32_t)M * K,  // MtKt
            (std::uint32_t)B       // batch
        };
    }
    in0_sender_compile_time_args.push_back((std::uint32_t)fuse_op);

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)in1_is_dram,
        (std::uint32_t)out_is_dram,

        // READER
        // in1 tensor args
        (std::uint32_t)1,                // in1_tensor_stride_w
        (std::uint32_t)N,                // in1_tensor_stride_h
        (std::uint32_t)in0_block_w * N,  // in1_tensor_next_block_stride
        (std::uint32_t)in1_block_w,      // in1_tensor_next_w_dim_block_stride
        // in1 block args
        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,                // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in1 mcast args
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,  // in1_mcast_num_dests
        (std::uint32_t)0,  // in1_mcast_num_cores
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
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)in3_is_dram);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
    } else {
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }

    in1_sender_writer_compile_time_args.push_back((std::uint32_t)fuse_op);

    std::vector<uint32_t> in0_receiver_compile_time_args = {
        // in0 block args
        (std::uint32_t)in0_block_num_tiles,  // in0_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in0 mcast args
        (std::uint32_t)in0_mcast_sender_semaphore_id,
        (std::uint32_t)in0_mcast_receiver_semaphore_id,
        // batch args
        (std::uint32_t)B  // batch
    };

    std::map<string, string> mm_kernel_defines;
    std::map<string, string> mm_kernel_in0_sender_writer_defines;
    std::map<string, string> mm_kernel_in1_sender_writer_defines;
    if (bias_buffer != nullptr) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
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
    if (in1_transpose_tile) {
        mm_kernel_defines["IN1_TRANSPOSE_TILE"] = "1";
    }

    bmm_op_utils::add_stagger_defines_if_needed(device->arch(), num_cores, mm_kernel_defines);

    if (in1_is_sharded) {
        mm_kernel_in1_sender_writer_defines["IN1_SHARDED"] = "1";
    }

    if (bias_is_sharded) {
        mm_kernel_in1_sender_writer_defines["BIAS_SHARDED"] = "1";
    }

    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
    }

    // TODO: SKIP_MCAST flag isn't used for the sharded reader kernel because internal mcast logic already works without
    // skipping We can use this flag to turn off unnecessary mcast overhead if necessary
    if (in0_mcast_receiver_num_cores == 1) {
        mm_kernel_in0_sender_writer_defines["SKIP_MCAST"] = "1";
    }

    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());

    if (fuse_op) {
        // Create semaphores
        fused_op_signaler->init_fused_op(
            program,
            device,
            in0_mcast_sender_cores,
            in0_is_sharded ? ttnn::experimental::ccl::FusedOpSignalerMode::SINGLE
                           : ttnn::experimental::ccl::FusedOpSignalerMode::MULTI);
    }

    auto mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id = tt_metal::CreateKernel(
        program,
        in0_is_sharded
            ? "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
              "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp"
            : "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        in0_mcast_cores_with_work_and_in_receiver_grid,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = mm_kernel_in0_sender_writer_defines});

    tt::tt_metal::KernelHandle mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid_id = 0;
    tt::tt_metal::KernelHandle mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id = 0;
    if (in0_is_sharded) {
        if (in0_mcast_cores_without_work_and_in_receiver_grid.num_cores() > 0) {
            in0_sender_compile_time_args[0] = 0;  // core_has_output_block_work
            in0_sender_compile_time_args[1] = 1;  // core_in_in0_receiver_mcast_grid
            mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",
                in0_mcast_cores_without_work_and_in_receiver_grid,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = in0_noc,
                    .compile_args = in0_sender_compile_time_args,
                    .defines = mm_kernel_in0_sender_writer_defines});
        }
        if (in0_mcast_cores_without_work_and_not_in_receiver_grid.num_cores() > 0) {
            in0_sender_compile_time_args[0] = 0;  // core_has_output_block_work
            in0_sender_compile_time_args[1] = 0;  // core_in_in0_receiver_mcast_grid
            mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id = tt_metal::CreateKernel(
                program,
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp",
                in0_mcast_cores_without_work_and_not_in_receiver_grid,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = in0_noc,
                    .compile_args = in0_sender_compile_time_args,
                    .defines = mm_kernel_in0_sender_writer_defines});
        }
    }

    tt::tt_metal::KernelHandle mm_kernel_in0_receiver_id = 0;
    if (!in0_is_sharded and in0_mcast_receivers.num_cores() > 0) {
        mm_kernel_in0_receiver_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
            in0_mcast_receivers,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_receiver_compile_time_args});
    }

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        all_cores_with_work,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_kernel_in1_sender_writer_defines});

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

        num_blocks,        // num_blocks
        out_num_blocks_x,  // out_num_blocks_x
        out_num_blocks_y,  // out_num_blocks_y

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out  // untilize_out
    };

    // Create compute kernel
    // bool fp32_dest_acc_en = false;
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
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
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

    if (in1_is_sharded) {
        src1_cb_config = src1_cb_config.set_globally_allocated_address(*in1_buffer);
    }

    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        in1_CB_size / in1_single_tile_size,
        in1_CB_size);

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CBHandle cb_src2 = 0;
    if (in0_is_sharded) {
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
                               .set_page_size(interm0_cb_index, interm0_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_tile_dims(interm0_cb_index, output_tile);
    }

    if (output_is_sharded) {
        output_cb_config = output_cb_config.set_globally_allocated_address(*out_buffer);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        output_cb_index,
        output_single_tile_size,
        out_CB_size / output_single_tile_size,
        out_CB_size);

    tt_metal::CBHandle cb_src3 = 0;
    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = tt::CBIndex::c_3;
        tt_metal::CircularBufferConfig cb_src3_config =
            tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
                .set_page_size(src3_cb_index, bias_single_tile_size)
                .set_tile_dims(src3_cb_index, bias_tile);

        if (bias_is_sharded) {
            cb_src3_config = cb_src3_config.set_globally_allocated_address(*bias_buffer);
        }

        cb_src3 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src3_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src3_cb_index,
            bias_single_tile_size,
            in3_CB_size / bias_single_tile_size,
            in3_CB_size);
    }

    // Parameters for last row, col, or block, no need to re-calc h-dim since there's no split on height
    uint32_t last_per_core_N = N % per_core_N == 0 ? per_core_N : N % per_core_N;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_out_num_blocks_w = (last_per_core_N - 1) / out_block_w + 1;
    uint32_t last_block_num_nonzero_subblocks_w = (last_out_block_w - 1) / out_subblock_w + 1;
    uint32_t last_subblock_of_last_block_w =
        last_out_block_w % out_subblock_w == 0 ? out_subblock_w : last_out_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        output_single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (out_block_w / out_subblock_w - last_block_num_nonzero_subblocks_w);

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i % num_blocks_x;
        uint32_t output_idx_y = i / num_blocks_x;

        if (in0_is_sharded) {
            std::vector<uint32_t> mm_in0_sender_args;
            mm_in0_sender_args.reserve(5 + in0_mcast_noc_x.size() + in0_mcast_noc_y.size());
            mm_in0_sender_args.push_back(i);
            mm_in0_sender_args.push_back(start_core_noc.x);
            mm_in0_sender_args.push_back(start_core_noc.y);
            mm_in0_sender_args.push_back(end_core_noc.x);
            mm_in0_sender_args.push_back(end_core_noc.y);
            mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
            mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());

            if (fuse_op) {
                fused_op_signaler->push_matmul_fused_op_rt_args(mm_in0_sender_args, false);
            }

            if (i < num_cores_with_work) {
                tt_metal::SetRuntimeArgs(
                    program,
                    mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id,
                    core,
                    mm_in0_sender_args);  // RISCV_0_default
            } else if (i < in0_mcast_receiver_num_dests) {
                tt_metal::SetRuntimeArgs(
                    program,
                    mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid_id,
                    core,
                    mm_in0_sender_args);  // RISCV_0_default
            } else {
                tt_metal::SetRuntimeArgs(
                    program,
                    mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id,
                    core,
                    mm_in0_sender_args);  // RISCV_0_default
            }
        }
        // in0 sender and in1 sender
        else if (core == start_core) {
            std::vector<uint32_t> mm_in0_sender_args = {
                // in0 tensor args
                (std::uint32_t)in0_buffer->address(),
                (std::uint32_t)K * per_core_M * output_idx_y,  // in0_tensor_start_tile_id
                // in0 mcast args
                (std::uint32_t)start_core_noc.x,  // in0_mcast_dest_noc_start_x
                (std::uint32_t)start_core_noc.y,  // in0_mcast_dest_noc_start_y
                (std::uint32_t)end_core_noc.x,    // in0_mcast_dest_noc_end_x
                (std::uint32_t)end_core_noc.y,    // in0_mcast_dest_noc_end_y

                // padding args
                (std::uint32_t)out_block_h  // last_block_h
            };

            if (fuse_op) {
                fused_op_signaler->push_matmul_fused_op_rt_args(mm_in0_sender_args, false);
            }

            tt_metal::SetRuntimeArgs(
                program,
                mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id,
                core,
                mm_in0_sender_args);  // RISCV_0_default
        }
        // in0 receiver and in 1 sender
        else {
            std::vector<uint32_t> mm_in0_receiver_args = {
                // in0 mcast args
                (std::uint32_t)top_left_core_physical.x,  // in0_mcast_sender_noc_x
                (std::uint32_t)top_left_core_physical.y   // in0_mcast_sender_noc_y
            };
            tt_metal::SetRuntimeArgs(
                program, mm_kernel_in0_receiver_id, core, mm_in0_receiver_args);  // RISCV_1_default
        }
        if (i < num_cores_with_work) {
            std::vector<uint32_t> mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)in1_buffer->address(),
                (std::uint32_t)per_core_N * output_idx_x,  // in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)0,  // in1_mcast_dest_noc_start_x
                (std::uint32_t)0,  // in1_mcast_dest_noc_start_y
                (std::uint32_t)0,  // in1_mcast_dest_noc_end_x
                (std::uint32_t)0,  // in1_mcast_dest_noc_end_y

                // WRITER
                // out tensor args
                (std::uint32_t)out_buffer->address(),
                (std::uint32_t)output_idx_x * per_core_N + output_idx_y * per_core_M * N  // out_tensor_start_tile_id
            };

            if (output_idx_x == num_blocks_x - 1) {
                // padding args (READER)
                mm_in1_sender_writer_args.push_back(last_out_block_w);

                // padding args (WRITER)
                mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);
                mm_in1_sender_writer_args.push_back(out_subblock_h);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);  // out_num_nonzero_subblocks_w
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
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);  // out_num_nonzero_subblocks_w
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);
                mm_in1_sender_writer_args.push_back(out_subblock_w);
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(0);
            }

            if (bias_buffer != nullptr) {
                mm_in1_sender_writer_args.push_back((std::uint32_t)bias_buffer->address());
                mm_in1_sender_writer_args.push_back(
                    (std::uint32_t)per_core_N * output_idx_x);  // in3_tensor_start_tile_id
            } else {                                            // Placeholder args
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(0);
            }
            if (!output_is_sharded) {
                if (output_idx_x == num_blocks_x - 1) {
                    mm_in1_sender_writer_args.push_back(last_out_num_blocks_w);
                } else {
                    mm_in1_sender_writer_args.push_back(out_num_blocks_x);
                }
            }

            if (fuse_op) {
                fused_op_signaler->push_matmul_fused_op_rt_args(mm_in1_sender_writer_args, true);
            }

            tt_metal::SetRuntimeArgs(
                program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);  // RISCV_0_default
        }
    }

    auto override_runtime_arguments_callback =
        [mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id,
         mm_kernel_in1_sender_writer_id,
         cb_src1,
         cb_src2,
         cb_src3,
         cb_output,
         start_core,
         cores,
         num_cores_with_work](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<tt::tt_metal::Tensor>& input_tensors,
            const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
            const std::vector<tt::tt_metal::Tensor>& output_tensors) {
            TT_ASSERT(input_tensors.size() + optional_input_tensors.size() == 3);
            TT_ASSERT(output_tensors.size() == 1);

            auto src_buffer_a = input_tensors.at(0).buffer();
            auto src_buffer_b = input_tensors.at(1).buffer();
            auto bias_tensor = optional_input_tensors.at(0);

            std::optional<tt::tt_metal::Buffer*> bias_buffer;
            if (bias_tensor.has_value()) {
                bias_buffer = bias_tensor.value().buffer();
            }

            auto dst_buffer = output_tensors.at(0).buffer();

            bool src0_sharded = input_tensors[0].is_sharded();
            bool src1_sharded = input_tensors[1].is_sharded();
            bool out_sharded = output_tensors[0].is_sharded();

            // Manually unroll sender core
            if (src0_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_src2, *src_buffer_a);
            } else {
                // in0 sender
                auto& reader_sender_runtime_args =
                    GetRuntimeArgs(program, mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id, start_core);
                reader_sender_runtime_args[0] = src_buffer_a->address();
            }

            if (src1_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_src1, *src_buffer_b);
            }

            if (bias_tensor.has_value() && bias_tensor.value().is_sharded()) {
                UpdateDynamicCircularBufferAddress(program, cb_src3, *bias_buffer.value());
            }

            auto& writer_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in1_sender_writer_id);

            for (uint32_t i = 0; i < num_cores_with_work; ++i) {
                const auto& core = cores[i];

                auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];

                // in1 sender
                writer_runtime_args[0] = src_buffer_b->address();
                writer_runtime_args[6] = dst_buffer->address();
                if (bias_tensor.has_value()) {
                    writer_runtime_args[17] = (*bias_buffer)->address();
                }
            }

            if (out_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

tt::tt_metal::operation::ProgramWithCallbacks create_program_mcast_in1(
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    CoreCoord compute_with_storage_grid_size,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    tt_metal::Buffer* bias_buffer,
    tt_metal::Buffer* out_buffer,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool in0_is_sharded,
    bool output_is_sharded,
    bool untilize_out) {
    // currently only support transpose of the full tile
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    tt_metal::Program program{};

    bool fuse_op = false;

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

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;
    uint32_t in0_num_blocks_y = per_core_M / out_block_h;
    uint32_t in1_num_blocks_x = per_core_N / out_block_w;
    uint32_t out_num_blocks_x = in1_num_blocks_x;
    uint32_t out_num_blocks_y = in0_num_blocks_y;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (in0_is_sharded) {
        in0_CB_tiles = num_blocks * per_core_M * in0_block_w * B;
    } else if (B * num_blocks > 1) {
        in0_CB_tiles = in0_CB_tiles * 2;  // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    bool extract_shard_sub_blocks = false;
    uint32_t in0_shard_height_in_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    if (in0_is_sharded) {
        in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_tile_shape()[0];
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
        // NOTE: Criteria for extract_shard_sub_blocks is different from mcast in0
        // In the reader kernel, always need to copy to cb0 even for height=1 shards since we may not always do mcast
        // In mcast in0 sharded reader kernel, this is handled by mcast with loopback src
        // For mcast in1, if we don't need to extract_shard_sub_blocks, set the sharded in0 cb to cb0
        // For mcast in0, sharded in0 cb is always cb2
        if (in0_shard_width_in_tiles / in0_block_w > 1) {
            extract_shard_sub_blocks = true;
        }
    }
    uint32_t in2_CB_tiles = in0_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in1_block_tiles = out_block_w * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles = in1_CB_tiles * 2;  // double buffer
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

    uint32_t in3_block_tiles = out_block_w;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    CoreCoord start_core = {0, 0};
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;

    uint32_t num_blocks_y = (M - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (N - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
    uint32_t num_cores = num_blocks_total;

    constexpr bool row_major = true;
    CoreRangeSet all_cores =
        tt::tt_metal::num_cores_to_corerangeset(start_core, num_cores, compute_with_storage_grid_size, row_major);
    CoreRange in1_mcast_receiver_cores_bounding_box = all_cores.bounding_box();
    uint32_t in1_mcast_receiver_num_cores = in1_mcast_receiver_cores_bounding_box.size();  // always mcast to full grid

    CoreRange in1_mcast_sender(start_core, start_core);
    CoreRangeSet in1_mcast_receivers;
    if (in1_mcast_receiver_num_cores > 1) {
        auto receiver_start_core = start_core.x != (compute_with_storage_grid_size.x - 1)
                                       ? CoreCoord{start_core.x + 1, start_core.y}
                                       : CoreCoord{start_core.x, start_core.y + 1};
        in1_mcast_receivers = tt::tt_metal::num_cores_to_corerangeset(
            receiver_start_core, num_cores - 1, compute_with_storage_grid_size, row_major);
    }

    // Mcast args
    auto in1_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in1_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    CoreCoord top_left_core = in1_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in1_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool in3_is_dram = true;
    if (bias_buffer != nullptr) {
        in3_is_dram = bias_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    }
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> in0_sender_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)in0_is_dram,

        // in0 tensor args
        (std::uint32_t)1,                // in0_tensor_stride_w
        (std::uint32_t)K,                // in0_tensor_stride_h
        (std::uint32_t)in0_block_w,      // in0_tensor_next_block_stride
        (std::uint32_t)K * in0_block_h,  // in0_tensor_next_h_dim_block_stride
        // in0 block args
        (std::uint32_t)in0_block_w,                // in0_block_w
        (std::uint32_t)in0_block_h,                // in0_block_h
        (std::uint32_t)in0_block_w * in0_block_h,  // in0_block_num_tiles
        (std::uint32_t)extract_shard_sub_blocks,
        (std::uint32_t)in0_shard_width_in_tiles,
        (std::uint32_t)in0_shard_height_in_tiles,
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in0 mcast args
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,  // in0_mcast_num_dests
        (std::uint32_t)0,  // in0_mcast_num_cores
        // batch args
        (std::uint32_t)M * K,  // MtKt
        (std::uint32_t)B       // batch
    };
    in0_sender_compile_time_args.push_back((std::uint32_t)fuse_op);

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)in1_is_dram,
        (std::uint32_t)out_is_dram,

        // READER
        // in1 tensor args
        (std::uint32_t)1,                // in1_tensor_stride_w
        (std::uint32_t)N,                // in1_tensor_stride_h
        (std::uint32_t)in0_block_w * N,  // in1_tensor_next_block_stride
        (std::uint32_t)in1_block_w,      // in1_tensor_next_w_dim_block_stride
        // in1 block args
        (std::uint32_t)in1_block_w,                // in1_block_w
        (std::uint32_t)in0_block_w,                // in1_block_h
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
        // in1 mcast args
        (std::uint32_t)in1_mcast_sender_semaphore_id,
        (std::uint32_t)in1_mcast_receiver_semaphore_id,
        (std::uint32_t)num_cores - 1,                     // in1_mcast_num_dests
        (std::uint32_t)in1_mcast_receiver_num_cores - 1,  // in1_mcast_num_cores
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
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)in3_is_dram);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
    } else {
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
        in1_sender_writer_compile_time_args.push_back(0);  // Placeholder; not used
    }

    in1_sender_writer_compile_time_args.push_back((std::uint32_t)fuse_op);

    std::vector<uint32_t> in1_receiver_writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)out_is_dram,

        // READER
        // in1 block args
        (std::uint32_t)in1_block_w * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,        // num_blocks
        (std::uint32_t)out_num_blocks_x,  // out_num_blocks_x
        (std::uint32_t)out_num_blocks_y,  // out_num_blocks_y
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
    }

    std::map<string, string> mm_kernel_defines;
    std::map<string, string> mm_kernel_in0_sender_defines;
    std::map<string, string> mm_kernel_in1_sender_writer_defines;
    std::map<string, string> mm_kernel_in1_receiver_writer_defines;
    if (bias_buffer != nullptr) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
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
    if (in1_transpose_tile) {
        mm_kernel_defines["IN1_TRANSPOSE_TILE"] = "1";
    }

    bmm_op_utils::add_stagger_defines_if_needed(device->arch(), num_cores, mm_kernel_defines);

    if (in0_is_sharded) {
        mm_kernel_in0_sender_defines["IN0_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
        mm_kernel_in1_receiver_writer_defines["OUT_SHARDED"] = "1";
    }

    mm_kernel_in0_sender_defines["SKIP_MCAST"] = "1";

    if (in1_mcast_receiver_num_cores == 1) {
        mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";
    }

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());

    auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = mm_kernel_in0_sender_defines});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        in1_mcast_sender,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_kernel_in1_sender_writer_defines});

    tt::tt_metal::KernelHandle mm_kernel_in1_receiver_writer_id = 0;
    if (in1_mcast_receivers.num_cores() > 0) {
        mm_kernel_in1_receiver_writer_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
            "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
            in1_mcast_receivers,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = in1_noc,
                .compile_args = in1_receiver_writer_compile_time_args,
                .defines = mm_kernel_in1_receiver_writer_defines});
    }

    // Compute kernel compile time args

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
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

        num_blocks,        // num_blocks
        out_num_blocks_x,  // out_num_blocks_x
        out_num_blocks_y,  // out_num_blocks_y

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out  // untilize_out
    };

    // Create compute kernel
    // bool fp32_dest_acc_en = false;
    // Gelu currently has better accuracy when run in approx mode
    // bool math_approx_mode = false;
    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores,
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
    if (in0_is_sharded and not extract_shard_sub_blocks) {
        src0_cb_config = src0_cb_config.set_globally_allocated_address(*in0_buffer);
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src0_cb_index,
        in0_single_tile_size,
        in0_CB_size / in0_single_tile_size,
        in0_CB_size);

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CBHandle cb_src2 = 0;
    if (in0_is_sharded and extract_shard_sub_blocks) {  // in0_is_sharded is technically redundant
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
    }

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        in1_CB_size / in1_single_tile_size,
        in1_CB_size);

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
                               .set_page_size(interm0_cb_index, interm0_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_tile_dims(interm0_cb_index, output_tile);
    }

    if (output_is_sharded) {
        output_cb_config = output_cb_config.set_globally_allocated_address(*out_buffer);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        output_cb_index,
        output_single_tile_size,
        out_CB_size / output_single_tile_size,
        out_CB_size);

    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = tt::CBIndex::c_3;
        tt_metal::CircularBufferConfig cb_src3_config =
            tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
                .set_page_size(src3_cb_index, bias_single_tile_size)
                .set_tile_dims(src3_cb_index, bias_tile);
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
    uint32_t last_per_core_M = M % per_core_M == 0 ? per_core_M : M % per_core_M;
    uint32_t last_out_block_h = last_per_core_M % out_block_h == 0 ? out_block_h : last_per_core_M % out_block_h;
    uint32_t last_out_num_blocks_h = (last_per_core_M - 1) / out_block_h + 1;
    uint32_t last_block_num_nonzero_subblocks_h = (last_out_block_h - 1) / out_subblock_h + 1;
    uint32_t last_subblock_of_last_block_h =
        last_out_block_h % out_subblock_h == 0 ? out_subblock_h : last_out_block_h % out_subblock_h;
    uint32_t last_block_padded_block_tiles_h_skip =
        (out_block_h / out_subblock_h - last_block_num_nonzero_subblocks_h) * (out_block_w * out_subblock_h);

    CoreCoord start_core_noc = bottom_right_core_physical;
    CoreCoord end_core_noc = top_left_core_physical;
    if (in1_noc == tt::tt_metal::NOC::NOC_0) {
        std::swap(start_core_noc, end_core_noc);
    }

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i / num_blocks_y;
        uint32_t output_idx_y = i % num_blocks_y;

        // in0 sender and in1 sender
        if (core == start_core) {
            std::vector<uint32_t> mm_in1_sender_writer_args = {
                // READER
                // in1 tensor args
                (std::uint32_t)in1_buffer->address(),
                (std::uint32_t)per_core_N * output_idx_x,  // in1_tensor_start_tile_id
                // in1 mcast args
                (std::uint32_t)start_core_noc.x,  // in1_mcast_dest_noc_start_x
                (std::uint32_t)start_core_noc.y,  // in1_mcast_dest_noc_start_y
                (std::uint32_t)end_core_noc.x,    // in1_mcast_dest_noc_end_x
                (std::uint32_t)end_core_noc.y,    // in1_mcast_dest_noc_end_y

                // WRITER
                // out tensor args
                (std::uint32_t)out_buffer->address(),
                (std::uint32_t)output_idx_x * per_core_N + output_idx_y * per_core_M * N,  // out_tensor_start_tile_id

                // padding args (READER)
                (std::uint32_t)out_block_w,  // last_block_w
                // padding args (WRITER)
                (std::uint32_t)out_block_h / out_subblock_h,
                (std::uint32_t)out_subblock_h,
                (std::uint32_t)0,
                (std::uint32_t)out_block_w / out_subblock_w,
                (std::uint32_t)out_block_w / out_subblock_w,
                (std::uint32_t)out_subblock_w,
                (std::uint32_t)0,
                (std::uint32_t)0};

            if (bias_buffer != nullptr) {
                mm_in1_sender_writer_args.push_back((std::uint32_t)bias_buffer->address());
                mm_in1_sender_writer_args.push_back(
                    (std::uint32_t)per_core_N * output_idx_x);  // in3_tensor_start_tile_id
            } else {
                mm_in1_sender_writer_args.push_back(0);
                mm_in1_sender_writer_args.push_back(0);
            }
            if (!output_is_sharded) {
                mm_in1_sender_writer_args.push_back(out_num_blocks_x);
            }

            tt_metal::SetRuntimeArgs(
                program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);  // RISCV_1_default
        }
        // in0 sender and in1 receiver
        else {
            std::vector<uint32_t> mm_in1_receiver_writer_args = {
                // READER
                // in1 mcast args
                (std::uint32_t)top_left_core_physical.x,  // in1_mcast_sender_noc_x
                (std::uint32_t)top_left_core_physical.y,  // in1_mcast_sender_noc_y

                // WRITER
                // out tensor args
                (std::uint32_t)out_buffer->address(),                                     // out_tensor_addr
                (std::uint32_t)output_idx_x * per_core_N + output_idx_y * per_core_M * N  // out_tensor_start_tile_id
            };

            if (output_idx_y == num_blocks_y - 1) {
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
                if (output_idx_y == num_blocks_y - 1) {
                    mm_in1_receiver_writer_args.push_back(last_out_num_blocks_h);
                    mm_in1_receiver_writer_args.push_back(out_num_blocks_x);
                } else {
                    mm_in1_receiver_writer_args.push_back(out_num_blocks_y);
                    mm_in1_receiver_writer_args.push_back(out_num_blocks_x);
                }
            }

            tt_metal::SetRuntimeArgs(
                program,
                mm_kernel_in1_receiver_writer_id,
                core,
                mm_in1_receiver_writer_args);  // RISCV_0_default
        }
        std::vector<uint32_t> mm_in0_sender_args = {
            // in0 tensor args
            (std::uint32_t)in0_buffer->address(),
            (std::uint32_t)K * per_core_M * output_idx_y,  // in0_tensor_start_tile_id
            // in0 mcast args
            (std::uint32_t)0,  // in0_mcast_dest_noc_start_x
            (std::uint32_t)0,  // in0_mcast_dest_noc_start_y
            (std::uint32_t)0,  // in0_mcast_dest_noc_end_x
            (std::uint32_t)0,  // in0_mcast_dest_noc_end_y

            // padding args
            (std::uint32_t)per_core_M  // last_block_h
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args);  // RISCV_1_default
    }

    auto override_runtime_arguments_callback =
        [mm_kernel_in0_sender_id,
         mm_kernel_in1_sender_writer_id,
         mm_kernel_in1_receiver_writer_id,
         extract_shard_sub_blocks,
         cb_src0,
         cb_src2,
         cb_output,
         start_core,
         cores](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<tt::tt_metal::Tensor>& input_tensors,
            const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
            const std::vector<tt::tt_metal::Tensor>& output_tensors) {
            TT_ASSERT(input_tensors.size() + optional_input_tensors.size() == 3);
            TT_ASSERT(output_tensors.size() == 1);

            auto src_buffer_a = input_tensors.at(0).buffer();
            auto src_buffer_b = input_tensors.at(1).buffer();
            auto bias_tensor = optional_input_tensors.at(0);

            std::optional<tt::tt_metal::Buffer*> bias_buffer;
            if (bias_tensor.has_value()) {
                bias_buffer = bias_tensor.value().buffer();
            }

            auto dst_buffer = output_tensors.at(0).buffer();

            bool src0_sharded = input_tensors[0].is_sharded();
            bool out_sharded = output_tensors[0].is_sharded();

            auto& reader_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in0_sender_id);

            // Manually unroll sender core
            {
                // in0 sender
                auto& reader_runtime_args = reader_runtime_args_by_core[start_core.x][start_core.y];
                reader_runtime_args[0] = src_buffer_a->address();

                // in1 sender
                auto& sender_writer_runtime_args = GetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, start_core);
                sender_writer_runtime_args[0] = src_buffer_b->address();
                sender_writer_runtime_args[6] = dst_buffer->address();
                if (bias_tensor.has_value()) {
                    sender_writer_runtime_args[17] = (*bias_buffer)->address();
                }
            }

            auto& receiver_writer_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in1_receiver_writer_id);

            for (uint32_t i = 1; i < cores.size(); ++i) {
                const CoreCoord& core = cores[i];

                auto& reader_runtime_args = reader_runtime_args_by_core[core.x][core.y];

                auto& writer_runtime_args = receiver_writer_runtime_args_by_core[core.x][core.y];

                // in0 sender
                reader_runtime_args[0] = src_buffer_a->address();
                // in1 receiver
                writer_runtime_args[2] = dst_buffer->address();
            }

            if (src0_sharded) {
                if (extract_shard_sub_blocks) {
                    UpdateDynamicCircularBufferAddress(program, cb_src2, *src_buffer_a);
                } else {
                    UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
                }
            }

            if (out_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
            }
        };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

enum class CORE_TYPE : uint32_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

tt::tt_metal::operation::ProgramWithCallbacks create_program_gather_in0(
    tt_metal::Program& program,
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    bool dst_full_sync_en,
    CoreCoord compute_with_storage_grid_size,
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
    std::optional<UnaryWithParam> fused_activation,
    const CoreRangeSet& hop_cores,
    tt_metal::Buffer* in0_buffer,
    tt_metal::Buffer* in1_buffer,
    tt_metal::Buffer* out_buffer,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    const bool in1_is_dram_interleaved = in1_buffer->is_dram() && !b.is_sharded();
    const bool in1_is_dram_sharded =
        in1_buffer->is_dram() && b.is_sharded() && !global_cb.has_value();  // read from DRAM directly

    /* Core setup */
    constexpr bool row_major = true;
    CoreRangeSet all_worker_cores = a.shard_spec().value().grid;
    CoreRangeSet non_idle_cores = all_worker_cores.merge(hop_cores);
    auto subdevice_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
    std::vector<CoreRange> non_idle_cores_vec;
    for (auto& cr : subdevice_cores.ranges()) {
        auto intersection = non_idle_cores.intersection(cr);
        if (intersection.size() > 0) {
            non_idle_cores_vec.push_back(intersection.bounding_box());
        }
    }
    CoreRangeSet all_cores = CoreRangeSet(non_idle_cores_vec);
    std::vector<CoreRange> ring_list = all_worker_cores.ranges();
    std::vector<CoreRange> hop_list = hop_cores.ranges();
    ring_list.insert(ring_list.end(), hop_list.begin(), hop_list.end());

    CoreRangeSet ring_cores = CoreRangeSet(ring_list);
    const uint32_t num_cores = all_worker_cores.num_cores();
    const uint32_t ring_size = num_cores;

    uint32_t num_hop_cores = hop_cores.num_cores();
    bool use_hop_cores = num_hop_cores > 0;

    /* Inner dim padding */
    const uint32_t Kt_pad = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1] * num_cores;
    in0_block_w = Kt_pad / num_cores;

    uint32_t num_blocks = Kt_pad / in0_block_w;
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    bool use_global_cb = global_cb.has_value();

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    /* in0 */
    uint32_t in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    uint32_t in0_CB_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    /* in1 */
    uint32_t in1_shard_height_in_tiles = 0;
    uint32_t in1_shard_width_in_tiles = 0;
    uint32_t in1_CB_tiles = 0;
    uint32_t in1_tensor_width_in_tiles = b.get_padded_shape()[-1] / in1_tile.get_tile_shape()[1];

    if (in1_is_dram_sharded || in1_is_dram_interleaved) {
        in1_CB_tiles = 2 * in0_shard_width_in_tiles * per_core_N;  // Double buffered
    } else {
        in1_shard_height_in_tiles = in1_buffer->shard_spec().shape()[0] / in1_tile.get_tile_shape()[0];
        in1_shard_width_in_tiles =
            in1_buffer->shard_spec().shape()[1] / in1_tile.get_tile_shape()[1] / num_global_cb_receivers;
        in1_CB_tiles = in1_shard_height_in_tiles * in1_shard_width_in_tiles;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    // get the max page size based on num tiles
    uint32_t per_core_N_size_bytes = per_core_N * in1_single_tile_size;
    uint32_t max_packet_size = 8192;
    uint32_t in1_block_page_size = per_core_N_size_bytes > max_packet_size ? max_packet_size : per_core_N_size_bytes;
    uint32_t in1_block_page_size_last =
        per_core_N_size_bytes > max_packet_size ? per_core_N_size_bytes % max_packet_size : per_core_N_size_bytes;
    uint32_t in1_block_width_num_pages = (per_core_N_size_bytes + in1_block_page_size - 1) / in1_block_page_size;
    uint32_t in1_shard_width_in_dram = 0;
    if (in1_is_dram_sharded) {
        in1_shard_width_in_dram = in1_buffer->shard_spec().shape()[1] / in1_tile.get_tile_shape()[1];
    }

    /* in2 */
    uint32_t in2_single_tile_size = in0_single_tile_size;
    uint32_t in2_CB_tiles = (ring_size - 1) * in0_CB_tiles;  // All shards except local
    uint32_t in2_CB_size = in2_CB_tiles * in2_single_tile_size;

    /* out */
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t K_ = K;
    std::vector<uint32_t> unpadded_in0_shard_widths_in_tiles(num_cores, 0);
    for (uint32_t i = 0; i < num_cores && K_ > 0; ++i) {
        unpadded_in0_shard_widths_in_tiles[i] = std::min(K_, in0_shard_width_in_tiles);
        K_ -= unpadded_in0_shard_widths_in_tiles[i];
    }

    /* semaphores */
    auto in0_signal_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t in1_block_height_in_tiles = in0_block_w;
    uint32_t in1_block_num_tiles = out_subblock_w * in1_block_height_in_tiles * in1_num_subblocks;
    uint32_t in1_block_size_bytes = in1_block_num_tiles * in1_single_tile_size;
    uint32_t in1_tensor_size_bytes = in1_block_num_tiles * num_blocks * in1_single_tile_size;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    /* Compile time args */
    std::vector<uint32_t> in0_sender_compile_time_args = {
        (std::uint32_t)in0_shard_width_in_tiles,
        (std::uint32_t)per_core_M,  // in0_shard_height_in_tiles
        (std::uint32_t)B,           // batch
        (std::uint32_t)ring_size,   // ring_size
        (std::uint32_t)in0_signal_semaphore_id,
    };

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        (std::uint32_t)in1_is_dram_interleaved,    // in1_is_dram_interleaved
        (std::uint32_t)in1_is_dram_sharded,        // in1_is_dram_sharded
        (std::uint32_t)in1_block_height_in_tiles,  // in1_block_height_in_tiles
        (std::uint32_t)per_core_N,                 // in1_block_width_in_tiles
        (std::uint32_t)in1_tensor_width_in_tiles,  // in1_tensor_width_in_tiles
        (std::uint32_t)num_blocks,                 // num_blocks
        (std::uint32_t)B,                          // batch
        (std::uint32_t)in1_block_page_size,
        (std::uint32_t)in1_block_page_size_last,
        (std::uint32_t)in1_block_width_num_pages,
        (std::uint32_t)in1_shard_width_in_dram,
    };

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,      // in1_num_subblocks
        in1_block_num_tiles,    // in1_block_num_tiles
        in1_block_size_bytes,   // in1_block_size_bytes
        in1_tensor_size_bytes,  // in1_tensor_size_bytes
        in1_per_core_w,         // in1_per_core_w

        num_blocks,  // num_blocks

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out,             // untilize_out
        in1_is_dram_interleaved,  // in1_is_dram_interleaved
        in1_is_dram_sharded,      // in1_is_dram_sharded
    };

    /* Kernel defines */
    std::map<string, string> mm_in1_kernel_defines;
    std::map<string, string> mm_kernel_defines;

    if (use_global_cb) {
        mm_in1_kernel_defines["ENABLE_GLOBAL_CB"] = "1";
        mm_kernel_defines["ENABLE_GLOBAL_CB"] = "1";
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
    bmm_op_utils::add_stagger_defines_if_needed(device->arch(), num_cores, mm_kernel_defines);

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());

    bool use_dedicated_noc = true;
    tt_metal::NOC_MODE noc_mode =
        use_dedicated_noc ? tt_metal::NOC_MODE::DM_DEDICATED_NOC : tt_metal::NOC_MODE::DM_DYNAMIC_NOC;

    /* Create the kernels */
    auto mm_kernel_in0_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .noc_mode = noc_mode,
            .compile_args = in0_sender_compile_time_args});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .noc_mode = noc_mode,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_in1_kernel_defines});

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    /* Create circular buffers */
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile)
            .set_globally_allocated_address(*in0_buffer);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CBHandle cb_src1;
    if (use_global_cb) {
        uint32_t in1_block_size_bytes = in1_single_tile_size * in1_block_num_tiles;
        uint32_t remote_cb_index = tt::CBIndex::c_31;
        tt_metal::CircularBufferConfig remote_cb_config =
            tt_metal::CircularBufferConfig((global_cb->size() / in1_block_size_bytes) * in1_block_size_bytes);
        remote_cb_config.remote_index(remote_cb_index)
            .set_page_size(in1_block_size_bytes)
            .set_data_format(in1_data_format);
        remote_cb_config.index(src1_cb_index).set_page_size(in1_single_tile_size).set_data_format(in1_data_format);
        cb_src1 = tt_metal::experimental::CreateCircularBuffer(program, all_cores, remote_cb_config, *global_cb);
    } else {
        tt_metal::CircularBufferConfig src1_cb_config =
            tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
                .set_page_size(src1_cb_index, in1_single_tile_size)
                .set_tile_dims(src1_cb_index, in1_tile);
        if (!in1_is_dram_interleaved && !in1_is_dram_sharded) {
            src1_cb_config = src1_cb_config.set_globally_allocated_address(*in1_buffer);
        }
        cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);
    }

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt_metal::CircularBufferConfig src2_cb_config =
        tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
            .set_page_size(src2_cb_index, in2_single_tile_size)
            .set_tile_dims(src2_cb_index, in0_tile);
    auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, src2_cb_config);

    uint32_t sync_cb_index = tt::CBIndex::c_3;
    uint32_t sync_cb_size_bytes = 16;
    tt_metal::CircularBufferConfig sync_cb_config =
        tt_metal::CircularBufferConfig(sync_cb_size_bytes, {{sync_cb_index, DataFormat::UInt16}})
            .set_page_size(sync_cb_index, sync_cb_size_bytes);
    auto cb_sync = tt_metal::CreateCircularBuffer(program, all_cores, sync_cb_config);

    uint32_t sync_cb2_index = tt::CBIndex::c_4;
    uint32_t sync_cb2_size_bytes = 16;
    tt_metal::CircularBufferConfig sync_cb2_config =
        tt_metal::CircularBufferConfig(sync_cb2_size_bytes, {{sync_cb2_index, DataFormat::UInt16}})
            .set_page_size(sync_cb2_index, sync_cb2_size_bytes);
    auto cb2_sync = tt_metal::CreateCircularBuffer(program, all_cores, sync_cb2_config);

    uint32_t output_cb_index = tt::CBIndex::c_5;  // output operands start at index 16
    uint32_t interm0_cb_index = tt::CBIndex::c_6;
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
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_globally_allocated_address(*out_buffer);
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{
            {interm0_cb_index, interm0_data_format},
        };
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                .set_tile_dims(interm0_cb_index, output_tile);

        auto cb_interm0 = tt_metal::CreateCircularBuffer(program, all_cores, interm0_cb_config);
    } else {
        // share buffer
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_page_size(interm0_cb_index, interm0_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_tile_dims(interm0_cb_index, output_tile)
                               .set_globally_allocated_address(*out_buffer);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

    // for all the cores in the rect grid, we send one rt arg to determine if they are worker core
    auto all_cores_vec = corerange_to_cores(all_cores, std::nullopt, row_major);
    auto worker_cores_vec = corerange_to_cores(all_worker_cores, std::nullopt, row_major);
    auto hop_cores_vec = corerange_to_cores(hop_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < all_cores_vec.size(); ++i) {
        auto core = all_cores_vec[i];

        auto all_worker_cores_iter = std::find(worker_cores_vec.begin(), worker_cores_vec.end(), core);
        auto hop_cores_iter = std::find(hop_cores_vec.begin(), hop_cores_vec.end(), core);
        bool core_is_in_all_worker_cores = all_worker_cores_iter != worker_cores_vec.end();
        bool core_is_in_hop_cores = hop_cores_iter != hop_cores_vec.end();
        if (!use_hop_cores) {
            core_is_in_hop_cores = false;
        }

        if (!core_is_in_all_worker_cores && !core_is_in_hop_cores) {  // not worker core and not hop core
            auto core_type = CORE_TYPE::IDLE_CORE;                    // idle core
            // in0
            std::vector<uint32_t> mm_kernel_in0_args;
            mm_kernel_in0_args.push_back((std::uint32_t)core_type);
            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_kernel_in0_args);

            // in1
            std::vector<uint32_t> mm_kernel_in1_sender_writer_args;
            mm_kernel_in1_sender_writer_args.push_back((std::uint32_t)core_type);
            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_kernel_in1_sender_writer_args);

            // compute
            std::vector<uint32_t> mm_kernel_args;
            mm_kernel_args.push_back((std::uint32_t)core_type);
            tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_args);
        }
    }

    /* Runtime args */
    std::map<uint32_t, uint32_t> worker_coord_y_to_dram_bank_first_col_mapping;
    std::map<uint32_t, uint32_t> worker_coord_y_to_dram_bank_second_col_mapping;
    if (in1_is_dram_sharded) {
        if (device->arch() == tt::ARCH::WORMHOLE_B0) {
            worker_coord_y_to_dram_bank_first_col_mapping[0] = 1;
            worker_coord_y_to_dram_bank_first_col_mapping[4] = 2;
            worker_coord_y_to_dram_bank_first_col_mapping[5] = 3;
            worker_coord_y_to_dram_bank_first_col_mapping[9] = 0;

            worker_coord_y_to_dram_bank_second_col_mapping[0] = 4;
            worker_coord_y_to_dram_bank_second_col_mapping[1] = 6;
            worker_coord_y_to_dram_bank_second_col_mapping[2] = 9;
            worker_coord_y_to_dram_bank_second_col_mapping[4] = 10;
            worker_coord_y_to_dram_bank_second_col_mapping[5] = 11;
            worker_coord_y_to_dram_bank_second_col_mapping[6] = 8;
            worker_coord_y_to_dram_bank_second_col_mapping[7] = 7;
            worker_coord_y_to_dram_bank_second_col_mapping[9] = 5;

        } else if (device->arch() == tt::ARCH::BLACKHOLE) {
            TT_THROW("ring gather MM currently not supporting blackhole when in1 is dram sharded");
        } else {
            TT_THROW("ring gather MM currently not supporting this device arch");
        }
    }

    uint32_t bank_id = 0;
    std::vector<uint32_t> bank_ids;
    for (uint32_t i = 0; i < num_cores; ++i) {
        bool send_to_hop_core = i == 0 && use_hop_cores;
        const auto& core = worker_cores_vec[i];
        const auto& core_noc = device->worker_core_from_logical_core(core);

        /* in0 */
        auto core_type = CORE_TYPE::WORKER_CORE;  // worker core
        CoreCoord next_core;
        if (send_to_hop_core) {
            next_core = hop_cores_vec[0];  // Send to first hop core
        } else {
            uint32_t next_i = i == 0 ? num_cores - 1 : i - 1;
            next_core = worker_cores_vec[next_i % num_cores];
        }
        const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
        uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

        std::vector<uint32_t> mm_in0_args = {
            (std::uint32_t)core_type,
            i,                // ring_index
            next_core_noc.x,  // next_core_noc_x
            next_core_noc.y,  // next_core_noc_y
            noc,
            (std::uint32_t)false,  // end_of_hop
        };

        mm_in0_args.insert(
            mm_in0_args.end(), unpadded_in0_shard_widths_in_tiles.begin(), unpadded_in0_shard_widths_in_tiles.end());
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_in0_args);

        /* in1 */
        std::vector<uint32_t> mm_in1_args = {
            (std::uint32_t)core_type,
            in1_buffer->address(),  // in1_tensor_addr
            i,                      // ring_idx
        };
        if (in1_is_dram_sharded) {
            if (core.x <= 3) {
                bank_id = worker_coord_y_to_dram_bank_first_col_mapping[core.y];
            } else {
                bank_id = worker_coord_y_to_dram_bank_second_col_mapping[core.y];
            }
            uint32_t dram_read_offset = 0;
            if (core.x % 2 == 0) {
                dram_read_offset = 1;
            }
            bank_ids.push_back(bank_id);
            uint32_t vc = 0;
            for (uint32_t j = 0; j < i; ++j) {
                auto core_prev = worker_cores_vec[j];
                if (core_prev.y == core.y) {
                    vc = (vc + 1) & 0x3;
                }
            }
            mm_in1_args.push_back((std::uint32_t)bank_id);
            mm_in1_args.push_back((std::uint32_t)vc);
            mm_in1_args.push_back((std::uint32_t)dram_read_offset);
        }
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_args);

        /* compute */
        std::vector<uint32_t> mm_kernel_compute_args = {
            (std::uint32_t)core_type,
            i,  // ring_idx
        };
        mm_kernel_compute_args.insert(
            mm_kernel_compute_args.end(),
            unpadded_in0_shard_widths_in_tiles.begin(),
            unpadded_in0_shard_widths_in_tiles.end());

        tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_compute_args);
    }

    // Runtime args for hop cores
    for (uint32_t i = 0; i < num_hop_cores; ++i) {
        bool end_of_hop = i == num_hop_cores - 1;

        auto core_type = CORE_TYPE::HOP_CORE;  // hop core
        const auto& core = hop_cores_vec[i];
        const auto& core_noc = device->worker_core_from_logical_core(core);

        /* in0 */
        CoreCoord next_core = end_of_hop ? worker_cores_vec[num_cores - 1] : hop_cores_vec[i + 1];
        const auto& next_core_noc = device->worker_core_from_logical_core(next_core);
        uint32_t noc = get_preferred_noc(core_noc, next_core_noc, device, use_dedicated_noc);

        std::vector<uint32_t> mm_in0_args = {
            (std::uint32_t)core_type,
            0,                // ring_index
            next_core_noc.x,  // next_core_noc_x
            next_core_noc.y,  // next_core_noc_y
            noc,
            (std::uint32_t)end_of_hop,  // end_of_hop
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_id, core, mm_in0_args);

        // in1
        std::vector<uint32_t> mm_kernel_in1_sender_writer_args;
        mm_kernel_in1_sender_writer_args.push_back((std::uint32_t)core_type);
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_kernel_in1_sender_writer_args);

        // compute
        std::vector<uint32_t> mm_kernel_args;
        mm_kernel_args.push_back((std::uint32_t)core_type);
        tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_kernel_args);
    }

    auto override_runtime_arguments_callback =
        [mm_kernel_in0_id, mm_kernel_in1_sender_writer_id, cb_src0, cb_src1, cb_output, num_cores, all_cores_vec](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<tt::tt_metal::Tensor>& input_tensors,
            const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
            const std::vector<tt::tt_metal::Tensor>& output_tensors) {
            TT_ASSERT(input_tensors.size() + optional_input_tensors.size() == 3);
            TT_ASSERT(output_tensors.size() == 1);

            auto& global_cb = static_cast<const ttnn::operations::matmul::Matmul*>(operation)->global_cb;

            auto src_buffer_a = input_tensors[0].buffer();
            auto src_buffer_b = input_tensors[1].buffer();
            auto dst_buffer = output_tensors[0].buffer();

            bool src0_sharded = input_tensors[0].is_sharded();
            bool src1_sharded = input_tensors[1].is_sharded();
            bool out_sharded = output_tensors[0].is_sharded();

            // Manually unroll sender core
            if (src0_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
            }
            if (src1_sharded) {
                if (!global_cb.has_value() && !src_buffer_b->is_dram()) {
                    UpdateDynamicCircularBufferAddress(program, cb_src1, *src_buffer_b);
                }
            }
            if (out_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
            }

            if (not src1_sharded) {
                auto& writer_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in1_sender_writer_id);
                for (uint32_t i = 0; i < all_cores_vec.size(); ++i) {
                    const auto& core = all_cores_vec[i];
                    auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];

                    /* in1 */
                    writer_runtime_args[1] = src_buffer_b->address();
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace reuse_mcast_1d_optimized_helpers

namespace ttnn {

namespace operations {

namespace matmul {

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized_(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    const std::optional<UnaryWithParam>& fused_activation,
    bool mcast_in0,
    bool gather_in0,
    const CoreRangeSet& hop_cores,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    const auto &ashape = a.get_padded_shape(), bshape = b.get_padded_shape();
    auto in0_tile = a.get_tensor_spec().tile();
    auto in1_tile = b.get_tensor_spec().tile();
    // cannot use the output tensor tile directly as that might be changed by user override
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile_shape[0], in1_tile_shape[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());          // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());          // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());  // output

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;  // bias; doesn't matter if bias=nullptr
    if (bias.has_value()) {
        auto& c = bias.value();
        TT_FATAL(c.storage_type() == StorageType::DEVICE, "Error");
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

        bias_buffer = c.buffer();

        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.get_dtype());
    }

    tt_metal::IDevice* device = a.device();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();
    TT_FATAL(in0_buffer->size() % in0_single_tile_size == 0, "Error");
    TT_FATAL(in1_buffer->size() % in1_single_tile_size == 0, "Error");

    TT_FATAL(
        ashape[-1] == bshape[-2],
        "Dimension K (A.shape[-1] and B.shape[-2]) must match for A and B in bmm_op");  // A.K == B.K
    TT_FATAL(ashape[-2] % in0_tile_shape[0] == 0, "Error");
    TT_FATAL(ashape[-1] % in0_tile_shape[1] == 0, "Error");
    TT_FATAL(bshape[-2] % in1_tile_shape[0] == 0, "Error");
    TT_FATAL(bshape[-1] % in1_tile_shape[1] == 0, "Error");

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Pads matmul input dims to 512 x 512 multiples (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];

    if (fuse_batch) {
        Mt = B * Mt;
        B = 1;
    }
    TT_FATAL(Kt % in0_block_w == 0, "Error");

    // This should allocate a DRAM buffer on the device
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;

    // Calculate number of blocks along x and y; tensor dims are padded up to 512
    uint32_t num_blocks_y = (Mt - 1) / per_core_M + 1;
    uint32_t num_blocks_x = (Nt - 1) / per_core_N + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;

    // TODO: Max used grid can actually exceed mcast receiver grid if in0 is sharded
    // TODO: Move these validates to op validate and properly check for this
    TT_FATAL(
        num_blocks_total <= num_cores,
        "Number of blocks exceeds number of cores: {} blocks > {} cores",
        num_blocks_total,
        num_cores);

    if (!gather_in0) {
        TT_FATAL(hop_cores.empty(), "Hop cores are not supported for any mode besides gather_in0.");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer* out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    if (gather_in0) {
        return reuse_mcast_1d_optimized_helpers::create_program_gather_in0(
            program,
            a,
            b,
            device,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode,
            packer_l1_acc,
            dst_full_sync_en,
            compute_with_storage_grid_size,
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
            fused_activation,
            hop_cores,
            in0_buffer,
            in1_buffer,
            out_buffer,
            in0_tile,
            in1_tile,
            output_tile,
            in0_data_format,
            in1_data_format,
            output_data_format,
            untilize_out,
            global_cb,
            num_global_cb_receivers,
            sub_device_id);
    }

    if (mcast_in0) {
        return reuse_mcast_1d_optimized_helpers::create_program_mcast_in0(
            program,
            a,
            device,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode,
            packer_l1_acc,
            compute_with_storage_grid_size,
            B,
            Mt,
            Nt,
            Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            out_block_h,
            out_block_w,
            per_core_M,
            per_core_N,
            fused_activation,
            in0_buffer,
            in1_buffer,
            bias_buffer,
            out_buffer,
            in0_tile,
            in1_tile,
            bias.has_value() ? bias->get_tensor_spec().tile() : output_tile,
            output_tile,
            in0_data_format,
            in1_data_format,
            bias_data_format,
            output_data_format,
            a.memory_config().is_sharded(),
            b.memory_config().is_sharded(),
            bias.has_value() ? bias->memory_config().is_sharded() : false,
            output.memory_config().is_sharded(),
            untilize_out,
            fused_op_signaler);
    } else {
        return reuse_mcast_1d_optimized_helpers::create_program_mcast_in1(
            device,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode,
            packer_l1_acc,
            compute_with_storage_grid_size,
            B,
            Mt,
            Nt,
            Kt,
            bcast_batch,
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            out_block_h,
            out_block_w,
            per_core_M,
            per_core_N,
            fused_activation,
            in0_buffer,
            in1_buffer,
            bias_buffer,
            out_buffer,
            in0_tile,
            in1_tile,
            bias.has_value() ? bias->get_tensor_spec().tile() : output_tile,
            output_tile,
            in0_data_format,
            in1_data_format,
            bias_data_format,
            output_data_format,
            a.memory_config().is_sharded(),
            output.memory_config().is_sharded(),
            untilize_out);
    }
}

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized(
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool broadcast_batch,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_block_h,
    uint32_t out_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    const std::optional<UnaryWithParam>& fused_activation,
    bool mcast_in0,
    bool gather_in0,
    const CoreRangeSet& hop_cores,
    bool untilize_out,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    uint32_t num_global_cb_receivers,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt_metal::Program program{}; /* Create a program */
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> empty_fused_op_signaler;

    return matmul_multi_core_reuse_mcast_1d_optimized_(
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
        out_block_h,
        out_block_w,
        per_core_M,
        per_core_N,
        fuse_batch,
        std::move(fused_activation),
        mcast_in0,
        gather_in0,
        std::move(hop_cores),
        untilize_out,
        empty_fused_op_signaler,
        global_cb,
        num_global_cb_receivers,
        sub_device_id);
}

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_mcast_1d_optimized_helper(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    bool broadcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const MatmulProgramConfig& program_config,
    bool untilize_out,
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& fused_op_signaler,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    MatmulMultiCoreReuseMultiCast1DProgramConfig config =
        std::get<MatmulMultiCoreReuseMultiCast1DProgramConfig>(program_config);

    return matmul_multi_core_reuse_mcast_1d_optimized_(
        program,
        a,
        b,
        bias,
        output_tensor,
        broadcast_batch,
        config.compute_with_storage_grid_size,
        compute_kernel_config,
        config.in0_block_w,
        config.out_subblock_h,
        config.out_subblock_w,
        config.out_block_h,
        config.out_block_w,
        config.per_core_M,
        config.per_core_N,
        config.fuse_batch,
        config.fused_activation,
        config.mcast_in0,
        config.gather_in0,
        config.hop_cores,
        untilize_out,
        fused_op_signaler,
        global_cb,
        config.num_global_cb_receivers,
        sub_device_id);
}

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn
