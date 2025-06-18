// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt;
using namespace tt::constants;

namespace reuse_dram_sharded_optimized_helpers {
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

void get_max_page_size_and_num_pages(uint32_t num_tiles, uint32_t tile_size, uint32_t& page_size, uint32_t& num_pages) {
    uint64_t total_size = static_cast<uint64_t>(num_tiles) * tile_size;

    page_size = (8192 / tile_size) * tile_size;
    while (total_size % page_size != 0 && page_size >= tile_size) {
        page_size -= tile_size;
    }
    num_pages = total_size / page_size;
}

void move_common_entries(std::vector<CoreCoord>& v1, std::vector<CoreCoord>& v2, std::vector<CoreCoord>& commons) {
    for (const CoreCoord& item : v2) {
        if (std::find(v1.begin(), v1.end(), item) != v1.end()) {
            commons.push_back(item);
        }
    }

    for (const CoreCoord& item : commons) {
        v2.erase(std::remove(v2.begin(), v2.end(), item), v2.end());
    }
}

void get_optimal_dram_bank_to_reader_assignment(
    tt::tt_metal::IDevice* device, std::vector<CoreCoord>& all_worker_cores_ordered, CoreRangeSet& all_worker_cores) {
    all_worker_cores_ordered = device->get_optimal_dram_bank_to_logical_worker_assignment();
    std::set<CoreRange> all_cores_set;
    for (const auto& worker_core : all_worker_cores_ordered) {
        all_cores_set.insert(CoreRange(worker_core));
    }
    all_worker_cores = CoreRangeSet(all_cores_set);
}

tt::tt_metal::operation::ProgramWithCallbacks create_program_dram_sharded(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& all_storage_cores,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    uint32_t B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t in0_block_w,
    uint32_t in0_last_ktile_w,
    uint32_t per_core_M,
    uint32_t per_core_N_storage,
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
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back) {
    log_debug(tt::LogOp, "math_fidelity: {}", math_fidelity);
    log_debug(tt::LogOp, "fp32_dest_acc_en: {}", fp32_dest_acc_en);
    log_debug(tt::LogOp, "math_approx_mode: {}", math_approx_mode);
    log_debug(tt::LogOp, "packer_l1_acc: {}", packer_l1_acc);
    log_debug(tt::LogOp, "M: {}, K: {}, N: {}", M, K, N);
    log_debug(tt::LogOp, "per_core_M: {}, per_core_N_storage: {}", per_core_M, per_core_N_storage);

    // currently only support transpose of the full tile
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    tt_metal::Program program{};

    // get the dram readers
    std::vector<CoreCoord> all_worker_cores_ordered;
    CoreRangeSet all_worker_cores;
    get_optimal_dram_bank_to_reader_assignment(device, all_worker_cores_ordered, all_worker_cores);

    // dram banks
    uint32_t num_dram_banks = all_worker_cores_ordered.size();
    for (auto core : corerange_to_cores(all_worker_cores)) {
        log_debug(tt::LogOp, "all_worker_cores_log: {}", core);
    }
    for (auto core : all_worker_cores_ordered) {
        log_debug(tt::LogOp, "all_worker_cores_ordered: {}", core);
    }

    uint32_t per_core_N_compute = (N + num_dram_banks - 1) / num_dram_banks;
    uint32_t per_core_N_in1_sender = per_core_N_compute;
    auto subblock_hw =
        bmm_op_utils::get_matmul_subblock_params(per_core_M, per_core_N_compute, false, false, fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    uint32_t max_subblock_w = fp32_dest_acc_en ? 4 : 8;
    // it is bad for compute, pad per_core_N_compute
    if (out_subblock_h == 1 and out_subblock_w < max_subblock_w) {
        uint32_t num_subblock_w_per_core_N = per_core_N_compute / out_subblock_w;
        uint32_t num_iter = max_subblock_w - out_subblock_w;
        uint32_t new_out_subblock_w = out_subblock_w;
        uint32_t preferred_out_subblock_w = out_subblock_w;

        for (uint32_t i = 0; i < num_iter; ++i) {
            new_out_subblock_w += 1;
            uint32_t new_num_subblock_w_per_core_N = (per_core_N_compute + new_out_subblock_w - 1) / new_out_subblock_w;

            if (new_num_subblock_w_per_core_N < num_subblock_w_per_core_N) {
                num_subblock_w_per_core_N = new_num_subblock_w_per_core_N;
                preferred_out_subblock_w = new_out_subblock_w;
            }
        }
        out_subblock_w = preferred_out_subblock_w;
        per_core_N_compute = out_subblock_w * num_subblock_w_per_core_N;
    }

    log_debug(
        tt::LogOp,
        "per_core_M: {}, per_core_N_compute: {}, per_core_N_in1_sender: {}",
        per_core_M,
        per_core_N_compute,
        per_core_N_in1_sender);
    log_debug(tt::LogOp, "out_subblock_h: {}, out_subblock_w: {}", out_subblock_h, out_subblock_w);

    uint32_t num_blocks = K / in0_block_w;
    // Only enable packer l1 accumulation when there are spills, otherwise
    // unnecessary overhead for reconfigs are added
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);
    interm0_data_format = tt::DataFormat::Float16_b;

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles = in0_CB_tiles * 2;  // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = per_core_N_in1_sender * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles = in1_CB_tiles * 3;  // tripple buffer
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    uint32_t out_block_tiles = per_core_M * per_core_N_compute;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t out_reshard_block_tiles = per_core_M * per_core_N_storage;
    uint32_t out_reshard_CB_tiles = out_reshard_block_tiles;  // No double buffer
    uint32_t out_reshard_CB_size = out_reshard_CB_tiles * output_single_tile_size;

    uint32_t in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    uint32_t in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_tile_shape()[0];
    uint32_t in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in3_block_tiles = per_core_N_in1_sender;
    uint32_t in3_CB_tiles = in3_block_tiles;  // No double buffer
    uint32_t in3_CB_size = in3_CB_tiles * bias_single_tile_size;

    // get the max page size based on num tiles
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(in1_block_tiles, in1_single_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(
        in3_block_tiles, bias_single_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    uint32_t num_worker_cores = num_dram_banks;

    // move conflict coord from mcast receiver to mcast sender
    std::vector<CoreCoord> all_storage_cores_vec = corerange_to_cores(all_storage_cores);
    std::vector<CoreCoord> all_worker_cores_vec = corerange_to_cores(all_worker_cores);
    std::vector<CoreCoord> storage_worker_common;
    move_common_entries(all_storage_cores_vec, all_worker_cores_vec, storage_worker_common);

    std::vector<CoreRange> all_storage_cores_range;
    all_storage_cores_range.reserve(all_storage_cores_vec.size());
    std::transform(
        all_storage_cores_vec.begin(),
        all_storage_cores_vec.end(),
        std::back_inserter(all_storage_cores_range),
        [](const CoreCoord& coord) { return CoreRange(coord); });

    std::vector<CoreRange> all_worker_cores_range;
    all_worker_cores_range.reserve(all_worker_cores_vec.size());
    std::transform(
        all_worker_cores_vec.begin(),
        all_worker_cores_vec.end(),
        std::back_inserter(all_worker_cores_range),
        [](const CoreCoord& coord) { return CoreRange(coord); });

    std::set<CoreRange> all_storage_cores_set(all_storage_cores_range.begin(), all_storage_cores_range.end());
    std::set<CoreRange> all_worker_cores_set(all_worker_cores_range.begin(), all_worker_cores_range.end());
    CoreRangeSet mcast_senders = CoreRangeSet(all_storage_cores_set);
    CoreRangeSet mcast_receivers = CoreRangeSet(all_worker_cores_set);

    for (auto core : corerange_to_cores(mcast_senders)) {
        log_debug(tt::LogOp, "mcast_senders: {}", core);
    }
    for (auto core : corerange_to_cores(mcast_receivers)) {
        log_debug(tt::LogOp, "mcast_receivers: {}", core);
    }

    // all cores
    std::set<CoreRange> all_cores_set;
    all_cores_set.insert(mcast_senders.ranges().begin(), mcast_senders.ranges().end());
    all_cores_set.insert(mcast_receivers.ranges().begin(), mcast_receivers.ranges().end());
    CoreRangeSet all_cores = CoreRangeSet(all_cores_set);

    for (auto core : corerange_to_cores(all_cores)) {
        log_debug(tt::LogOp, "all_cores: {}", core);
    }

    // grid bounding box
    CoreRange bounding_box = all_cores.bounding_box();
    std::set<CoreRange> bounding_box_set;
    bounding_box_set.insert(bounding_box);
    CoreRangeSet all_cores_in_rect_grid(bounding_box_set);
    std::vector<CoreCoord> all_cores_in_rect_grid_vec = corerange_to_cores(all_cores_in_rect_grid);
    log_debug(tt::LogOp, "bounding_box: {}", bounding_box);

    // Mcast args
    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores_in_rect_grid, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores_in_rect_grid, INVALID);
    auto in0_mcast_sender_valid_semaphore_id = tt_metal::CreateSemaphore(program, all_cores_in_rect_grid, VALID);

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    uint32_t num_mcast_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    CoreCoord top_left_core = {(std::size_t)start_core_x, (std::size_t)start_core_y};
    CoreCoord bottom_right_core = {
        (std::size_t)start_core_x + compute_with_storage_grid_size.x - 1,
        (std::size_t)start_core_y + compute_with_storage_grid_size.y - 1};
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    bool in0_is_dram = false;
    bool in1_is_dram = true;
    bool in3_is_dram = true;

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMWrite(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::GetPreferredNOCForDRAMRead(device->arch());

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    uint32_t num_blocks_per_shard = num_blocks / all_storage_cores_vec.size();
    log_debug(tt::LogOp, "num_blocks_per_shard: {}", num_blocks_per_shard);
    if (per_core_M > 1) {
        TT_FATAL(
            num_blocks_per_shard == 1,
            "currently not support per_core_M larger than 1, while split one shard into multiple blocks (per_core_M "
            "{}, num_blocks_per_shard {})",
            per_core_M,
            num_blocks_per_shard);
    }

    std::vector<uint32_t> in0_sender_compile_time_args = {
        (std::uint32_t)in0_block_num_tiles,                         // in0_block_num_tiles
        (std::uint32_t)in0_block_num_tiles * in0_single_tile_size,  // in0_block_size_bytes
        (std::uint32_t)in0_last_ktile_w,                            // in0_last_ktile_w
        // in0 mcast args
        (std::uint32_t)in0_mcast_sender_semaphore_id,
        (std::uint32_t)in0_mcast_receiver_semaphore_id,
        (std::uint32_t)num_worker_cores,  // in0_mcast_num_dests
        (std::uint32_t)num_mcast_cores,   // in0_mcast_num_cores
        // block
        (std::uint32_t)num_blocks,
        // mcast noc coords
        (std::uint32_t)start_core_noc.x,
        (std::uint32_t)start_core_noc.y,
        (std::uint32_t)end_core_noc.x,
        (std::uint32_t)end_core_noc.y,
        // semahpre valid
        (std::uint32_t)in0_mcast_sender_valid_semaphore_id,
        //
        (std::uint32_t)num_blocks_per_shard};

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        (std::uint32_t)in1_buffer_page_size,
        (std::uint32_t)in1_buffer_num_pages,
        // in1 block args
        (std::uint32_t)per_core_N_in1_sender,                // in1_block_w
        (std::uint32_t)per_core_N_in1_sender * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,                                    // num_blocks
        (std::uint32_t)out_block_tiles,                               // out_block_num_tiles
        (std::uint32_t)per_core_N_compute * output_single_tile_size,  // out_tensor_stride_w_bytes
        (std::uint32_t)per_core_N_storage * output_single_tile_size,  // out_reshard_tensor_stride_w_bytes
        (std::uint32_t)per_core_M};
    if (bias_buffer != nullptr) {
        in1_sender_writer_compile_time_args.push_back(bias_buffer_page_size);
        in1_sender_writer_compile_time_args.push_back(bias_buffer_num_pages);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
    }

    std::map<string, string> mm_kernel_defines;
    std::map<string, string> mm_kernel_in0_sender_define;
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
    mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    if (skip_compute) {
        mm_kernel_defines["SKIP_COMPUTE"] = "1";
    }
    if (skip_in0_mcast) {
        mm_kernel_in0_sender_define["SKIP_MCAST"] = "1";
    }
    if (skip_write_back) {
        mm_kernel_in1_sender_writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    mm_kernel_defines["MATMUL_DRAM_SHARDED"] = "1";
    if (in1_transpose_tile) {
        mm_kernel_defines["IN1_TRANSPOSE_TILE"] = "1";
    }

    auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp",
        all_cores_in_rect_grid,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = mm_kernel_in0_sender_define});

    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp",
        all_cores_in_rect_grid,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_kernel_in1_sender_writer_defines});

    // Compute kernel compile time args
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N_compute / out_subblock_w);
    uint32_t in1_per_core_w = per_core_N_in1_sender;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,  // in1_num_subblocks
        in1_block_tiles,    // in1_block_num_tiles
        in1_per_core_w,     // in1_per_core_w

        num_blocks,  // num_blocks
        1,           // out_num_blocks_x
        1,           // out_num_blocks_y

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out  // untilize_out
    };

    // Create compute kernel
    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        // all_worker_cores,
        all_cores_in_rect_grid,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    log_debug(LogOp, "in1_single_tile_size: {}", in1_single_tile_size);

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, src0_cb_config);
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
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, src1_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src1_cb_index,
        in1_single_tile_size,
        in1_CB_size / in1_single_tile_size,
        in1_CB_size);

    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt_metal::CircularBufferConfig src2_cb_config =
        tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
            .set_page_size(src2_cb_index, in0_single_tile_size)
            .set_tile_dims(src2_cb_index, in0_tile)
            .set_globally_allocated_address(*in0_buffer);
    auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, src2_cb_config);
    log_debug(
        LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        src2_cb_index,
        in0_single_tile_size,
        in2_CB_size / in0_single_tile_size,
        in2_CB_size);

    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
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
                               .set_tile_dims(output_cb_index, output_tile);
        // interm0
        std::map<uint8_t, tt::DataFormat> interm0_cb_data_format_spec{
            {interm0_cb_index, interm0_data_format},
        };
        interm0_cb_config = tt_metal::CircularBufferConfig(interm0_CB_size, interm0_cb_data_format_spec)
                                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                                .set_tile_dims(interm0_cb_index, output_tile);

        auto cb_interm0 = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, interm0_cb_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            interm0_cb_index,
            interm0_single_tile_size,
            interm0_CB_size / interm0_single_tile_size,
            interm0_CB_size);
    } else {
        log_debug(tt::LogOp, "inplace interm and outout cb");
        // share buffer
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
        output_cb_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
                               .set_page_size(output_cb_index, output_single_tile_size)
                               .set_page_size(interm0_cb_index, interm0_single_tile_size)
                               .set_tile_dims(output_cb_index, output_tile)
                               .set_tile_dims(interm0_cb_index, output_tile);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, output_cb_config);
    log_debug(
        tt::LogOp,
        "CB {} :: PS = {}, NP = {}, TOTAL = {}",
        output_cb_index,
        output_single_tile_size,
        out_CB_size / output_single_tile_size,
        out_CB_size);

    // resharded output
    uint32_t output_reshard_cb_index = tt::CBIndex::c_6;
    std::map<uint8_t, tt::DataFormat> output_reshard_cb_data_format_spec{
        {output_reshard_cb_index, output_data_format},
    };
    tt_metal::CircularBufferConfig output_reshard_cb_config =
        tt_metal::CircularBufferConfig(out_reshard_CB_size, output_reshard_cb_data_format_spec)
            .set_page_size(output_reshard_cb_index, output_single_tile_size)
            .set_tile_dims(output_reshard_cb_index, output_tile);
    output_reshard_cb_config = output_reshard_cb_config.set_globally_allocated_address(*out_buffer);
    auto cb_output_reshard = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, output_reshard_cb_config);

    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = tt::CBIndex::c_3;
        tt_metal::CircularBufferConfig cb_src3_config =
            tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
                .set_page_size(src3_cb_index, bias_single_tile_size)
                .set_tile_dims(src3_cb_index, bias_tile);
        auto cb_src3 = tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, cb_src3_config);
        log_debug(
            LogOp,
            "CB {} :: PS = {}, NP = {}, TOTAL = {}",
            src3_cb_index,
            bias_single_tile_size,
            in3_CB_size / bias_single_tile_size,
            in3_CB_size);
    }

    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;

    std::vector<uint32_t> in0_mcast_sender_noc_x;
    std::vector<uint32_t> in0_mcast_sender_noc_y;
    std::vector<CoreCoord> mcast_senders_coords = corerange_to_cores(mcast_senders);
    std::sort(mcast_senders_coords.begin(), mcast_senders_coords.end(), [](const CoreCoord& a, const CoreCoord& b) {
        if (a.y != b.y) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });
    for (auto core : mcast_senders_coords) {
        in0_mcast_sender_noc_x.push_back((std::uint32_t)device->worker_core_from_logical_core(core).x);
    }
    for (auto core : mcast_senders_coords) {
        in0_mcast_sender_noc_y.push_back((std::uint32_t)device->worker_core_from_logical_core(core).y);
    }

    uint32_t sender_id = 0;
    for (auto core : mcast_senders_coords) {
        std::vector<uint32_t> mm_in0_sender_args;

        // mcast sender - 1, mcast sender + compute core - 2
        uint32_t worker_core_type;
        if (find(storage_worker_common.begin(), storage_worker_common.end(), core) != storage_worker_common.end()) {
            worker_core_type = 2;
        } else {
            worker_core_type = 1;
        }

        mm_in0_sender_args.push_back((std::uint32_t)worker_core_type);
        mm_in0_sender_args.push_back((std::uint32_t)sender_id);
        mm_in0_sender_args.push_back(
            (std::uint32_t)((core == all_storage_cores_vec.back()) and (in0_last_ktile_w > 0)));
        mm_in0_sender_args.insert(
            mm_in0_sender_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_sender_args.insert(
            mm_in0_sender_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());

        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args);
        reader_kernel_ids.push_back(mm_kernel_in0_sender_id);

        sender_id++;
    }

    std::vector<CoreCoord> mcast_receiver_coords = corerange_to_cores(mcast_receivers);
    for (uint32_t i = 0; i < mcast_receiver_coords.size(); ++i) {
        auto core = mcast_receiver_coords[i];

        // in0 receivers rt args
        std::vector<uint32_t> mm_in0_receiver_args;
        // mcast receiver - 3
        uint32_t worker_core_type = 3;
        mm_in0_receiver_args.push_back((std::uint32_t)worker_core_type);
        mm_in0_receiver_args.push_back((std::uint32_t)0);
        mm_in0_receiver_args.push_back((std::uint32_t)0);  // in0_last_ktile_w
        mm_in0_receiver_args.insert(
            mm_in0_receiver_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_receiver_args.insert(
            mm_in0_receiver_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());

        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_receiver_args);
        reader_kernel_ids.push_back(mm_kernel_in0_sender_id);
    }

    for (auto core : all_cores_in_rect_grid_vec) {
        if (std::find(mcast_senders_coords.begin(), mcast_senders_coords.end(), core) == mcast_senders_coords.end() and
            std::find(mcast_receiver_coords.begin(), mcast_receiver_coords.end(), core) ==
                mcast_receiver_coords.end()) {
            // in0 receivers rt args
            std::vector<uint32_t> mm_in0_idle_args;
            // idle core - 0
            uint32_t worker_core_type = 0;
            mm_in0_idle_args.push_back((std::uint32_t)worker_core_type);

            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_idle_args);
        }
    }

    uint32_t bank_id = 0;
    std::vector<uint32_t> bank_ids;
    uint32_t curr_storage_core_idx = 0;
    uint32_t per_core_N_storage_curr_stride = 0;

    uint32_t worker_core_stride = 0;
    uint32_t storage_core_stride = 0;
    uint32_t curr_worker_core = 0;
    uint32_t curr_storage_core = 0;

    // for all the cores in the rect grid, we send one rt arg to determine if they are worker core
    for (uint32_t i = 0; i < all_cores_in_rect_grid_vec.size(); ++i) {
        auto core = all_cores_in_rect_grid_vec[i];

        if (std::find(all_worker_cores.ranges().begin(), all_worker_cores.ranges().end(), core) ==
            all_worker_cores.ranges().end()) {  // not worker
            // in1 reader rt args
            bool is_worker_core = false;
            std::vector<uint32_t> mm_in1_sender_writer_args;
            mm_in1_sender_writer_args.push_back((std::uint32_t)is_worker_core);

            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);

            // compute rt args
            std::vector<uint32_t> mm_compute_args;
            mm_compute_args.push_back((std::uint32_t)is_worker_core);

            tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_compute_args);
        } else {
            // compute rt args
            bool is_worker_core = true;
            std::vector<uint32_t> mm_compute_args;
            mm_compute_args.push_back((std::uint32_t)is_worker_core);

            tt_metal::SetRuntimeArgs(program, mm_kernel, core, mm_compute_args);
        }
    }

    uint32_t num_cores_written_back = (N + per_core_N_storage - 1) / per_core_N_storage;
    uint32_t expected_max_total_width = num_cores_written_back * per_core_N_storage;
    log_debug(tt::LogOp, "per_core_N_storage: {}", per_core_N_storage);
    log_debug(tt::LogOp, "num_cores_written_back: {}", num_cores_written_back);
    uint32_t total_tensor_width_written_back = 0;
    for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
        auto core = all_worker_cores_ordered[i];

        // in1 reader rt args
        bool is_worker_core = true;
        std::vector<uint32_t> mm_in1_sender_writer_args;
        mm_in1_sender_writer_args.push_back((std::uint32_t)is_worker_core);
        mm_in1_sender_writer_args.push_back(in1_buffer->address());
        if (bias_buffer != nullptr) {
            mm_in1_sender_writer_args.push_back(bias_buffer->address());
        } else {
            mm_in1_sender_writer_args.push_back(0);
        }

        uint32_t vc = bank_id & 0x3;
        bank_ids.push_back(bank_id);
        for (uint32_t j = 0; j < i; ++j) {
            auto core_prev = all_worker_cores_ordered[j];

            if (core_prev.y == core.y and ((bank_id & 0x3) == (bank_ids[j] & 0x3))) {  // same vc and same row
                vc = (vc + 1) & 0x3;
                break;
            }
        }
        mm_in1_sender_writer_args.push_back((std::uint32_t)bank_id);
        mm_in1_sender_writer_args.push_back((std::uint32_t)vc);

        bank_id = (bank_id + 1) % num_dram_banks;

        if (per_core_N_in1_sender < per_core_N_storage) {
            if (curr_storage_core_idx < num_cores_written_back) {
                uint32_t remaining_per_core_N_storage = (per_core_N_storage - per_core_N_storage_curr_stride);
                uint32_t per_core_N_reshard_1 = (remaining_per_core_N_storage > per_core_N_in1_sender)
                                                    ? per_core_N_in1_sender
                                                    : remaining_per_core_N_storage;
                uint32_t per_core_N_reshard_2 = per_core_N_in1_sender - per_core_N_reshard_1;

                if (per_core_N_reshard_2 != 0 and (curr_storage_core_idx + 1) < num_cores_written_back) {
                    mm_in1_sender_writer_args.push_back(2);
                } else {
                    mm_in1_sender_writer_args.push_back(1);
                }

                log_debug(
                    tt::LogOp,
                    "curr worker core: {}, send back: {} tiles to storage core: {}, coord: {}",
                    i,
                    per_core_N_reshard_1,
                    curr_storage_core_idx,
                    mcast_senders_coords[curr_storage_core_idx]);

                mm_in1_sender_writer_args.push_back(
                    per_core_N_storage_curr_stride * output_single_tile_size);  // reshard_tensor_start_offset
                mm_in1_sender_writer_args.push_back(
                    per_core_N_reshard_1 * output_single_tile_size);  // per_core_N_reshard_bytes_1
                mm_in1_sender_writer_args.push_back(
                    in0_mcast_sender_noc_x[curr_storage_core_idx]);  // in0_mcast_sender_noc_x
                mm_in1_sender_writer_args.push_back(
                    in0_mcast_sender_noc_y[curr_storage_core_idx]);  // in0_mcast_sender_noc_y

                total_tensor_width_written_back += per_core_N_reshard_1;

                if (per_core_N_reshard_2 != 0 and (curr_storage_core_idx + 1) < num_cores_written_back) {
                    log_debug(
                        tt::LogOp,
                        "curr worker core: {}, send back: {} tiles to storage core: {}, coord: {}",
                        i,
                        per_core_N_reshard_2,
                        curr_storage_core_idx + 1,
                        mcast_senders_coords[curr_storage_core_idx + 1]);

                    mm_in1_sender_writer_args.push_back(
                        per_core_N_reshard_2 * output_single_tile_size);  // per_core_N_reshard_bytes_2
                    mm_in1_sender_writer_args.push_back(
                        in0_mcast_sender_noc_x[curr_storage_core_idx + 1]);  // in0_mcast_sender_noc_x
                    mm_in1_sender_writer_args.push_back(
                        in0_mcast_sender_noc_y[curr_storage_core_idx + 1]);  // in0_mcast_sender_noc_y

                    total_tensor_width_written_back += per_core_N_reshard_2;
                }

                curr_storage_core_idx += (per_core_N_storage_curr_stride + per_core_N_in1_sender) / per_core_N_storage;
                per_core_N_storage_curr_stride =
                    (per_core_N_storage_curr_stride + per_core_N_in1_sender) % per_core_N_storage;
            }
        } else {
            uint32_t num_cores_write_back = 0;

            if (curr_storage_core < num_cores_written_back) {
                num_cores_write_back++;

                worker_core_stride = per_core_N_storage - storage_core_stride;

                log_debug(
                    tt::LogOp,
                    "curr worker core: {}, send back: {} tiles to storage core: {}, coord: {}",
                    curr_worker_core,
                    worker_core_stride,
                    curr_storage_core,
                    mcast_senders_coords[curr_storage_core]);

                mm_in1_sender_writer_args.push_back(
                    storage_core_stride * output_single_tile_size);  // reshard_tensor_start_offset
                mm_in1_sender_writer_args.push_back(
                    worker_core_stride * output_single_tile_size);  // per_core_N_reshard
                mm_in1_sender_writer_args.push_back(
                    in0_mcast_sender_noc_x[curr_storage_core]);  // in0_mcast_sender_noc_x
                mm_in1_sender_writer_args.push_back(
                    in0_mcast_sender_noc_y[curr_storage_core]);  // in0_mcast_sender_noc_y

                curr_storage_core += (storage_core_stride + worker_core_stride) / per_core_N_storage;
                storage_core_stride = (storage_core_stride + worker_core_stride) % per_core_N_storage;

                if (worker_core_stride >= per_core_N_in1_sender) {
                    curr_worker_core += 1;
                }

                total_tensor_width_written_back += worker_core_stride;

                while (curr_worker_core <= i and curr_storage_core < num_cores_written_back) {
                    num_cores_write_back++;

                    bool increment_worker_core = (worker_core_stride + per_core_N_storage) >= per_core_N_in1_sender;
                    uint32_t current_worker_stride_total =
                        increment_worker_core ? per_core_N_in1_sender : worker_core_stride + per_core_N_storage;
                    uint32_t current_worker_write_back_tiles = current_worker_stride_total - worker_core_stride;

                    log_debug(
                        tt::LogOp,
                        "curr worker core: {}, send back: {} tiles to storage core: {}, coord: {}",
                        curr_worker_core,
                        current_worker_write_back_tiles,
                        curr_storage_core,
                        mcast_senders_coords[curr_storage_core]);

                    if (increment_worker_core) {
                        curr_worker_core += 1;
                    }

                    mm_in1_sender_writer_args.push_back(
                        current_worker_write_back_tiles * output_single_tile_size);  // per_core_N_reshard
                    mm_in1_sender_writer_args.push_back(
                        in0_mcast_sender_noc_x[curr_storage_core]);  // in0_mcast_sender_noc_x
                    mm_in1_sender_writer_args.push_back(
                        in0_mcast_sender_noc_y[curr_storage_core]);  // in0_mcast_sender_noc_y

                    total_tensor_width_written_back += current_worker_write_back_tiles;

                    storage_core_stride = current_worker_write_back_tiles % per_core_N_storage;
                    curr_storage_core += current_worker_write_back_tiles / per_core_N_storage;
                    worker_core_stride = current_worker_stride_total;
                }
            }

            mm_in1_sender_writer_args.insert(mm_in1_sender_writer_args.begin() + 5, num_cores_write_back);
        }

        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);
        writer_kernel_ids.push_back(mm_kernel_in1_sender_writer_id);
    }

    TT_FATAL(
        total_tensor_width_written_back <= expected_max_total_width,
        "more datums written back to sharded tensor, L1 corruption, expected: {}, actual: {}",
        expected_max_total_width,
        total_tensor_width_written_back);

    auto override_runtime_arguments_callback =
        [writer_kernel_ids, all_worker_cores_ordered, cb_src2, cb_output_reshard](
            const void* operation,
            tt::tt_metal::Program& program,
            const std::vector<tt::tt_metal::Tensor>& input_tensors,
            const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
            const std::vector<tt::tt_metal::Tensor>& output_tensors) {
            TT_FATAL(
                input_tensors.size() + optional_input_tensors.size() == 3,
                "Total number of input tensors (required + optional) must be 3, but got {} + {} = {}",
                input_tensors.size(),
                optional_input_tensors.size(),
                input_tensors.size() + optional_input_tensors.size());
            TT_FATAL(
                output_tensors.size() == 1, "Number of output tensors must be 1, but got {}", output_tensors.size());

            auto src_buffer_a = input_tensors.at(0).buffer();
            auto src_buffer_b = input_tensors.at(1).buffer();
            auto bias_tensor = optional_input_tensors.at(0);

            auto dst_buffer = output_tensors.at(0).buffer();

            UpdateDynamicCircularBufferAddress(program, cb_src2, *src_buffer_a);
            UpdateDynamicCircularBufferAddress(program, cb_output_reshard, *dst_buffer);

            for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
                auto core = all_worker_cores_ordered[i];
                auto writer_kernel_id = writer_kernel_ids[i];
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                writer_runtime_args[1] = src_buffer_b->address();
                if (bias_tensor.has_value()) {
                    writer_runtime_args[2] = bias_tensor.value().buffer()->address();
                } else {
                    writer_runtime_args[2] = 0;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}
}  // namespace reuse_dram_sharded_optimized_helpers

namespace ttnn {

namespace operations {

namespace matmul {

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_dram_sharded_optimized_(
    const ttnn::MeshCoordinate& mesh_coord,
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back) {
    const auto &ashape = a.padded_shape(), bshape = b.padded_shape();
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    // cannot use the output tensor tile directly as that might be changed by user override
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile.get_tile_shape()[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());          // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());          // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());  // output

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;  // bias; doesn't matter if bias=nullptr
    if (bias.has_value()) {
        auto& c = bias.value();
        TT_FATAL(c.storage_type() == StorageType::DEVICE, "Error");
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

        bias_buffer = c.buffer();

        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    tt::tt_metal::IDevice* device = a.mesh_device()->get_device(mesh_coord);

    TT_FATAL(a.shard_spec().has_value() && output.shard_spec().has_value(), "Error");
    CoreRangeSet all_cores_storage = a.shard_spec().value().grid;

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
    uint32_t B = 1;
    uint32_t Mt = get_batch_size(ashape) * ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];
    uint32_t in0_last_ktile_w = a.logical_shape()[-1] % in0_tile_shape[1];

    TT_FATAL(Kt % in0_block_w == 0, "Error");

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Buffer* out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    return reuse_dram_sharded_optimized_helpers::create_program_dram_sharded(
        device,
        all_cores_storage,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        B,
        Mt,
        Nt,
        Kt,
        in0_block_w,
        in0_last_ktile_w,
        per_core_M,
        per_core_N,
        std::move(fused_activation),
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
        skip_compute,
        skip_in0_mcast,
        skip_write_back);
}

tt::tt_metal::operation::ProgramWithCallbacks matmul_multi_core_reuse_dram_sharded_optimized(
    const ttnn::MeshCoordinate& mesh_coord,
    const Tensor& a,
    const Tensor& b,
    const std::optional<const Tensor>& bias,
    Tensor& output_tensor,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    std::optional<UnaryWithParam> fused_activation,
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back) {
    return matmul_multi_core_reuse_dram_sharded_optimized_(
        mesh_coord,
        a,
        b,
        bias,
        output_tensor,
        compute_kernel_config,
        in0_block_w,
        per_core_M,
        per_core_N,
        std::move(fused_activation),
        untilize_out,
        skip_compute,
        skip_in0_mcast,
        skip_write_back);
}

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn
