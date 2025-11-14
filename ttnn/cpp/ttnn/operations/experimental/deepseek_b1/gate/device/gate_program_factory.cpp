// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_program_factory.hpp"

#include <algorithm>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

using namespace tt;
using namespace tt::constants;

namespace ttnn {

namespace operations {

namespace experimental {

namespace deepseek_b1 {

namespace gate {

gate_common_override_variables_t deepseek_b1_gate_(
    tt_metal::Program& program,
    const Tensor& a,
    const Tensor& b,
    const Tensor& expert_bias,
    const Tensor& output_tensor,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config) {
    const auto& output = output_tensor;

    // Hardcoded parameters
    bool fuse_batch = true;
    bool untilize_out = false;
    uint32_t start_cb_index = tt::CBIndex::c_0;
    auto throttle_level = ttnn::get_throttle_level(compute_kernel_config);

    // Get tensor shapes and tiles
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    // cannot use the output tensor tile directly as that might be changed by user override
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile_shape[0], in1_tile_shape[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());          // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());          // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());  // output

    tt_metal::IDevice* device = a.device();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();

    auto [math_fidelity, math_approx_mode, _, __, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                      Gate Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // Calculate dimensions in tiles
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / in0_tile_shape[0];  // M dimension in tiles
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];  // K dimension in tiles (inner dim)
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];  // N dimension in tiles

    if (fuse_batch) {
        Mt = B * Mt;
        B = 1;
    }

    // Gate operation parameters
    uint32_t in0_block_w = Kt;  // Process full K dimension
    uint32_t out_subblock_h = 1;
    uint32_t out_subblock_w = 1;
    uint32_t per_core_M = 1;
    uint32_t per_core_N = 1;
    uint32_t out_block_h = 1;
    uint32_t out_block_w = 1;

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;

    // Calculate number of blocks along x and y
    uint32_t num_blocks_y = ((Mt - 1) / per_core_M) + 1;
    uint32_t num_blocks_x = ((Nt - 1) / per_core_N) + 1;
    uint32_t num_blocks_total = num_blocks_y * num_blocks_x;

    TT_FATAL(
        num_blocks_total <= num_cores,
        "Number of blocks exceeds number of cores: {} blocks > {} cores",
        num_blocks_total,
        num_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    TT_FATAL(start_cb_index == tt::CBIndex::c_0, "mcast does not support a non-zero start cb index");

    using tt::tt_metal::num_cores_to_corerangeset;

    uint32_t num_blocks = Kt / in0_block_w;

    tt::DataFormat interm0_data_format = output_data_format;

    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    bool in0_is_sharded = a.memory_config().is_sharded();
    bool in1_is_sharded = b.memory_config().is_sharded();
    bool output_is_sharded = output.memory_config().is_sharded();

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);

    uint32_t in0_block_h = out_block_h;
    uint32_t in1_block_w = out_block_w;

    uint32_t in0_block_tiles = in0_block_h * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles *= ttnn::operations::matmul::MCAST_INPUT_BUFFERING_DEPTH;
    }

    uint32_t in0_shard_width_in_tiles = 0;
    if (in0_is_sharded) {
        // CB size needs to be full shard size when sharded
        uint32_t in0_shard_height_in_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_tile_shape()[0];
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
        in0_CB_tiles = in0_shard_height_in_tiles * in0_shard_width_in_tiles;
    }

    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

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

    CoreCoord start_core = {0, 0};
    uint32_t start_core_x = start_core.x;
    uint32_t start_core_y = start_core.y;
    uint32_t num_cores_c = compute_with_storage_grid_size.x;

    uint32_t num_cores_with_work = num_blocks_total;

    uint32_t in0_sender_num_cores = in0_is_sharded ? a.shard_spec().value().grid.num_cores() : 1;
    num_cores = in0_is_sharded ? std::max(num_cores_with_work, in0_sender_num_cores) : num_cores_with_work;

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
    std::vector<uint32_t> in0_mcast_noc_x;
    std::vector<uint32_t> in0_mcast_noc_y;

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

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernel Creation
    ////////////////////////////////////////////////////////////////////////////

    // Mcast args
    auto in0_mcast_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);
    auto in0_mcast_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, INVALID);

    CoreCoord top_left_core = in0_mcast_receiver_cores_bounding_box.start_coord;
    CoreCoord bottom_right_core = in0_mcast_receiver_cores_bounding_box.end_coord;
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    std::vector<uint32_t> in0_sender_compile_time_args = {
        (std::uint32_t)in0_block_num_tiles,                         // 0: in0_block_num_tiles
        (std::uint32_t)in0_block_num_tiles * in0_single_tile_size,  // 1: in0_block_size_bytes
        (std::uint32_t)num_blocks,                                  // 2: num_blocks_inner_dim
        (std::uint32_t)in0_mcast_sender_semaphore_id,               // 3: in0_mcast_sender_semaphore_id
        (std::uint32_t)in0_mcast_receiver_semaphore_id,             // 4: in0_mcast_receiver_semaphore_id
        (std::uint32_t)in0_mcast_receiver_num_dests,                // 5: in0_mcast_num_dests
        (std::uint32_t)in0_mcast_receiver_num_cores,                // 6: in0_mcast_num_cores
        (std::uint32_t)(in0_mcast_sender_cores_grid.x),             // 7: num_x
        (std::uint32_t)(in0_mcast_sender_cores_grid.y),             // 8: num_y
        (std::uint32_t)(false),                                     // 9: transpose_mcast
        (std::uint32_t)(in0_shard_width_in_tiles),                  // 10: shard_width_in_tiles
        (std::uint32_t)(in0_block_w),                               // 11: in0_block_w
        (std::uint32_t)B                                            // 12: batch
    };

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        (std::uint32_t)in1_block_w * in0_block_w,  // 0: in1_block_num_tiles
        (std::uint32_t)num_blocks,                 // 1: num_blocks_inner_dim
        (std::uint32_t)B,                          // 2: batch
        (std::uint32_t)out_subblock_w,             // 3: out_subblock_w
        (std::uint32_t)out_subblock_h              // 4: out_subblock_h
    };

    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_writer_defines;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;

    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    if (in1_is_sharded) {
        mm_kernel_in1_sender_writer_defines["IN1_SHARDED"] = "1";
    }

    if (output_is_sharded) {
        mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
    }

    if (in0_mcast_receiver_num_cores == 1) {
        mm_kernel_in0_sender_writer_defines["SKIP_MCAST"] = "1";
    }

    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    // Intermediate CB read (Blackhole alignment workaround)
    bool in0_needs_intermediate_cb_read = false;
    bool in1_needs_intermediate_cb_read = false;
    if (device->arch() == tt::ARCH::BLACKHOLE) {
        in0_needs_intermediate_cb_read = ((in0_single_tile_size % 64) != 0);
        if (in0_needs_intermediate_cb_read) {
            mm_kernel_in0_sender_writer_defines["INTERMEDIATE_CB_READ"] = "1";
        }
        in1_needs_intermediate_cb_read = ((in1_single_tile_size % 64) != 0);
        if (in1_needs_intermediate_cb_read) {
            mm_kernel_in1_sender_writer_defines["INTERMEDIATE_CB_READ"] = "1";
        }
    }

    // NOC selection
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    // Create reader kernel (RISCV_1)
    auto mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/gate/device/kernels/reader.cpp",
        in0_mcast_cores_with_work_and_in_receiver_grid,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = mm_kernel_in0_sender_writer_defines});

    tt::tt_metal::KernelHandle mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid_id = 0;
    tt::tt_metal::KernelHandle mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id = 0;
    if (in0_mcast_cores_without_work_and_in_receiver_grid.num_cores() > 0) {
        mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/gate/device/kernels/reader.cpp",
            in0_mcast_cores_without_work_and_in_receiver_grid,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_sender_compile_time_args,
                .defines = mm_kernel_in0_sender_writer_defines});
    }
    if (in0_mcast_cores_without_work_and_not_in_receiver_grid.num_cores() > 0) {
        mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/gate/device/kernels/reader.cpp",
            in0_mcast_cores_without_work_and_not_in_receiver_grid,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = in0_noc,
                .compile_args = in0_sender_compile_time_args,
                .defines = mm_kernel_in0_sender_writer_defines});
    }

    // Create writer kernel (RISCV_0)
    auto mm_kernel_in1_sender_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/gate/device/kernels/writer.cpp",
        all_cores_with_work,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_sender_writer_compile_time_args,
            .defines = mm_kernel_in1_sender_writer_defines});

    // Compute kernel compile time args
    uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // 0: in0_block_w
        in0_block_num_tiles,     // 1: in0_block_num_tiles
        in1_block_num_tiles,     // 2: in1_block_num_tiles
        in1_per_core_w,          // 3: in1_block_w
        out_subblock_h,          // 4: out_subblock_h
        out_subblock_w,          // 5: out_subblock_w
        out_subblock_num_tiles,  // 6: out_subblock_num_tiles
        untilize_out             // 7: untilize_out
    };

    // Create compute kernel
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_b1/gate/device/kernels/gate.cpp",
        all_cores_with_work,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = false,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    ////////////////////////////////////////////////////////////////////////////
    //                      Circular Buffer Creation
    ////////////////////////////////////////////////////////////////////////////

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);

    // Directly use cb0 for sharded
    if (in0_is_sharded) {
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

    // Local L1 to store temp vars (only when sharded)
    if (in0_is_sharded) {
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

    tt_metal::Buffer* out_buffer = output.buffer();
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

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime Args Setup
    ////////////////////////////////////////////////////////////////////////////

    // Parameters for last row, col, or block
    uint32_t last_per_core_N = Nt % per_core_N == 0 ? per_core_N : Nt % per_core_N;
    uint32_t last_out_block_w = last_per_core_N % out_block_w == 0 ? out_block_w : last_per_core_N % out_block_w;
    uint32_t last_block_num_nonzero_subblocks_w = ((last_out_block_w - 1) / out_subblock_w) + 1;

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i % num_blocks_x;

        std::vector<uint32_t> mm_in0_sender_args;
        mm_in0_sender_args.reserve(5 + in0_mcast_noc_x.size() + in0_mcast_noc_y.size());
        mm_in0_sender_args.push_back(i);
        mm_in0_sender_args.push_back(start_core_noc.x);
        mm_in0_sender_args.push_back(start_core_noc.y);
        mm_in0_sender_args.push_back(end_core_noc.x);
        mm_in0_sender_args.push_back(end_core_noc.y);
        mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_x.begin(), in0_mcast_noc_x.end());
        mm_in0_sender_args.insert(mm_in0_sender_args.end(), in0_mcast_noc_y.begin(), in0_mcast_noc_y.end());

        if (i < num_cores_with_work) {
            tt_metal::SetRuntimeArgs(
                program, mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id, core, mm_in0_sender_args);
        } else if (i < in0_mcast_receiver_num_dests) {
            tt_metal::SetRuntimeArgs(
                program, mm_kernel_in0_mcast_cores_without_work_and_in_receiver_grid_id, core, mm_in0_sender_args);
        } else {
            tt_metal::SetRuntimeArgs(
                program, mm_kernel_in0_mcast_cores_without_work_and_not_in_receiver_grid_id, core, mm_in0_sender_args);
        }
        if (i < num_cores_with_work) {
            std::vector<uint32_t> mm_in1_sender_writer_args;

            // Writer runtime args
            mm_in1_sender_writer_args.push_back(out_block_h / out_subblock_h);  // out_num_nonzero_subblocks_h
            if (output_idx_x == num_blocks_x - 1) {
                mm_in1_sender_writer_args.push_back(last_block_num_nonzero_subblocks_w);  // out_num_nonzero_subblocks_w
            } else {
                mm_in1_sender_writer_args.push_back(out_block_w / out_subblock_w);  // out_num_nonzero_subblocks_w
            }

            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_sender_writer_id, core, mm_in1_sender_writer_args);
        }
    }

    return gate_common_override_variables_t{
        {mm_kernel_in0_mcast_cores_with_work_and_in_receiver_grid_id, mm_kernel_in1_sender_writer_id},
        {cb_src0, cb_src1, cb_output},
        false,
        start_core,
        cores,
        num_cores_with_work,
    };
}

tt::tt_metal::operation::ProgramWithCallbacks deepseek_b1_gate(
    const Tensor& a,
    const Tensor& b,
    const Tensor& expert_bias,
    const Tensor& output_tensor,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config) {
    tt_metal::Program program{};

    gate_common_override_variables_t shared_vars = deepseek_b1_gate_(
        program, a, b, expert_bias, output_tensor, compute_with_storage_grid_size, compute_kernel_config);
    auto override_runtime_arguments_callback =
        [shared_vars](
            const void* operation,
            tt_metal::Program& program,
            const std::vector<tt::tt_metal::Tensor>& input_tensors,
            const std::vector<std::optional<const tt::tt_metal::Tensor>>& optional_input_tensors,
            const std::vector<tt::tt_metal::Tensor>& output_tensors) {
            TT_FATAL(input_tensors.size() == 3, "gate requires 3 input tensors, {} provided", input_tensors.size());
            TT_FATAL(output_tensors.size() == 1, "gate requires 1 output tensor, {} provided", output_tensors.size());

            auto src_buffer_a = input_tensors.at(0).buffer();
            auto src_buffer_b = input_tensors.at(1).buffer();
            // auto expert_bias_buffer = input_tensors.at(2).buffer();

            auto dst_buffer = output_tensors.at(0).buffer();

            bool src0_sharded = input_tensors[0].is_sharded();
            bool src1_sharded = input_tensors[1].is_sharded();
            bool out_sharded = output_tensors[0].is_sharded();

            // Update in0 sharded buffer (cb_src0 is at index 0)
            if (src0_sharded) {
                UpdateDynamicCircularBufferAddress(program, shared_vars.cbs.at(0), *src_buffer_a);
            }

            if (src1_sharded) {
                UpdateDynamicCircularBufferAddress(program, shared_vars.cbs.at(1), *src_buffer_b);
            }

            if (out_sharded) {
                UpdateDynamicCircularBufferAddress(program, shared_vars.cbs.at(2), *dst_buffer);
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace gate

}  // namespace deepseek_b1

}  // namespace experimental

}  // namespace operations

}  // namespace ttnn
