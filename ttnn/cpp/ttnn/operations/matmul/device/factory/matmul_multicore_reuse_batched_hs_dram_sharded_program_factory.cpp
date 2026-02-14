// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

namespace ttnn::prim {
namespace reuse_batched_hs_dram_sharded_optimized_helpers {

using dram_sharded_helpers::get_device_for_dram_banks;
using dram_sharded_helpers::get_max_page_size_and_num_pages;
using dram_sharded_helpers::get_optimal_dram_bank_to_reader_assignment;

// Batch-sharded DRAM matmul
// For batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
// Sharded by batch dimension - each worker handles B/num_workers complete matmuls
std::pair<tt::tt_metal::Program, MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::shared_variables_t>
create_program_batch_sharded(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& input_all_storage_cores,
    const CoreRangeSet& output_all_storage_cores,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    uint32_t B,            // Total batch size
    uint32_t M,            // M dimension (rows of A, rows of output)
    uint32_t K,            // K dimension (cols of A, rows of B - contracted dimension)
    uint32_t N,            // N dimension (cols of B, cols of output)
    uint32_t in0_block_w,  // Block width for inner loop over K
    uint32_t per_core_M,   // M tiles per core (should equal M for batch sharding)
    uint32_t per_core_N,   // N tiles per core (should equal N for batch sharding)
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
    bool skip_write_back) {
    log_debug(tt::LogOp, "Batch-sharded DRAM matmul");
    log_debug(tt::LogOp, "B: {}, M: {}, K: {}, N: {}", B, M, K, N);
    log_debug(tt::LogOp, "per_core_M: {}, per_core_N: {}, in0_block_w: {}", per_core_M, per_core_N, in0_block_w);

    tt_metal::Program program{};

    // Get the optimal DRAM bank to worker core assignment
    // Workers MUST be on these cores for efficient DRAM reads
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    std::vector<CoreCoord> all_worker_cores_ordered;
    CoreRangeSet all_worker_cores;
    get_optimal_dram_bank_to_reader_assignment(device, all_worker_cores_ordered, all_worker_cores, in1_noc);

    log_debug(tt::LogOp, "=== Worker cores (optimal DRAM readers) ===");
    for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
        log_debug(
            tt::LogOp,
            "  Worker/DRAM bank {} -> core ({}, {})",
            i,
            all_worker_cores_ordered[i].x,
            all_worker_cores_ordered[i].y);
    }

    // Input storage cores - where in0 L1 shards physically reside
    std::vector<CoreCoord> input_storage_cores_ordered =
        corerange_to_cores(input_all_storage_cores, std::nullopt, true);
    log_debug(tt::LogOp, "=== Input storage cores (in0 L1 shards) ===");
    for (uint32_t i = 0; i < input_storage_cores_ordered.size(); ++i) {
        log_debug(
            tt::LogOp,
            "  Input shard {} -> core ({}, {})",
            i,
            input_storage_cores_ordered[i].x,
            input_storage_cores_ordered[i].y);
    }

    // Output storage cores - where output L1 shards need to go
    std::vector<CoreCoord> output_storage_cores_ordered =
        corerange_to_cores(output_all_storage_cores, std::nullopt, true);
    log_debug(tt::LogOp, "=== Output storage cores (output L1 shards) ===");
    for (uint32_t i = 0; i < output_storage_cores_ordered.size(); ++i) {
        log_debug(
            tt::LogOp,
            "  Output shard {} -> core ({}, {})",
            i,
            output_storage_cores_ordered[i].x,
            output_storage_cores_ordered[i].y);
    }

    // Validate core counts
    uint32_t num_workers = all_worker_cores_ordered.size();
    uint32_t num_input_storage_cores = input_storage_cores_ordered.size();
    uint32_t num_output_storage_cores = output_storage_cores_ordered.size();

    log_debug(tt::LogOp, "num_workers (DRAM banks): {}", num_workers);
    log_debug(tt::LogOp, "num_input_storage_cores: {}", num_input_storage_cores);
    log_debug(tt::LogOp, "num_output_storage_cores: {}", num_output_storage_cores);

    // Currently, both input and output storage cores must match the number of workers (DRAM banks).
    // This is because the factory uses a 1:1 mapping: worker[i] reads from input_storage[i] and
    // writes to output_storage[i]. Supporting different numbers would require more complex mapping, where multiple
    // workers read from the same storage core.
    TT_FATAL(
        num_input_storage_cores == num_workers,
        "Input storage cores ({}) must match number of workers/DRAM banks ({}). "
        "Use an L1 shard grid with {} cores.",
        num_input_storage_cores,
        num_workers,
        num_workers);

    TT_FATAL(
        num_output_storage_cores == num_workers,
        "Output storage cores ({}) must match number of workers/DRAM banks ({}). "
        "Use an L1 shard grid with {} cores.",
        num_output_storage_cores,
        num_workers,
        num_workers);

    // Verify that storage cores match worker cores in the same order.
    // The factory maps worker[i] -> input_storage[i] -> output_storage[i].
    // If the orderings don't match, data gets misrouted to the wrong cores/batches
    for (uint32_t i = 0; i < num_workers; ++i) {
        TT_FATAL(
            input_storage_cores_ordered[i] == all_worker_cores_ordered[i],
            "Input storage core ordering mismatch at index {}! "
            "Storage core ({}, {}) != Worker core ({}, {}). "
            "The L1 shard grid must use the same core ordering as "
            "device.get_optimal_dram_bank_to_logical_worker_assignment(). "
            "Using e.g. a simple rectangular CoreRange will cause data misrouting.",
            i,
            input_storage_cores_ordered[i].x,
            input_storage_cores_ordered[i].y,
            all_worker_cores_ordered[i].x,
            all_worker_cores_ordered[i].y);

        TT_FATAL(
            output_storage_cores_ordered[i] == all_worker_cores_ordered[i],
            "Output storage core ordering mismatch at index {}! "
            "Storage core ({}, {}) != Worker core ({}, {}). "
            "The L1 shard grid must use the same core ordering as "
            "device.get_optimal_dram_bank_to_logical_worker_assignment(). "
            "Using e.g. a simple rectangular CoreRange will cause data misrouting.",
            i,
            output_storage_cores_ordered[i].x,
            output_storage_cores_ordered[i].y,
            all_worker_cores_ordered[i].x,
            all_worker_cores_ordered[i].y);
    }

    // Build NOC coordinate vectors for input and output storage cores
    std::vector<uint32_t> input_storage_noc_x, input_storage_noc_y;
    std::vector<uint32_t> output_storage_noc_x, output_storage_noc_y;
    for (const auto& core : input_storage_cores_ordered) {
        auto phys_core = device->worker_core_from_logical_core(core);
        input_storage_noc_x.push_back(phys_core.x);
        input_storage_noc_y.push_back(phys_core.y);
    }
    for (const auto& core : output_storage_cores_ordered) {
        auto phys_core = device->worker_core_from_logical_core(core);
        output_storage_noc_x.push_back(phys_core.x);
        output_storage_noc_y.push_back(phys_core.y);
    }

    // Compute the bounding box of all cores (workers + storage) for CB creation
    std::set<CoreRange> all_cores_set;
    for (const auto& core : all_worker_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    for (const auto& core : input_storage_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    for (const auto& core : output_storage_cores_ordered) {
        all_cores_set.insert(CoreRange(core));
    }
    CoreRangeSet all_cores(all_cores_set);
    CoreRange bounding_box = all_cores.bounding_box();
    CoreRangeSet all_cores_in_rect_grid({bounding_box});
    log_debug(
        tt::LogOp,
        "Bounding box: ({}, {}) to ({}, {})",
        bounding_box.start_coord.x,
        bounding_box.start_coord.y,
        bounding_box.end_coord.x,
        bounding_box.end_coord.y);

    uint32_t num_cores = num_workers;
    uint32_t num_dram_banks = device->num_dram_channels();
    uint32_t batches_per_core = (B + num_cores - 1) / num_cores;

    TT_FATAL(
        num_cores <= num_dram_banks,
        "Number of worker cores ({}) cannot exceed number of DRAM banks ({})",
        num_cores,
        num_dram_banks);

    log_debug(
        tt::LogOp,
        "num_cores: {}, num_dram_banks: {}, batches_per_core: {}",
        num_cores,
        num_dram_banks,
        batches_per_core);

    // Subblock parameters
    auto subblock_hw = operations::matmul::bmm_op_utils::get_matmul_subblock_params(
        per_core_M, per_core_N, false, false, fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    uint32_t num_blocks = K / in0_block_w;  // Number of inner loop iterations
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    // Tile sizes
    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    // CB sizes
    // in0: M x in0_block_w tiles per block
    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2;  // double buffer
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;

    // in1: in0_block_w x N tiles per block
    uint32_t in1_block_tiles = in0_block_w * per_core_N;
    uint32_t in1_CB_tiles = in1_block_tiles * 3;  // triple buffer
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;

    // output: M x N tiles
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t interm0_CB_size = out_block_tiles * interm0_single_tile_size;

    // Sharded input buffer (in0 in L1)
    uint32_t in0_shard_tiles = in0_buffer->shard_spec().shape()[0] / in0_tile.get_tile_shape()[0] *
                               in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    uint32_t in2_CB_size = in0_shard_tiles * in0_single_tile_size;

    // Bias CB
    uint32_t in3_block_tiles = per_core_N;
    uint32_t in3_CB_size = in3_block_tiles * bias_single_tile_size;

    // Output reshard buffer
    uint32_t out_shard_tiles = out_buffer->shard_spec().shape()[0] / output_tile.get_tile_shape()[0] *
                               out_buffer->shard_spec().shape()[1] / output_tile.get_tile_shape()[1];
    uint32_t out_reshard_CB_size = out_shard_tiles * output_single_tile_size;

    // Page sizes for DRAM reads
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in1_block_tiles, in1_single_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in3_block_tiles, bias_single_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    // Tensor stride calculations (bytes per batch)
    // M, K, N are already in tiles, so stride = num_tiles * tile_size
    uint32_t in0_batch_stride_bytes = per_core_M * K * in0_single_tile_size;
    uint32_t in1_batch_stride_bytes = K * per_core_N * in1_single_tile_size;
    uint32_t out_batch_stride_bytes = per_core_M * per_core_N * output_single_tile_size;

    log_debug(
        tt::LogOp,
        "in0_batch_stride_bytes: {}, in1_batch_stride_bytes: {}, out_batch_stride_bytes: {}",
        in0_batch_stride_bytes,
        in1_batch_stride_bytes,
        out_batch_stride_bytes);

    // Create CBs on different core sets:
    // - CB0, CB1, CB3 (compute buffers): on all cores in bounding box
    // - CB2 (in0 sharded): backed by in0_buffer, on INPUT storage cores only
    // - CB4/5 (output/intermediate): on worker cores (not backed by buffer)
    // - CB6 (output reshard): backed by out_buffer, on OUTPUT storage cores only

    // CB 0: in0 (activations) - on all cores in bounding box
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, src0_cb_config);

    // CB 1: in1 (weights) - on all cores in bounding box
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig src1_cb_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, src1_cb_config);

    // CB 2: sharded in0 buffer - on INPUT storage cores, backed by in0_buffer
    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt_metal::CircularBufferConfig src2_cb_config =
        tt_metal::CircularBufferConfig(in2_CB_size, {{src2_cb_index, in0_data_format}})
            .set_page_size(src2_cb_index, in0_single_tile_size)
            .set_tile_dims(src2_cb_index, in0_tile)
            .set_globally_allocated_address(*in0_buffer);
    auto cb_src2 = tt_metal::CreateCircularBuffer(program, input_all_storage_cores, src2_cb_config);

    // CB 3: bias (if fused) - on all cores in bounding box
    if (bias_buffer != nullptr) {
        uint32_t src3_cb_index = tt::CBIndex::c_3;
        tt_metal::CircularBufferConfig cb_src3_config =
            tt_metal::CircularBufferConfig(in3_CB_size, {{src3_cb_index, bias_data_format}})
                .set_page_size(src3_cb_index, bias_single_tile_size)
                .set_tile_dims(src3_cb_index, bias_tile);
        tt_metal::CreateCircularBuffer(program, all_cores_in_rect_grid, cb_src3_config);
    }

    // CB 4 & 5: output and intermediate - on worker cores (NOT backed by buffer)
    // Workers compute locally, then NOC write to output storage cores
    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
    if (interm0_data_format != output_data_format) {
        // Need separate CBs for output and intermediate
        tt_metal::CircularBufferConfig output_cb_config =
            tt_metal::CircularBufferConfig(out_reshard_CB_size, {{output_cb_index, output_data_format}})
                .set_page_size(output_cb_index, output_single_tile_size)
                .set_tile_dims(output_cb_index, output_tile);
        tt_metal::CreateCircularBuffer(program, all_worker_cores, output_cb_config);

        tt_metal::CircularBufferConfig interm0_cb_config =
            tt_metal::CircularBufferConfig(interm0_CB_size, {{interm0_cb_index, interm0_data_format}})
                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                .set_tile_dims(interm0_cb_index, output_tile);
        tt_metal::CreateCircularBuffer(program, all_worker_cores, interm0_cb_config);
    } else {
        // Output and intermediate share the same buffer
        std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec{
            {output_cb_index, output_data_format}, {interm0_cb_index, interm0_data_format}};
        tt_metal::CircularBufferConfig output_cb_config =
            tt_metal::CircularBufferConfig(out_reshard_CB_size, output_cb_data_format_spec)
                .set_page_size(output_cb_index, output_single_tile_size)
                .set_page_size(interm0_cb_index, interm0_single_tile_size)
                .set_tile_dims(output_cb_index, output_tile)
                .set_tile_dims(interm0_cb_index, output_tile);
        tt_metal::CreateCircularBuffer(program, all_worker_cores, output_cb_config);
    }

    // CB 6: output reshard buffer - on OUTPUT storage cores, backed by out_buffer
    uint32_t output_reshard_cb_index = tt::CBIndex::c_6;
    tt_metal::CircularBufferConfig output_reshard_cb_config =
        tt_metal::CircularBufferConfig(out_reshard_CB_size, {{output_reshard_cb_index, output_data_format}})
            .set_page_size(output_reshard_cb_index, output_single_tile_size)
            .set_tile_dims(output_reshard_cb_index, output_tile)
            .set_globally_allocated_address(*out_buffer);
    auto cb_output_reshard =
        tt_metal::CreateCircularBuffer(program, output_all_storage_cores, output_reshard_cb_config);

    // Kernel defines
    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;

    if (bias_buffer != nullptr) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        writer_defines["FUSE_BIAS"] = "1";
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
    if (skip_compute) {
        mm_kernel_defines["SKIP_COMPUTE"] = "1";
    }
    if (skip_write_back) {
        writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    // MATMUL_DRAM_SHARDED enables the is_worker_core check in compute kernel
    // so that idle cores in the bounding box return early instead of waiting for CB data
    mm_kernel_defines["MATMUL_DRAM_SHARDED"] = "1";

    // in0 reader compile time args
    // Now includes info for NOC read from remote input storage cores
    std::vector<uint32_t> in0_reader_compile_args = {
        (uint32_t)in0_block_tiles,                         // in0_block_num_tiles
        (uint32_t)in0_block_tiles * in0_single_tile_size,  // in0_block_size_bytes
        (uint32_t)num_blocks,                              // num_blocks (K / in0_block_w)
        (uint32_t)batches_per_core,                        // num_batches_per_core
        (uint32_t)in0_batch_stride_bytes,                  // in0_tensor_stride_batch_bytes
        (uint32_t)in2_CB_size,                             // in0_shard_size_bytes (full shard)
    };

    // in1 reader/writer compile time args
    // Now includes info for NOC write to remote output storage cores
    std::vector<uint32_t> in1_writer_compile_args = {
        (uint32_t)in1_buffer_page_size,
        (uint32_t)in1_buffer_num_pages,
        (uint32_t)per_core_N,              // in1_block_w (N tiles)
        (uint32_t)in1_block_tiles,         // in1_block_num_tiles
        (uint32_t)num_blocks,              // num_blocks (K / in0_block_w)
        (uint32_t)out_block_tiles,         // out_block_num_tiles
        (uint32_t)batches_per_core,        // num_batches_per_core
        (uint32_t)in1_batch_stride_bytes,  // in1_tensor_stride_batch_bytes
        (uint32_t)out_batch_stride_bytes,  // out_tensor_stride_batch_bytes (for NOC write)
        (uint32_t)out_reshard_CB_size,     // out_shard_size_bytes (full shard)
    };
    if (bias_buffer != nullptr) {
        in1_writer_compile_args.push_back(bias_buffer_page_size);
        in1_writer_compile_args.push_back(bias_buffer_num_pages);
        in1_writer_compile_args.push_back(in3_block_tiles);
    }

    // Compute kernel compile time args
    uint32_t in0_num_subblocks = per_core_M / out_subblock_h;
    uint32_t in1_num_subblocks = per_core_N / out_subblock_w;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_tiles,         // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles
        in1_num_subblocks,       // in1_num_subblocks
        in1_block_tiles,         // in1_block_num_tiles
        per_core_N,              // in1_per_core_w
        num_blocks,              // num_blocks
        1,                       // out_num_blocks_x
        1,                       // out_num_blocks_y
        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        batches_per_core,        // batch (batches per core)
        out_block_tiles,         // out_block_num_tiles
        untilize_out ? 1u : 0u,  // untilize_out
        0u,                      // get_batch_from_reader
        0u,                      // in0_transpose_tile
    };

    // Create kernels on all cores in bounding box
    // Runtime args control which cores are active workers vs idle
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());

    // Add define to indicate we need remote NOC reads/writes
    writer_defines["OUT_SHARDED"] = "1";  // Output goes to sharded buffer on storage cores

    auto mm_kernel_in0_reader_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp",
        all_cores_in_rect_grid,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_reader_compile_args,
            .defines = reader_defines});

    auto mm_kernel_in1_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
        "reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp",
        all_cores_in_rect_grid,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = in1_noc,
            .compile_args = in1_writer_compile_args,
            .defines = writer_defines});

    auto mm_kernel_compute_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores_in_rect_grid,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    // Set runtime args - each core only gets SetRuntimeArgs called ONCE per kernel
    // Following the pattern from the mcast DRAM sharded factory
    std::vector<CoreCoord> all_cores_in_rect_grid_vec = corerange_to_cores(all_cores_in_rect_grid);

    // Build a set of worker cores for fast lookup
    std::set<CoreCoord> worker_cores_set(all_worker_cores_ordered.begin(), all_worker_cores_ordered.end());

    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    std::vector<uint32_t> bank_ids;  // Track bank_ids for vc calculation

    // First, set idle args for non-worker cores in the bounding box
    for (const auto& core : all_cores_in_rect_grid_vec) {
        bool is_worker = worker_cores_set.contains(core);

        if (!is_worker) {
            // Idle core - set minimal args (1 arg each)
            std::vector<uint32_t> in0_idle_args = {0u};  // worker_core_type = 0
            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_reader_id, core, in0_idle_args);

            std::vector<uint32_t> in1_idle_args = {0u};  // is_worker_core = 0
            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_writer_id, core, in1_idle_args);

            std::vector<uint32_t> compute_idle_args = {0u};  // is_worker_core = 0
            tt_metal::SetRuntimeArgs(program, mm_kernel_compute_id, core, compute_idle_args);
        }
    }

    // Then, iterate over workers in all_worker_cores_ordered order (like mcast factory)
    // This ensures bank_ids[i] corresponds to all_worker_cores_ordered[i]
    for (uint32_t worker_idx = 0; worker_idx < all_worker_cores_ordered.size(); ++worker_idx) {
        auto core = all_worker_cores_ordered[worker_idx];

        uint32_t bank_id = worker_idx;  // Worker i reads from DRAM bank i (optimal assignment)

        // Calculate VC (virtual channel) to avoid conflicts
        uint32_t vc = bank_id & 0x3;
        bank_ids.push_back(bank_id);
        for (uint32_t j = 0; j < worker_idx; ++j) {
            auto core_prev = all_worker_cores_ordered[j];
            if (core_prev.y == core.y && ((bank_id & 0x3) == (bank_ids[j] & 0x3))) {
                vc = (vc + 1) & 0x3;
                break;
            }
        }

        log_debug(
            tt::LogOp,
            "Worker {} at core ({}, {}): in0 from storage ({}, {}), DRAM bank {}, out to storage ({}, {})",
            worker_idx,
            core.x,
            core.y,
            input_storage_cores_ordered[worker_idx].x,
            input_storage_cores_ordered[worker_idx].y,
            bank_id,
            output_storage_cores_ordered[worker_idx].x,
            output_storage_cores_ordered[worker_idx].y);

        // in0 reader runtime args
        std::vector<uint32_t> in0_reader_runtime_args = {
            1u,                               // worker_core_type (1 = active worker)
            input_storage_noc_x[worker_idx],  // input storage core NOC x
            input_storage_noc_y[worker_idx],  // input storage core NOC y
            in0_buffer->address(),            // L1 address of in0 shard on storage core
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_reader_id, core, in0_reader_runtime_args);
        reader_kernel_ids.push_back(mm_kernel_in0_reader_id);

        // in1 writer runtime args
        std::vector<uint32_t> in1_writer_runtime_args = {
            1u,  // is_worker_core
            in1_buffer->address(),
            bias_buffer != nullptr ? bias_buffer->address() : 0u,
            bank_id,
            vc,
            output_storage_noc_x[worker_idx],  // output storage core NOC x
            output_storage_noc_y[worker_idx],  // output storage core NOC y
            out_buffer->address(),             // L1 address of output shard on storage core
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_writer_id, core, in1_writer_runtime_args);
        writer_kernel_ids.push_back(mm_kernel_in1_writer_id);

        // Compute runtime args
        std::vector<uint32_t> compute_runtime_args = {
            1u,  // is_worker_core
        };
        tt_metal::SetRuntimeArgs(program, mm_kernel_compute_id, core, compute_runtime_args);
    }

    return {
        std::move(program),
        MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::shared_variables_t{
            reader_kernel_ids, writer_kernel_ids, all_worker_cores_ordered, cb_src2, cb_output_reshard}};
}

}  // namespace reuse_batched_hs_dram_sharded_optimized_helpers

std::pair<tt::tt_metal::Program, MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::shared_variables_t>
matmul_multi_core_reuse_batched_hs_dram_sharded_optimized_(
    const ttnn::MeshCoordinate& mesh_coord,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::prim::MatmulParams& operation_attributes) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;
    const auto& output_tensors = tensor_return_value;

    const auto& a = input_tensors.at(0);
    const auto& b = input_tensors.at(1);
    const auto& bias = optional_input_tensors.at(0);
    const auto& output = output_tensors.at(0);
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile.get_tile_shape()[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt_metal::Buffer* bias_buffer = nullptr;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(c.storage_type() == StorageType::DEVICE, "Bias tensor must be on device");
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");
        TT_FATAL(c.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");
        bias_buffer = c.buffer();
        bias_data_format = tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    tt::tt_metal::IDevice* device =
        reuse_batched_hs_dram_sharded_optimized_helpers::get_device_for_dram_banks(a, mesh_coord);

    TT_FATAL(
        a.shard_spec().has_value() && output.shard_spec().has_value(), "Both input A and output must have shard specs");
    CoreRangeSet input_all_cores_storage = a.shard_spec().value().grid;
    CoreRangeSet output_all_cores_storage = output.shard_spec().value().grid;

    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();
    tt_metal::Buffer* out_buffer = output.buffer();

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(tt_metal::datatype_to_dataformat_converter(a.dtype()));
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(tt_metal::datatype_to_dataformat_converter(b.dtype()));

    // Buffer size validation
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
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    // Shape compatibility checks
    TT_FATAL(
        ashape[-1] == bshape[-2],
        "Dimension K (A.shape[-1] = {}, B.shape[-2] = {}) must match for matmul",
        ashape[-1],
        bshape[-2]);
    TT_FATAL(
        ashape[-2] % in0_tile_shape[0] == 0,
        "A.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        ashape[-2],
        in0_tile_shape[0]);
    TT_FATAL(
        ashape[-1] % in0_tile_shape[1] == 0,
        "A.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        ashape[-1],
        in0_tile_shape[1]);
    TT_FATAL(
        bshape[-2] % in1_tile_shape[0] == 0,
        "B.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        bshape[-2],
        in1_tile_shape[0]);
    TT_FATAL(
        bshape[-1] % in1_tile_shape[1] == 0,
        "B.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        bshape[-1],
        in1_tile_shape[1]);

    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();
    const auto& program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>(
            operation_attributes.program_config.value());
    const auto& in0_block_w = program_config.in0_block_w;
    const auto& per_core_M = program_config.per_core_M;
    const auto& per_core_N = program_config.per_core_N;
    const auto& fused_activation = program_config.fused_activation;
    const auto& untilize_out = operation_attributes.untilize_out;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    // For batch sharding: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
    // ashape = [1, B, M, K], bshape = [1, B, K, N]
    uint32_t B = ashape[1];
    uint32_t M = ashape[-2] / in0_tile_shape[0];  // M in tiles
    uint32_t K = ashape[-1] / in0_tile_shape[1];  // K in tiles (contracted dimension)
    uint32_t N = bshape[-1] / in1_tile_shape[1];  // N in tiles

    // Batch sharding validation: per_core_M and per_core_N should equal M and N respectively
    // because each core handles complete matmuls, not partial tiles
    TT_FATAL(
        per_core_M == M,
        "For batch sharding, per_core_M ({}) must equal M ({}) - each core handles complete matmuls",
        per_core_M,
        M);
    TT_FATAL(
        per_core_N == N,
        "For batch sharding, per_core_N ({}) must equal N ({}) - each core handles complete matmuls",
        per_core_N,
        N);

    TT_FATAL(K % in0_block_w == 0, "K ({}) must be divisible by in0_block_w ({})", K, in0_block_w);

    return reuse_batched_hs_dram_sharded_optimized_helpers::create_program_batch_sharded(
        device,
        input_all_cores_storage,
        output_all_cores_storage,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        in0_block_w,
        per_core_M,
        per_core_N,
        fused_activation,
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
        false,   // skip_compute
        false);  // skip_write_back
}

MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::cached_mesh_workload_t
MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::create_mesh_workload(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto [program, shared_variable] = matmul_multi_core_reuse_batched_hs_dram_sharded_optimized_(
                mesh_coord, tensor_args, tensor_return_value, operation_attributes);
            shared_variables[single_coord_range] = shared_variable;
            workload.add_program(single_coord_range, std::move(program));
        }
    }
    return {std::move(workload), std::move(shared_variables)};
}

void MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ttnn::prim::MatmulParams& /*operation_attributes*/,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;
    const auto& output_tensors = tensor_return_value;

    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        auto* src_buffer_a = input_tensors.at(0).buffer();
        auto* src_buffer_b = input_tensors.at(1).buffer();
        const auto& bias_tensor = optional_input_tensors.at(0);
        auto* dst_buffer = output_tensors.at(0).buffer();
        auto shared_variables = cached_workload.shared_variables.at(mesh_coord_range);

        UpdateDynamicCircularBufferAddress(program, shared_variables.cb_src2, *src_buffer_a);
        UpdateDynamicCircularBufferAddress(program, shared_variables.cb_output_reshard, *dst_buffer);

        const auto& all_worker_cores_ordered = shared_variables.all_worker_cores_ordered;
        const auto& reader_kernel_ids = shared_variables.reader_kernel_ids;
        const auto& writer_kernel_ids = shared_variables.writer_kernel_ids;

        for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
            auto core = all_worker_cores_ordered[i];

            // Update reader kernel runtime args - arg[3] is the L1 address of in0 shard
            auto reader_kernel_id = reader_kernel_ids[i];
            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            reader_runtime_args[3] = src_buffer_a->address();

            // Update writer kernel runtime args
            auto writer_kernel_id = writer_kernel_ids[i];
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            writer_runtime_args[1] = src_buffer_b->address();
            if (bias_tensor.has_value()) {
                writer_runtime_args[2] = bias_tensor.value().mesh_buffer()->address();
            } else {
                writer_runtime_args[2] = 0;
            }
            writer_runtime_args[7] = dst_buffer->address();  // L1 address of output shard on storage core
        }
    }
}

}  // namespace ttnn::prim
