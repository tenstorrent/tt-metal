// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_optimized_program_factory.hpp"

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;
using tt::tt_metal::Tensor;

namespace ttnn::prim {
namespace reuse_optimized_helpers {

MatmulMultiCoreReuseOptimizedProgramFactory::cached_program_t create_program(
    tt_metal::IDevice* device,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    CoreCoord core_range,
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
    uint32_t per_core_M,
    uint32_t per_core_N,
    const Tensor& in0,
    const Tensor& in1,
    const Tensor& output,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out) {
    tt_metal::Program program{};

    // TODO: We can generalize this into some special form of fuse batch, where we have B /= batch_scale_factor and M *=
    // batch_scale_factor
    uint32_t batch_scale_factor = per_core_M > M ? per_core_M / M : 1;
    uint32_t per_core_M_per_batch = per_core_M > M ? M : per_core_M;

    uint32_t num_blocks = (K / in0_block_w);

    // Only enable packer l1 accumulation when there are num_blocks > 2, otherwise
    // unnecessary overhead for reconfigs are added. Last iteration of l1 accumulation
    // does a spill and reload, so need more than 2 blocks to use l1 acc for packer
    bool packer_l1_acc_en = packer_l1_acc && (num_blocks > 2);

    // if fp32 enabled then we pack fp32 in l1, if not, then we pack fp16 in l1
    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    // currently only support transpose of the full tile
    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    // cannot use the output tensor tile directly as that might be changed by user override
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});
    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    tt_metal::Buffer* in0_buffer = in0.buffer();
    tt_metal::Buffer* in1_buffer = in1.buffer();
    tt_metal::Buffer* out_buffer = output.buffer();
    bool in0_is_sharded = in0.is_sharded();
    bool in1_is_sharded = in1.is_sharded();
    bool output_is_sharded = output.is_sharded();

    uint32_t in0_block_num_tiles = per_core_M_per_batch * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_num_tiles;
    if (in0_is_sharded) {
        in0_CB_tiles = per_core_M * K;
    } else {
        in0_CB_tiles *= 2;  // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_num_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_num_tiles;
    if (in1_is_sharded) {
        in1_CB_tiles *= num_blocks * batch_scale_factor;
    } else {
        in1_CB_tiles *= 2;  // double buffer
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    // Compute kernel compile time args
    uint32_t in0_num_subblocks = (per_core_M_per_batch / out_subblock_h);
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    uint32_t out_num_subblocks_h = per_core_M_per_batch / out_subblock_h;
    uint32_t out_num_subblocks_w = in1_num_subblocks;

    uint32_t num_tiles_per_block_out = per_core_M_per_batch * per_core_N;
    uint32_t num_output_blocks_total = (B * M / per_core_M) * (N / per_core_N);
    std::optional<tt::tt_metal::ShardSpec> shard_spec = std::nullopt;
    if (in0_is_sharded) {
        shard_spec = in0.shard_spec().value();
    } else if (in1_is_sharded) {
        shard_spec = in1.shard_spec().value();
    } else if (output_is_sharded) {
        shard_spec = output.shard_spec().value();
    }

    uint32_t num_cores = 0, num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores, core_group_1, core_group_2;

    if (shard_spec.has_value()) {
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = num_output_blocks_total / num_cores * batch_scale_factor;
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) = tt::tt_metal::split_work_to_cores(core_range, num_output_blocks_total);
        num_blocks_per_core_group_1 *= batch_scale_factor;
        num_blocks_per_core_group_2 *= batch_scale_factor;
    }
    uint32_t g1_numcores = core_group_1.num_cores();
    // TODO: This contains same information as above; refactor this?
    uint32_t num_evenly_divided_output_blocks = num_output_blocks_total / num_cores;

    // Assume all of core_range is used (ie. num_evenly_divided_output_blocks > 0)
    TT_FATAL(num_evenly_divided_output_blocks > 0, "Not all cores from core_range was used!");

    const auto in0_tensor_stride_w = transpose_a ? M : 1;
    const auto in0_tensor_stride_h = transpose_a ? 1 : K;
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;

    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;

    // Compile time args
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)in0_tensor_stride_w,
        (std::uint32_t)in0_tensor_stride_h,
        (std::uint32_t)in0_tensor_next_block_stride,
        (std::uint32_t)in0_block_w,
        (std::uint32_t)per_core_M_per_batch,  // in0_block_h
        (std::uint32_t)in0_block_num_tiles,
        (std::uint32_t)in0_last_ktile_w,
        (std::uint32_t)num_blocks,
        (std::uint32_t)bcast_batch,
        (std::uint32_t)M * K,  // MtKt
    };
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> reader_writer_compile_time_args = {
        (std::uint32_t)in1_tensor_stride_w,
        (std::uint32_t)in1_tensor_stride_h,
        (std::uint32_t)in1_tensor_next_block_stride,
        (std::uint32_t)per_core_N,           // in1_block_w
        (std::uint32_t)in0_block_w,          // in1_block_h
        (std::uint32_t)in1_block_num_tiles,  // in1_block_num_tiles
        (std::uint32_t)num_blocks,
        (std::uint32_t)bcast_batch,
        (std::uint32_t)K * N,  // KtNt

        (std::uint32_t)1,                                  // out_tensor_stride_w
        (std::uint32_t)N,                                  // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,                     // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * N,                 // out_tensor_next_subblock_stride_h
        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblock_tile_count
        (std::uint32_t)out_num_subblocks_w,                // out_num_subblocks_w
        (std::uint32_t)out_num_subblocks_h,                // out_num_subblocks_h
        (std::uint32_t)M * N,                              // MtNt
    };
    tt::tt_metal::TensorAccessorArgs(*in1_buffer).append_to(reader_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(reader_writer_compile_time_args);
    std::map<std::string, std::string> mm_kernel_in0_reader_defines;
    std::map<std::string, std::string> mm_kernel_in1_reader_writer_defines;
    if (in0_is_sharded) {
        mm_kernel_in0_reader_defines["IN0_SHARDED"] = "1";
    }
    if (in1_is_sharded) {
        mm_kernel_in1_reader_writer_defines["IN1_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        mm_kernel_in1_reader_writer_defines["OUT_SHARDED"] = "1";
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
            mm_kernel_in0_reader_defines["INTERMEDIATE_CB_READ"] = "1";
        }
        in1_needs_intermediate_cb_read = ((in1_single_tile_size % 64) != 0);
        if (in1_needs_intermediate_cb_read) {
            mm_kernel_in1_reader_writer_defines["INTERMEDIATE_CB_READ"] = "1";
        }
    }

    tt::tt_metal::KernelHandle mm_kernel_in0_reader_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, mm_kernel_in0_reader_defines));

    tt::tt_metal::KernelHandle mm_kernel_in1_reader_writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(reader_writer_compile_time_args, mm_kernel_in1_reader_writer_defines));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,  // num_blocks
        1,           // out_num_blocks_x
        1,           // out_num_blocks_y

        out_subblock_h,               // out_subblock_h
        out_subblock_w,               // out_subblock_w
        out_subblock_num_tiles,       // out_subblock_num_tiles
        num_blocks_per_core_group_1,  // batch
        out_block_tiles,

        untilize_out,  // untilize_out
        false,         // get_batch_from_reader
        in0_transpose_tile,
    };

    std::map<std::string, std::string> mm_kernel_defines;
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
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    // Create compute kernel
    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1,
            .defines = mm_kernel_defines});
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            in0_block_w,             // in0_block_w
            in0_num_subblocks,       // in0_num_subblocks
            in0_block_num_tiles,     // in0_block_num_tiles
            in0_subblock_num_tiles,  // in0_subblock_num_tiles

            in1_num_subblocks,    // in1_num_subblocks
            in1_block_num_tiles,  // in1_block_num_tiles
            in1_per_core_w,       // in1_per_core_w

            num_blocks,  // num_blocks
            1,           // out_num_blocks_x
            1,           // out_num_blocks_y

            out_subblock_h,               // out_subblock_h
            out_subblock_w,               // out_subblock_w
            out_subblock_num_tiles,       // out_subblock_num_tiles
            num_blocks_per_core_group_2,  // batch
            out_block_tiles,

            untilize_out,  // untilize_out
            false,         // get_batch_from_reader
            in0_transpose_tile,
        };
        tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2,
                .defines = mm_kernel_defines});
    }

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile);
    if (in0_is_sharded) {
        cb_src0_config = cb_src0_config.set_globally_allocated_address(*in0_buffer);
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
            .set_page_size(src1_cb_index, in1_single_tile_size)
            .set_tile_dims(src1_cb_index, in1_tile);
    if (in1_is_sharded) {
        cb_src1_config = cb_src1_config.set_globally_allocated_address(*in1_buffer);
    }
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

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

        tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), interm0_cb_config);
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
    // std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
    //     {output_cb_index, output_data_format},
    //     {interm0_cb_index, output_data_format}
    // };
    // tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(out_CB_size,
    // output_cb_data_format_spec) 	.set_page_size(output_cb_index, output_single_tile_size)
    //     .set_page_size(interm0_cb_index, output_single_tile_size);
    if (output_is_sharded) {
        output_cb_config = output_cb_config.set_globally_allocated_address(*out_buffer);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), output_cb_config);

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
        const uint32_t in0_transpose_cb_index = tt::CBIndex::c_10;
        auto in0_transpose_cb_config =
            tt_metal::CircularBufferConfig(in0_CB_size, {{in0_transpose_cb_index, in0_data_format}})
                .set_page_size(in0_transpose_cb_index, in0_single_tile_size)
                .set_tile_dims(in0_transpose_cb_index, in0_tile);
        tt_metal::CreateCircularBuffer(program, all_cores, in0_transpose_cb_config);
    }

    // Write runtime args to device
    std::vector<uint32_t> mm_reader_args = {
        (std::uint32_t)in0_buffer->address(),  // in0_tensor_addr
        (std::uint32_t)0,                      // in0_tensor_start_tile_id placeholder
        (std::uint32_t)0,                      // batch placeholder
    };

    std::vector<uint32_t> mm_writer_args = {
        (std::uint32_t)in1_buffer->address(),  // in1_tensor_addr
        (std::uint32_t)0,                      // in1_tensor_start_tile_id placeholder
        (std::uint32_t)0,                      // batch placeholder
        (std::uint32_t)out_buffer->address(),  // out_tensor_addr
        (std::uint32_t)0,                      // out_tensor_start_tile_id placeholder
    };
    bool row_major = false;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
    }
    const auto cores = grid_to_cores(num_cores, core_range.x, core_range.y, row_major);

    // Work can be split in two ways:
    // (a) by batch: each core processes all M,N dimensions of b number of batches
    // (b) by M dimension: each core processes a subset of M dimension within batches
    // In the general case, some cores may have work split by batch while others have work
    // split over the M dimension. We compute each core's start tile based on its global
    // position in the work distribution.

    // Compute the number of M and N blocks per batch
    uint32_t m_blocks_per_batch = M / per_core_M_per_batch;
    uint32_t n_blocks_per_batch = N / per_core_N;
    uint32_t blocks_per_batch = m_blocks_per_batch * n_blocks_per_batch;

    // Strides for computing start tile IDs
    uint32_t in0_batch_stride = M * K;
    uint32_t in1_batch_stride = K * N;
    uint32_t in0_m_block_stride = per_core_M_per_batch * (transpose_a ? 1 : K);
    uint32_t in1_n_block_stride = per_core_N * (transpose_b ? K : 1);

    for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_output_blocks_per_core =
            i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Compute starting batch and position within batch based on global block position
        uint32_t start_batch = num_blocks_written / blocks_per_batch;
        uint32_t block_within_batch = num_blocks_written % blocks_per_batch;
        uint32_t start_m_block = block_within_batch / n_blocks_per_batch;
        uint32_t start_n_block = block_within_batch % n_blocks_per_batch;

        // Compute start tile IDs based on batch and block position
        uint32_t in0_start_tile_id = (start_batch * in0_batch_stride) + (start_m_block * in0_m_block_stride);
        uint32_t in1_start_tile_id =
            (bcast_batch ? 0 : (start_batch * in1_batch_stride)) + (start_n_block * in1_n_block_stride);

        // Write runtime args to device
        mm_reader_args[1] = in0_start_tile_id;           // in0_tensor_start_tile_id
        mm_reader_args[2] = num_output_blocks_per_core;  // batch

        mm_writer_args[1] = in1_start_tile_id;                             // in1_tensor_start_tile_id
        mm_writer_args[2] = num_output_blocks_per_core;                    // batch
        mm_writer_args[4] = num_blocks_written * num_tiles_per_block_out;  // out_tensor_start_tile_id

        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_reader_id, core, mm_reader_args);
        tt_metal::SetRuntimeArgs(program, mm_kernel_in1_reader_writer_id, core, mm_writer_args);

        num_blocks_written += num_output_blocks_per_core;
    }

    return {
        std::move(program),
        {mm_kernel_in0_reader_id, mm_kernel_in1_reader_writer_id, cb_src0, cb_src1, cb_output, num_cores, cores}};
}

}  // namespace reuse_optimized_helpers

MatmulMultiCoreReuseOptimizedProgramFactory::cached_program_t matmul_multi_core_reuse_optimized_(
    const Tensor& a,
    const Tensor& b,
    Tensor& output,
    bool bcast_batch,
    bool transpose_a,
    bool transpose_b,
    CoreCoord compute_with_storage_grid_size,
    tt::tt_metal::DataType output_dtype,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    bool fuse_batch,
    bool untilize_out) {
    const auto& ashape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& bshape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, transpose_b);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(a, transpose_a);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(b, transpose_b);

    TT_FATAL(
        (bcast_batch == false) or (ashape[0] == 1) or (ashape.rank() == 2),
        "Bcast batch not supported for this parallelization");

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());        // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());        // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output_dtype);  // output

    tt_metal::IDevice* device = a.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    if (fp32_dest_acc_en) {
        TT_FATAL(
            out_subblock_h * out_subblock_w <= 4,
            "Total number of tiles in a subblock must be less than 4 when in fp32_dest_acc mode");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = operations::matmul::utilities::get_M_dim(ashape, in0_tile, fuse_batch);
    uint32_t Kt = operations::matmul::utilities::get_K_dim(ashape, in0_tile);
    uint32_t Nt = operations::matmul::utilities::get_N_dim(bshape, in1_tile);

    const auto ashape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
    const auto in0_last_ktile_w = ashape_logical[-1] % in0_tile.get_width();

    // TODO: Generalize
    TT_FATAL(!fuse_batch, "Only fuse_batch=false is supported for optimized bmm!");

    // Get large matmul params

    CoreCoord core_range = compute_with_storage_grid_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // Pass in cshape instead
    tt_metal::Buffer* out_buffer = output.buffer();
    TT_FATAL(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    return reuse_optimized_helpers::create_program(
        device,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        core_range,
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
        per_core_M,
        per_core_N,
        a,
        b,
        output,
        in0_tile,
        in1_tile,
        in0_data_format,
        in1_data_format,
        output_data_format,
        untilize_out);
}

// TODO: Get rid of no-op reshapes when we generalize
// matmul_multi_core_reuse_optimized_bert_large not used
MatmulMultiCoreReuseOptimizedProgramFactory::cached_program_t MatmulMultiCoreReuseOptimizedProgramFactory::create(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    /*
     * For pre-softmax and post-softmax bmm, do an additional no-op reshape by changing cshape and ashape
     * - pre-softmax: [9, 16, 384, 64] x [9, 16, 64, 384] = ([9, 16, 384, 384] -> [9, 1, 6144, 384])
     * - post-softmax: ([9, 1, 6144, 384] -> [9, 16, 384, 384]) x [9, 16, 384, 64] = [9, 16, 384, 64]
     * NOTE: Only need to pass in the right cshape and ashape for these no-op reshapes.
     * The actual bmm op works on [9, 16, 384, 64] x [9, 16, 64, 384] and [9, 16, 384, 384] x [9, 16, 384, 64].
     */

    const auto& program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseProgramConfig>(operation_attributes.program_config.value());

    TT_FATAL(operation_attributes.output_dtype.has_value(), "Output dtype should have been provided");
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config should have been provided");
    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Bcast batch should have been provided");

    return matmul_multi_core_reuse_optimized_(
        tensor_args.input_tensors.at(0),
        tensor_args.input_tensors.at(1),
        tensor_return_value.at(0),
        operation_attributes.bcast_batch.value(),
        operation_attributes.transpose_a,
        operation_attributes.transpose_b,
        program_config.compute_with_storage_grid_size,
        operation_attributes.output_dtype.value(),
        operation_attributes.compute_kernel_config.value(),
        program_config.in0_block_w,
        program_config.out_subblock_h,
        program_config.out_subblock_w,
        program_config.per_core_M,
        program_config.per_core_N,
        false,
        operation_attributes.untilize_out);
}

void MatmulMultiCoreReuseOptimizedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ttnn::prim::MatmulParams& /*operation_attributes*/,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;
    auto mm_kernel_in0_reader_id = shared_variables.mm_kernel_in0_reader_id;
    auto mm_kernel_in1_reader_writer_id = shared_variables.mm_kernel_in1_reader_writer_id;
    auto cb_src0 = shared_variables.cb_src0;
    auto cb_src1 = shared_variables.cb_src1;
    auto cb_output = shared_variables.cb_output;
    auto cores = shared_variables.cores;

    const auto& input_tensors = tensor_args.input_tensors;
    const auto& output_tensors = tensor_return_value;

    auto* src_buffer_a = input_tensors.at(0).buffer();
    auto* src_buffer_b = input_tensors.at(1).buffer();

    auto* dst_buffer = output_tensors.at(0).buffer();

    const bool src0_sharded = input_tensors[0].memory_config().is_sharded();
    const bool src1_sharded = input_tensors[1].memory_config().is_sharded();
    const bool out_sharded = output_tensors[0].memory_config().is_sharded();

    const bool update_reader_args = !src0_sharded;

    const bool update_writer_args = !(src1_sharded and out_sharded);

    if (update_reader_args || update_writer_args) {
        auto& reader_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in0_reader_id);

        auto& writer_runtime_args_by_core = GetRuntimeArgs(program, mm_kernel_in1_reader_writer_id);

        for (const auto& core : cores) {
            if (update_reader_args) {
                auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer_a->address();  // in0_tensor_addr
            }

            if (update_writer_args) {
                auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
                runtime_args[0] = src_buffer_b->address();  // in1_tensor_addr
                runtime_args[3] = dst_buffer->address();    // out_tensor_addr
            }
        }
    }
    if (src0_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
    }

    if (src1_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_src1, *src_buffer_b);
    }

    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    }
}

MatmulMeshWorkloadMultiCoreReuseOptimizedProgramFactory::cached_mesh_workload_t
MatmulMeshWorkloadMultiCoreReuseOptimizedProgramFactory::create_mesh_workload(
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
                MatmulMultiCoreReuseOptimizedProgramFactory::create(attributes, tensor_args, tensor_return_value);
            shared_variables[mesh_coord_range] = single_device_program.shared_variables;
            workload.add_program(mesh_coord_range, std::move(single_device_program.program));
        }
    }
    return {std::move(workload), std::move(shared_variables)};
}

void MatmulMeshWorkloadMultiCoreReuseOptimizedProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ttnn::prim::MatmulParams& attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    for (auto& [mesh_coord_range, program] : cached_workload.workload.get_programs()) {
        auto cached_program_proxy = MatmulMultiCoreReuseOptimizedProgramFactory::cached_program_t::proxy(
            program, cached_workload.shared_variables.at(mesh_coord_range));
        MatmulMultiCoreReuseOptimizedProgramFactory::override_runtime_arguments(
            cached_program_proxy, attributes, tensor_args, tensor_return_value);
    }
}

}  // namespace ttnn::prim
