// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt::constants;
using namespace tt;

namespace reuse_optimized_helpers {

using tt::tt_metal::Tensor;

tt::tt_metal::ProgramDescriptor create_program(
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
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    const Tensor& in0,
    const Tensor& in1,
    const Tensor& output,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out) {
    tt_metal::ProgramDescriptor program;

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

    auto in0_tile = in0.get_tensor_spec().tile();
    auto in1_tile = in1.get_tensor_spec().tile();
    // currently only support transpose of the full tile
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    // cannot use the output tensor tile directly as that might be changed by user override
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile_shape[1]});
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

    uint32_t num_tiles_per_block_in0 = per_core_M_per_batch * K;
    uint32_t num_tiles_per_block_in1 = K * per_core_N;
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
    uint32_t g2_numcores = core_group_2.num_cores();
    // TODO: This contains same information as above; refactor this?
    uint32_t num_evenly_divided_output_blocks = num_output_blocks_total / num_cores;

    // Assume all of core_range is used (ie. num_evenly_divided_output_blocks > 0)
    TT_FATAL(num_evenly_divided_output_blocks > 0, "Not all cores from core_range was used!");
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    // Compile time args
    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;
    bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? true : false;

    tt_metal::KernelDescriptor::Defines mm_kernel_in0_reader_defines;
    tt_metal::KernelDescriptor::Defines mm_kernel_in1_reader_writer_defines;
    if (in0_is_sharded) {
        mm_kernel_in0_reader_defines.emplace_back("IN0_SHARDED", "1");
    }
    if (in1_is_sharded) {
        mm_kernel_in1_reader_writer_defines.emplace_back("IN1_SHARDED", "1");
    }
    if (output_is_sharded) {
        mm_kernel_in1_reader_writer_defines.emplace_back("OUT_SHARDED", "1");
    }

    constexpr auto max_num_kernels = 4;
    program.kernels.resize(max_num_kernels);
    size_t num_kernels = 0;

    auto& mm_kernel_in0_reader = program.kernels[num_kernels++];
    mm_kernel_in0_reader.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0.cpp";
    mm_kernel_in0_reader.core_ranges = all_cores.ranges();
    mm_kernel_in0_reader.compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)in0_is_dram,
    };
    mm_kernel_in0_reader.defines = std::move(mm_kernel_in0_reader_defines);
    mm_kernel_in0_reader.config = tt_metal::ReaderConfigDescriptor{};
    mm_kernel_in0_reader.reserve_runtime_args();

    auto& mm_kernel_in1_reader_writer = program.kernels[num_kernels++];
    mm_kernel_in1_reader_writer.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp";
    mm_kernel_in1_reader_writer.core_ranges = all_cores.ranges();
    mm_kernel_in1_reader_writer.compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)in1_is_dram,
        (std::uint32_t)out_is_dram,
    };
    mm_kernel_in1_reader_writer.defines = std::move(mm_kernel_in1_reader_writer_defines);
    mm_kernel_in1_reader_writer.config = tt_metal::WriterConfigDescriptor{};
    mm_kernel_in1_reader_writer.reserve_runtime_args();

    tt_metal::KernelDescriptor::CompileTimeArgs compute_kernel_args_group_1 = {
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

        untilize_out};

    tt_metal::KernelDescriptor::Defines mm_kernel_defines;
    if (packer_l1_acc_en) {
        mm_kernel_defines.emplace_back("PACKER_L1_ACC", "1");
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    if (in1_transpose_tile) {
        mm_kernel_defines.emplace_back("IN1_TRANSPOSE_TILE", "1");
    }

    bmm_op_utils::add_stagger_defines_if_needed(device->arch(), num_cores, mm_kernel_defines);

    // Create compute kernel
    auto& mm_kernel = program.kernels[num_kernels++];
    mm_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
    mm_kernel.core_ranges = core_group_1.ranges();
    mm_kernel.compile_time_args = compute_kernel_args_group_1;
    mm_kernel.defines = mm_kernel_defines;
    mm_kernel.config = tt_metal::ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
    };
    if (!core_group_2.ranges().empty()) {
        tt_metal::KernelDescriptor::CompileTimeArgs compute_kernel_args_group_2 = {
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

            untilize_out};
        auto& mm_kernel_group_2 = program.kernels[num_kernels++];
        mm_kernel_group_2.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
        mm_kernel_group_2.core_ranges = core_group_2.ranges();
        mm_kernel_group_2.compile_time_args = compute_kernel_args_group_2;
        mm_kernel_group_2.defines = mm_kernel_defines;
        mm_kernel_group_2.config = tt_metal::ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
        };
    }

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = in0_CB_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {tt_metal::CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = in0_tile,
        }},
        .buffer = in0_is_sharded ? in0_buffer : nullptr,
    });

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    program.cbs.push_back(tt_metal::CBDescriptor{
        .total_size = in1_CB_size,
        .core_ranges = all_cores.ranges(),
        .format_descriptors = {tt_metal::CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_single_tile_size,
            .tile = in1_tile,
        }},
        .buffer = in1_is_sharded ? in1_buffer : nullptr,
    });

    uint32_t output_cb_index = tt::CBIndex::c_4;
    uint32_t interm0_cb_index = tt::CBIndex::c_5;
    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = out_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors = {tt_metal::CBFormatDescriptor{
                .buffer_index = output_cb_index,
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = output_tile,
            }},
            .buffer = output_is_sharded ? out_buffer : nullptr,
        });

        // interm0
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = interm0_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors = {tt_metal::CBFormatDescriptor{
                .buffer_index = interm0_cb_index,
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = output_tile,
            }},
        });
    } else {
        // share buffer
        program.cbs.push_back(tt_metal::CBDescriptor{
            .total_size = out_CB_size,
            .core_ranges = all_cores.ranges(),
            .format_descriptors =
                {tt_metal::CBFormatDescriptor{
                     .buffer_index = output_cb_index,
                     .data_format = output_data_format,
                     .page_size = output_single_tile_size,
                     .tile = output_tile,
                 },
                 tt_metal::CBFormatDescriptor{
                     .buffer_index = interm0_cb_index,
                     .data_format = interm0_data_format,
                     .page_size = interm0_single_tile_size,
                     .tile = output_tile,
                 }},
            .buffer = output_is_sharded ? out_buffer : nullptr,
        });
    }
    // std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
    //     {output_cb_index, output_data_format},
    //     {interm0_cb_index, output_data_format}
    // };
    // tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(out_CB_size,
    // output_cb_data_format_spec) 	.set_page_size(output_cb_index, output_single_tile_size)
    //     .set_page_size(interm0_cb_index, output_single_tile_size);

    // Write runtime args to device
    tt_metal::KernelDescriptor::CoreRuntimeArgs mm_reader_args = {
        (std::uint32_t)num_blocks,  // num_blocks

        (std::uint32_t)0,            // batch placeholder
        (std::uint32_t)bcast_batch,  // bcast_B
        (std::uint32_t)M * K,        // MtKt

        (std::uint32_t)in0_buffer->address(),  // in0_tensor_addr
        (std::uint32_t)0,                      // in0_tensor_start_tile_id placeholder
        (std::uint32_t)1,                      // in0_tensor_stride_w
        (std::uint32_t)K,                      // in0_tensor_stride_h
        (std::uint32_t)in0_block_w,            // in0_tensor_next_block_stride

        (std::uint32_t)in0_block_w,           // in0_block_w
        (std::uint32_t)per_core_M_per_batch,  // in0_block_h
        (std::uint32_t)in0_block_num_tiles,   // in0_block_num_tiles
    };

    tt_metal::KernelDescriptor::CoreRuntimeArgs mm_writer_args = {
        (std::uint32_t)num_blocks,   // num_blocks
        (std::uint32_t)0,            // batch placeholder
        (std::uint32_t)bcast_batch,  // bcast_B
        (std::uint32_t)M * N,        // MtNt
        (std::uint32_t)K * N,        // KtNt

        (std::uint32_t)in1_buffer->address(),  // in1_tensor_addr
        (std::uint32_t)0,                      // in1_tensor_start_tile_id placeholder
        (std::uint32_t)1,                      // in1_tensor_stride_w
        (std::uint32_t)N,                      // in1_tensor_stride_h
        (std::uint32_t)in0_block_w * N,        // in1_tensor_next_block_stride

        (std::uint32_t)per_core_N,           // in1_block_w
        (std::uint32_t)in0_block_w,          // in1_block_h
        (std::uint32_t)in1_block_num_tiles,  // in1_block_num_tiles

        (std::uint32_t)out_buffer->address(),  // out_tensor_addr
        (std::uint32_t)0,                      // out_tensor_start_tile_id placeholder
        (std::uint32_t)1,                      // out_tensor_stride_w
        (std::uint32_t)N,                      // out_tensor_stride_h
        (std::uint32_t)out_subblock_w,         // out_tensor_next_subblock_stride_w
        (std::uint32_t)out_subblock_h * N,     // out_tensor_next_subblock_stride_h

        (std::uint32_t)out_subblock_w,                     // out_subblock_w
        (std::uint32_t)out_subblock_h,                     // out_subblock_h
        (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
        (std::uint32_t)out_num_subblocks_w,                // out_num_subblocks_w
        (std::uint32_t)out_num_subblocks_h,                // out_num_subblocks_h
    };
    bool row_major = false;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
    }
    const auto cores = grid_to_cores(num_cores, core_range.x, core_range.y, row_major);

    for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t core_idx_x = core.x;
        uint32_t core_idx_y = core.y;
        uint32_t num_output_blocks_per_core =
            i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Write runtime args to device
        mm_reader_args[1] = num_output_blocks_per_core;
        mm_reader_args[5] = num_blocks_written * num_tiles_per_block_in0;

        mm_writer_args[1] = num_output_blocks_per_core;
        mm_writer_args[6] =
            (num_blocks_written * per_core_M_per_batch / M) * num_tiles_per_block_in1;  // in1_tensor_start_tile_id
        mm_writer_args[14] = num_blocks_written * num_tiles_per_block_out;              // out_tensor_start_tile_id

        mm_kernel_in0_reader.runtime_args[core.x][core.y] = mm_reader_args;
        mm_kernel_in1_reader_writer.runtime_args[core.x][core.y] = mm_writer_args;

        num_blocks_written += num_output_blocks_per_core;
    }

    program.kernels.resize(num_kernels);
    return program;
}

}  // namespace reuse_optimized_helpers

namespace ttnn {

namespace operations {

namespace matmul {

tt::tt_metal::ProgramDescriptor matmul_multi_core_reuse_optimized_(
    const Tensor& a,
    const Tensor& b,
    Tensor& output,
    bool bcast_batch,
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
    const auto& ashape = a.get_padded_shape();
    const auto& bshape = b.get_padded_shape();
    auto in0_tile_shape = a.get_tensor_spec().tile().get_tile_shape();
    auto in1_tile_shape = b.get_tensor_spec().tile().get_tile_shape();

    TT_FATAL(
        (bcast_batch == false) or (ashape[0] == 1) or (ashape.rank() == 2),
        "Bcast batch not supported for this parallelization");

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());    // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());    // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output_dtype);  // output

    tt_metal::IDevice* device = a.device();

    tt_metal::Buffer* in0_buffer = a.buffer();
    tt_metal::Buffer* in1_buffer = b.buffer();

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
    uint32_t Mt = ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];

    // TODO: Generalize
    TT_FATAL(!fuse_batch, "Only fuse_batch=false is supported for optimized bmm!");

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Get large matmul params

    uint32_t num_blocks_total = (B * Mt / per_core_M) * (Nt / per_core_N);
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
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        a,
        b,
        output,
        in0_data_format,
        in1_data_format,
        output_data_format,
        untilize_out);
}

// TODO: Get rid of no-op reshapes when we generalize
// matmul_multi_core_reuse_optimized_bert_large not used
tt::tt_metal::ProgramDescriptor bmm_multi_core_reuse_optimized(
    const Tensor& a,
    const Tensor& b,
    Tensor& output,
    bool bcast_batch,
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
    /*
     * For pre-softmax and post-softmax bmm, do an additional no-op reshape by changing cshape and ashape
     * - pre-softmax: [9, 16, 384, 64] x [9, 16, 64, 384] = ([9, 16, 384, 384] -> [9, 1, 6144, 384])
     * - post-softmax: ([9, 1, 6144, 384] -> [9, 16, 384, 384]) x [9, 16, 384, 64] = [9, 16, 384, 64]
     * NOTE: Only need to pass in the right cshape and ashape for these no-op reshapes.
     * The actual bmm op works on [9, 16, 384, 64] x [9, 16, 64, 384] and [9, 16, 384, 384] x [9, 16, 384, 64].
     */
    return matmul_multi_core_reuse_optimized_(
        a,
        b,
        output,
        bcast_batch,
        compute_with_storage_grid_size,
        output_dtype,
        compute_kernel_config,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        fuse_batch,
        untilize_out);
}

}  // namespace matmul

}  // namespace operations

}  // namespace ttnn
