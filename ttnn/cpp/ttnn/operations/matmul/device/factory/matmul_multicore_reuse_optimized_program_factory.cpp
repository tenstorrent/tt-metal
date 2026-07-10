// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_optimized_program_factory.hpp"

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

#include <map>
#include <string>
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;
using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::ReaderConfigDescriptor;
using tt::tt_metal::Tensor;
using tt::tt_metal::WriterConfigDescriptor;

namespace ttnn::prim {

CoreRangeSet MatmulMultiCoreReuseOptimizedProgramFactory::default_core_range(IDevice* device) {
    auto grid_size = device->compute_with_storage_grid_size();
    return CoreRangeSet({CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1})});
}

tt::tt_metal::ProgramDescriptor MatmulMultiCoreReuseOptimizedProgramFactory::create_descriptor(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const std::optional<CoreRangeSet>& core_range_set) {
    TT_FATAL(operation_attributes.program_config.has_value(), "program_config must be provided for create_descriptor");
    const auto& program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseProgramConfig>(operation_attributes.program_config.value());

    TT_FATAL(operation_attributes.output_dtype.has_value(), "Output dtype should have been provided");
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config should have been provided");
    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Bcast batch should have been provided");

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& output = tensor_return_value.at(0).mesh_tensor();

    bool bcast_batch = operation_attributes.bcast_batch.value();
    bool transpose_a = operation_attributes.transpose_a;
    bool transpose_b = operation_attributes.transpose_b;
    bool untilize_out = operation_attributes.untilize_out;

    uint32_t in0_block_w = program_config.in0_block_w;
    uint32_t out_subblock_h = program_config.out_subblock_h;
    uint32_t out_subblock_w = program_config.out_subblock_w;
    uint32_t per_core_M = program_config.per_core_M;
    uint32_t per_core_N = program_config.per_core_N;

    const auto& ashape = operations::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& bshape = operations::matmul::utilities::get_matmul_tensor_padded_shape(b, transpose_b);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(a, transpose_a);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(b, transpose_b);

    const auto& in0_buffer = a.mesh_tensor();
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(in0_buffer.dtype());
    const auto& in1_buffer = b.mesh_tensor();
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(in1_buffer.dtype());
    tt::DataFormat output_data_format =
        tt_metal::datatype_to_dataformat_converter(operation_attributes.output_dtype.value());

    tt_metal::IDevice* device = &in0_buffer.mutable_device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config.value());

    if (fp32_dest_acc_en) {
        TT_FATAL(
            out_subblock_h * out_subblock_w <= 4,
            "Total number of tiles in a subblock must be less than 4 when in fp32_dest_acc mode");
    }

    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = operations::matmul::utilities::get_M_dim(ashape, in0_tile, false);
    uint32_t Kt = operations::matmul::utilities::get_K_dim(ashape, in0_tile);
    uint32_t Nt = operations::matmul::utilities::get_N_dim(bshape, in1_tile);
    uint32_t M = Mt;
    uint32_t N = Nt;
    uint32_t K = Kt;

    const auto ashape_logical = operations::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
    // When transpose_a is true, the K dimension maps to the row dimension of the raw tile,
    // which is already zero-padded during tile layout conversion. pad_last_ktile operates on
    // columns, so applying it would incorrectly zero valid data that becomes output rows
    // after the compute kernel transposes the tile.
    const auto in0_last_ktile_w = transpose_a ? 0 : ashape_logical[-1] % in0_tile.get_width();
    const auto in0_last_ktile_h = transpose_a ? ashape_logical[-1] % in0_tile.get_width() : 0;
    TT_FATAL(
        in0_last_ktile_w == 0 || in0_last_ktile_h == 0,
        "At most one of in0_last_ktile_w ({}) and in0_last_ktile_h ({}) can be non-zero",
        in0_last_ktile_w,
        in0_last_ktile_h);

    // Derived parameters
    uint32_t batch_scale_factor = per_core_M > M ? per_core_M / M : 1;
    uint32_t per_core_M_per_batch = per_core_M > M ? M : per_core_M;
    uint32_t num_blocks = (K / in0_block_w);
    bool packer_l1_acc_en = packer_l1_acc && (num_blocks > 2);

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    bool in0_transpose_tile = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();

    auto output_tile = tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});
    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    bool in0_is_sharded = in0_buffer.is_sharded();
    bool in1_is_sharded = in1_buffer.is_sharded();
    bool output_is_sharded = output.is_sharded();

    // Tiles whose size is not a multiple of the DRAM alignment (e.g. bfp8 32x16 = 544B on Blackhole's
    // 64B alignment) are padded to it in DRAM. The interleaved reader copies tiles at that padded
    // stride, so the in0/in1 CBs must hold pages at the aligned stride and the reader/unpacker walk
    // tiles at the same stride. This is a no-op when the tile is already aligned (all bf16 tiles,
    // 32-wide bfp8, and everything on Wormhole's 32B alignment) and replaces the staging-CB workaround.
    // Sharded CBs are backed by the tensor buffer and keep their natural page size.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size =
        in1_is_sharded ? in1_single_tile_size : tt::align(in1_single_tile_size, dram_alignment);

    // CB sizes
    uint32_t in0_block_num_tiles = per_core_M_per_batch * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_num_tiles;
    if (in0_is_sharded) {
        in0_CB_tiles = per_core_M * K;
    } else {
        in0_CB_tiles *= 2;
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_aligned_tile_size;
    uint32_t in1_block_num_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_num_tiles;
    if (in1_is_sharded) {
        in1_CB_tiles *= num_blocks * batch_scale_factor;
    } else {
        in1_CB_tiles *= 2;
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_aligned_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    // Compute kernel args
    uint32_t in0_num_subblocks = (per_core_M_per_batch / out_subblock_h);
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = (per_core_N / out_subblock_w);
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    uint32_t out_num_subblocks_h = per_core_M_per_batch / out_subblock_h;
    uint32_t out_num_subblocks_w = in1_num_subblocks;
    uint32_t num_output_blocks_total = (B * M / per_core_M) * (N / per_core_N);

    std::optional<tt::tt_metal::ShardSpec> shard_spec = std::nullopt;
    if (in0_is_sharded) {
        shard_spec = in0_buffer.shard_spec().value();
    } else if (in1_is_sharded) {
        shard_spec = in1_buffer.shard_spec().value();
    } else if (output_is_sharded) {
        shard_spec = output.shard_spec().value();
    }

    // Core splitting
    uint32_t num_cores = 0, num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores, core_group_1, core_group_2;

    if (shard_spec.has_value()) {
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = num_output_blocks_total / num_cores * batch_scale_factor;
    } else if (core_range_set.has_value()) {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(core_range_set.value(), num_output_blocks_total);
        num_blocks_per_core_group_1 *= batch_scale_factor;
        num_blocks_per_core_group_2 *= batch_scale_factor;
    } else {
        if (!program_config.allowed_worker_cores.has_value()) {
            log_warning(
                tt::LogOp,
                "MatmulMultiCoreReuseOptimizedProgramFactory: program_config.allowed_worker_cores not populated; "
                "falling back to compute_with_storage_grid_size. Callers that bypass ttnn::prim::matmul() should "
                "invoke ttnn::operations::matmul::normalize_program_config() on the program config first. This "
                "will become a hard error in a future release.");
        }
        // Use the CoreRangeSet overload so the output core ranges carry the actual
        // absolute coordinates (e.g. (4,0)-(7,0)) rather than always starting at (0,0).
        if (program_config.allowed_worker_cores.has_value()) {
            std::tie(
                num_cores,
                all_cores,
                core_group_1,
                core_group_2,
                num_blocks_per_core_group_1,
                num_blocks_per_core_group_2) =
                tt::tt_metal::split_work_to_cores(program_config.allowed_worker_cores.value(), num_output_blocks_total);
        } else {
            CoreCoord grid = program_config.compute_with_storage_grid_size;
            std::tie(
                num_cores,
                all_cores,
                core_group_1,
                core_group_2,
                num_blocks_per_core_group_1,
                num_blocks_per_core_group_2) = tt::tt_metal::split_work_to_cores(grid, num_output_blocks_total);
        }
        num_blocks_per_core_group_1 *= batch_scale_factor;
        num_blocks_per_core_group_2 *= batch_scale_factor;
    }
    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t num_evenly_divided_output_blocks = num_output_blocks_total / num_cores;
    TT_FATAL(num_evenly_divided_output_blocks > 0, "Not all cores from core_range was used!");

    const auto in0_tensor_stride_w = transpose_a ? M : 1;
    const auto in0_tensor_stride_h = transpose_a ? 1 : K;
    const auto in0_tensor_next_block_stride = in0_block_w * in0_tensor_stride_w;
    const auto in1_tensor_stride_w = transpose_b ? K : 1;
    const auto in1_tensor_stride_h = transpose_b ? 1 : N;
    const auto in1_tensor_next_block_stride = in0_block_w * in1_tensor_stride_h;

    // Compile time args for reader
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)in0_tensor_stride_w,
        (std::uint32_t)in0_tensor_stride_h,
        (std::uint32_t)in0_tensor_next_block_stride,
        (std::uint32_t)in0_block_w,
        (std::uint32_t)per_core_M_per_batch,
        (std::uint32_t)in0_block_num_tiles,
        (std::uint32_t)in0_last_ktile_w,
        (std::uint32_t)in0_last_ktile_h,
        (std::uint32_t)num_blocks,
        (std::uint32_t)bcast_batch,
        (std::uint32_t)M * K,
    };
    tt::tt_metal::TensorAccessorArgs(in0_buffer).append_to(reader_compile_time_args);

    // Compile time args for reader/writer
    std::vector<uint32_t> reader_writer_compile_time_args = {
        (std::uint32_t)in1_tensor_stride_w,
        (std::uint32_t)in1_tensor_stride_h,
        (std::uint32_t)in1_tensor_next_block_stride,
        (std::uint32_t)per_core_N,
        (std::uint32_t)in0_block_w,
        (std::uint32_t)in1_block_num_tiles,
        (std::uint32_t)num_blocks,
        (std::uint32_t)bcast_batch,
        (std::uint32_t)K * N,
        (std::uint32_t)1,
        (std::uint32_t)N,
        (std::uint32_t)out_subblock_w,
        (std::uint32_t)out_subblock_h * N,
        (std::uint32_t)out_subblock_w,
        (std::uint32_t)out_subblock_h,
        (std::uint32_t)(out_subblock_w * out_subblock_h),
        (std::uint32_t)out_num_subblocks_w,
        (std::uint32_t)out_num_subblocks_h,
        (std::uint32_t)M * N,
    };
    tt::tt_metal::TensorAccessorArgs(in1_buffer).append_to(reader_writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output).append_to(reader_writer_compile_time_args);

    // Reader defines
    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines reader_writer_defines;
    if (in0_is_sharded) {
        reader_defines.emplace_back("IN0_SHARDED", "1");
    }
    if (in1_is_sharded) {
        reader_writer_defines.emplace_back("IN1_SHARDED", "1");
    }
    if (output_is_sharded) {
        reader_writer_defines.emplace_back("OUT_SHARDED", "1");
    }

    // Named compile-time args for CB indices (enables fusion/chaining)
    KernelDescriptor::NamedCompileTimeArgs cb_named_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_intermed0", tt::CBIndex::c_5},
        {"cb_in0_transposed", tt::CBIndex::c_10},
    };

    // Compute kernel compile time args (group 1)
    std::vector<uint32_t> compute_kernel_args_group_1 = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        1,  // out_num_blocks_x
        1,  // out_num_blocks_y
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        num_blocks_per_core_group_1,
        out_block_tiles,
        untilize_out,
        false,  // get_batch_from_reader
        in0_transpose_tile,
    };

    // Compute defines
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
    const auto throttle_level = ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);
    KernelDescriptor::Defines compute_defines{mm_kernel_defines.begin(), mm_kernel_defines.end()};

    // Build per-core runtime args
    bool row_major = false;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
    }
    const auto cores = corerange_to_cores(all_cores, num_cores, row_major);

    uint32_t m_blocks_per_batch = M / per_core_M_per_batch;
    uint32_t n_blocks_per_batch = N / per_core_N;
    uint32_t blocks_per_batch = m_blocks_per_batch * n_blocks_per_batch;
    uint32_t in0_batch_stride = M * K;
    uint32_t in1_batch_stride = K * N;
    uint32_t in0_m_block_stride = per_core_M_per_batch * (transpose_a ? 1 : K);
    uint32_t in1_n_block_stride = per_core_N * (transpose_b ? K : 1);

    ////////////////////////////////////////////////////////////////////////////
    //                      Build ProgramDescriptor
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor program_descriptor;

    // Reader kernel descriptor (created before loop so emplace_runtime_args can be called in loop)
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = reader_compile_time_args;
    reader_kernel_desc.named_compile_time_args = cb_named_args;
    reader_kernel_desc.defines = reader_defines;
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    // Reader/Writer kernel descriptor (reads in1, writes output)
    KernelDescriptor reader_writer_kernel_desc;
    reader_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp";
    reader_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_writer_kernel_desc.core_ranges = all_cores;
    reader_writer_kernel_desc.compile_time_args = reader_writer_compile_time_args;
    reader_writer_kernel_desc.named_compile_time_args = cb_named_args;
    reader_writer_kernel_desc.defines = reader_writer_defines;
    reader_writer_kernel_desc.config = WriterConfigDescriptor{};

    KernelDescriptor::RuntimeArgs compute_runtime_args_g1;
    KernelDescriptor::RuntimeArgs compute_runtime_args_g2;
    compute_runtime_args_g1.reserve(g1_numcores);

    for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t num_output_blocks_per_core =
            i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        uint32_t start_batch = num_blocks_written / blocks_per_batch;
        uint32_t block_within_batch = num_blocks_written % blocks_per_batch;
        uint32_t start_m_block = block_within_batch / n_blocks_per_batch;
        uint32_t start_n_block = block_within_batch % n_blocks_per_batch;

        uint32_t in0_start_tile_id = (start_batch * in0_batch_stride) + (start_m_block * in0_m_block_stride);
        uint32_t in1_start_tile_id =
            (bcast_batch ? 0 : (start_batch * in1_batch_stride)) + (start_n_block * in1_n_block_stride);

        reader_kernel_desc.emplace_runtime_args(core, {in0_buffer, in0_start_tile_id, num_output_blocks_per_core});

        uint32_t out_start_tile_id =
            (start_batch * M * N) + (start_m_block * per_core_M_per_batch * N) + (start_n_block * per_core_N);
        reader_writer_kernel_desc.emplace_runtime_args(
            core, {in1_buffer, in1_start_tile_id, num_output_blocks_per_core, output, out_start_tile_id});

        // Compute kernels have no per-core runtime args
        if (i < g1_numcores) {
            compute_runtime_args_g1.emplace_back(core, std::vector<uint32_t>{});
        } else {
            compute_runtime_args_g2.emplace_back(core, std::vector<uint32_t>{});
        }

        num_blocks_written += num_output_blocks_per_core;
    }

    program_descriptor.kernels.push_back(std::move(reader_kernel_desc));
    program_descriptor.kernels.push_back(std::move(reader_writer_kernel_desc));

    // Compute kernel descriptor (core group 1)
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = core_group_1;
    compute_kernel_desc.compile_time_args = compute_kernel_args_group_1;
    compute_kernel_desc.named_compile_time_args = cb_named_args;
    compute_kernel_desc.defines = compute_defines;
    compute_kernel_desc.runtime_args = std::move(compute_runtime_args_g1);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};
    program_descriptor.kernels.push_back(std::move(compute_kernel_desc));

    // Core group 2 compute kernel (if needed)
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            in0_subblock_num_tiles,
            in1_num_subblocks,
            in1_block_num_tiles,
            in1_per_core_w,
            num_blocks,
            1,  // out_num_blocks_x
            1,  // out_num_blocks_y
            out_subblock_h,
            out_subblock_w,
            out_subblock_num_tiles,
            num_blocks_per_core_group_2,
            out_block_tiles,
            untilize_out,
            false,  // get_batch_from_reader
            in0_transpose_tile,
        };

        KernelDescriptor compute_kernel_desc_g2;
        compute_kernel_desc_g2.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
            "bmm_large_block_zm_fused_bias_activation.cpp";
        compute_kernel_desc_g2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_kernel_desc_g2.core_ranges = core_group_2;
        compute_kernel_desc_g2.compile_time_args = compute_kernel_args_group_2;
        compute_kernel_desc_g2.named_compile_time_args = cb_named_args;
        compute_kernel_desc_g2.defines = compute_defines;
        compute_kernel_desc_g2.runtime_args = std::move(compute_runtime_args_g2);
        compute_kernel_desc_g2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode};
        program_descriptor.kernels.push_back(std::move(compute_kernel_desc_g2));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    auto make_cb_descriptor = [&all_cores](
                                  uint32_t total_size,
                                  uint8_t buffer_index,
                                  tt::DataFormat data_format,
                                  uint32_t page_size,
                                  const tt::tt_metal::Tile& tile,
                                  const tt_metal::MeshTensor* tensor = nullptr) {
        CBDescriptor cb_desc;
        cb_desc.total_size = total_size;
        cb_desc.core_ranges = all_cores;
        tt::tt_metal::TileDescriptor tile_desc{tile};
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = buffer_index, .data_format = data_format, .page_size = page_size, .tile = tile_desc});
        cb_desc.tensor = tensor;
        return cb_desc;
    };

    // CB 0: Input A
    program_descriptor.cbs.push_back(make_cb_descriptor(
        in0_CB_size,
        tt::CBIndex::c_0,
        in0_data_format,
        in0_aligned_tile_size,
        in0_tile,
        in0_is_sharded ? &in0_buffer : nullptr));

    // CB 1: Input B
    program_descriptor.cbs.push_back(make_cb_descriptor(
        in1_CB_size,
        tt::CBIndex::c_1,
        in1_data_format,
        in1_aligned_tile_size,
        in1_tile,
        in1_is_sharded ? &in1_buffer : nullptr));

    // CB 4 and CB 5: Output and intermediate accumulator
    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        // Separate output and intermediate CBs
        program_descriptor.cbs.push_back(make_cb_descriptor(

            out_CB_size,
            tt::CBIndex::c_4,
            output_data_format,
            output_single_tile_size,
            output_tile,
            output_is_sharded ? &output : nullptr));
        program_descriptor.cbs.push_back(make_cb_descriptor(
            interm0_CB_size, tt::CBIndex::c_5, interm0_data_format, interm0_single_tile_size, output_tile));
    } else {
        // Shared output+intermediate CB
        CBDescriptor output_cb_desc;
        output_cb_desc.total_size = out_CB_size;
        output_cb_desc.core_ranges = all_cores;
        tt::tt_metal::TileDescriptor output_tile_desc{output_tile};
        output_cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_4,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        output_cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_5,
            .data_format = interm0_data_format,
            .page_size = interm0_single_tile_size,
            .tile = output_tile_desc});
        output_cb_desc.tensor = output_is_sharded ? &output : nullptr;
        program_descriptor.cbs.push_back(std::move(output_cb_desc));
    }

    // Optional transpose CB
    if (in0_transpose_tile) {
        program_descriptor.cbs.push_back(
            make_cb_descriptor(in0_CB_size, tt::CBIndex::c_10, in0_data_format, in0_aligned_tile_size, in0_tile));
    }

    return program_descriptor;
}

}  // namespace ttnn::prim
