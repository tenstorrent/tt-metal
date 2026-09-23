// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_optimized_program_factory.hpp"

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

#include <map>
#include <string>
#include <utility>
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"

using namespace tt;

using tt::tt_metal::KernelBuildOptLevel;
using tt::tt_metal::UnpackMode;
using tt::tt_metal::experimental::AddRuntimeArgsForNode;
using tt::tt_metal::experimental::DataflowBufferSpec;
using tt::tt_metal::experimental::DFBBinding;
using tt::tt_metal::experimental::DFBEndpointType;
using tt::tt_metal::experimental::DFBSpecName;
using tt::tt_metal::experimental::Group;
using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpec;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::ProgramRunArgs;
using tt::tt_metal::experimental::ProgramSpec;
using tt::tt_metal::experimental::TensorBinding;
using tt::tt_metal::experimental::TensorParameter;
using tt::tt_metal::experimental::TensorParamName;
using tt::tt_metal::experimental::unpack_modes;
using tt::tt_metal::experimental::WorkUnitSpec;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts MatmulMultiCoreReuseOptimizedProgramFactory::create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    TT_FATAL(
        operation_attributes.program_config.has_value(),
        "program_config must be provided for create_program_artifacts");
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
    // stride, so the in0/in1 DFBs must hold entries at the aligned stride and the reader/unpacker walk
    // tiles at the same stride. This is a no-op when the tile is already aligned (all bf16 tiles,
    // 32-wide bfp8, and everything on Wormhole's 32B alignment) and replaces the staging-DFB workaround.
    // Borrowed DFBs are backed by the tensor buffer and keep their natural entry size.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size =
        in1_is_sharded ? in1_single_tile_size : tt::align(in1_single_tile_size, dram_alignment);

    // DFB sizes
    uint32_t in0_block_num_tiles = per_core_M_per_batch * in0_block_w;
    uint32_t in0_DFB_tiles = in0_block_num_tiles;
    if (in0_is_sharded) {
        in0_DFB_tiles = per_core_M * K;
    } else {
        in0_DFB_tiles *= 2;
    }
    uint32_t in1_block_num_tiles = per_core_N * in0_block_w;
    uint32_t in1_DFB_tiles = in1_block_num_tiles;
    if (in1_is_sharded) {
        in1_DFB_tiles *= num_blocks * batch_scale_factor;
    } else {
        in1_DFB_tiles *= 2;
    }
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_DFB_tiles = out_block_tiles;

    // Optional fused full-tile bias. The whole per-batch [M, N] bias block
    // is loaded once and reused across the core's batch iterations.
    const auto bias = ttnn::as_optional_mesh_tensor(tensor_args.optional_input_tensors.at(0));
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    tt::tt_metal::Tile bias_tile = output_tile;
    uint32_t bias_single_tile_size = 0;
    if (bias.has_value()) {
        // Defence in depth: guarantees the whole [M, N] bias fits one block so the load-once and
        // reuse-over-batch scheme is correct. Unreachable via ttnn.linear: get_post_process_bias
        // only routes a bias here when this precondition already holds.
        TT_FATAL(
            N == per_core_N && per_core_M_per_batch == M,
            "Fused bias in matmul_multicore_reuse requires each batch element's matrix to be a single "
            "block: N ({}) == per_core_N ({}) and M ({}) == per_core_M_per_batch ({}).",
            N,
            per_core_N,
            M,
            per_core_M_per_batch);
        bias_data_format = tt_metal::datatype_to_dataformat_converter(bias->dtype());
        bias_tile = bias->tensor_spec().tile();
        bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    }
    // Full [M, N] per-batch bias block
    uint32_t in3_block_tiles = per_core_M_per_batch * per_core_N;

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

    ////////////////////////////////////////////////////////////////////////////
    //                      Spec resource names
    ////////////////////////////////////////////////////////////////////////////
    const KernelSpecName READER{"reader"};
    const KernelSpecName READER_WRITER{"reader_writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName BIAS_DFB{"bias"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName INTERMED0_DFB{"intermed0"};
    const DFBSpecName IN0_TRANSPOSED_DFB{"in0_transposed"};

    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName BIAS{"bias"};
    const TensorParamName OUTPUT{"output"};

    ////////////////////////////////////////////////////////////////////////////
    //                      Build DataflowBufferSpecs
    ////////////////////////////////////////////////////////////////////////////
    DataflowBufferSpec in0_dfb_spec{
        .unique_id = IN0_DFB,
        .entry_size = in0_aligned_tile_size,
        .num_entries = in0_DFB_tiles,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
    };
    if (in0_is_sharded) {
        in0_dfb_spec.borrowed_from = IN0;
    }

    DataflowBufferSpec in1_dfb_spec{
        .unique_id = IN1_DFB,
        .entry_size = in1_aligned_tile_size,
        .num_entries = in1_DFB_tiles,
        .data_format_metadata = in1_data_format,
        .tile_format_metadata = in1_tile,
    };
    if (in1_is_sharded) {
        in1_dfb_spec.borrowed_from = IN1;
    }

    // Output and intermediate accumulator. Whenever their formats match — and the untilize path
    // isn't splitting the output across several W subblocks — the two occupy one L1 region, which
    // is expressed as two mutually-aliased buffers of identical total size. Otherwise they are
    // independent buffers with their own allocations.
    const bool share_out_interm_buffer =
        !((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1)));

    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = out_DFB_tiles,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
    };
    DataflowBufferSpec intermed0_dfb_spec{
        .unique_id = INTERMED0_DFB,
        .entry_size = interm0_single_tile_size,
        .num_entries = out_DFB_tiles,
        .data_format_metadata = interm0_data_format,
        .tile_format_metadata = output_tile,
    };
    if (output_is_sharded) {
        out_dfb_spec.borrowed_from = OUTPUT;
        if (share_out_interm_buffer) {
            intermed0_dfb_spec.borrowed_from = OUTPUT;
        }
    }
    if (share_out_interm_buffer) {
        out_dfb_spec.advanced_options.alias_with = {INTERMED0_DFB};
        intermed0_dfb_spec.advanced_options.alias_with = {OUT_DFB};
    }

    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.reserve(4 + (bias.has_value() ? 1 : 0) + (in0_transpose_tile ? 1 : 0));
    dataflow_buffers.push_back(std::move(in0_dfb_spec));
    dataflow_buffers.push_back(std::move(in1_dfb_spec));
    dataflow_buffers.push_back(std::move(out_dfb_spec));
    dataflow_buffers.push_back(std::move(intermed0_dfb_spec));
    if (bias.has_value()) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = BIAS_DFB,
            .entry_size = bias_single_tile_size,
            .num_entries = in3_block_tiles,
            .data_format_metadata = bias_data_format,
            .tile_format_metadata = bias_tile,
        });
    }
    if (in0_transpose_tile) {
        // Transpose target: compute reads in0, transposes each tile into this buffer, and matmuls
        // out of it.
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = IN0_TRANSPOSED_DFB,
            .entry_size = in0_aligned_tile_size,
            .num_entries = in0_DFB_tiles,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernel defines
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> reader_writer_defines;
    if (in0_is_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    }
    if (in1_is_sharded) {
        reader_writer_defines["IN1_SHARDED"] = "1";
    }
    if (output_is_sharded) {
        reader_writer_defines["OUT_SHARDED"] = "1";
    }
    if (bias.has_value()) {
        reader_writer_defines["FUSE_BIAS"] = "1";
    }

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
    if (in0_transpose_tile) {
        // The shared compute kernel selects its in0 buffer at the preprocessor stage, because
        // dfb::in0_transposed only exists when the transpose is wanted and a ternary over the two
        // buffer tokens would name-look-up both regardless of the condition.
        mm_kernel_defines["IN0_TRANSPOSE_TILE"] = "1";
    }
    if (bias.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        // This factory loads the whole per-batch [M, N] bias block, so the compute indexes the bias
        // by (M-tile-row, N-tile). Other callers of this shared kernel load a single bias row.
        mm_kernel_defines["BIAS_FULL_BLOCK"] = "1";
    }
    const auto throttle_level = ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);

    ////////////////////////////////////////////////////////////////////////////
    //                      Build KernelSpecs
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
        .compiler_options =
            {
                .defines = KernelSpec::CompilerOptions::Defines(reader_defines),
            },
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN0_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = IN0,
                    .accessor_name = "in0",
                },
            },
        .compile_time_args =
            {
                {"in0_tensor_stride_w", static_cast<uint32_t>(in0_tensor_stride_w)},
                {"in0_tensor_stride_h", static_cast<uint32_t>(in0_tensor_stride_h)},
                {"in0_tensor_next_block_stride", static_cast<uint32_t>(in0_tensor_next_block_stride)},
                {"in0_block_w", in0_block_w},
                {"in0_block_h", per_core_M_per_batch},
                {"in0_block_num_tiles", in0_block_num_tiles},
                {"last_ktile_w", static_cast<uint32_t>(in0_last_ktile_w)},
                {"last_ktile_h", static_cast<uint32_t>(in0_last_ktile_h)},
                {"num_blocks", num_blocks},
                {"bcast_B", static_cast<uint32_t>(bcast_batch)},
                {"MtKt", M * K},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"in0_tensor_start_tile_id", "batch"},
            },
        .hw_config =
            ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    KernelSpec reader_writer{
        .unique_id = READER_WRITER,
        .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp",
        .compiler_options =
            {
                .defines = KernelSpec::CompilerOptions::Defines(reader_writer_defines),
            },
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN1_DFB,
                    .accessor_name = "in1",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUT_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = IN1,
                    .accessor_name = "in1",
                },
                TensorBinding{
                    .tensor_parameter_name = OUTPUT,
                    .accessor_name = "out",
                },
            },
        .compile_time_args =
            {
                {"in1_tensor_stride_w", static_cast<uint32_t>(in1_tensor_stride_w)},
                {"in1_tensor_stride_h", static_cast<uint32_t>(in1_tensor_stride_h)},
                {"in1_tensor_next_block_stride", static_cast<uint32_t>(in1_tensor_next_block_stride)},
                {"in1_block_w", per_core_N},
                {"in1_block_h", in0_block_w},
                {"in1_block_num_tiles", in1_block_num_tiles},
                {"num_blocks", num_blocks},
                {"bcast_B", static_cast<uint32_t>(bcast_batch)},
                {"KtNt", K * N},
                {"out_tensor_stride_w", 1u},
                {"out_tensor_stride_h", N},
                {"out_tensor_next_subblock_stride_w", out_subblock_w},
                {"out_tensor_next_subblock_stride_h", out_subblock_h * N},
                {"out_subblock_w", out_subblock_w},
                {"out_subblock_h", out_subblock_h},
                {"out_subblock_tile_count", out_subblock_w * out_subblock_h},
                {"out_num_subblocks_w", out_num_subblocks_w},
                {"out_num_subblocks_h", out_num_subblocks_h},
                {"MtNt", M * N},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"in1_tensor_start_tile_id", "batch", "out_tensor_start_tile_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };
    if (bias.has_value()) {
        reader_writer.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = BIAS_DFB,
            .accessor_name = "bias",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_writer.tensor_bindings.push_back(TensorBinding{
            .tensor_parameter_name = BIAS,
            .accessor_name = "bias",
        });
        reader_writer.runtime_arg_schema.runtime_arg_names.push_back("in3_tensor_start_tile_id");
    }

    // Compute kernel. Two specs of one source, one per work-split core group, differing only in the
    // per-group block count; they cover disjoint node sets, so each node still sees exactly one.
    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t blocks_per_core_group) {
        auto compute_hw =
            ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config.value());

        // Legacy set no unpack_to_dest_mode at all, i.e. UnpackToDestMode::Default for every
        // buffer, which is UnpackMode::UnpackToSrc here. Stated explicitly because Metal 2.0
        // requires the choice for a Float32 buffer a compute kernel consumes under
        // enable_32_bit_dest, which the intermediate hits whenever fp32_dest_acc_en is set.
        unpack_modes(compute_hw) = {
            {IN0_DFB, UnpackMode::UnpackToSrc},
            {IN1_DFB, UnpackMode::UnpackToSrc},
            {INTERMED0_DFB, UnpackMode::UnpackToSrc},
        };
        if (bias.has_value()) {
            unpack_modes(compute_hw).insert({BIAS_DFB, UnpackMode::UnpackToSrc});
        }
        if (in0_transpose_tile) {
            unpack_modes(compute_hw).insert({IN0_TRANSPOSED_DFB, UnpackMode::UnpackToSrc});
        }

        KernelSpec compute{
            .unique_id = unique_id,
            .source =
                "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
                "bmm_large_block_zm_fused_bias_activation_metal2.cpp",
            .compiler_options =
                {
                    .defines = KernelSpec::CompilerOptions::Defines(mm_kernel_defines),
                    .opt_level = KernelBuildOptLevel::O3,
                },
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN0_DFB,
                        .accessor_name = "in0",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB,
                        .accessor_name = "in1",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB,
                        .accessor_name = "out",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    // The compute kernel writes partials into the intermediate and reads them back
                    // across inner-dimension blocks, so it is both endpoints of that buffer.
                    DFBBinding{
                        .dfb_spec_name = INTERMED0_DFB,
                        .accessor_name = "intermed0",
                        .endpoint_type = DFBEndpointType::PRODUCER,
                    },
                    DFBBinding{
                        .dfb_spec_name = INTERMED0_DFB,
                        .accessor_name = "intermed0",
                        .endpoint_type = DFBEndpointType::CONSUMER,
                    },
                },
            .compile_time_args =
                {
                    {"in0_block_w", in0_block_w},
                    {"in0_num_subblocks", in0_num_subblocks},
                    {"in0_block_num_tiles", in0_block_num_tiles},
                    {"in0_subblock_num_tiles", in0_subblock_num_tiles},
                    {"in1_num_subblocks", in1_num_subblocks},
                    {"in1_block_num_tiles", in1_block_num_tiles},
                    {"in1_block_w", in1_per_core_w},
                    {"num_blocks_inner_dim", num_blocks},
                    {"num_blocks_w_dim", 1u},
                    {"num_blocks_h_dim", 1u},
                    {"out_subblock_h", out_subblock_h},
                    {"out_subblock_w", out_subblock_w},
                    {"out_subblock_num_tiles", out_subblock_num_tiles},
                    // The shared kernel's batch counter is this factory's per-group output-block
                    // count: each core walks that many [per_core_M, per_core_N] output blocks.
                    {"batch", blocks_per_core_group},
                    {"out_block_num_tiles", out_block_tiles},
                    {"untilize_out", static_cast<uint32_t>(untilize_out)},
                    {"get_batch_from_reader", 0u},
                },
            .hw_config = std::move(compute_hw),
        };
        if (in0_transpose_tile) {
            compute.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = IN0_TRANSPOSED_DFB,
                .accessor_name = "in0_transposed",
                .endpoint_type = DFBEndpointType::PRODUCER,
            });
            compute.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = IN0_TRANSPOSED_DFB,
                .accessor_name = "in0_transposed",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }
        if (bias.has_value()) {
            compute.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = BIAS_DFB,
                .accessor_name = "bias",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
            compute.compile_time_args.insert({"bias_ntiles", in3_block_tiles});
            // Full-tile bias block, indexed by (M-tile-row, N-tile) rather than broadcast per row.
            compute.compile_time_args.insert({"row_broadcast_bias", 0u});
        }
        return compute;
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Build per-node runtime args
    ////////////////////////////////////////////////////////////////////////////
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

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs reader_writer_run_args{.kernel = READER_WRITER};

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

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"in0_tensor_start_tile_id", in0_start_tile_id}, {"batch", num_output_blocks_per_core}});

        uint32_t out_start_tile_id =
            (start_batch * M * N) + (start_m_block * per_core_M_per_batch * N) + (start_n_block * per_core_N);
        AddRuntimeArgsForNode(
            reader_writer_run_args.runtime_arg_values,
            core,
            {{"in1_tensor_start_tile_id", in1_start_tile_id},
             {"batch", num_output_blocks_per_core},
             {"out_tensor_start_tile_id", out_start_tile_id}});
        if (bias.has_value()) {
            // Broadcast over batch, single block per element (start_m_block == start_n_block == 0
            // under the bias FATAL): the whole [M, N] bias starts at tile 0.
            AddRuntimeArgsForNode(reader_writer_run_args.runtime_arg_values, core, {{"in3_tensor_start_tile_id", 0u}});
        }

        num_blocks_written += num_output_blocks_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////
    const bool has_group_2 = !core_group_2.ranges().empty();

    Group<KernelSpec> kernels;
    kernels.reserve(has_group_2 ? 4 : 3);
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(reader_writer));
    kernels.push_back(make_compute(COMPUTE_G1, num_blocks_per_core_group_1));
    if (has_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_blocks_per_core_group_2));
    }

    Group<TensorParameter> tensor_parameters;
    tensor_parameters.push_back(TensorParameter{.unique_id = IN0, .spec = in0_buffer.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = IN1, .spec = in1_buffer.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()});
    if (bias.has_value()) {
        tensor_parameters.push_back(TensorParameter{.unique_id = BIAS, .spec = bias->tensor_spec()});
    }

    Group<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{
        .name = "core_group_1",
        .kernels = {READER, READER_WRITER, COMPUTE_G1},
        .target_nodes = core_group_1,
    });
    if (has_group_2) {
        work_units.push_back(WorkUnitSpec{
            .name = "core_group_2",
            .kernels = {READER, READER_WRITER, COMPUTE_G2},
            .target_nodes = core_group_2,
        });
    }

    ProgramSpec spec{
        .name = "matmul_multi_core_reuse_optimized",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(reader_writer_run_args));
    run_args.tensor_args = {
        {IN0, in0_buffer},
        {IN1, in1_buffer},
        {OUTPUT, output},
    };
    if (bias.has_value()) {
        run_args.tensor_args.insert({BIAS, *bias});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
