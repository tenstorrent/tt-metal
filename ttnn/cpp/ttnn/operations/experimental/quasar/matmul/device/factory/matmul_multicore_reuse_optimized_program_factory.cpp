// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_optimized_program_factory.hpp"

#include <filesystem>
#include <map>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

// DFB / tensor / kernel spec names. The accessor names chosen here surface kernel-side as the
// dfb::* / tensor:: tokens used by the ported kernels (reader_bmm_tile_layout_in0.cpp,
// reader_writer_bmm_tile_layout_in1.cpp, bmm_large_block_zm_fused_bias_activation_metal2.cpp).
// The C++ constants are RO_-prefixed to avoid anonymous-namespace duplicate-symbol collisions with
// the sibling matmul_multicore_program_factory.cpp under unity builds (port_patterns "Unity-build
// hygiene"); the StrongType string *values* are still the plain accessor names the kernels expect.
const DFBSpecName RO_IN0_DFB{"in0"};
const DFBSpecName RO_IN1_DFB{"in1"};
const DFBSpecName RO_OUT_DFB{"out"};
const DFBSpecName RO_INTERM0_DFB{"intermed0"};
const DFBSpecName RO_IN0_TRANSPOSE_DFB{"in0_transposed"};

const TensorParamName RO_IN0_TENSOR{"in0"};
const TensorParamName RO_IN1_TENSOR{"in1"};
const TensorParamName RO_OUT_TENSOR{"out"};

const KernelSpecName RO_READER_KERNEL{"reader"};
const KernelSpecName RO_READER_WRITER_KERNEL{"reader_writer"};
const KernelSpecName RO_COMPUTE_KERNEL_G1{"compute_g1"};
const KernelSpecName RO_COMPUTE_KERNEL_G2{"compute_g2"};

}  // namespace

ttnn::device_operation::ProgramArtifacts MatmulMultiCoreReuseOptimizedProgramFactory::create_program_artifacts(
    const ttnn::prim::qsr::MatmulParams& operation_attributes,
    const ttnn::prim::qsr::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    TT_FATAL(
        operation_attributes.program_config.has_value(),
        "program_config must be provided for create_program_artifacts");
    const auto& program_config = std::get<operations::experimental::quasar::matmul::MatmulMultiCoreReuseProgramConfig>(
        operation_attributes.program_config.value());

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

    const auto& ashape =
        operations::experimental::quasar::matmul::utilities::get_matmul_tensor_padded_shape(a, transpose_a);
    const auto& bshape =
        operations::experimental::quasar::matmul::utilities::get_matmul_tensor_padded_shape(b, transpose_b);
    auto in0_tile = operations::experimental::quasar::matmul::utilities::get_matmul_tile(a, transpose_a);
    auto in1_tile = operations::experimental::quasar::matmul::utilities::get_matmul_tile(b, transpose_b);

    const auto& in0_buffer = a.mesh_tensor();
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(in0_buffer.dtype());
    const auto& in1_buffer = b.mesh_tensor();
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(in1_buffer.dtype());
    tt::DataFormat output_data_format =
        tt_metal::datatype_to_dataformat_converter(operation_attributes.output_dtype.value());

    tt_metal::IDevice* device = &in0_buffer.mutable_device();

    auto fp32_dest_acc_en = operation_attributes.compute_kernel_config->fp32_dest_acc_en;
    auto packer_l1_acc = operation_attributes.compute_kernel_config->packer_l1_acc;

    if (fp32_dest_acc_en) {
        TT_FATAL(
            out_subblock_h * out_subblock_w <= 4,
            "Total number of tiles in a subblock must be less than 4 when in fp32_dest_acc mode");
    }

    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = operations::experimental::quasar::matmul::utilities::get_M_dim(ashape, in0_tile, false);
    uint32_t Kt = operations::experimental::quasar::matmul::utilities::get_K_dim(ashape, in0_tile);
    uint32_t Nt = operations::experimental::quasar::matmul::utilities::get_N_dim(bshape, in1_tile);
    uint32_t M = Mt;
    uint32_t N = Nt;
    uint32_t K = Kt;

    const auto ashape_logical =
        operations::experimental::quasar::matmul::utilities::get_matmul_tensor_logical_shape(a, transpose_a);
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
    // stride, so the in0/in1 DFBs must hold pages at the aligned stride and the reader/unpacker walk
    // tiles at the same stride. This is a no-op when the tile is already aligned. Sharded DFBs are
    // backed by the tensor buffer and keep their natural page size.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in0_aligned_tile_size =
        in0_is_sharded ? in0_single_tile_size : tt::align(in0_single_tile_size, dram_alignment);
    uint32_t in1_aligned_tile_size =
        in1_is_sharded ? in1_single_tile_size : tt::align(in1_single_tile_size, dram_alignment);

    // DFB sizes
    uint32_t in0_block_num_tiles = per_core_M_per_batch * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_num_tiles;
    if (in0_is_sharded) {
        in0_CB_tiles = per_core_M * K;
    } else {
        in0_CB_tiles *= 2;
    }
    uint32_t in1_block_num_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_num_tiles;
    if (in1_is_sharded) {
        in1_CB_tiles *= num_blocks * batch_scale_factor;
    } else {
        in1_CB_tiles *= 2;
    }
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles;

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

    // Core splitting. The legacy create_descriptor carried an extra `core_range_set` parameter that
    // existed only to drive a pybind hook (now deleted); production always took the sharded /
    // allowed_worker_cores / compute_with_storage_grid_size paths below.
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
                "falling back to compute_with_storage_grid_size. Callers that bypass ttnn::prim::qsr::matmul() should "
                "invoke ttnn::operations::experimental::quasar::matmul::normalize_program_config() on the program "
                "config first. This will become a hard error in a future release.");
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

    // ---- Tensor parameters (replace the legacy buffer-address RTAs + TensorAccessorArgs plumbing) ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = RO_IN0_TENSOR, .spec = in0_buffer.tensor_spec()},
        TensorParameter{.unique_id = RO_IN1_TENSOR, .spec = in1_buffer.tensor_spec()},
        TensorParameter{.unique_id = RO_OUT_TENSOR, .spec = output.tensor_spec()},
    };

    // ---- Dataflow buffers ----
    // in0 (c_0) and in1 (c_1): sharded -> borrowed_from the backing tensor; otherwise allocated.
    // out (c_4) and intermed0 (c_5): when their formats/tile match (and not untilize-with-multi-subblock)
    // they shared one legacy CB buffer (two format_descriptors) -> two aliased DFBs sharing memory.
    // in0_transposed (c_10): only present when in0_transpose_tile.
    Group<DataflowBufferSpec> dataflow_buffers;

    {
        DataflowBufferSpec in0_dfb{
            .unique_id = RO_IN0_DFB,
            .entry_size = in0_aligned_tile_size,
            .num_entries = in0_CB_tiles,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        };
        if (in0_is_sharded) {
            in0_dfb.borrowed_from = RO_IN0_TENSOR;
        }
        dataflow_buffers.push_back(std::move(in0_dfb));
    }
    {
        DataflowBufferSpec in1_dfb{
            .unique_id = RO_IN1_DFB,
            .entry_size = in1_aligned_tile_size,
            .num_entries = in1_CB_tiles,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = in1_tile,
        };
        if (in1_is_sharded) {
            in1_dfb.borrowed_from = RO_IN1_TENSOR;
        }
        dataflow_buffers.push_back(std::move(in1_dfb));
    }

    // out / intermed0: separate buffers, or aliased (shared memory) when the legacy factory shared
    // one CB across c_4 and c_5. Total sizes of aliased DFBs must be equal; in the shared legacy
    // case the formats match so the out and interm0 entry sizes are equal by construction.
    const bool separate_out_interm =
        (interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1));
    {
        DataflowBufferSpec out_dfb{
            .unique_id = RO_OUT_DFB,
            .entry_size = output_single_tile_size,
            .num_entries = out_CB_tiles,
            .data_format_metadata = output_data_format,
            .tile_format_metadata = output_tile,
        };
        if (output_is_sharded) {
            out_dfb.borrowed_from = RO_OUT_TENSOR;
        }
        DataflowBufferSpec interm0_dfb{
            .unique_id = RO_INTERM0_DFB,
            .entry_size = interm0_single_tile_size,
            .num_entries = out_CB_tiles,
            .data_format_metadata = interm0_data_format,
            .tile_format_metadata = output_tile,
        };
        if (!separate_out_interm) {
            // Aliased DFBs: each must name every other member (strict clique of 2). The legacy
            // shared CB backed its single L1 region with the output tensor when output was sharded;
            // the aliased-DFB validator requires borrowed_from to be consistent across the group, so
            // both members borrow from RO_OUT_TENSOR (same backing memory) when output is sharded.
            out_dfb.advanced_options.alias_with = {RO_INTERM0_DFB};
            interm0_dfb.advanced_options.alias_with = {RO_OUT_DFB};
            if (output_is_sharded) {
                interm0_dfb.borrowed_from = RO_OUT_TENSOR;
            }
        }
        dataflow_buffers.push_back(std::move(out_dfb));
        dataflow_buffers.push_back(std::move(interm0_dfb));
    }

    if (in0_transpose_tile) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RO_IN0_TRANSPOSE_DFB,
            .entry_size = in0_aligned_tile_size,
            .num_entries = in0_CB_tiles,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile,
        });
    }

    // ---- Defines ----
    KernelSpec::CompilerOptions::Defines reader_defines;
    KernelSpec::CompilerOptions::Defines reader_writer_defines;
    if (in0_is_sharded) {
        reader_defines.insert({"IN0_SHARDED", "1"});
    }
    if (in1_is_sharded) {
        reader_writer_defines.insert({"IN1_SHARDED", "1"});
    }
    if (output_is_sharded) {
        reader_writer_defines.insert({"OUT_SHARDED", "1"});
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
        // Drives the conditional dfb::cb_in0_transposed reference in the _metal2 compute kernel; must
        // be a preprocessor define (not a CTA) so the token reference is gated before C++ parsing.
        mm_kernel_defines["IN0_TRANSPOSE_TILE_PATH"] = "1";
    }
    const auto throttle_level = ttnn::get_throttle_level(operation_attributes.compute_kernel_config);
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_cores, mm_kernel_defines, throttle_level);
    KernelSpec::CompilerOptions::Defines compute_defines(mm_kernel_defines);

    // ---- Reader kernel (reads in0) ----
    // RTA carries the per-core in0 start tile id; the legacy 3rd RTA (per-core block count) is named
    // "batch" kernel-side (preserved verbatim from the legacy kernel).
    KernelSpec reader{
        .unique_id = RO_READER_KERNEL,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
                                        "reader_bmm_tile_layout_in0.cpp"),
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = RO_IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = RO_IN0_TENSOR, .accessor_name = "in0"},
            },
        .compile_time_args =
            {
                {"in0_tensor_stride_w", (uint32_t)in0_tensor_stride_w},
                {"in0_tensor_stride_h", (uint32_t)in0_tensor_stride_h},
                {"in0_tensor_next_block_stride", (uint32_t)in0_tensor_next_block_stride},
                {"in0_block_w", in0_block_w},
                {"in0_block_h", per_core_M_per_batch},
                {"in0_block_num_tiles", in0_block_num_tiles},
                {"in0_last_ktile_w", (uint32_t)in0_last_ktile_w},
                {"in0_last_ktile_h", (uint32_t)in0_last_ktile_h},
                {"num_blocks", num_blocks},
                {"bcast_B", (uint32_t)bcast_batch},
                {"MtKt", M * K},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"in0_tensor_start_tile_id", "batch"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- Reader/Writer kernel (reads in1, writes output) ----
    KernelSpec reader_writer{
        .unique_id = RO_READER_WRITER_KERNEL,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/"
                                        "reader_writer_bmm_tile_layout_in1.cpp"),
        .compiler_options = {.defines = reader_writer_defines},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = RO_IN1_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = RO_OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = RO_IN1_TENSOR, .accessor_name = "in1"},
                TensorBinding{.tensor_parameter_name = RO_OUT_TENSOR, .accessor_name = "out"},
            },
        .compile_time_args =
            {
                {"in1_tensor_stride_w", (uint32_t)in1_tensor_stride_w},
                {"in1_tensor_stride_h", (uint32_t)in1_tensor_stride_h},
                {"in1_tensor_next_block_stride", (uint32_t)in1_tensor_next_block_stride},
                {"in1_block_w", per_core_N},
                {"in1_block_h", in0_block_w},
                {"in1_block_num_tiles", in1_block_num_tiles},
                {"num_blocks", num_blocks},
                {"bcast_B", (uint32_t)bcast_batch},
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

    ComputeHardwareConfig compute_hw_config =
        ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config.value());

    // ---- Compute kernel(s) — one KernelSpec per core group, preserving the per-group block-count CTA ----
    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t num_blocks_per_core_group) {
        std::vector<DFBBinding> dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = RO_IN0_DFB, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = RO_IN1_DFB, .accessor_name = "cb_in1", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = RO_OUT_DFB, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = RO_INTERM0_DFB,
                .accessor_name = "cb_intermed0",
                .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = RO_INTERM0_DFB,
                .accessor_name = "cb_intermed0",
                .endpoint_type = DFBEndpointType::CONSUMER},
        };
        if (in0_transpose_tile) {
            // The compute kernel both produces (transposes into) and consumes the transposed in0 DFB.
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RO_IN0_TRANSPOSE_DFB,
                .accessor_name = "cb_in0_transposed",
                .endpoint_type = DFBEndpointType::PRODUCER});
            dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = RO_IN0_TRANSPOSE_DFB,
                .accessor_name = "cb_in0_transposed",
                .endpoint_type = DFBEndpointType::CONSUMER});
        }
        return KernelSpec{
            .unique_id = unique_id,
            .source =
                std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/compute/"
                                      "bmm_large_block_zm_fused_bias_activation_metal2.cpp"),
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings = std::move(dfb_bindings),
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
                    {"batch", num_blocks_per_core_group},
                    {"out_block_num_tiles", out_block_tiles},
                    {"untilize_out", (uint32_t)untilize_out},
                    {"get_batch_from_reader", 0u},
                },
            .hw_config = compute_hw_config,
        };
    };

    const bool group_2_present = !core_group_2.ranges().empty();

    Group<KernelSpec> kernels = {
        reader, reader_writer, make_compute(RO_COMPUTE_KERNEL_G1, num_blocks_per_core_group_1)};
    if (group_2_present) {
        kernels.push_back(make_compute(RO_COMPUTE_KERNEL_G2, num_blocks_per_core_group_2));
    }

    // ---- Work units: reader + reader_writer + the matching compute kernel per core group ----
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{
            .name = "wu_g1",
            .kernels = {RO_READER_KERNEL, RO_READER_WRITER_KERNEL, RO_COMPUTE_KERNEL_G1},
            .target_nodes = core_group_1,
        },
    };
    if (group_2_present) {
        work_units.push_back(WorkUnitSpec{
            .name = "wu_g2",
            .kernels = {RO_READER_KERNEL, RO_READER_WRITER_KERNEL, RO_COMPUTE_KERNEL_G2},
            .target_nodes = core_group_2,
        });
    }

    // ---- Per-core runtime args ----
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

    ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = RO_READER_KERNEL};
    ProgramRunArgs::KernelRunArgs reader_writer_run_args{.kernel = RO_READER_WRITER_KERNEL};

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
        uint32_t out_start_tile_id =
            (start_batch * M * N) + (start_m_block * per_core_M_per_batch * N) + (start_n_block * per_core_N);

        ProgramRunArgs::KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run_args.runtime_arg_values;
        ProgramRunArgs::KernelRunArgs::RuntimeArgValues& reader_writer_rtas = reader_writer_run_args.runtime_arg_values;
        AddRuntimeArgsForNode(
            reader_rtas,
            core,
            {
                {"in0_tensor_start_tile_id", in0_start_tile_id},
                {"batch", num_output_blocks_per_core},
            });
        AddRuntimeArgsForNode(
            reader_writer_rtas,
            core,
            {
                {"in1_tensor_start_tile_id", in1_start_tile_id},
                {"batch", num_output_blocks_per_core},
                {"out_tensor_start_tile_id", out_start_tile_id},
            });

        num_blocks_written += num_output_blocks_per_core;
    }

    // ---- Assemble spec + run args ----
    ProgramSpec spec{
        .name = "matmul_multicore_reuse_optimized",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(reader_writer_run_args)},
        .tensor_args =
            {
                {RO_IN0_TENSOR, in0_buffer},
                {RO_IN1_TENSOR, in1_buffer},
                {RO_OUT_TENSOR, output},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
