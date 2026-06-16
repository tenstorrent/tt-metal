// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_optimized_program_factory.hpp"

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

#include <map>
#include <string>
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt;
using tt::tt_metal::Tensor;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

CoreRangeSet MatmulMultiCoreReuseOptimizedProgramFactory::default_core_range(IDevice* device) {
    auto grid_size = device->compute_with_storage_grid_size();
    return CoreRangeSet({CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1})});
}

ttnn::device_operation::ProgramArtifacts MatmulMultiCoreReuseOptimizedProgramFactory::create_program_spec(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    TT_FATAL(
        operation_attributes.program_config.has_value(), "program_config must be provided for create_program_spec");
    const auto& program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseProgramConfig>(operation_attributes.program_config.value());

    TT_FATAL(operation_attributes.output_dtype.has_value(), "Output dtype should have been provided");
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config should have been provided");
    TT_FATAL(operation_attributes.bcast_batch.has_value(), "Bcast batch should have been provided");

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    auto& output_tensor = tensor_return_value.at(0);
    const auto& output = output_tensor.mesh_tensor();

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

    // CB sizes
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

    // Blackhole intermediate CB read workaround
    bool in0_needs_intermediate_cb_read = false;
    bool in1_needs_intermediate_cb_read = false;
    if (device->arch() == tt::ARCH::BLACKHOLE) {
        in0_needs_intermediate_cb_read = ((in0_single_tile_size % 64) != 0);
        in1_needs_intermediate_cb_read = ((in1_single_tile_size % 64) != 0);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramSpec (immutable)
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec;
    spec.name = "matmul_multi_core_reuse_optimized";

    // ---- DataflowBufferSpecs (one per legacy CBDescriptor) ----
    // Sharded input/output CBs borrow their backing L1 from the corresponding io tensor
    // (legacy CBDescriptor::tensor = &buffer); model as borrowed-memory DFBs.
    spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = m2::DFBSpecName{"in0"},
        .entry_size = in0_single_tile_size,
        .num_entries = in0_CB_tiles,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
        .borrowed_from = in0_is_sharded ? std::optional<m2::TensorParamName>{m2::TensorParamName{"a"}} : std::nullopt});
    spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = m2::DFBSpecName{"in1"},
        .entry_size = in1_single_tile_size,
        .num_entries = in1_CB_tiles,
        .data_format_metadata = in1_data_format,
        .tile_format_metadata = in1_tile,
        .borrowed_from = in1_is_sharded ? std::optional<m2::TensorParamName>{m2::TensorParamName{"b"}} : std::nullopt});

    // CB 4 and CB 5: Output and intermediate accumulator. Legacy keeps them as a single CB
    // (one CBDescriptor, two format_descriptors) when the formats match; that aliasing is
    // expressed as two DFBs sharing backing memory via advanced_options.alias_with.
    const bool separate_out_interm =
        (interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1));
    if (separate_out_interm) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out"},
            .entry_size = output_single_tile_size,
            .num_entries = out_CB_tiles,
            .data_format_metadata = output_data_format,
            .tile_format_metadata = output_tile,
            .borrowed_from =
                output_is_sharded ? std::optional<m2::TensorParamName>{m2::TensorParamName{"out"}} : std::nullopt});
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"intermed0"},
            .entry_size = interm0_single_tile_size,
            .num_entries = out_CB_tiles,
            .data_format_metadata = interm0_data_format,
            .tile_format_metadata = output_tile});
    } else {
        // Shared output+intermediate region: two aliased DFBs over one L1 allocation. The
        // borrowed-memory consistency rule requires aliased members to agree on borrowed_from, so
        // when the output is sharded the intermediate alias borrows from the same TensorParameter.
        std::optional<m2::TensorParamName> out_borrow =
            output_is_sharded ? std::optional<m2::TensorParamName>{m2::TensorParamName{"out"}} : std::nullopt;
        m2::DataflowBufferSpec out_dfb{
            .unique_id = m2::DFBSpecName{"out"},
            .entry_size = output_single_tile_size,
            .num_entries = out_CB_tiles,
            .data_format_metadata = output_data_format,
            .tile_format_metadata = output_tile,
            .borrowed_from = out_borrow};
        out_dfb.advanced_options.alias_with = {m2::DFBSpecName{"intermed0"}};
        m2::DataflowBufferSpec interm_dfb{
            .unique_id = m2::DFBSpecName{"intermed0"},
            .entry_size = interm0_single_tile_size,
            .num_entries = out_CB_tiles,
            .data_format_metadata = interm0_data_format,
            .tile_format_metadata = output_tile,
            .borrowed_from = out_borrow};
        interm_dfb.advanced_options.alias_with = {m2::DFBSpecName{"out"}};
        spec.dataflow_buffers.push_back(std::move(out_dfb));
        spec.dataflow_buffers.push_back(std::move(interm_dfb));
    }

    // Optional CBs for Blackhole intermediate reads
    if (in1_needs_intermediate_cb_read) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in1_intermediate"},
            .entry_size = in1_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = in1_tile});
    }
    if (in0_needs_intermediate_cb_read) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in0_intermediate"},
            .entry_size = in0_single_tile_size,
            .num_entries = 1,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile});
    }
    // Optional transpose CB
    if (in0_transpose_tile) {
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in0_transposed"},
            .entry_size = in0_single_tile_size,
            .num_entries = in0_CB_tiles,
            .data_format_metadata = in0_data_format,
            .tile_format_metadata = in0_tile});
    }

    // ---- TensorParameters (one per distinct accessed tensor) ----
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"a"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"b"}, .spec = b.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"out"}, .spec = output_tensor.tensor_spec()},
    };

    // ---- Reader kernel (in0) ----
    m2::KernelSpec::CompilerOptions::Defines reader_defines;
    if (in0_is_sharded) {
        reader_defines.insert({"IN0_SHARDED", "1"});
    }
    if (in0_needs_intermediate_cb_read) {
        reader_defines.insert({"INTERMEDIATE_CB_READ", "1"});
    }
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                                        "reader_bmm_tile_layout_in0.cpp"},
        .compiler_options = {.defines = reader_defines},
        .compile_time_args =
            {{"in0_tensor_stride_w", static_cast<uint32_t>(in0_tensor_stride_w)},
             {"in0_tensor_stride_h", static_cast<uint32_t>(in0_tensor_stride_h)},
             {"in0_tensor_next_block_stride", static_cast<uint32_t>(in0_tensor_next_block_stride)},
             {"in0_block_w", in0_block_w},
             {"in0_block_h", per_core_M_per_batch},
             {"in0_block_num_tiles", in0_block_num_tiles},
             {"last_ktile_w", static_cast<uint32_t>(in0_last_ktile_w)},
             {"last_ktile_h", static_cast<uint32_t>(in0_last_ktile_h)},
             {"num_blocks", num_blocks},
             {"bcast_B", static_cast<uint32_t>(bcast_batch)},
             {"MtKt", M * K}},
        .runtime_arg_schema = {.runtime_arg_names = {"in0_tensor_start_tile_id", "batch"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };
    // in0 DFB binding: PRODUCER (reader fills it). Bind tensor `a` only on the non-sharded
    // (NoC-read) path — when sharded, the in0 DFB borrows tensor `a`'s memory directly.
    reader.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"in0"},
        .accessor_name = "cb_in0",
        .endpoint_type = m2::DFBEndpointType::PRODUCER});
    if (!in0_is_sharded) {
        reader.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"a"}, .accessor_name = "a"});
    }
    if (in0_needs_intermediate_cb_read) {
        // Helper CB used as a real producer/consumer FIFO inside the reader (reserve/push then
        // wait/pop), so it self-loops on this kernel.
        reader.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0_intermediate"},
            .accessor_name = "cb_in0_intermediate",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0_intermediate"},
            .accessor_name = "cb_in0_intermediate",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }

    // ---- Reader/Writer kernel (reads in1, writes output) ----
    m2::KernelSpec::CompilerOptions::Defines rw_defines;
    if (in1_is_sharded) {
        rw_defines.insert({"IN1_SHARDED", "1"});
    }
    if (output_is_sharded) {
        rw_defines.insert({"OUT_SHARDED", "1"});
    }
    if (in1_needs_intermediate_cb_read) {
        rw_defines.insert({"INTERMEDIATE_CB_READ", "1"});
    }
    m2::KernelSpec reader_writer{
        .unique_id = m2::KernelSpecName{"reader_writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                                        "reader_writer_bmm_tile_layout_in1.cpp"},
        .compiler_options = {.defines = rw_defines},
        .compile_time_args =
            {{"in1_tensor_stride_w", static_cast<uint32_t>(in1_tensor_stride_w)},
             {"in1_tensor_stride_h", static_cast<uint32_t>(in1_tensor_stride_h)},
             {"in1_tensor_next_block_stride", static_cast<uint32_t>(in1_tensor_next_block_stride)},
             {"in1_block_w", per_core_N},
             {"in1_block_h", in0_block_w},
             {"in1_block_num_tiles", in1_block_num_tiles},
             {"num_blocks", num_blocks},
             {"bcast_B", static_cast<uint32_t>(bcast_batch)},
             {"KtNt", K * N},
             {"out_tensor_stride_w", 1},
             {"out_tensor_stride_h", N},
             {"out_tensor_next_subblock_stride_w", out_subblock_w},
             {"out_tensor_next_subblock_stride_h", out_subblock_h * N},
             {"out_subblock_w", out_subblock_w},
             {"out_subblock_h", out_subblock_h},
             {"out_subblock_tile_count", out_subblock_w * out_subblock_h},
             {"out_num_subblocks_w", out_num_subblocks_w},
             {"out_num_subblocks_h", out_num_subblocks_h},
             {"MtNt", M * N}},
        .runtime_arg_schema = {.runtime_arg_names = {"in1_tensor_start_tile_id", "batch", "out_tensor_start_tile_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };
    // in1 DFB: PRODUCER (reader fills it); out DFB: CONSUMER (writer drains it).
    reader_writer.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"in1"},
        .accessor_name = "cb_in1",
        .endpoint_type = m2::DFBEndpointType::PRODUCER});
    reader_writer.dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = m2::DFBSpecName{"out"},
        .accessor_name = "cb_out",
        .endpoint_type = m2::DFBEndpointType::CONSUMER});
    if (!in1_is_sharded) {
        reader_writer.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"b"}, .accessor_name = "b"});
    }
    if (!output_is_sharded) {
        reader_writer.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"out"}, .accessor_name = "out"});
    }
    if (in1_needs_intermediate_cb_read) {
        reader_writer.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in1_intermediate"},
            .accessor_name = "cb_in1_intermediate",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader_writer.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in1_intermediate"},
            .accessor_name = "cb_in1_intermediate",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }

    // ---- Compute kernel(s) (one KernelSpec per core group) ----
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
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    for (const auto& [k, v] : mm_kernel_defines) {
        compute_defines.insert({k, v});
    }
    // cb_in0_transposed is bound only on the in0-transpose path; gate the kernel-side handle
    // on a matching define so its dfb:: token isn't name-looked-up in the non-transpose build.
    if (in0_transpose_tile) {
        compute_defines.insert({"IN0_TRANSPOSE_TILE_CB", "1"});
    }

    const char* COMPUTE_SRC =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_m2.cpp";
    auto make_compute = [&](const std::string& id, uint32_t num_blocks_per_core_group) {
        m2::KernelSpec compute{
            .unique_id = m2::KernelSpecName{id},
            .source = std::filesystem::path{COMPUTE_SRC},
            .compiler_options = {.defines = compute_defines},
            // Compute CTA layout matches the legacy positional emission order; here num_blocks_w_dim
            // and num_blocks_h_dim are both 1, batch is the per-group output-block count, and
            // out_block_num_tiles is out_block_tiles.
            .compile_time_args =
                {{"in0_block_w", in0_block_w},
                 {"in0_num_subblocks", in0_num_subblocks},
                 {"in0_block_num_tiles", in0_block_num_tiles},
                 {"in0_subblock_num_tiles", in0_subblock_num_tiles},
                 {"in1_num_subblocks", in1_num_subblocks},
                 {"in1_block_num_tiles", in1_block_num_tiles},
                 {"in1_block_w", in1_per_core_w},
                 {"num_blocks_inner_dim", num_blocks},
                 {"num_blocks_w_dim", 1},
                 {"num_blocks_h_dim", 1},
                 {"out_subblock_h", out_subblock_h},
                 {"out_subblock_w", out_subblock_w},
                 {"out_subblock_num_tiles", out_subblock_num_tiles},
                 {"batch", num_blocks_per_core_group},
                 {"out_block_num_tiles", out_block_tiles},
                 {"untilize_out", static_cast<uint32_t>(untilize_out)},
                 {"get_batch_from_reader", 0},
                 {"in0_transpose_tile", static_cast<uint32_t>(in0_transpose_tile)}},
            .hw_config =
                m2::ComputeHardwareConfig{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en,
                    .math_approx_mode = math_approx_mode},
        };
        // in0 / in1: CONSUMER; out + intermed0: PRODUCER (intermed0 also CONSUMER — the kernel
        // spills partials into it and reloads them, a real self-loop). cb_in0_transposed
        // self-loops on the compute kernel (reads cb_in0, transposes, writes cb_in0_transposed,
        // then reads it back as matmul input).
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in0"},
            .accessor_name = "cb_in0",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"in1"},
            .accessor_name = "cb_in1",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"out"},
            .accessor_name = "cb_out",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"intermed0"},
            .accessor_name = "cb_intermed0",
            .endpoint_type = m2::DFBEndpointType::PRODUCER});
        compute.dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = m2::DFBSpecName{"intermed0"},
            .accessor_name = "cb_intermed0",
            .endpoint_type = m2::DFBEndpointType::CONSUMER});
        if (in0_transpose_tile) {
            compute.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"in0_transposed"},
                .accessor_name = "cb_in0_transposed",
                .endpoint_type = m2::DFBEndpointType::PRODUCER});
            compute.dfb_bindings.push_back(m2::DFBBinding{
                .dfb_spec_name = m2::DFBSpecName{"in0_transposed"},
                .accessor_name = "cb_in0_transposed",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        return compute;
    };
    m2::KernelSpec compute_1 = make_compute("compute_1", num_blocks_per_core_group_1);

    // ---- Kernels + WorkUnitSpecs ----
    // Local DFBs require producer and consumer KernelSpecs to share the same WorkUnitSpec(s):
    // every node hosting a DFB must host both endpoints. Each core group becomes one
    // WorkUnitSpec containing reader + reader_writer + compute_<group> (reader/reader_writer are
    // shared across both groups; the per-group compute differs only in its batch CTA).
    const bool has_group_2 = !core_group_2.ranges().empty();
    if (has_group_2) {
        m2::KernelSpec compute_2 = make_compute("compute_2", num_blocks_per_core_group_2);
        spec.kernels = {reader, reader_writer, compute_1, compute_2};
        spec.work_units = std::vector<m2::WorkUnitSpec>{
            m2::WorkUnitSpec{
                .name = "g1",
                .kernels =
                    {m2::KernelSpecName{"reader"},
                     m2::KernelSpecName{"reader_writer"},
                     m2::KernelSpecName{"compute_1"}},
                .target_nodes = core_group_1},
            m2::WorkUnitSpec{
                .name = "g2",
                .kernels =
                    {m2::KernelSpecName{"reader"},
                     m2::KernelSpecName{"reader_writer"},
                     m2::KernelSpecName{"compute_2"}},
                .target_nodes = core_group_2},
        };
    } else {
        spec.kernels = {reader, reader_writer, compute_1};
        spec.work_units = std::vector<m2::WorkUnitSpec>{
            m2::WorkUnitSpec{
                .name = "g1",
                .kernels =
                    {m2::KernelSpecName{"reader"},
                     m2::KernelSpecName{"reader_writer"},
                     m2::KernelSpecName{"compute_1"}},
                .target_nodes = core_group_1},
        };
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      ProgramRunArgs (mutable)
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

    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs reader_writer_run{.kernel = m2::KernelSpecName{"reader_writer"}};

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

        // The reader/reader-writer kernels iterate their "batch" loop num_output_blocks_per_core
        // times (the legacy reader's RTA #2), one per output block this core handles.
        reader_run.runtime_arg_values.push_back(
            {core, {{"in0_tensor_start_tile_id", in0_start_tile_id}, {"batch", num_output_blocks_per_core}}});
        reader_writer_run.runtime_arg_values.push_back(
            {core,
             {{"in1_tensor_start_tile_id", in1_start_tile_id},
              {"batch", num_output_blocks_per_core},
              {"out_tensor_start_tile_id", out_start_tile_id}}});

        num_blocks_written += num_output_blocks_per_core;
    }
    run.kernel_run_args = {reader_run, reader_writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"a"}, in0_buffer},
        {m2::TensorParamName{"b"}, in1_buffer},
        {m2::TensorParamName{"out"}, output},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
