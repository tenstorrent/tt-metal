// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::operations::experimental::quasar {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

ttnn::device_operation::ProgramArtifacts fold_multi_core_tiled_interleaved(
    const Tensor& input_tensor, const Tensor& output, const uint32_t stride_h, const uint32_t stride_w) {
    auto* device = input_tensor.device();

    const uint32_t input_width = input_tensor.logical_shape()[2];

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt::DataFormat out_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t out_single_tile_size = tt::tile_size(out_cb_data_format);

    ttnn::Shape output_padded_shape = output.padded_shape();
    ttnn::Shape input_padded_shape = input_tensor.padded_shape();

    log_debug(tt::LogOp, "in_cb_data_format: {}", cb_data_format);
    log_debug(tt::LogOp, "out_cb_data_format: {}", out_cb_data_format);
    log_debug(tt::LogOp, "single_tile_size: {}", single_tile_size);
    log_debug(tt::LogOp, "input_tensor_shape: {}", input_padded_shape);
    log_debug(tt::LogOp, "output_tensor_shape: {}", output_padded_shape);

    // Memory layout parameters
    auto stick_nbytes = output_padded_shape[3] * tt::datum_size(datatype_to_dataformat_converter(output.dtype()));
    uint32_t ntiles = input_tensor.physical_volume() / TILE_HW;
    uint32_t tiles_per_channel_dim = tt::div_up(input_padded_shape[-1], TILE_WIDTH);
    uint32_t tiles_per_width_dim = tt::div_up(input_padded_shape[-2], TILE_HEIGHT);
    uint32_t tiles_per_complete_row = tiles_per_width_dim * tiles_per_channel_dim;
    // Total number of blocks for batch * height
    uint32_t num_blocks = std::ceil(static_cast<float>(ntiles) / (tiles_per_complete_row));

    uint32_t aligned_stick_nbytes = tt::align(stick_nbytes, TILE_WIDTH * tt::datum_size(out_cb_data_format));
    log_debug(
        tt::LogOp, "tiles_per_channel_dim: {}, ntiles: {}, num_blocks: {}", tiles_per_channel_dim, ntiles, num_blocks);

    // Split work across cores for parallel processing
    auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, num_blocks);

    log_debug(
        tt::LogOp,
        "ncores: {}, nblocks_per_core: {}, nblocks_per_core_cliff: {}",
        ncores,
        nblocks_per_core,
        nblocks_per_core_cliff);

    const uint32_t num_input_tiles = tiles_per_channel_dim;

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName SRC0{"src0"};
    const DFBSpecName SRC1{"src1"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_MAIN{"compute_main"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = input_tensor.mesh_tensor().tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.mesh_tensor().tensor_spec()};

    // ---- Dataflow buffers ----
    // Source CB (c_0): reader -> compute
    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };
    // Untilized CB (c_1): compute -> writer
    DataflowBufferSpec src1_dfb{
        .unique_id = SRC1,
        .entry_size = out_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = out_cb_data_format,
    };

    // ---- Reader kernel (DRAM -> SRC0) ----
    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/experimental/quasar/fold/device/kernels/dataflow/reader_dram2cb_tiled.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0, .accessor_name = "src0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .compile_time_args =
            {{"tiles_per_channel_dim", tiles_per_channel_dim}, {"tiles_per_width_dim", tiles_per_width_dim}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_block_id", "num_blocks"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ---- Writer kernel (SRC1 -> DRAM) ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/fold/device/kernels/dataflow/"
            "writer_cb2dram_for_tiled_input.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC1, .accessor_name = "src1", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .compile_time_args =
            {{"input_width", input_width},
             {"stride_height", stride_h},
             {"stride_width", stride_w},
             {"stick_nbytes", stick_nbytes},
             {"aligned_stick_nbytes", aligned_stick_nbytes},
             {"tiles_per_channel_dim", tiles_per_channel_dim},
             {"tiles_per_width_dim", tiles_per_width_dim},
             {"element_size", datum_size(out_cb_data_format)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_block_id", "num_blocks", "patch_height_offset", "output_offset"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ---- Compute kernels (untilize SRC0 -> SRC1) ----
    const bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32;
    auto make_compute = [&](KernelSpecName unique_id, uint32_t per_core_block_cnt) {
        return KernelSpec{
            .unique_id = std::move(unique_id),
            .source = "ttnn/cpp/ttnn/operations/experimental/quasar/fold/device/kernels/compute/untilize.cpp",
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "src0", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = SRC1, .accessor_name = "src1", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"per_core_block_cnt", per_core_block_cnt}, {"per_core_block_tile_cnt", tiles_per_channel_dim}},
            .hw_config = ComputeHardwareConfig{.fp32_dest_acc_en = fp32_dest_acc_en},
        };
    };
    KernelSpec compute_main = make_compute(COMPUTE_MAIN, nblocks_per_core * tiles_per_width_dim);

    const bool has_cliff = !core_range_cliff.ranges().empty();

    // ---- Per-core runtime args ----
    // Determine the "full" core set vs. the cliff core for runtime arg distribution.
    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        ncores_full -= 1;
        full_cores = core_range;
    }

    uint32_t block_start_id = 0;
    auto ncores_x = grid_size.x;
    auto ncores_y = std::ceil(static_cast<float>(ncores) / ncores_x);
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, true);

    const uint32_t patch_size = stride_h * stride_w;       // Size of each patch
    const uint32_t output_width = input_width / stride_w;  // Output width

    Group<ProgramRunArgs::KernelRunArgs::NodeRuntimeArgs> reader_rta;
    Group<ProgramRunArgs::KernelRunArgs::NodeRuntimeArgs> writer_rta;

    for (auto core : cores) {
        uint32_t curr_input_height_idx = block_start_id;
        uint32_t curr_output_height_idx = curr_input_height_idx / stride_h;
        uint32_t patch_height_offset = curr_input_height_idx % stride_h;
        // Total output height * width
        uint32_t output_offset =
            (patch_size * curr_output_height_idx * output_width) + (patch_height_offset * stride_w);
        if (!full_cores.contains(core)) {
            continue;
        }
        reader_rta.push_back(
            {.node = core, .args = {{"start_block_id", block_start_id}, {"num_blocks", nblocks_per_core}}});
        writer_rta.push_back(
            {.node = core,
             .args = {
                 {"start_block_id", block_start_id},
                 {"num_blocks", nblocks_per_core},
                 {"patch_height_offset", patch_height_offset},
                 {"output_offset", output_offset}}});
        block_start_id += nblocks_per_core;
    }

    if (ncores_full < ncores) {
        uint32_t curr_input_height_idx = block_start_id;
        uint32_t curr_output_height_idx = curr_input_height_idx / stride_h;
        uint32_t patch_height_offset = curr_input_height_idx % stride_h;
        uint32_t output_offset =
            (patch_size * curr_output_height_idx * output_width) + (patch_height_offset * stride_w);
        CoreCoord core = CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x};
        reader_rta.push_back(
            {.node = core, .args = {{"start_block_id", block_start_id}, {"num_blocks", nblocks_per_core_cliff}}});
        writer_rta.push_back(
            {.node = core,
             .args = {
                 {"start_block_id", block_start_id},
                 {"num_blocks", nblocks_per_core_cliff},
                 {"patch_height_offset", patch_height_offset},
                 {"output_offset", output_offset}}});
    }

    // ---- Assemble the spec ----
    ProgramSpec spec;
    spec.name = "fold_multi_core_tiled_interleaved";
    spec.dataflow_buffers = {std::move(src0_dfb), std::move(src1_dfb)};
    spec.tensor_parameters = {std::move(input_param), std::move(output_param)};
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute_main)};
    // Reader/writer cover all_cores; the compute work splits into a main core group
    // and an optional cliff core group (same source, different per_core_block_cnt CTA).
    spec.work_units = {
        WorkUnitSpec{.name = "wu_main", .kernels = {READER, WRITER, COMPUTE_MAIN}, .target_nodes = core_range}};
    if (has_cliff) {
        spec.kernels.push_back(make_compute(COMPUTE_CLIFF, nblocks_per_core_cliff * tiles_per_width_dim));
        spec.work_units.push_back(WorkUnitSpec{
            .name = "wu_cliff", .kernels = {READER, WRITER, COMPUTE_CLIFF}, .target_nodes = core_range_cliff});
    }

    // ---- Run args ----
    // The compute kernels have no runtime args; an entry with empty runtime_arg_values
    // satisfies the "a KernelRunArgs for every kernel" contract trivially.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_rta)},
        ProgramRunArgs::KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_rta)},
        ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE_MAIN},
    };
    if (has_cliff) {
        run_args.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE_CLIFF});
    }
    run_args.tensor_args.insert({INPUT, input_tensor.mesh_tensor()});
    run_args.tensor_args.insert({OUTPUT, output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts fold_multi_core_row_major_interleaved(
    const Tensor& input_tensor, const Tensor& output, const uint32_t stride_h, const uint32_t stride_w) {
    auto* device = input_tensor.device();

    const uint32_t batch_size = input_tensor.logical_shape()[0];
    const uint32_t input_height = input_tensor.logical_shape()[1];
    const uint32_t input_width = input_tensor.logical_shape()[2];

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    // Total input work
    uint32_t total_patches = (batch_size * input_height * input_width) / (stride_h * stride_w);

    auto compute_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_grid_size.x;
    uint32_t num_cores_y = compute_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    log_debug(tt::LogOp, "input_tensor_shape: {}", input_tensor.padded_shape());
    log_debug(tt::LogOp, "output_tensor_shape: {}", output.padded_shape());

    uint32_t patches_per_core = tt::div_up(total_patches, num_cores_total);

    log_debug(
        tt::LogOp,
        "total_patches: {}, num_cores_total: {}, patches_per_core: {}",
        total_patches,
        num_cores_total,
        patches_per_core);

    CoreRangeSet all_cores{CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1})};
    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, true);

    uint32_t stick_nbytes = input_tensor.padded_shape()[3] * tt::datum_size(cb_data_format);
    // Align to DRAM read alignment.
    uint32_t aligned_stick_nbytes = tt::align(stick_nbytes, hal::get_dram_alignment());

    log_debug(
        tt::LogOp,
        "stick_nbytes: {}, aligned_stick_nbytes: {}, dram_alignment: {}",
        stick_nbytes,
        aligned_stick_nbytes,
        hal::get_dram_alignment());

    const bool is_l1_aligned = stick_nbytes == aligned_stick_nbytes;

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName SRC0{"src0"};
    const DFBSpecName SRC1{"src1"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = input_tensor.mesh_tensor().tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.mesh_tensor().tensor_spec()};

    // ---- Dataflow buffers ----
    const int double_buffer = 2;
    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = aligned_stick_nbytes * stride_w * stride_h,
        .num_entries = double_buffer,
        .data_format_metadata = cb_data_format,
    };

    // ---- Compile-time args ----
    // The cb_src0 / cb_src1 indices become DFB bindings; the rest are named CTAs.
    // Each kernel only declares the CTAs it actually reads.
    KernelSpec::CompileTimeArgs reader_cta{
        {"stick_nbytes", stick_nbytes},
        {"aligned_stick_nbytes", aligned_stick_nbytes},
        {"stride_h", stride_h},
        {"stride_w", stride_w},
        {"input_width", input_width},
        {"work_per_core", patches_per_core},
    };
    KernelSpec::CompileTimeArgs writer_cta{
        {"stick_nbytes", stick_nbytes},
        {"aligned_stick_nbytes", aligned_stick_nbytes},
        {"stride_h", stride_h},
        {"stride_w", stride_w},
        {"input_width", input_width},
        {"work_per_core", patches_per_core},
        {"is_l1_aligned", static_cast<uint32_t>(is_l1_aligned)},
    };

    // ---- Reader kernel (DRAM -> SRC0) ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/fold/device/kernels/dataflow/reader_dram2cb_for_rm_input.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0, .accessor_name = "src0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .compile_time_args = std::move(reader_cta),
        .runtime_arg_schema = {.runtime_arg_names = {"src_index", "curr_src_row_index"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ---- Writer kernel (SRC0 [+ SRC1 scratch] -> DRAM) ----
    // SRC1 is an intermediate L1 scratch buffer used only on the !is_l1_aligned path.
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/fold/device/kernels/dataflow/writer_cb2dram_for_rm_input.cpp",
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .compile_time_args = std::move(writer_cta),
        .runtime_arg_schema = {.runtime_arg_names = {"dst_index"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };
    // SRC0 is consumed by the writer.
    writer.dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "src0", .endpoint_type = DFBEndpointType::CONSUMER});

    ProgramSpec spec;
    spec.name = "fold_multi_core_row_major_interleaved";
    spec.dataflow_buffers = {std::move(src0_dfb)};
    spec.tensor_parameters = {std::move(input_param), std::move(output_param)};

    DataflowBufferSpec src1_dfb;
    if (!is_l1_aligned) {
        // If not L1 aligned, use a separate scratch buffer for src1.
        // It is written and read by the writer kernel only (no cross-kernel FIFO),
        // so bind it as a self-loop (PRODUCER + CONSUMER) on the writer.
        log_debug(tt::LogOp, "Using intermediate L1 scratch buffer for src1");
        src1_dfb = DataflowBufferSpec{
            .unique_id = SRC1,
            .entry_size = stick_nbytes * stride_w * stride_h,
            .num_entries = 1,
            .data_format_metadata = cb_data_format,
        };
        spec.dataflow_buffers.push_back(std::move(src1_dfb));
        writer.dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = SRC1, .accessor_name = "src1", .endpoint_type = DFBEndpointType::PRODUCER});
        writer.dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = SRC1, .accessor_name = "src1", .endpoint_type = DFBEndpointType::CONSUMER});
        // The kernel-side reference to dfb::src1 is gated on USE_SCRATCH_SRC1.
        writer.compiler_options.defines.insert({"USE_SCRATCH_SRC1", "1"});
    }

    // ---- Per-core runtime args ----
    const uint32_t output_height = input_height / stride_h;
    const uint32_t output_width = input_width / stride_w;
    const uint32_t patch_size = stride_h * stride_w;
    const uint32_t output_hw = output_height * output_width;
    uint32_t curr_patches = 0;
    uint32_t src_idx = 0;
    uint32_t dst_idx = 0;
    uint32_t src_col_offset = 0;

    Group<ProgramRunArgs::KernelRunArgs::NodeRuntimeArgs> reader_rta;
    Group<ProgramRunArgs::KernelRunArgs::NodeRuntimeArgs> writer_rta;

    for (uint32_t i = 0; i < cores.size(); i++) {
        CoreCoord core = cores[i];

        if (curr_patches < total_patches) {
            uint32_t output_offset = i * patches_per_core;
            uint32_t batch_idx = output_offset / output_hw;
            uint32_t batch_offset = output_offset % output_hw;
            uint32_t out_height = batch_offset / output_width;
            uint32_t out_width = batch_offset % output_width;

            uint32_t src_batch_offset = batch_idx * output_height * output_width * patch_size;
            uint32_t src_row_offset = out_height * stride_h * input_width;
            src_col_offset = out_width * stride_w;

            src_idx = src_batch_offset + src_row_offset + src_col_offset;
            dst_idx = output_offset;
        }

        curr_patches += patches_per_core;
        reader_rta.push_back({.node = core, .args = {{"src_index", src_idx}, {"curr_src_row_index", src_col_offset}}});
        writer_rta.push_back({.node = core, .args = {{"dst_index", dst_idx}}});
    }

    spec.kernels = {std::move(reader), std::move(writer)};
    spec.work_units = {WorkUnitSpec{.name = "wu", .kernels = {READER, WRITER}, .target_nodes = all_cores}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_rta)},
        ProgramRunArgs::KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_rta)},
    };
    run_args.tensor_args.insert({INPUT, input_tensor.mesh_tensor()});
    run_args.tensor_args.insert({OUTPUT, output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace

ttnn::device_operation::ProgramArtifacts Fold::MultiCoreDRAMFold::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    if (tensor_args.input_tensor.layout() == Layout::TILE) {
        log_debug(tt::LogOp, "Fold operation with DRAM tiled input");
        return fold_multi_core_tiled_interleaved(
            tensor_args.input_tensor, output_tensor, operation_attributes.stride_h, operation_attributes.stride_w);
    }
    log_debug(tt::LogOp, "Fold operation with DRAM row major input");
    return fold_multi_core_row_major_interleaved(
        tensor_args.input_tensor, output_tensor, operation_attributes.stride_h, operation_attributes.stride_w);
}

}  // namespace ttnn::operations::experimental::quasar
