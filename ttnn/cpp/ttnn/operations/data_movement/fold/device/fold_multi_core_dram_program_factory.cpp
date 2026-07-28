// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
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

namespace ttnn::operations::data_movement {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

// ---- Metal 2.0 named resource handles: tiled (TILE-input) sub-program ----
const DFBSpecName FOLD_T_SRC0{"fold_t_src0"};
const DFBSpecName FOLD_T_SRC1{"fold_t_src1"};
const TensorParamName FOLD_T_INPUT{"fold_t_input"};
const TensorParamName FOLD_T_OUTPUT{"fold_t_output"};
const KernelSpecName FOLD_T_READER{"fold_t_reader"};
const KernelSpecName FOLD_T_WRITER{"fold_t_writer"};
const KernelSpecName FOLD_T_COMPUTE{"fold_t_compute"};
const KernelSpecName FOLD_T_COMPUTE_CLIFF{"fold_t_compute_cliff"};

// ---- Metal 2.0 named resource handles: row-major-input sub-program ----
const DFBSpecName FOLD_RM_SRC0{"fold_rm_src0"};
const DFBSpecName FOLD_RM_SRC1{"fold_rm_src1"};
const TensorParamName FOLD_RM_INPUT{"fold_rm_input"};
const TensorParamName FOLD_RM_OUTPUT{"fold_rm_output"};
const KernelSpecName FOLD_RM_READER{"fold_rm_reader"};
const KernelSpecName FOLD_RM_WRITER{"fold_rm_writer"};

constexpr const char* READER_TILED =
    "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/reader_dram2cb_tiled.cpp";
constexpr const char* WRITER_TILED =
    "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2dram_for_tiled_input.cpp";
// Metal 2.0 fork of untilize's compute kernel. The legacy fold DRAM (tiled) factory
// file-path-instantiated untilize/device/kernels/compute/untilize.cpp, which is shared with the
// untilize op (still on the legacy API). Per the shared-kernel port strategy, the `_metal2` fork
// lives *beside the original* in the untilize op's directory (created by this port, reused by
// future Metal 2.0 consumers) — not copied into fold's tree. See METAL2_PORT_REPORT.md.
constexpr const char* COMPUTE_UNTILIZE =
    "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp";
constexpr const char* READER_RM =
    "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/reader_dram2cb_for_rm_input.cpp";
constexpr const char* WRITER_RM =
    "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/writer_cb2dram_for_rm_input.cpp";

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

    // Source DFB (formerly src0 CB) and untilized-output DFB (formerly src1 CB).
    DataflowBufferSpec src0_dfb{
        .unique_id = FOLD_T_SRC0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec src1_dfb{
        .unique_id = FOLD_T_SRC1,
        .entry_size = out_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = out_cb_data_format,
    };

    TensorParameter input_param{.unique_id = FOLD_T_INPUT, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = FOLD_T_OUTPUT, .spec = output.tensor_spec()};

    // Reader kernel: DRAM -> DFB. Input tensor is bound (its buffer address + TensorAccessorArgs
    // plumbing disappear); src0 CB index becomes the SRC0 DFB binding.
    KernelSpec reader_spec{
        .unique_id = FOLD_T_READER,
        .source = std::filesystem::path{READER_TILED},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = FOLD_T_SRC0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = FOLD_T_INPUT, .accessor_name = "src"}},
        .compile_time_args =
            {{"tiles_per_channel_dim", tiles_per_channel_dim}, {"tiles_per_width_dim", tiles_per_width_dim}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_block_id", "num_blocks"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // Writer kernel: DFB -> DRAM.
    KernelSpec writer_spec{
        .unique_id = FOLD_T_WRITER,
        .source = std::filesystem::path{WRITER_TILED},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = FOLD_T_SRC1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = FOLD_T_OUTPUT, .accessor_name = "dst"}},
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
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Compute kernel (untilize). One KernelSpec per legacy compute KernelDescriptor (main + cliff),
    // preserving the per-group block-count multiplicity. ComputeConfig set directly (Style B):
    // only fp32_dest_acc_en was set by the legacy op -> enable_32_bit_dest.
    const bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32;
    auto make_compute_cfg = [&]() {
        ComputeGen1Config cfg{.enable_32_bit_dest = fp32_dest_acc_en};
        // The compute kernel consumes SRC0. When SRC0 is Float32 and enable_32_bit_dest is set,
        // the validator requires an explicit unpack mode. Legacy set none (default) -> UnpackToSrc.
        if (fp32_dest_acc_en) {
            cfg.unpack_modes.insert({FOLD_T_SRC0, UnpackMode::UnpackToSrc});
        }
        return cfg;
    };

    auto make_compute_spec = [&](const KernelSpecName& id, uint32_t nblocks) {
        return KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path{COMPUTE_UNTILIZE},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = FOLD_T_SRC0, .accessor_name = "src", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = FOLD_T_SRC1, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"per_core_block_cnt", nblocks * tiles_per_width_dim},
                 {"per_core_block_tile_cnt", tiles_per_channel_dim}},
            .hw_config = make_compute_cfg(),
        };
    };

    KernelSpec compute_spec = make_compute_spec(FOLD_T_COMPUTE, nblocks_per_core);
    const bool cliff_present = !core_range_cliff.ranges().empty();
    KernelSpec compute_cliff_spec =
        cliff_present ? make_compute_spec(FOLD_T_COMPUTE_CLIFF, nblocks_per_core_cliff) : KernelSpec{};

    // Determine the "full" core set vs. the cliff core for runtime arg distribution.
    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        ncores_full -= 1;
        full_cores = core_range;
    }

    KernelRunArgs reader_run{.kernel = FOLD_T_READER};
    KernelRunArgs writer_run{.kernel = FOLD_T_WRITER};

    uint32_t block_start_id = 0;
    auto ncores_x = grid_size.x;
    auto ncores_y = std::ceil(static_cast<float>(ncores) / ncores_x);
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, true);

    const uint32_t patch_size = stride_h * stride_w;       // Size of each patch
    const uint32_t output_width = input_width / stride_w;  // Output width
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
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"start_block_id", block_start_id}, {"num_blocks", nblocks_per_core}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"start_block_id", block_start_id},
             {"num_blocks", nblocks_per_core},
             {"patch_height_offset", patch_height_offset},
             {"output_offset", output_offset}});
        block_start_id += nblocks_per_core;
    }

    if (ncores_full < ncores) {
        uint32_t curr_input_height_idx = block_start_id;
        uint32_t curr_output_height_idx = curr_input_height_idx / stride_h;
        uint32_t patch_height_offset = curr_input_height_idx % stride_h;
        uint32_t output_offset =
            (patch_size * curr_output_height_idx * output_width) + (patch_height_offset * stride_w);
        CoreCoord core = CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x};
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"start_block_id", block_start_id}, {"num_blocks", nblocks_per_core_cliff}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"start_block_id", block_start_id},
             {"num_blocks", nblocks_per_core_cliff},
             {"patch_height_offset", patch_height_offset},
             {"output_offset", output_offset}});
    }

    ProgramSpec spec{
        .name = "fold_multi_core_tiled_interleaved",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb, src1_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {FOLD_T_READER, FOLD_T_WRITER, FOLD_T_COMPUTE},
            .target_nodes = core_range,
        }},
    };
    if (cliff_present) {
        spec.kernels.push_back(compute_cliff_spec);
        spec.work_units.push_back(WorkUnitSpec{
            .name = "cliff",
            .kernels = {FOLD_T_READER, FOLD_T_WRITER, FOLD_T_COMPUTE_CLIFF},
            .target_nodes = core_range_cliff,
        });
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run, KernelRunArgs{.kernel = FOLD_T_COMPUTE}};
    if (cliff_present) {
        run_args.kernel_run_args.push_back(KernelRunArgs{.kernel = FOLD_T_COMPUTE_CLIFF});
    }
    run_args.tensor_args = {
        {FOLD_T_INPUT, TensorArgument{input_tensor.mesh_tensor()}},
        {FOLD_T_OUTPUT, TensorArgument{output.mesh_tensor()}},
    };

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

    const int double_buffer = 2;
    DataflowBufferSpec src0_dfb{
        .unique_id = FOLD_RM_SRC0,
        .entry_size = aligned_stick_nbytes * stride_w * stride_h,
        .num_entries = static_cast<uint32_t>(double_buffer),
        .data_format_metadata = cb_data_format,
    };

    const bool is_l1_aligned = stick_nbytes == aligned_stick_nbytes;

    // src1 is an intermediate L1 scratch, present only when the stick is not L1-aligned.
    // It is touched by a single kernel (the writer, by raw pointer) -> self-loop DFB, and its
    // binding is conditional on !is_l1_aligned (matched by a kernel-side #ifdef).
    DataflowBufferSpec src1_dfb{
        .unique_id = FOLD_RM_SRC1,
        .entry_size = stick_nbytes * stride_w * stride_h,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    };

    TensorParameter input_param{.unique_id = FOLD_RM_INPUT, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = FOLD_RM_OUTPUT, .spec = output.tensor_spec()};

    const KernelSpec::CompileTimeArgs common_cta{
        {"stick_nbytes", stick_nbytes},
        {"aligned_stick_nbytes_dram", aligned_stick_nbytes},
        {"stride_h", stride_h},
        {"stride_w", stride_w},
        {"input_width", input_width},
        {"work_per_core", patches_per_core},
    };

    // Emit the NOT_L1_ALIGNED define to the writer only when the src1 scratch is bound
    // (the define and the binding share one condition — Pattern: Conditional / optional DFB bindings).
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (!is_l1_aligned) {
        writer_defines.insert({"FOLD_RM_NOT_L1_ALIGNED", "1"});
    }

    KernelSpec reader_spec{
        .unique_id = FOLD_RM_READER,
        .source = std::filesystem::path{READER_RM},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = FOLD_RM_SRC0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = FOLD_RM_INPUT, .accessor_name = "src"}},
        .compile_time_args = common_cta,
        .runtime_arg_schema = {.runtime_arg_names = {"src_index", "curr_src_row_index"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // Writer: consumes src0; when !is_l1_aligned it also uses src1 as a self-loop scratch.
    Group<DFBBinding> writer_dfbs{
        DFBBinding{.dfb_spec_name = FOLD_RM_SRC0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER}};
    if (!is_l1_aligned) {
        writer_dfbs.push_back(DFBBinding{
            .dfb_spec_name = FOLD_RM_SRC1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER});
        writer_dfbs.push_back(DFBBinding{
            .dfb_spec_name = FOLD_RM_SRC1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER});
    }
    KernelSpec writer_spec{
        .unique_id = FOLD_RM_WRITER,
        .source = std::filesystem::path{WRITER_RM},
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = writer_dfbs,
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = FOLD_RM_OUTPUT, .accessor_name = "dst"}},
        .compile_time_args = common_cta,
        .runtime_arg_schema = {.runtime_arg_names = {"dst_index"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Per-core runtime args (name-first tables built from the legacy node-first loop).
    KernelRunArgs reader_run{.kernel = FOLD_RM_READER};
    KernelRunArgs writer_run{.kernel = FOLD_RM_WRITER};

    const uint32_t output_height = input_height / stride_h;
    const uint32_t output_width = input_width / stride_w;
    const uint32_t patch_size = stride_h * stride_w;
    const uint32_t output_hw = output_height * output_width;
    uint32_t curr_patches = 0;
    uint32_t src_idx = 0;
    uint32_t dst_idx = 0;
    uint32_t src_col_offset = 0;
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
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values, core, {{"src_index", src_idx}, {"curr_src_row_index", src_col_offset}});
        AddRuntimeArgsForNode(writer_run.runtime_arg_values, core, {{"dst_index", dst_idx}});
    }

    ProgramSpec spec{
        .name = "fold_multi_core_row_major_interleaved",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {src0_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {FOLD_RM_READER, FOLD_RM_WRITER},
            .target_nodes = all_cores,
        }},
    };
    if (!is_l1_aligned) {
        spec.dataflow_buffers.push_back(src1_dfb);
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {FOLD_RM_INPUT, TensorArgument{input_tensor.mesh_tensor()}},
        {FOLD_RM_OUTPUT, TensorArgument{output.mesh_tensor()}},
    };

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

}  // namespace ttnn::operations::data_movement
