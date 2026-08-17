// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

const DFBSpecName WH_SRC0_DFB{"wh_src0"};      // c_0
const DFBSpecName WH_OUT_DFB{"wh_out"};        // c_16
const DFBSpecName WH_TILIZE_DFB{"wh_tilize"};  // c_24 (row-major tilize intermediate)
const TensorParamName WH_INPUT{"wh_input"};
const TensorParamName WH_OUTPUT{"wh_output"};
const KernelSpecName WH_READER{"wh_reader"};
const KernelSpecName WH_COMPUTE{"wh_compute"};
const KernelSpecName WH_WRITER{"wh_writer"};

ttnn::device_operation::ProgramArtifacts build_wh_tiled(const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    IDevice* device = input_tensor.device();

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32 ||
                            src0_cb_data_format == tt::DataFormat::Int32 ||
                            src0_cb_data_format == tt::DataFormat::UInt32;

    auto grid = device->compute_with_storage_grid_size();

    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(grid, num_tensor_tiles);

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "transpose_wh";

    spec.tensor_parameters = {
        TensorParameter{.unique_id = WH_INPUT, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = WH_OUTPUT, .spec = output_tensor.tensor_spec()},
    };

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WH_SRC0_DFB,
        .entry_size = src0_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = src0_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WH_OUT_DFB,
        .entry_size = dst_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = dst_cb_data_format,
    });

    ComputeGen1Config compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        // Legacy set unpack_to_dest_mode[c_0] = UnpackToDestFp32 for the Float32 input CB.
        compute_cfg.unpack_modes.emplace(WH_SRC0_DFB, UnpackMode::UnpackToDest);
    }

    KernelSpec reader{
        .unique_id = WH_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_start_id.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = WH_SRC0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = WH_INPUT, .accessor_name = "src"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"num_tiles", "start_id", "start_ht", "start_wt", "Ht", "Wt", "HtWt"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec compute{
        .unique_id = WH_COMPUTE,
        // Lent kernel: transpose_wh.cpp is cross-op shared with legacy peers (permute,
        // nlp_create_qkv_heads{,_boltz,_vit}, split_query_key_value_and_split_heads), so the legacy
        // source must stay non-Metal-2.0 for them; this factory binds the Metal 2.0 fork beside it.
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/"
            "transpose_wh_metal2.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = WH_SRC0_DFB, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = WH_OUT_DFB, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"NHtWt"}},
        .hw_config = ComputeHardwareConfig{compute_cfg},
    };

    KernelSpec writer{
        .unique_id = WH_WRITER,
        // Borrowed kernel: bound from the Metal 2.0 fork that lives beside its legacy original
        // under eltwise/unary. The fork's binding names are this factory's constraint.
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = WH_OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = WH_OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    spec.kernels = {reader, compute, writer};
    spec.work_units = {
        WorkUnitSpec{.name = "main", .kernels = {WH_READER, WH_COMPUTE, WH_WRITER}, .target_nodes = all_cores}};

    // ---- ProgramRunArgs ----
    auto input_shape = input_tensor.padded_shape();
    uint32_t W = input_shape[3], H = input_shape[2];
    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = WH_READER};
    KernelRunArgs compute_run{.kernel = WH_COMPUTE};
    KernelRunArgs writer_run{.kernel = WH_WRITER};

    auto cores = corerange_to_cores(all_cores, std::nullopt);

    uint32_t num_tiles_read = 0;
    for (const auto& core : cores) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t h = num_tiles_read % Ht;
        uint32_t w = num_tiles_read / Ht % Wt;

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core},
             {"start_id", tt::round_down(num_tiles_read, HtWt) + (h * Wt) + w},
             {"start_ht", h},
             {"start_wt", w},
             {"Ht", Ht},
             {"Wt", Wt},
             {"HtWt", HtWt}});
        AddRuntimeArgsForNode(compute_run.runtime_arg_values, core, {{"NHtWt", num_tiles_per_core}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_read}});

        num_tiles_read += num_tiles_per_core;
    }

    run_args.kernel_run_args = {reader_run, compute_run, writer_run};
    run_args.tensor_args.emplace(WH_INPUT, TensorArgument{input_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(WH_OUTPUT, TensorArgument{output_tensor.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts build_wh_rm(const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    IDevice* device = input_tensor.device();

    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t NC = input_tensor.logical_shape()[1] * input_tensor.logical_shape()[0];

    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32 ||
                            src0_cb_data_format == tt::DataFormat::Int32 ||
                            src0_cb_data_format == tt::DataFormat::UInt32;

    auto grid = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        split_work_to_cores(grid, NC);

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "transpose_wh_rm";

    spec.tensor_parameters = {
        TensorParameter{.unique_id = WH_INPUT, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = WH_OUTPUT, .spec = output_tensor.tensor_spec()},
    };

    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WH_SRC0_DFB,
        .entry_size = src0_single_tile_size,
        .num_entries = wt * 2,
        .data_format_metadata = src0_cb_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WH_OUT_DFB,
        .entry_size = dst_single_tile_size,
        .num_entries = ht * 2,
        .data_format_metadata = dst_cb_data_format,
    });
    // Tilize intermediate (legacy c_24): produced and consumed within the compute kernel → self-loop.
    // (Legacy also allocated a dead c_25 "im2" here; no kernel bound by this factory references it, so
    // it is dropped rather than carried over.)
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WH_TILIZE_DFB,
        .entry_size = src0_single_tile_size,
        .num_entries = ht * wt,
        .data_format_metadata = src0_cb_data_format,
    });

    ComputeGen1Config compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        // Legacy set unpack_to_dest_mode[c_0] and [c_24] = UnpackToDestFp32 for the Float32 path.
        compute_cfg.unpack_modes.emplace(WH_SRC0_DFB, UnpackMode::UnpackToDest);
        compute_cfg.unpack_modes.emplace(WH_TILIZE_DFB, UnpackMode::UnpackToDest);
    }

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (input_tensor.dtype() == DataType::UINT32 || input_tensor.dtype() == DataType::INT32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }

    KernelSpec reader{
        .unique_id = WH_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_start_id_rm.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = WH_SRC0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = WH_INPUT, .accessor_name = "src"}},
        .compile_time_args =
            {{"Ht", ht},
             {"H_per_tile", H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT},
             {"H_per_tile_last", H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT},
             {"Wt", wt},
             {"W", W},
             {"HtWt", ht * wt},
             {"W_size_bytes", W * input_tensor.element_size()},
             {"l1_write_offset_bytes", wt * input_tensor.element_size() * TILE_WIDTH}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_hw_blocks_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec compute{
        .unique_id = WH_COMPUTE,
        // Forked from transpose_wh_rm.cpp (shared top-level entry point with the gated
        // WH-Sharded-RM factory); the SHARDED branch is stripped in the fork.
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/"
            "transpose_wh_rm_metal2.cpp",
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = WH_SRC0_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = WH_TILIZE_DFB, .accessor_name = "tilize", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = WH_TILIZE_DFB, .accessor_name = "tilize", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = WH_OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args = {{"Ht", ht}, {"Wt", wt}, {"HtWt", ht * wt}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_hw_blocks_per_core"}},
        .hw_config = ComputeHardwareConfig{compute_cfg},
    };

    KernelSpec writer{
        .unique_id = WH_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "writer_unary_transpose_wh_interleaved_start_id_rm.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = WH_OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = WH_OUTPUT, .accessor_name = "dst"}},
        .compile_time_args =
            {{"Ht", ht},
             {"H", H},
             {"Wt", wt},
             {"W_per_tile", W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH},
             {"W_per_tile_last", W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH},
             {"HtWt", ht * wt},
             {"H_size_bytes", H * output_tensor.element_size()},
             {"l1_read_offset_bytes", ht * output_tensor.element_size() * TILE_HEIGHT}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_hw_blocks_per_core"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    spec.kernels = {reader, compute, writer};
    spec.work_units = {
        WorkUnitSpec{.name = "main", .kernels = {WH_READER, WH_COMPUTE, WH_WRITER}, .target_nodes = all_cores}};

    // ---- ProgramRunArgs ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_run{.kernel = WH_READER};
    KernelRunArgs compute_run{.kernel = WH_COMPUTE};
    KernelRunArgs writer_run{.kernel = WH_WRITER};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t num_sticks_read = 0, num_sticks_write = 0;
    for (const auto& core : cores) {
        uint32_t num_hw_blocks_per_core;
        if (core_group_1.contains(core)) {
            num_hw_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_hw_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"start_id", num_sticks_read}, {"num_hw_blocks_per_core", num_hw_blocks_per_core}});
        AddRuntimeArgsForNode(
            compute_run.runtime_arg_values, core, {{"num_hw_blocks_per_core", num_hw_blocks_per_core}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"start_id", num_sticks_write}, {"num_hw_blocks_per_core", num_hw_blocks_per_core}});

        num_sticks_read += num_hw_blocks_per_core * H;
        num_sticks_write += num_hw_blocks_per_core * W;
    }

    run_args.kernel_run_args = {reader_run, compute_run, writer_run};
    run_args.tensor_args.emplace(WH_INPUT, TensorArgument{input_tensor.mesh_tensor()});
    run_args.tensor_args.emplace(WH_OUTPUT, TensorArgument{output_tensor.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TransposeWHProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");
    TT_ASSERT(output_tensor.buffer() != nullptr, "Output buffer should be allocated on device!");

    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;
    return row_major ? build_wh_rm(tensor_args, output_tensor) : build_wh_tiled(tensor_args, output_tensor);
}

}  // namespace ttnn::prim
