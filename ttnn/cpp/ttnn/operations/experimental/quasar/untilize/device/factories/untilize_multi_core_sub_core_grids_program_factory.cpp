// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_sub_core_grids_program_factory.hpp"

#include <filesystem>

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreSubCoreGridsProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids.value();
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t ntiles = a.physical_volume() / TILE_HW;
    uint32_t ncores = sub_core_grids.num_cores();
    for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
        if (ntiles % ncores == 0) {
            break;
        }
        ncores--;
    }

    TT_ASSERT(ntiles % (ncores) == 0);

    uint32_t max_tiles = 1;
    uint32_t ntiles_per_block = ntiles / ncores;
    uint32_t stick_s = a.padded_shape()[-1];
    uint32_t ntiles_per_row = stick_s / TILE_WIDTH;
    uint32_t stick_size = stick_s * output.element_size();
    uint32_t ntiles_per_column = ntiles / ntiles_per_row;
    uint32_t starting_tile = ntiles_per_block;
    if (ntiles_per_row > max_tiles) {
        starting_tile = max_tiles;
    }
    ntiles_per_block = untilize_helper::get_largest_divisor(ntiles_per_row, starting_tile);
    TT_ASSERT(
        ntiles_per_row % ntiles_per_block == 0 and ntiles_per_block >= 1 and ntiles_per_block <= ntiles_per_row and
        ntiles % ntiles_per_block == 0);

    uint32_t nblocks = (ntiles / ntiles_per_block);

    auto cores = corerange_to_cores(sub_core_grids, ncores, true);
    auto all_cores = num_cores_to_corerangeset_in_subcoregrids(cores[0], ncores, sub_core_grids, true);
    uint32_t nblocks_per_core = nblocks / ncores;

    uint32_t num_input_tiles = ntiles_per_block * 2;
    uint32_t num_output_tiles = ntiles_per_block * 2;
    uint32_t tile_width_size = TILE_WIDTH * output.element_size();

    // ---- Resource names ----
    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- DataflowBuffers (legacy CB c_0 / c_16) ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Reader (Metal 2.0 fork of reader_unary_interleaved_start_id) ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/dataflow/"
                                        "reader_unary_interleaved_start_id_metal2.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(a.device()->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Writer (Metal 2.0 fork of writer_..._interleaved_parallel_columns) ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/dataflow/"
                                        "writer_unary_stick_layout_split_rows_interleaved_parallel_columns_metal2.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"stick_size", stick_size}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks", "num_tiles_per_core", "tile_width_size", "start_stick_id", "offset_within_stick"}},
        .hw_config =
            ttnn::create_writer_datamovement_config(a.device()->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Compute (Metal 2.0 fork of untilize; uniform across all sub-core-grid cores) ----
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(a.device()->arch(), compute_config);
    if (fp32_dest_acc_en) {
        std::visit(
            [&](auto& c) { c.unpack_to_dest_mode.emplace(IN, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32); },
            compute_hw);
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/compute/untilize_metal2.cpp"),
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args = {{"per_core_block_cnt", nblocks_per_core}, {"per_core_block_tile_cnt", ntiles_per_block}},
        .hw_config = compute_hw,
    };

    Group<KernelSpec> kernels = {reader, writer, compute};
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{.name = "wu", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}};

    // ---- Per-core runtime args (uniform: no cliff for sub-core-grids) ----
    uint32_t tile_start_id = 0;
    uint32_t offset_within_stick = 0;
    auto nsticks_per_core = ntiles_per_column * TILE_HEIGHT;
    uint32_t ntiles_per_core = ntiles_per_block * nblocks_per_core;

    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;

    for (const auto& core : cores) {
        AddRuntimeArgsForNode(
            reader_node_args,
            core,
            {
                {"num_pages", ntiles_per_core},
                {"start_id", tile_start_id},
            });
        AddRuntimeArgsForNode(
            writer_node_args,
            core,
            {
                {"num_sticks", nsticks_per_core},
                {"num_tiles_per_core", ntiles_per_core},
                {"tile_width_size", tile_width_size},
                {"start_stick_id", 0u},
                {"offset_within_stick", offset_within_stick},
            });
        tile_start_id += ntiles_per_core;
        offset_within_stick += ntiles_per_core * tile_width_size;
    }

    ProgramSpec spec{
        .name = "untilize_multi_core_sub_core_grids",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    KernelRunArgs reader_args{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)};
    KernelRunArgs writer_args{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)};
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args)};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
