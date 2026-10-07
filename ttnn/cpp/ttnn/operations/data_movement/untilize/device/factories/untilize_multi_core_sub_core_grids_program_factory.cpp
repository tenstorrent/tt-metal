// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt_stl/reflection.hpp>
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/allocator.hpp>
#include "untilize_multi_core_sub_core_grids_program_factory.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreSubCoreGridsProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids.value();
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const DFBSpecName SRC0{"src0"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    constexpr const char* READER_SRC =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp";
    constexpr const char* WRITER_SRC =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp";
    constexpr const char* COMPUTE_SRC =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp";

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

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

    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = output_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path{READER_SRC},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path{WRITER_SRC},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .compile_time_args = {{"stick_size", stick_size}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks", "num_tiles_per_core", "tile_width_size", "start_stick_id", "offset_within_stick"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    ComputeHardwareConfig compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        compute_cfg.unpack_modes.insert({SRC0, UnpackMode::UnpackToDest});
    }
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.insert({"DST_ACCUM_MODE", "1"});
    }
    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path{COMPUTE_SRC},
        .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "src", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args = {{"per_core_block_cnt", nblocks_per_core}, {"per_core_block_tile_cnt", ntiles_per_block}},
        .hw_config = std::move(compute_cfg),
    };

    ProgramSpec spec{
        .name = "untilize_multi_core_sub_core_grids",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {src0_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = all_cores,
        }},
    };

    // Per-core runtime args (name-first tables built from the legacy node-first loop).
    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};

    uint32_t tile_start_id = 0;
    uint32_t offset_within_stick = 0;
    auto nsticks_per_core = ntiles_per_column * TILE_HEIGHT;
    for (auto core : cores) {
        auto ntiles_per_core = ntiles_per_block * nblocks_per_core;
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values, core, {{"num_pages", ntiles_per_core}, {"start_id", tile_start_id}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"num_sticks", nsticks_per_core},
             {"num_tiles_per_core", ntiles_per_core},
             {"tile_width_size", static_cast<uint32_t>(TILE_WIDTH * output.element_size())},
             {"start_stick_id", 0u},
             {"offset_within_stick", offset_within_stick}});
        tile_start_id += ntiles_per_core;
        offset_within_stick += ntiles_per_core * TILE_WIDTH * output.element_size();
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run};
    run_args.tensor_args = {
        {INPUT, TensorArgument{a.mesh_tensor()}},
        {OUTPUT, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}
}  // namespace ttnn::prim
