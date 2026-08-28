// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_parallelize_column_program_factory.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/untilize/device/untilize_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreParallelizeColumnProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const DFBSpecName SRC0{"src0"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};

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

    IDevice* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t ntiles = a.physical_volume() / TILE_HW;
    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = grid_size.y;

    ncores_x = untilize_helper::get_largest_divisor(ntiles, ncores_x);
    ncores_y = untilize_helper::get_largest_divisor(ntiles, ncores_y, ncores_x);

    TT_ASSERT(ntiles % (ncores_x * ncores_y) == 0);
    uint32_t ntiles_per_block = ntiles / (ncores_x * ncores_y);

    // TODO increase block size to increase untilize performance, currently each untilize block is a single tile
    uint32_t max_tiles = 1;

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

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(CoreCoord(ncores_x, ncores_y), nblocks);

    bool row_major = true;

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
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
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
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.insert({"DST_ACCUM_MODE", "1"});
    }
    // One KernelSpec per legacy compute KernelDescriptor (full + cliff), preserving the per-group
    // block-count multiplicity across disjoint WorkUnitSpecs.
    auto make_compute = [&](const KernelSpecName& id, uint32_t per_core_block_cnt) {
        ComputeGen1Config compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            compute_cfg.unpack_modes.insert({SRC0, UnpackMode::UnpackToDest});
        }
        return KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path{COMPUTE_SRC},
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "src", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"per_core_block_cnt", per_core_block_cnt}, {"per_core_block_tile_cnt", ntiles_per_block}},
            .hw_config = std::move(compute_cfg),
        };
    };

    const bool full_present = !core_range.ranges().empty();
    const bool cliff_present = !core_range_cliff.ranges().empty();

    ProgramSpec spec{
        .name = "untilize_multi_core_parallelize_column",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {src0_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
    };
    if (full_present) {
        spec.kernels.push_back(make_compute(COMPUTE_FULL, nblocks_per_core));
        spec.work_units.push_back(
            WorkUnitSpec{.name = "wu_full", .kernels = {READER, WRITER, COMPUTE_FULL}, .target_nodes = core_range});
    }
    if (cliff_present) {
        spec.kernels.push_back(make_compute(COMPUTE_CLIFF, nblocks_per_core_cliff));
        spec.work_units.push_back(WorkUnitSpec{
            .name = "wu_cliff", .kernels = {READER, WRITER, COMPUTE_CLIFF}, .target_nodes = core_range_cliff});
    }

    // Per-core runtime args (name-first tables built from the legacy node-first loop).
    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};

    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        // unequal case with cliff
        ncores_full -= 1;
        full_cores = core_range;
    }
    uint32_t tile_start_id = 0;
    uint32_t offset_within_stick = 0;
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);
    auto nsticks_per_core = ntiles_per_column * TILE_HEIGHT;
    const uint32_t tile_width_size = static_cast<uint32_t>(TILE_WIDTH * output.element_size());

    for (const auto& core : cores) {
        if (!full_cores.contains(core)) {
            continue;
        }
        uint32_t ntiles_per_core = ntiles_per_block * nblocks_per_core;
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values, core, {{"num_pages", ntiles_per_core}, {"start_id", tile_start_id}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"num_sticks", nsticks_per_core},
             {"num_tiles_per_core", ntiles_per_core},
             {"tile_width_size", tile_width_size},
             {"start_stick_id", 0u},
             {"offset_within_stick", offset_within_stick}});
        tile_start_id += ntiles_per_core;
        offset_within_stick += ntiles_per_core * TILE_WIDTH * output.element_size();
    }
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        CoreCoord core = row_major ? CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x}
                                   : CoreCoord{ncores_full / ncores_y, ncores_full % ncores_y};
        uint32_t ntiles_per_core_cliff = ntiles_per_block * nblocks_per_core_cliff;
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values, core, {{"num_pages", ntiles_per_core_cliff}, {"start_id", tile_start_id}});
        // NOTE: The legacy factory passed the cliff writer 7 positional RTAs (an extra stick_size at
        // index 2) while the writer kernel reads 6 — a latent, likely-unreached misalignment bug. The
        // named-argument model has no slot for the extra value, so it is dropped and the cliff core now
        // receives the same well-formed arg set as a full core (with the cliff tile count). This is not
        // a deliberate fix; it is what a faithful name-based translation produces.
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"num_sticks", nsticks_per_core},
             {"num_tiles_per_core", ntiles_per_core_cliff},
             {"tile_width_size", tile_width_size},
             {"start_stick_id", 0u},
             {"offset_within_stick", offset_within_stick}});
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
