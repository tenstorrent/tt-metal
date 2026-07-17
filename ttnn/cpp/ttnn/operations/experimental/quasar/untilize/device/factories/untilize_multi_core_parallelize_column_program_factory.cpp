// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_parallelize_column_program_factory.hpp"

#include <filesystem>

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreParallelizeColumnProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

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
    uint32_t tile_width_size = TILE_WIDTH * output.element_size();

    // ---- Resource names ----
    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};

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
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
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
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Compute (Metal 2.0 fork of untilize; full + cliff) ----
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    auto make_compute_hw = [&]() -> ComputeHardwareConfig {
        ttnn::ComputeKernelConfig hw{
            .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
        ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(device->arch(), hw);
        if (fp32_dest_acc_en) {
            std::visit(
                [&](auto& c) { c.unpack_modes.emplace(IN, tt::tt_metal::UnpackMode::UnpackToDest); }, compute_hw);
        }
        return compute_hw;
    };
    const std::filesystem::path compute_source(
        "ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/compute/untilize_metal2.cpp");
    auto make_compute = [&](const KernelSpecName& id, uint32_t nblocks_cnt) {
        return KernelSpec{
            .unique_id = id,
            .source = compute_source,
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args = {{"per_core_block_cnt", nblocks_cnt}, {"per_core_block_tile_cnt", ntiles_per_block}},
            .hw_config = make_compute_hw(),
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    if (!core_range.ranges().empty()) {
        kernels.push_back(make_compute(COMPUTE_FULL, nblocks_per_core));
        work_units.push_back(
            WorkUnitSpec{.name = "wu_full", .kernels = {READER, WRITER, COMPUTE_FULL}, .target_nodes = core_range});
    }
    if (!core_range_cliff.ranges().empty()) {
        kernels.push_back(make_compute(COMPUTE_CLIFF, nblocks_per_core_cliff));
        work_units.push_back(WorkUnitSpec{
            .name = "wu_cliff", .kernels = {READER, WRITER, COMPUTE_CLIFF}, .target_nodes = core_range_cliff});
    }

    // ---- Per-core runtime args (mirror legacy distribution) ----
    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        ncores_full -= 1;
        full_cores = core_range;
    }
    uint32_t tile_start_id = 0;
    uint32_t offset_within_stick = 0;
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);
    auto nsticks_per_core = ntiles_per_column * TILE_HEIGHT;

    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;

    for (const auto& core : cores) {
        if (!full_cores.contains(core)) {
            continue;
        }
        uint32_t ntiles_per_core = ntiles_per_block * nblocks_per_core;
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
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        CoreCoord core = row_major ? CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x}
                                   : CoreCoord{ncores_full / ncores_y, ncores_full % ncores_y};
        uint32_t ntiles_per_core_cliff = ntiles_per_block * nblocks_per_core_cliff;
        AddRuntimeArgsForNode(
            reader_node_args,
            core,
            {
                {"num_pages", ntiles_per_core_cliff},
                {"start_id", tile_start_id},
            });
        // NOTE: the legacy cliff writer passed an extra positional `stick_size` arg the kernel never
        // read, mis-aligning all subsequent cliff-core writer args (a latent bug — the full-core path
        // is correct). Metal 2.0 named args bind by name, so this port emits the cliff writer with the
        // same named args the kernel actually reads, correcting the cliff-core behavior. Flagged for
        // owner review in METAL2_PORT_REPORT.md.
        AddRuntimeArgsForNode(
            writer_node_args,
            core,
            {
                {"num_sticks", nsticks_per_core},
                {"num_tiles_per_core", ntiles_per_core_cliff},
                {"tile_width_size", tile_width_size},
                {"start_stick_id", 0u},
                {"offset_within_stick", offset_within_stick},
            });
    }

    ProgramSpec spec{
        .name = "untilize_multi_core_parallelize_column",
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
