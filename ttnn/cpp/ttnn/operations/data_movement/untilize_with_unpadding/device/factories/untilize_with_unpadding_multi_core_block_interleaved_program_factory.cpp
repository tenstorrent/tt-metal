// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp"

#include <algorithm>
#include <filesystem>
#include <vector>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName IN_DFB{"in"};    // legacy c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy c_16
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_FULL_KERNEL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF_COL_ROW_KERNEL{"compute_cliff_col_row"};
    const KernelSpecName COMPUTE_CLIFF_ROW_KERNEL{"compute_cliff_row"};
    const KernelSpecName COMPUTE_CLIFF_COL_KERNEL{"compute_cliff_col"};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t max_l1_size = operations::data_movement::get_max_l1_space(a);
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / TILE_WIDTH;
    uint32_t num_tiles_per_col = a.padded_shape()[-2] / TILE_HEIGHT;

    uint32_t num_blocks = (a.padded_shape()[-1] * a.padded_shape()[-2]) / (TILE_HEIGHT * TILE_WIDTH);
    uint32_t cb_block_size_limit = max_l1_size / (input_single_tile_size + output_single_tile_size);

    auto
        [ncores,
         all_cores,
         core_range,
         cliff_row_core_range,
         cliff_col_core_range,
         cliff_col_row_core_range,
         nblocks_per_core,
         single_block_size,
         single_block_size_cliff_row,
         single_block_size_cliff_col,
         has_cliff_row,
         has_cliff_col,
         full_cores_per_row,
         full_cores_per_col,
         single_sub_block_size] =
            ttnn::split_blocks_for_tilize_wh(
                available_grid, num_blocks, num_tiles_per_row, num_tiles_per_col, cb_block_size_limit);

    if (single_sub_block_size > 0 && single_block_size % single_sub_block_size) {
        TT_FATAL(false, "single_block_size is not divided by single_sub_block_size");
    }

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);
    uint32_t padded_row_size_bytes;
    uint32_t unpadded_row_size_bytes;

    uint32_t el_size;
    if (a.dtype() == DataType::BFLOAT8_B) {
        padded_row_size_bytes = input_shape[-1] * output.element_size();
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        padded_row_size_bytes = input_shape[-1] * a.element_size();
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
        el_size = a.element_size();
    }

    // ------------------------------------------------------------------------
    // Dataflow buffers (formerly CB c_0 / c_16). The legacy factory emitted a CB pair per non-empty
    // sub-region with a per-region size (single_sub_block_size for full/cliff-col,
    // single_block_size_cliff_row for the row-cliff regions). A DataflowBuffer carries one size set
    // at construction, so the IN/OUT DFBs are sized to the MAX present region size — a superset of
    // each legacy per-region buffer (the FIFO depth need only be >= the block consumed at once).
    // ------------------------------------------------------------------------
    uint32_t max_block_tiles = 0;
    if (!core_range.empty()) {
        max_block_tiles = std::max(max_block_tiles, single_sub_block_size);
    }
    if (has_cliff_col && has_cliff_row) {
        max_block_tiles = std::max(max_block_tiles, single_block_size_cliff_row);
    }
    if (has_cliff_row) {
        max_block_tiles = std::max(max_block_tiles, single_block_size_cliff_row);
    }
    if (has_cliff_col) {
        max_block_tiles = std::max(max_block_tiles, single_sub_block_size);
    }

    DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = max_block_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = max_block_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    TensorParameter input_param{.unique_id = INPUT_TENSOR, .spec = input_mesh.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT_TENSOR, .spec = output_mesh.tensor_spec()};

    uint32_t num_tiles_2d = a.padded_shape()[-1] * a.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    uint32_t total_num_rows = output.logical_shape()[-2];

    // ------------------------------------------------------------------------
    // Kernels. Reader/writer span all working cores; compute is one KernelSpec per non-empty
    // sub-region (preserving the legacy per-region CTA multiplicity), each in its own WorkUnitSpec.
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                        "reader_unary_interleaved_wh_multicore_metal2.cpp"},
        .dfb_bindings = {ProducerOf(IN_DFB, "in")},
        .tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args =
            {{"num_tiles_per_2d", num_tiles_2d},
             {"third_dim", third_dim},
             {"total_tiles_per_row", total_tiles_per_row}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_id", "single_block_size_row_arg", "single_block_size_col_arg"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                "writer_unary_stick_layout_wh_multicore_metal2.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .compile_time_args =
            {{"total_num_rows", total_num_rows},
             {"third_dim", third_dim},
             {"tile_height", TILE_HEIGHT},
             {"unpadded_X_size", unpadded_row_size_bytes}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"width_size",
                  "start_row_id",
                  "start_column_id",
                  "single_block_size_row_arg",
                  "single_block_size_col_arg",
                  "sub_block_width_size",
                  "single_sub_block_size_row_arg"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    uint32_t single_sub_block_size_wh = single_block_size * single_block_size / single_sub_block_size;
    uint32_t single_sub_block_size_cliff_col_wh =
        single_block_size_cliff_col * single_block_size / single_sub_block_size;

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_defines.insert({"DST_ACCUM_MODE", "1"});
    }

    ComputeHardwareConfig::UnpackToDestModes unpack_to_dest_modes;
    if (fp32_dest_acc_en) {
        unpack_to_dest_modes.insert({IN_DFB, UnpackToDestMode::UnpackToDestFp32});
    }

    // Each compute KernelSpec keeps its sub-region's per-group CTAs (block_size_col, block_size_row)
    // — no CTA->RTA demotion (preserves compile-time loop unrolling).
    auto make_compute = [&](const KernelSpecName& id, uint32_t block_size_col, uint32_t block_size_row) {
        return KernelSpec{
            .unique_id = id,
            .source =
                std::filesystem::path{
                    "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh_metal2.cpp"},
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings = {ConsumerOf(IN_DFB, "in"), ProducerOf(OUT_DFB, "out")},
            .compile_time_args =
                {{"block_size_col", block_size_col}, {"block_size_row", block_size_row}, {"third_dim", third_dim}},
            .hw_config =
                ComputeHardwareConfig{
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .unpack_to_dest_mode = unpack_to_dest_modes,
                },
        };
    };

    std::vector<KernelSpec> kernels = {reader_spec, writer_spec};
    std::vector<WorkUnitSpec> work_units;
    if (!core_range.empty()) {
        kernels.push_back(make_compute(COMPUTE_FULL_KERNEL, single_sub_block_size_wh, single_sub_block_size));
        work_units.push_back(WorkUnitSpec{
            .name = "untilize_with_unpadding_block_full",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_FULL_KERNEL},
            .target_nodes = core_range,
        });
    }
    if (has_cliff_col && has_cliff_row) {
        kernels.push_back(
            make_compute(COMPUTE_CLIFF_COL_ROW_KERNEL, single_block_size_cliff_col, single_block_size_cliff_row));
        work_units.push_back(WorkUnitSpec{
            .name = "untilize_with_unpadding_block_cliff_col_row",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_CLIFF_COL_ROW_KERNEL},
            .target_nodes = cliff_col_row_core_range,
        });
    }
    if (has_cliff_row) {
        kernels.push_back(make_compute(COMPUTE_CLIFF_ROW_KERNEL, single_block_size, single_block_size_cliff_row));
        work_units.push_back(WorkUnitSpec{
            .name = "untilize_with_unpadding_block_cliff_row",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_CLIFF_ROW_KERNEL},
            .target_nodes = cliff_row_core_range,
        });
    }
    if (has_cliff_col) {
        kernels.push_back(
            make_compute(COMPUTE_CLIFF_COL_KERNEL, single_sub_block_size_cliff_col_wh, single_sub_block_size));
        work_units.push_back(WorkUnitSpec{
            .name = "untilize_with_unpadding_block_cliff_col",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_CLIFF_COL_KERNEL},
            .target_nodes = cliff_col_core_range,
        });
    }

    // ------------------------------------------------------------------------
    // Per-node runtime args. Replicates the legacy per-core work-distribution loop verbatim; the
    // src/dst buffer-address RTAs are dropped (carried by the TensorAccessor bindings).
    // ------------------------------------------------------------------------
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    reader_run.runtime_arg_values.reserve(ncores);
    writer_run.runtime_arg_values.reserve(ncores);

    const auto& cores = corerange_to_cores(available_grid);
    uint32_t start_row_id = 0;
    uint32_t start_column_id = 0;
    uint32_t tile_start_id = 0;
    uint32_t single_block_size_row_arg;
    uint32_t single_block_size_col_arg;
    uint32_t single_sub_block_size_row_arg;

    uint32_t total_row_cores = full_cores_per_row;
    if (has_cliff_row) {
        total_row_cores++;
    }
    uint32_t cores_col_count = 1;

    for (uint32_t i = 0; i < ncores; ++i) {
        const NodeCoord node = cores[i];

        if (has_cliff_col && has_cliff_row && i == ncores - 1) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size_cliff_col;
            single_sub_block_size_row_arg = single_block_size_cliff_row;

        } else if (has_cliff_row && i != 0 && ((i + 1) % (full_cores_per_row + 1)) == 0) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size;
            single_sub_block_size_row_arg = single_block_size_cliff_row;

        } else if (i < total_row_cores * full_cores_per_col) {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size;
            single_sub_block_size_row_arg = single_sub_block_size;

        } else {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size_cliff_col;
            single_sub_block_size_row_arg = single_sub_block_size;
        }

        reader_run.runtime_arg_values.push_back(
            {node,
             {{"start_id", tile_start_id},
              {"single_block_size_row_arg", single_block_size_row_arg},
              {"single_block_size_col_arg", single_block_size_col_arg}}});

        writer_run.runtime_arg_values.push_back(
            {node,
             {{"width_size", TILE_WIDTH * el_size * single_block_size_row_arg},
              {"start_row_id", start_row_id},
              {"start_column_id", start_column_id},
              {"single_block_size_row_arg", single_block_size_row_arg},
              {"single_block_size_col_arg", single_block_size_col_arg},
              {"sub_block_width_size", TILE_WIDTH * el_size * single_sub_block_size_row_arg},
              {"single_sub_block_size_row_arg", single_sub_block_size_row_arg}}});

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * TILE_WIDTH * el_size);
        start_column_id = end_column_id % padded_row_size_bytes;
        if (end_column_id % padded_row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * TILE_HEIGHT;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_block_interleaved",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_mesh)}}, {OUTPUT_TENSOR, TensorArgument{std::cref(output_mesh)}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
