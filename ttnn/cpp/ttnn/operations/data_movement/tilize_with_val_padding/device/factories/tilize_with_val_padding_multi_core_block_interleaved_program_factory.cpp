// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_block_interleaved_program_factory.hpp"

#include <cmath>
#include <algorithm>
#include <vector>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts TilizeWithValPaddingMultiCoreBlockInterleavedFactory::create_program_artifacts(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    const auto& input_mesh = a.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();

    // Metal 2.0 named resource handles.
    const DFBSpecName SRC_DFB{"src"};      // legacy c_0
    const DFBSpecName STAGE_DFB{"stage"};  // legacy c_1: reader-local alignment staging scratch
    const DFBSpecName OUT_DFB{"out"};      // legacy c_16
    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};
    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t max_l1_size = operations::data_movement::get_max_l1_space(a);
    uint32_t num_tiles_per_col = output.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = output.padded_shape()[-1] / TILE_WIDTH;
    uint32_t num_blocks = (output.padded_shape()[-1] * output.padded_shape()[-2]) / (TILE_HEIGHT * TILE_WIDTH);
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

    uint32_t unpadded_row_size_bytes = a.padded_shape()[-1] * a.element_size();
    uint32_t padded_row_size_bytes = output.padded_shape()[-1] * a.element_size();

    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    // ------------------------------------------------------------------------
    // DataflowBuffers. The legacy factory sized c_0/c_1/c_16 per core group; a DFBSpec carries a
    // single program-scope size, so we size each DFB to the max block across active groups (still
    // bounded by cb_block_size_limit). Per-core push/wait counts come from the per-node runtime
    // args and per-group compute CTAs, so correctness is unaffected.
    // ------------------------------------------------------------------------
    uint32_t max_block_tiles = single_sub_block_size;
    if (has_cliff_row) {
        max_block_tiles = std::max(max_block_tiles, single_block_size_cliff_row);
    }
    uint32_t input_row_bytes = input_single_tile_size / TILE_HEIGHT;
    uint32_t temp_cb_size = input_row_bytes * max_block_tiles + 2 * dram_alignment;

    DataflowBufferSpec stage_dfb_spec{
        .unique_id = STAGE_DFB,
        .entry_size = temp_cb_size,
        .num_entries = 1,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec src_dfb_spec{
        .unique_id = SRC_DFB,
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

    // ------------------------------------------------------------------------
    // Reader / writer (uniform CTAs; per-core values are runtime args).
    // ------------------------------------------------------------------------
    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);
    uint32_t num_tiles_2d = output.padded_shape()[-1] * output.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    uint32_t tile_height = output.tensor_spec().tile().get_height();
    uint32_t total_num_rows = a.logical_shape()[-2];
    if (output.padded_shape()[-2] > tt::round_up(total_num_rows, tile_height)) {
        total_num_rows = output.padded_shape()[-2];
    }

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
                "reader_unary_pad_multicore_both_dims.cpp"},
        .dfb_bindings = {ProducerOf(SRC_DFB, "in"), ProducerOf(STAGE_DFB, "stage"), ConsumerOf(STAGE_DFB, "stage")},
        .tensor_bindings = {{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"}},
        .compile_time_args =
            {{"total_num_rows", total_num_rows},
             {"third_dim", third_dim},
             {"tile_height", tile_height},
             {"element_size", a.element_size()},
             {"unpadded_X_size", unpadded_row_size_bytes},
             {"dram_alignment", dram_alignment}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"pad_value",
                  "width_size",
                  "start_row_id",
                  "start_column_id",
                  "single_block_size_row_arg",
                  "single_block_size_col_arg",
                  "sub_block_width_size",
                  "single_sub_block_size_row_arg"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_wh.cpp"},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"}},
        .compile_time_args =
            {{"num_tiles_per_2d", num_tiles_2d},
             {"third_dim", third_dim},
             {"total_tiles_per_row", total_tiles_per_row}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_id", "single_block_size_row_arg", "single_block_size_col_arg"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ------------------------------------------------------------------------
    // Compute: one KernelSpec per core group, each in its own WorkUnit.
    // ------------------------------------------------------------------------
    uint32_t single_sub_block_wh = single_block_size * single_block_size / single_sub_block_size;
    uint32_t single_sub_block_cliff_col_wh = single_block_size_cliff_col * single_block_size / single_sub_block_size;

    ComputeHardwareConfig::UnpackToDestModes unpack_to_dest_modes;
    if (fp32_llk_acc && input_cb_data_format == tt::DataFormat::Float32) {
        unpack_to_dest_modes.insert({SRC_DFB, UnpackToDestMode::UnpackToDestFp32});
    }

    auto make_compute = [&](const KernelSpecName& id, uint32_t block_size_col, uint32_t block_size_row) {
        return KernelSpec{
            .unique_id = id,
            .source =
                std::filesystem::path{
                    "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp"},
            .dfb_bindings = {ConsumerOf(SRC_DFB, "in"), ProducerOf(OUT_DFB, "out")},
            .compile_time_args =
                {{"block_size_col", block_size_col}, {"block_size_row", block_size_row}, {"third_dim", third_dim}},
            .hw_config =
                ComputeHardwareConfig{
                    .fp32_dest_acc_en = fp32_llk_acc,
                    .unpack_to_dest_mode = unpack_to_dest_modes,
                },
        };
    };

    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF_COL_ROW{"compute_cliff_col_row"};
    const KernelSpecName COMPUTE_CLIFF_ROW{"compute_cliff_row"};
    const KernelSpecName COMPUTE_CLIFF_COL{"compute_cliff_col"};

    std::vector<KernelSpec> kernels = {reader_spec, writer_spec};
    std::vector<WorkUnitSpec> work_units;
    if (!core_range.ranges().empty()) {
        kernels.push_back(make_compute(COMPUTE_FULL, single_sub_block_wh, single_sub_block_size));
        work_units.push_back(WorkUnitSpec{
            .name = "twp_block_full",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_FULL},
            .target_nodes = core_range});
    }
    if (has_cliff_col && has_cliff_row) {
        kernels.push_back(
            make_compute(COMPUTE_CLIFF_COL_ROW, single_block_size_cliff_col, single_block_size_cliff_row));
        work_units.push_back(WorkUnitSpec{
            .name = "twp_block_cliff_col_row",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_CLIFF_COL_ROW},
            .target_nodes = cliff_col_row_core_range});
    }
    if (has_cliff_row) {
        kernels.push_back(make_compute(COMPUTE_CLIFF_ROW, single_block_size, single_block_size_cliff_row));
        work_units.push_back(WorkUnitSpec{
            .name = "twp_block_cliff_row",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_CLIFF_ROW},
            .target_nodes = cliff_row_core_range});
    }
    if (has_cliff_col) {
        kernels.push_back(make_compute(COMPUTE_CLIFF_COL, single_sub_block_cliff_col_wh, single_sub_block_size));
        work_units.push_back(WorkUnitSpec{
            .name = "twp_block_cliff_col",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_CLIFF_COL},
            .target_nodes = cliff_col_core_range});
    }

    // ------------------------------------------------------------------------
    // Per-core runtime args (reproduces the legacy 2D block walk exactly).
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
        const auto& core = cores[i];
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

        const NodeCoord node = core;
        reader_run.runtime_arg_values.push_back(
            {node,
             {{"pad_value", packed_pad_value},
              {"width_size", TILE_WIDTH * a.element_size() * single_block_size_row_arg},
              {"start_row_id", start_row_id},
              {"start_column_id", start_column_id},
              {"single_block_size_row_arg", single_block_size_row_arg},
              {"single_block_size_col_arg", single_block_size_col_arg},
              {"sub_block_width_size", TILE_WIDTH * a.element_size() * single_sub_block_size_row_arg},
              {"single_sub_block_size_row_arg", single_sub_block_size_row_arg}}});

        writer_run.runtime_arg_values.push_back(
            {node,
             {{"start_id", tile_start_id},
              {"single_block_size_row_arg", single_block_size_row_arg},
              {"single_block_size_col_arg", single_block_size_col_arg}}});

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * TILE_WIDTH * a.element_size());
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
        .name = "tilize_with_val_padding_multi_core_block_interleaved",
        .kernels = std::move(kernels),
        .dataflow_buffers = {src_dfb_spec, stage_dfb_spec, out_dfb_spec},
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
