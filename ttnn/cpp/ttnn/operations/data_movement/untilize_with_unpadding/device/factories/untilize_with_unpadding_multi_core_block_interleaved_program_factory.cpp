// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp"

#include <algorithm>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory::create_program_spec(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

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

    // ---- Resource names ----
    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF_COL_ROW{"compute_cliff_col_row"};
    const KernelSpecName COMPUTE_CLIFF_ROW{"compute_cliff_row"};
    const KernelSpecName COMPUTE_CLIFF_COL{"compute_cliff_col"};

    // ---- Dataflow buffers (legacy CBs c_0 / c_16) ----
    // CB-SIZE CONSOLIDATION DECISION:
    //   Legacy emitted a PAIR of (c_0 input, c_16 output) CBs on each non-empty core sub-region via
    //   push_cb_pair, and the per-region num_tiles differed: core_range and cliff_col_core_range used
    //   `single_sub_block_size`; cliff_col_row_core_range and cliff_row_core_range used
    //   `single_block_size_cliff_row`. A Metal 2.0 DataflowBufferSpec fixes entry_size/num_entries
    //   ONCE per spec (dfb_run_overrides cannot resize), so one DFB per logical buffer must pick a
    //   num_entries that is safe (>=) for ALL emitted regions. We use the MAX of the per-region
    //   num-tiles that the legacy push_cb_pair calls actually used, guarded by the SAME emission
    //   conditions so an unused region never inflates the reservation. This collapses the legacy
    //   per-region CB sizes to a single max-sized DFB; L1 footprint per core can only INCREASE (to
    //   the max), never decrease, so no region under-reserves — correctness is preserved.
    uint32_t max_cb_num_tiles = 0;
    if (!core_range.empty()) {
        max_cb_num_tiles = std::max(max_cb_num_tiles, single_sub_block_size);
    }
    if (has_cliff_col && has_cliff_row) {
        max_cb_num_tiles = std::max(max_cb_num_tiles, single_block_size_cliff_row);
    }
    if (has_cliff_row) {
        max_cb_num_tiles = std::max(max_cb_num_tiles, single_block_size_cliff_row);
    }
    if (has_cliff_col) {
        max_cb_num_tiles = std::max(max_cb_num_tiles, single_sub_block_size);
    }

    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = input_single_tile_size,
        .num_entries = max_cb_num_tiles,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = max_cb_num_tiles,
        .data_format_metadata = output_cb_data_format,
    };

    // ---- Tensor parameters ----
    TensorParameter src_param{.unique_id = SRC, .spec = input.tensor_spec()};
    TensorParameter dst_param{.unique_id = DST, .spec = output.tensor_spec()};

    // reader

    uint32_t num_tiles_2d = a.padded_shape()[-1] * a.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    // ---- Reader kernel (tilized reader; forked m2 copy of the shared eltwise/unary wh reader) ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "reader_unary_interleaved_wh_multicore_m2.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"},
            },
        .compile_time_args =
            {
                {"num_tiles_per_2d", num_tiles_2d},
                {"third_dim", third_dim},
                {"total_tiles_per_row", total_tiles_per_row},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"start_id", "single_block_size_row_arg", "single_block_size_col_arg"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // writer
    uint32_t total_num_rows = output.logical_shape()[-2];

    // ---- Writer kernel (untilized wh split-rows writer; forked m2 copy — the legacy original is
    // ---- still used by untilize's block factory, so it is forked rather than edited in place) ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_stick_layout_wh_multicore_m2.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"},
            },
        .compile_time_args =
            {
                {"total_num_rows", total_num_rows},
                {"third_dim", third_dim},
                {"tile_height", TILE_HEIGHT},
                {"unpadded_X_size", unpadded_row_size_bytes},
            },
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

    // compute
    uint32_t single_sub_block_size_wh = single_block_size * single_block_size / single_sub_block_size;
    uint32_t single_sub_block_size_cliff_col_wh =
        single_block_size_cliff_col * single_block_size / single_sub_block_size;
    KernelSpec::CompilerOptions::Defines compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.insert({"DST_ACCUM_MODE", "1"});
    }

    // One compute KernelSpec per work-split region (full / cliff-col-row / cliff-row / cliff-col),
    // each with its own CTA values (block_size_col, block_size_row, third_dim) and its own WorkUnitSpec.
    // Reader + writer run on all_cores, so they must be members of EVERY WorkUnitSpec (Local-DFB rule:
    // IN/OUT producer & consumer share the same WUs). The per-region block sizes stay CTAs — no
    // CTA->RTA demotion — preserving compile-time loop unrolling.
    Group<KernelSpec> kernels;
    std::vector<WorkUnitSpec> work_units;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));

    auto make_compute = [&](const KernelSpecName& unique_id,
                            const CoreRangeSet& target,
                            uint32_t block_size_col,
                            uint32_t block_size_row,
                            const char* wu_name) {
        ComputeHardwareConfig compute_hw_config{.fp32_dest_acc_en = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            compute_hw_config.unpack_to_dest_mode.insert({IN, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
        }
        kernels.push_back(KernelSpec{
            .unique_id = unique_id,
            .source = "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/compute/"
                      "untilize_wh_m2.cpp",
            .compiler_options = {.defines = compute_kernel_defines},
            .dfb_bindings =
                {
                    DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                    DFBBinding{
                        .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
                },
            .compile_time_args =
                {
                    {"block_size_col", block_size_col},
                    {"block_size_row", block_size_row},
                    {"third_dim", third_dim},
                },
            .hw_config = std::move(compute_hw_config),
        });
        work_units.push_back(
            WorkUnitSpec{.name = wu_name, .kernels = {READER, WRITER, unique_id}, .target_nodes = target});
    };

    if (!core_range.empty()) {
        make_compute(COMPUTE_FULL, core_range, single_sub_block_size_wh, single_sub_block_size, "uwu_full");
    }
    if (has_cliff_col && has_cliff_row) {
        make_compute(
            COMPUTE_CLIFF_COL_ROW,
            cliff_col_row_core_range,
            single_block_size_cliff_col,
            single_block_size_cliff_row,
            "uwu_cliff_col_row");
    }
    if (has_cliff_row) {
        make_compute(
            COMPUTE_CLIFF_ROW, cliff_row_core_range, single_block_size, single_block_size_cliff_row, "uwu_cliff_row");
    }
    if (has_cliff_col) {
        make_compute(
            COMPUTE_CLIFF_COL,
            cliff_col_core_range,
            single_sub_block_size_cliff_col_wh,
            single_sub_block_size,
            "uwu_cliff_col");
    }

    // RUNTIME ARGS
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

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    reader_run_args.runtime_arg_values.reserve(ncores);
    writer_run_args.runtime_arg_values.reserve(ncores);
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

        // reader runtime args (the legacy src0_buffer->address() slot is folded into ta::src)
        reader_run_args.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core,
            .args = {
                {"start_id", tile_start_id},
                {"single_block_size_row_arg", single_block_size_row_arg},
                {"single_block_size_col_arg", single_block_size_col_arg}}});

        // writer runtime args (the legacy dst_buffer->address() slot is folded into ta::dst)
        writer_run_args.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core,
            .args = {
                {"width_size", TILE_WIDTH * el_size * single_block_size_row_arg},
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
        .dataflow_buffers = {std::move(in_dfb), std::move(out_dfb)},
        .tensor_parameters = {std::move(src_param), std::move(dst_param)},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args =
            {
                {SRC, TensorArgument{input.mesh_tensor()}},
                {DST, TensorArgument{output.mesh_tensor()}},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
