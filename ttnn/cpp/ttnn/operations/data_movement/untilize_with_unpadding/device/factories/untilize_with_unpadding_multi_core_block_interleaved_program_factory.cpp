// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

using ttnn::operations::data_movement::BlockBufferSet;
using ttnn::operations::data_movement::BlockCoreOrder;
using ttnn::operations::data_movement::BlockDirection;
using ttnn::operations::data_movement::BlockPlan;
using ttnn::operations::data_movement::buffer_set_for_core;
using ttnn::operations::data_movement::make_block_plan;

// Spec names are prefixed per factory: all five factory .cpp files land in one unity-build
// translation unit, where every anonymous namespace merges into a single scope.
const KernelSpecName BI_READER_FULL{"bi_reader_full"};
const KernelSpecName BI_WRITER_FULL{"bi_writer_full"};
const KernelSpecName BI_READER_CLIFFROW{"bi_reader_cliffrow"};
const KernelSpecName BI_WRITER_CLIFFROW{"bi_writer_cliffrow"};
const KernelSpecName BI_COMPUTE_FULL{"bi_compute_full"};
const KernelSpecName BI_COMPUTE_CLIFF_COL_ROW{"bi_compute_cliff_col_row"};
const KernelSpecName BI_COMPUTE_CLIFF_ROW{"bi_compute_cliff_row"};
const KernelSpecName BI_COMPUTE_CLIFF_COL{"bi_compute_cliff_col"};
const DFBSpecName BI_IN_FULL{"bi_in_full"};
const DFBSpecName BI_OUT_FULL{"bi_out_full"};
const DFBSpecName BI_IN_CLIFFROW{"bi_in_cliffrow"};
const DFBSpecName BI_OUT_CLIFFROW{"bi_out_cliffrow"};
const TensorParamName BI_INPUT{"bi_input"};
const TensorParamName BI_OUTPUT{"bi_output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    tt::DataFormat input_dfb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_dfb_data_format);
    tt::DataFormat output_dfb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_dfb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    // ColumnMajor: this factory's runtime-arg loop walks corerange_to_cores(available_grid), which
    // is the order the CoreRangeSet work-split overload assigns in. Untilize direction, so the
    // split follows the *input* padded shape -- the output here is the unpadded one.
    const BlockPlan plan = make_block_plan(
        BlockDirection::Untilize,
        BlockCoreOrder::ColumnMajor,
        a,
        output,
        input_single_tile_size,
        output_single_tile_size,
        TILE_HEIGHT,
        TILE_WIDTH,
        sub_core_grids);
    const BlockBufferSet& full_set = plan.full;
    const BlockBufferSet& cliffrow_set = plan.cliffrow;
    const auto& [ncores, all_cores, core_range, cliff_row_core_range, cliff_col_core_range, cliff_col_row_core_range, nblocks_per_core, single_block_size, single_block_size_cliff_row, single_block_size_cliff_col, has_cliff_row, has_cliff_col, full_cores_per_row, full_cores_per_col, single_sub_block_size] =
        plan.split;

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

    // One buffer pair per block width, each on its own disjoint cores. This replaces the legacy
    // layout of one (input, output) index pair re-used at two different sizes across four regions.
    // The sizing rules are the shared `push_buffer_set` helper's, reproduced here because that
    // helper emits into the legacy descriptor API and so cannot be called from a spec factory; an
    // untilize set has no staging buffer, so only the input and output pair is emitted.
    Group<DataflowBufferSpec> dataflow_buffers;
    auto push_buffer_pair = [&](const BlockBufferSet& set, const DFBSpecName& in_name, const DFBSpecName& out_name) {
        TT_FATAL(
            set.block_tiles > 0,
            "Buffer set on cores {} has a zero block width; its buffers would be empty",
            set.core_ranges.str());
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = in_name,
            .entry_size = input_single_tile_size,
            .num_entries = set.block_tiles,
            .data_format_metadata = input_dfb_data_format,
        });
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = out_name,
            .entry_size = output_single_tile_size,
            .num_entries = set.block_tiles,
            .data_format_metadata = output_dfb_data_format,
        });
    };
    if (!full_set.empty()) {
        push_buffer_pair(full_set, BI_IN_FULL, BI_OUT_FULL);
    }
    if (!cliffrow_set.empty()) {
        push_buffer_pair(cliffrow_set, BI_IN_CLIFFROW, BI_OUT_CLIFFROW);
    }

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // reader

    uint32_t num_tiles_2d = a.padded_shape()[-1] * a.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    uint32_t total_num_rows = output.logical_shape()[-2];

    // One reader and one writer per buffer set, each over that set's cores and bound to that set's
    // buffers. A set's cores are exactly the cores whose block width its buffers are sized for, so
    // every writer instance's contiguous walk from `get_read_ptr()` stays inside a buffer that is an
    // exact multiple of the block it drains.
    auto make_reader_kernel = [&](const KernelSpecName& unique_id, const DFBSpecName& in_name) {
        return KernelSpec{
            .unique_id = unique_id,
            .source =
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                "reader_unary_interleaved_wh_multicore_metal2.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = in_name,
                .accessor_name = "in",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = BI_INPUT,
                .accessor_name = "src",
            }},
            .compile_time_args =
                {{"num_tiles_per_2d", num_tiles_2d},
                 {"third_dim", third_dim},
                 {"total_tiles_per_row", total_tiles_per_row}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"start_id", "single_block_size_row_arg", "single_block_size_col_arg"}},
            .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        };
    };

    auto make_writer_kernel = [&](const KernelSpecName& unique_id, const DFBSpecName& out_name) {
        return KernelSpec{
            .unique_id = unique_id,
            .source =
                "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                "writer_unary_stick_layout_wh_multicore_metal2.cpp",
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = out_name,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = BI_OUTPUT,
                .accessor_name = "dst",
            }},
            .compile_time_args =
                {{"total_num_rows", total_num_rows},
                 {"third_dim", third_dim},
                 {"tile_height", (uint32_t)TILE_HEIGHT},
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
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };
    };

    // compute
    uint32_t single_sub_block_size_wh = single_block_size * single_block_size / single_sub_block_size;
    uint32_t single_sub_block_size_cliff_col_wh =
        single_block_size_cliff_col * single_block_size / single_sub_block_size;
    KernelSpec::CompilerOptions::Defines compute_kernel_defines;
    if (input_dfb_data_format == tt::DataFormat::Int32 || input_dfb_data_format == tt::DataFormat::UInt32 ||
        input_dfb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace("DST_ACCUM_MODE", "1");
    }

    const std::string compute_kernel_path(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh_metal2.cpp");

    Group<KernelSpec> kernels;
    Group<WorkUnitSpec> work_units;

    // The compute kernel stays split per region -- each region has its own block *count* -- but each
    // instance binds the buffer set matching its cores' block *width*. The region's block-width CTA
    // must equal that set's `block_tiles`, since it is the entry count the kernel waits on and pops;
    // the assertion below keeps the two from drifting apart.
    auto push_compute = [&](const KernelSpecName& unique_id,
                            const CoreRangeSet& cr,
                            const BlockBufferSet& set,
                            const DFBSpecName& in_name,
                            const DFBSpecName& out_name,
                            const KernelSpecName& reader_id,
                            const KernelSpecName& writer_id,
                            uint32_t block_size_col,
                            uint32_t block_size_row) {
        TT_FATAL(
            block_size_row == set.block_tiles,
            "Compute on cores {} expects a block width of {} tiles but its buffers hold {}",
            cr.str(),
            block_size_row,
            set.block_tiles);
        // fp32 unpack is marked for exactly the buffer this kernel reads. Marking both sets'
        // buffers would set it on the cliffrow set's input even when that set is empty -- a
        // buffer that exists on no core -- and would do so in the full-set kernels too.
        ComputeGen1Config compute_hw_config{.enable_32_bit_dest = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            compute_hw_config.unpack_modes = {{in_name, UnpackMode::UnpackToDest}};
        }

        kernels.push_back(KernelSpec{
            .unique_id = unique_id,
            .source = compute_kernel_path,
            .compiler_options =
                {
                    .defines = compute_kernel_defines,
                    .opt_level = KernelBuildOptLevel::O3,
                },
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = in_name,
                     .accessor_name = "src",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = out_name,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .compile_time_args =
                {{"block_size_col", block_size_col}, {"block_size_row", block_size_row}, {"third_dim", third_dim}},
            .hw_config = compute_hw_config,
        });
        work_units.push_back(WorkUnitSpec{
            .name = std::string(unique_id.get()),
            .kernels = {reader_id, writer_id, unique_id},
            .target_nodes = cr,
        });
    };

    if (!full_set.empty()) {
        kernels.push_back(make_reader_kernel(BI_READER_FULL, BI_IN_FULL));
        kernels.push_back(make_writer_kernel(BI_WRITER_FULL, BI_OUT_FULL));
    }
    if (!cliffrow_set.empty()) {
        kernels.push_back(make_reader_kernel(BI_READER_CLIFFROW, BI_IN_CLIFFROW));
        kernels.push_back(make_writer_kernel(BI_WRITER_CLIFFROW, BI_OUT_CLIFFROW));
    }

    if (!core_range.empty()) {
        push_compute(
            BI_COMPUTE_FULL,
            core_range,
            full_set,
            BI_IN_FULL,
            BI_OUT_FULL,
            BI_READER_FULL,
            BI_WRITER_FULL,
            single_sub_block_size_wh,
            single_sub_block_size);
    }
    if (has_cliff_col && has_cliff_row) {
        push_compute(
            BI_COMPUTE_CLIFF_COL_ROW,
            cliff_col_row_core_range,
            cliffrow_set,
            BI_IN_CLIFFROW,
            BI_OUT_CLIFFROW,
            BI_READER_CLIFFROW,
            BI_WRITER_CLIFFROW,
            single_block_size_cliff_col,
            single_block_size_cliff_row);
    }
    if (has_cliff_row) {
        push_compute(
            BI_COMPUTE_CLIFF_ROW,
            cliff_row_core_range,
            cliffrow_set,
            BI_IN_CLIFFROW,
            BI_OUT_CLIFFROW,
            BI_READER_CLIFFROW,
            BI_WRITER_CLIFFROW,
            single_block_size,
            single_block_size_cliff_row);
    }
    if (has_cliff_col) {
        push_compute(
            BI_COMPUTE_CLIFF_COL,
            cliff_col_core_range,
            full_set,
            BI_IN_FULL,
            BI_OUT_FULL,
            BI_READER_FULL,
            BI_WRITER_FULL,
            single_sub_block_size_cliff_col_wh,
            single_sub_block_size);
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

    KernelRunArgs full_reader_run_args{.kernel = BI_READER_FULL};
    KernelRunArgs full_writer_run_args{.kernel = BI_WRITER_FULL};
    KernelRunArgs cliffrow_reader_run_args{.kernel = BI_READER_CLIFFROW};
    KernelRunArgs cliffrow_writer_run_args{.kernel = BI_WRITER_CLIFFROW};

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

        // Route this core's args to the reader/writer instance for its buffer set. Membership is read
        // from the work split's own core assignment rather than re-derived from the branch above, so
        // the args and the buffers they drive can never disagree about which set a core is in. The
        // assertion then checks the one thing that must hold: the set's buffers are sized for exactly
        // the sub-block width being passed here, which is what keeps the writer's contiguous walk
        // inside its buffer.
        const BlockBufferSet& set = buffer_set_for_core(plan, core);
        const bool is_cliff_row_core = (&set == &cliffrow_set);
        KernelRunArgs& reader_run_args = is_cliff_row_core ? cliffrow_reader_run_args : full_reader_run_args;
        KernelRunArgs& writer_run_args = is_cliff_row_core ? cliffrow_writer_run_args : full_writer_run_args;
        TT_FATAL(
            single_sub_block_size_row_arg == set.block_tiles,
            "Core {} is fed a sub-block of {} tiles but the buffers on it hold {}. The work split "
            "assigned this core a block width that disagrees with its runtime args",
            core.str(),
            single_sub_block_size_row_arg,
            set.block_tiles);

        // reader runtime args — the input tensor's base address rides on its TensorBinding, which
        // the framework refreshes on every dispatch.
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"start_id", tile_start_id},
             {"single_block_size_row_arg", single_block_size_row_arg},
             {"single_block_size_col_arg", single_block_size_col_arg}});

        //  writer runtime args
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"width_size", TILE_WIDTH * el_size * single_block_size_row_arg},
             {"start_row_id", start_row_id},
             {"start_column_id", start_column_id},
             {"single_block_size_row_arg", single_block_size_row_arg},
             {"single_block_size_col_arg", single_block_size_col_arg},
             {"sub_block_width_size", TILE_WIDTH * el_size * single_sub_block_size_row_arg},
             {"single_sub_block_size_row_arg", single_sub_block_size_row_arg}});

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
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {TensorParameter{.unique_id = BI_INPUT, .spec = input_mesh_tensor.tensor_spec()},
             TensorParameter{.unique_id = BI_OUTPUT, .spec = output_mesh_tensor.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    if (!full_set.empty()) {
        run_args.kernel_run_args.push_back(std::move(full_reader_run_args));
        run_args.kernel_run_args.push_back(std::move(full_writer_run_args));
    }
    if (!cliffrow_set.empty()) {
        run_args.kernel_run_args.push_back(std::move(cliffrow_reader_run_args));
        run_args.kernel_run_args.push_back(std::move(cliffrow_writer_run_args));
    }
    run_args.tensor_args = {
        {BI_INPUT, input_mesh_tensor},
        {BI_OUTPUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
