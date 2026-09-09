// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_block_program_factory.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>

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

}  // namespace

ttnn::device_operation::ProgramArtifacts UntilizeMultiCoreBlockProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    // Spec names are function-local: the op's factories are unity-built, and same-named
    // anonymous-namespace constants across them would redefine.
    const DFBSpecName IN_FULL{"in_full"};
    const DFBSpecName OUT_FULL{"out_full"};
    const DFBSpecName IN_CLIFFROW{"in_cliffrow"};
    const DFBSpecName OUT_CLIFFROW{"out_cliffrow"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER_FULL{"reader_full"};
    const KernelSpecName WRITER_FULL{"writer_full"};
    const KernelSpecName READER_CLIFFROW{"reader_cliffrow"};
    const KernelSpecName WRITER_CLIFFROW{"writer_cliffrow"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF_COL_ROW{"compute_cliff_col_row"};
    const KernelSpecName COMPUTE_CLIFF_ROW{"compute_cliff_row"};
    const KernelSpecName COMPUTE_CLIFF_COL{"compute_cliff_col"};

    constexpr const char* READER_SRC =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
        "reader_unary_interleaved_wh_multicore_metal2.cpp";
    constexpr const char* WRITER_SRC =
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_stick_layout_wh_multicore_metal2.cpp";
    constexpr const char* COMPUTE_SRC =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh_metal2.cpp";

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_data_format);
    tt::DataFormat output_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    const auto& input_shape = a.padded_shape();
    const uint32_t a_tile_height = a.tensor_spec().tile().get_height();

    // RowMajor: this factory's runtime-arg loop walks grid_to_cores(..., row_wise=true), so the
    // split has to hand cores out in that same order. The block factory is never reached with
    // sub_core_grids set -- that routes to the sub-core-grid factory instead.
    //
    // `make_block_plan` reads live L1 occupancy, so it is only valid on a program-cache miss --
    // which is the only time this function runs.
    const BlockPlan plan = make_block_plan(
        BlockDirection::Untilize,
        BlockCoreOrder::RowMajor,
        a,
        output,
        input_single_tile_size,
        output_single_tile_size,
        a_tile_height,
        a.tensor_spec().tile().get_width(),
        /*sub_core_grids=*/std::nullopt);
    const BlockBufferSet& full_set = plan.full;
    const BlockBufferSet& cliffrow_set = plan.cliffrow;
    const auto& [ncores, all_cores, core_range, cliff_row_core_range, cliff_col_core_range, cliff_col_row_core_range, nblocks_per_core, single_block_size, single_block_size_cliff_row, single_block_size_cliff_col, has_cliff_row, has_cliff_col, full_cores_per_row, full_cores_per_col, single_sub_block_size] =
        plan.split;

    // Same grid `make_block_plan` split over -- the runtime-arg loop below walks it in core order.
    IDevice* device = a.device();
    const CoreCoord grid_size = device->compute_with_storage_grid_size();

    if (single_sub_block_size > 0 && single_block_size % single_sub_block_size) {
        TT_FATAL(false, "single_block_size is not divided by single_sub_block_size");
    }

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);
    uint32_t row_size_bytes;

    uint32_t el_size = a.element_size();
    if (a.dtype() == DataType::BFLOAT8_B) {
        row_size_bytes = input_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        row_size_bytes = input_shape[-1] * a.element_size();
    }

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    ProgramSpec spec{.name = "untilize_multi_core_block"};

    // Each non-empty buffer set gets its own input/output DFB pair, sized for that set's block width
    // (see BlockBufferSet in data_movement/common). The cliff-row pair exists only when the split
    // produced a cliff row. The sizes mirror `push_buffer_set` -- `block_tiles` pages of one tile
    // each, default tile shape, nothing borrowed -- so `make_block_plan` stays the single source of
    // the split and the sizing.
    const auto in_dfb_of = [&](const BlockBufferSet& set) -> const DFBSpecName& {
        return (&set == &cliffrow_set) ? IN_CLIFFROW : IN_FULL;
    };
    const auto out_dfb_of = [&](const BlockBufferSet& set) -> const DFBSpecName& {
        return (&set == &cliffrow_set) ? OUT_CLIFFROW : OUT_FULL;
    };
    for (const BlockBufferSet* set : {&full_set, &cliffrow_set}) {
        if (set->empty()) {
            continue;
        }
        TT_FATAL(
            set->block_tiles > 0,
            "Buffer set on cores {} has a zero block width; its buffers would be empty",
            set->core_ranges.str());
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = in_dfb_of(*set),
            .entry_size = input_single_tile_size,
            .num_entries = set->block_tiles,
            .data_format_metadata = input_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = out_dfb_of(*set),
            .entry_size = output_single_tile_size,
            .num_entries = set->block_tiles,
            .data_format_metadata = output_data_format,
        });
    }

    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
    };

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
    // DFB pair. A set's cores are exactly the cores whose block width its buffers are sized for, so
    // every writer instance's contiguous walk from `get_read_ptr()` stays inside a buffer that is an
    // exact multiple of the block it drains. The set's cores are the union of the compute regions
    // below, so the reader/writer specs are shared by that set's work units.
    auto make_reader_spec = [&](const KernelSpecName& id, const BlockBufferSet& set) {
        return KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path{READER_SRC},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = in_dfb_of(set),
                .accessor_name = "in",
                .endpoint_type = DFBEndpointType::PRODUCER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = INPUT,
                .accessor_name = "src",
            }},
            .compile_time_args =
                {
                    {"num_tiles_per_2d", num_tiles_2d},
                    {"third_dim", third_dim},
                    {"total_tiles_per_row", total_tiles_per_row},
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"start_id", "single_block_size_row_arg", "single_block_size_col_arg"},
                },
            .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        };
    };

    auto make_writer_spec = [&](const KernelSpecName& id, const BlockBufferSet& set) {
        return KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path{WRITER_SRC},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = out_dfb_of(set),
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            }},
            .tensor_bindings = {TensorBinding{
                .tensor_parameter_name = OUTPUT,
                .accessor_name = "dst",
            }},
            .compile_time_args =
                {
                    {"total_num_rows", total_num_rows},
                    {"third_dim", third_dim},
                    {"tile_height", TILE_HEIGHT},
                    {"unpadded_X_size", row_size_bytes},
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names =
                        {"width_size",
                         "start_row_id",
                         "start_column_id",
                         "single_block_size_row_arg",
                         "single_block_size_col_arg",
                         "sub_block_width_size",
                         "single_sub_block_size_row_arg"},
                },
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };
    };

    // Push each non-empty set's reader and writer.
    if (!full_set.empty()) {
        spec.kernels.push_back(make_reader_spec(READER_FULL, full_set));
        spec.kernels.push_back(make_writer_spec(WRITER_FULL, full_set));
    }
    if (!cliffrow_set.empty()) {
        spec.kernels.push_back(make_reader_spec(READER_CLIFFROW, cliffrow_set));
        spec.kernels.push_back(make_writer_spec(WRITER_CLIFFROW, cliffrow_set));
    }

    // compute
    uint32_t single_sub_block_size_wh = single_block_size * single_block_size / single_sub_block_size;
    uint32_t single_sub_block_size_cliff_col_wh =
        single_block_size_cliff_col * single_block_size / single_sub_block_size;

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (input_data_format == tt::DataFormat::Int32 || input_data_format == tt::DataFormat::UInt32 ||
        input_data_format == tt::DataFormat::Float32) {
        compute_defines.insert({"DST_ACCUM_MODE", "1"});
    }

    // The compute kernel stays split per region -- each region has its own block *count* -- but each
    // instance binds the buffer set matching its cores' block *width*. The region's block-width CTA
    // (`block_size_row`) must equal that set's `block_tiles`, since it is the page count the kernel
    // waits on and pops; the assertion below keeps the two from drifting apart. Each region is its
    // own work unit: the same source with different compile-time block sizes over disjoint cores.
    auto add_compute_region = [&](const KernelSpecName& id,
                                  const char* work_unit_name,
                                  const CoreRangeSet& cores,
                                  const BlockBufferSet& set,
                                  uint32_t block_size_col,
                                  uint32_t block_size_row) {
        TT_FATAL(
            block_size_row == set.block_tiles,
            "Compute on cores {} expects a block width of {} tiles but its buffers hold {}",
            cores.str(),
            block_size_row,
            set.block_tiles);
        // fp32 unpack is marked for exactly the buffer this kernel reads. Marking both sets' DFBs
        // would name `cliffrow_set`'s input even when that set is empty -- an operand with no buffer
        // on any core -- and would do so in the full-set kernels too.
        ComputeGen1Config compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            compute_cfg.unpack_modes.insert({in_dfb_of(set), UnpackMode::UnpackToDest});
        }

        const bool is_cliff_row_set = (&set == &cliffrow_set);
        spec.kernels.push_back(KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path{COMPUTE_SRC},
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = in_dfb_of(set),
                     .accessor_name = "src",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = out_dfb_of(set),
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .compile_time_args =
                {
                    {"block_size_col", block_size_col},
                    {"block_size_row", block_size_row},
                    {"third_dim", third_dim},
                },
            .hw_config = std::move(compute_cfg),
        });
        spec.work_units.push_back(WorkUnitSpec{
            .name = work_unit_name,
            .kernels =
                {is_cliff_row_set ? READER_CLIFFROW : READER_FULL,
                 is_cliff_row_set ? WRITER_CLIFFROW : WRITER_FULL,
                 id},
            .target_nodes = cores,
        });
    };

    if (!core_range.empty()) {
        add_compute_region(
            COMPUTE_FULL, "wu_full", core_range, full_set, single_sub_block_size_wh, single_sub_block_size);
    }
    if (has_cliff_col && has_cliff_row) {
        add_compute_region(
            COMPUTE_CLIFF_COL_ROW,
            "wu_cliff_col_row",
            cliff_col_row_core_range,
            cliffrow_set,
            single_block_size_cliff_col,
            single_block_size_cliff_row);
    }
    if (has_cliff_row) {
        add_compute_region(
            COMPUTE_CLIFF_ROW,
            "wu_cliff_row",
            cliff_row_core_range,
            cliffrow_set,
            single_block_size,
            single_block_size_cliff_row);
    }
    if (has_cliff_col) {
        add_compute_region(
            COMPUTE_CLIFF_COL,
            "wu_cliff_col",
            cliff_col_core_range,
            full_set,
            single_sub_block_size_cliff_col_wh,
            single_sub_block_size);
    }

    // RUNTIME ARGS
    KernelRunArgs full_reader_run{.kernel = READER_FULL};
    KernelRunArgs full_writer_run{.kernel = WRITER_FULL};
    KernelRunArgs cliffrow_reader_run{.kernel = READER_CLIFFROW};
    KernelRunArgs cliffrow_writer_run{.kernel = WRITER_CLIFFROW};

    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
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

        // Route this core's args to the reader/writer instance for its buffer set.
        const BlockBufferSet& set = buffer_set_for_core(plan, core);
        const bool is_cliff_row_core = (&set == &cliffrow_set);
        KernelRunArgs& reader_run = is_cliff_row_core ? cliffrow_reader_run : full_reader_run;
        KernelRunArgs& writer_run = is_cliff_row_core ? cliffrow_writer_run : full_writer_run;
        TT_FATAL(
            single_sub_block_size_row_arg == set.block_tiles,
            "Core {} is fed a sub-block of {} tiles but the buffers on it hold {}. The work split "
            "assigned this core a block width that disagrees with its runtime args",
            core.str(),
            single_sub_block_size_row_arg,
            set.block_tiles);

        // reader runtime args -- the input address is not an arg: it travels through the `src`
        // tensor binding, which the framework refreshes on every program-cache hit.
        AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"start_id", tile_start_id},
             {"single_block_size_row_arg", single_block_size_row_arg},
             {"single_block_size_col_arg", single_block_size_col_arg}});

        // writer runtime args (the output address likewise travels through the `dst` binding)
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"width_size", TILE_WIDTH * el_size * single_block_size_row_arg},
             {"start_row_id", start_row_id},
             {"start_column_id", start_column_id},
             {"single_block_size_row_arg", single_block_size_row_arg},
             {"single_block_size_col_arg", single_block_size_col_arg},
             {"sub_block_width_size", TILE_WIDTH * el_size * single_sub_block_size_row_arg},
             {"single_sub_block_size_row_arg", single_sub_block_size_row_arg}});

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * TILE_WIDTH * el_size);
        start_column_id = end_column_id % row_size_bytes;
        if (end_column_id % row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * TILE_HEIGHT;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    ProgramRunArgs run_args;
    if (!full_set.empty()) {
        run_args.kernel_run_args.push_back(std::move(full_reader_run));
        run_args.kernel_run_args.push_back(std::move(full_writer_run));
    }
    if (!cliffrow_set.empty()) {
        run_args.kernel_run_args.push_back(std::move(cliffrow_reader_run));
        run_args.kernel_run_args.push_back(std::move(cliffrow_writer_run));
    }
    run_args.tensor_args = {
        {INPUT, TensorArgument{a.mesh_tensor()}},
        {OUTPUT, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
