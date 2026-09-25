// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_interleaved_program_factory.hpp"

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

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Spec names are prefixed per factory: all five factory .cpp files land in one unity-build
// translation unit, where every anonymous namespace merges into a single scope.
const KernelSpecName MCI_READER{"mci_reader"};
const KernelSpecName MCI_WRITER{"mci_writer"};
const KernelSpecName MCI_COMPUTE_FULL{"mci_compute_full"};
const KernelSpecName MCI_COMPUTE_CLIFF{"mci_compute_cliff"};
const DFBSpecName MCI_IN{"mci_in"};
const DFBSpecName MCI_OUT{"mci_out"};
const TensorParamName MCI_INPUT{"mci_input"};
const TensorParamName MCI_OUTPUT{"mci_output"};

}  // namespace

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create_program_artifacts(
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

    uint32_t num_blocks = input_shape[-1] == 0 ? 0 : a.physical_volume() / input_shape[-1] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.empty();

    uint32_t padded_row_size_bytes;
    uint32_t unpadded_row_size_bytes;

    if (a.dtype() == DataType::BFLOAT8_B) {
        padded_row_size_bytes = input_shape[-1] * output.element_size();
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
    } else {
        padded_row_size_bytes = input_shape[-1] * a.element_size();
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
    }

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    DataflowBufferSpec in_dfb{
        .unique_id = MCI_IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = input_dfb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = MCI_OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = output_dfb_data_format,
    };

    /** reader
     */
    KernelSpec reader{
        .unique_id = MCI_READER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "reader_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = MCI_IN,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = MCI_INPUT,
            .accessor_name = "src",
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    /** writer
     */
    // The destination's per-shard page size — the shard width for BLOCK/WIDTH-sharded output (a
    // logical row spans multiple shards), the buffer's aligned page size for HEIGHT-sharded output
    // (one row per page, padded up to the buffer alignment), the full unpadded row for interleaved —
    // is supplied by the tensor binding, which hands the kernel the tensor's aligned page size.
    KernelSpec writer{
        .unique_id = MCI_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_stick_layout_split_rows_multicore.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = MCI_OUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{
            .tensor_parameter_name = MCI_OUTPUT,
            .accessor_name = "dst",
        }},
        .compile_time_args =
            {{"float32_dtype",
              (uint32_t)(input_dfb_data_format == tt::DataFormat::Float32 or
                         input_dfb_data_format == tt::DataFormat::UInt32 or
                         input_dfb_data_format == tt::DataFormat::Int32)},
             {"unpadded_X_size", unpadded_row_size_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"padded_X_size", "start_stick_id", "n_block_reps"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    /** compute
     */
    KernelSpec::CompilerOptions::Defines compute_kernel_defines;
    if (input_dfb_data_format == tt::DataFormat::Int32 || input_dfb_data_format == tt::DataFormat::UInt32 ||
        input_dfb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace("DST_ACCUM_MODE", "1");
    }
    ComputeGen1Config compute_hw_config{.enable_32_bit_dest = fp32_dest_acc_en};
    if (fp32_dest_acc_en) {
        compute_hw_config.unpack_modes = {{MCI_IN, UnpackMode::UnpackToDest}};
    }
    const std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp");

    // One compute KernelSpec per core group, each keeping its group's block count as a compile-time
    // arg (the groups cover disjoint node sets, so each node still sees exactly one instance).
    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t per_core_block_cnt) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = compute_kernel,
            .compiler_options =
                {
                    .defines = compute_kernel_defines,
                    .opt_level = KernelBuildOptLevel::O3,
                },
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = MCI_IN,
                     .accessor_name = "src",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = MCI_OUT,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .compile_time_args =
                {{"per_core_block_cnt", per_core_block_cnt}, {"per_core_block_tile_cnt", num_tiles_per_row}},
            .hw_config = compute_hw_config,
        };
    };

    // The writer joins after the core loop below, which fills in its per-node vararg counts.
    Group<KernelSpec> kernels = {std::move(reader)};
    Group<WorkUnitSpec> work_units;
    if (!core_range.empty()) {
        kernels.push_back(make_compute(MCI_COMPUTE_FULL, nblocks_per_core));
        work_units.push_back(WorkUnitSpec{
            .name = "full",
            .kernels = {MCI_READER, MCI_WRITER, MCI_COMPUTE_FULL},
            .target_nodes = core_range,
        });
    }
    if (has_cliff) {
        kernels.push_back(make_compute(MCI_COMPUTE_CLIFF, nblocks_per_core_cliff));
        work_units.push_back(WorkUnitSpec{
            .name = "cliff",
            .kernels = {MCI_READER, MCI_WRITER, MCI_COMPUTE_CLIFF},
            .target_nodes = core_range_cliff,
        });
    }

    uint32_t tile_height = input.tensor_spec().tile().get_height();
    auto core_assignments = ttnn::distribute_work(
        output_shape, input_shape, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff, tile_height);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;

    const auto& cores = corerange_to_cores(available_grid);
    KernelRunArgs reader_run_args{.kernel = MCI_READER};
    KernelRunArgs writer_run_args{.kernel = MCI_WRITER};
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // writer runtime args. This core's first stick is the running total from the cores before
        // it, so capture it before the loop below advances `row_start_id` past this core's blocks.
        AdvancedKernelRunArgs::Varargs writer_varargs;
        const uint32_t core_row_start_id = row_start_id;

        uint32_t nblocks_per_core_core = 0;

        BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;  // will be incremented in first iteration of the loop
        for (const auto& el : assignment) {
            nblocks_per_core_core += el.block_count();
            row_start_id += el.data_row_count();
            if (compare_assignments(ref_el, el)) {
                count_repeated++;
            } else {
                // push back information for previous elements
                writer_varargs.push_back(ref_el.n_data);
                writer_varargs.push_back(ref_el.n_mixed);
                writer_varargs.push_back(ref_el.n_pads);
                writer_varargs.push_back(ref_el.times);
                writer_varargs.push_back(count_repeated);
                // Set up assignment for this element
                ref_el = el;
                count_repeated = 1;
            }
        }
        writer_varargs.push_back(ref_el.n_data);
        writer_varargs.push_back(ref_el.n_mixed);
        writer_varargs.push_back(ref_el.n_pads);
        writer_varargs.push_back(ref_el.times);
        writer_varargs.push_back(count_repeated);

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core_core;

        // reader runtime args
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"num_pages", num_tiles_per_core}, {"start_id", tile_start_id}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"padded_X_size", padded_row_size_bytes},
             {"start_stick_id", core_row_start_id},
             {"n_block_reps", static_cast<uint32_t>(assignment.size())}});
        // Each core's BlockRep runs are a runtime-length payload, so they ride as runtime varargs.
        // The run count differs per core, which is what the per-node vararg-count override says.
        writer.advanced_options.num_runtime_varargs_per_node[core] = static_cast<uint32_t>(writer_varargs.size());
        writer_run_args.advanced_options.runtime_varargs[core] = std::move(writer_varargs);

        tile_start_id += num_tiles_per_core;
    }

    kernels.push_back(std::move(writer));

    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_interleaved",
        .kernels = std::move(kernels),
        .dataflow_buffers = {std::move(in_dfb), std::move(out_dfb)},
        .tensor_parameters =
            {TensorParameter{.unique_id = MCI_INPUT, .spec = input_mesh_tensor.tensor_spec()},
             TensorParameter{.unique_id = MCI_OUTPUT, .spec = output_mesh_tensor.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {MCI_INPUT, input_mesh_tensor},
        {MCI_OUTPUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
