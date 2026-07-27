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
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    using namespace tt::tt_metal::experimental;

    // Metal 2.0 named resources. Function-local: ttnn_op_data_movement is a unity-build target, so
    // anonymous-namespace symbols from every factory .cpp would merge into one scope.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};
    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};

    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

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

    DataflowBufferSpec in_dfb_spec{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = output_cb_data_format,
    };

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */
    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                        "reader_unary_interleaved_start_id_metal2.cpp"},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::PRODUCER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "src"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    /** writer
     */
    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                "writer_unary_stick_layout_split_rows_multicore.cpp"},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .compile_time_args =
            {{"float32_dtype",
              static_cast<uint32_t>(
                  input_cb_data_format == tt::DataFormat::Float32 or input_cb_data_format == tt::DataFormat::UInt32 or
                  input_cb_data_format == tt::DataFormat::Int32)},
             {"unpadded_X_size", unpadded_row_size_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"padded_X_size", "start_stick_id", "n_block_reps"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    /** compute
     */
    KernelSpec::CompilerOptions::Defines compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace("DST_ACCUM_MODE", "1");
    }
    // Legacy set unpack_to_dest_mode[c_0] = UnpackToDestFp32 exactly when fp32_dest_acc_en; every
    // other CB stayed Default (== UnpackMode::UnpackToSrc, expressed by omitting the entry).
    ComputeUnpackModes unpack_modes;
    if (fp32_dest_acc_en) {
        unpack_modes.emplace(IN_DFB, UnpackMode::UnpackToDest);
    }
    const std::filesystem::path compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp");

    // One KernelSpec per legacy compute KernelDescriptor, preserving the work-split multiplicity:
    // the per-group block count stays a compile-time arg (never demoted to an RTA).
    auto make_compute_spec = [&](const KernelSpecName& unique_id, uint32_t per_core_block_cnt) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = compute_kernel,
            .compiler_options = {.defines = compute_kernel_defines},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = IN_DFB,
                     .accessor_name = "in",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB,
                     .accessor_name = "out",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 }},
            .compile_time_args =
                {{"per_core_block_cnt", per_core_block_cnt}, {"per_core_block_tile_cnt", num_tiles_per_row}},
            .hw_config =
                ComputeGen1Config{
                    .enable_32_bit_dest = fp32_dest_acc_en,
                    .unpack_modes = unpack_modes,
                },
        };
    };

    uint32_t tile_height = input.tensor_spec().tile().get_height();
    auto core_assignments = ttnn::distribute_work(
        output_shape, input_shape, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff, tile_height);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;

    const auto& cores = corerange_to_cores(available_grid);

    KernelRunArgs reader_run{.kernel = READER};
    KernelRunArgs writer_run{.kernel = WRITER};
    // The writer's per-group 5-tuple block is a genuine vararg: its length varies per node with the
    // block assignment, so the count is declared per node rather than by the scalar schema field.
    Table<Nodes, uint32_t> writer_varargs_per_node;

    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // start_stick_id is the running row cursor *before* this core's blocks are accounted for.
        const uint32_t core_start_stick_id = row_start_id;

        std::vector<uint32_t> writer_varargs;

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
            reader_run.runtime_arg_values, core, {{"num_pages", num_tiles_per_core}, {"start_id", tile_start_id}});
        // writer runtime args
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"padded_X_size", padded_row_size_bytes},
             {"start_stick_id", core_start_stick_id},
             {"n_block_reps", static_cast<uint32_t>(assignment.size())}});
        writer_varargs_per_node.emplace(Nodes{core}, static_cast<uint32_t>(writer_varargs.size()));
        writer_run.advanced_options.runtime_varargs.emplace(core, std::move(writer_varargs));

        tile_start_id += num_tiles_per_core;
    }

    writer_spec.advanced_options.num_runtime_varargs_per_node = std::move(writer_varargs_per_node);

    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_interleaved",
        .kernels = {std::move(reader_spec), std::move(writer_spec)},
        .dataflow_buffers = {std::move(in_dfb_spec), std::move(out_dfb_spec)},
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()}},
    };

    if (!core_range.empty()) {
        spec.kernels.push_back(make_compute_spec(COMPUTE_FULL, nblocks_per_core));
        spec.work_units.push_back(WorkUnitSpec{
            .name = "full",
            .kernels = {READER, WRITER, COMPUTE_FULL},
            .target_nodes = core_range,
        });
    }
    if (has_cliff) {
        spec.kernels.push_back(make_compute_spec(COMPUTE_CLIFF, nblocks_per_core_cliff));
        spec.work_units.push_back(WorkUnitSpec{
            .name = "cliff",
            .kernels = {READER, WRITER, COMPUTE_CLIFF},
            .target_nodes = core_range_cliff,
        });
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
