// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_interleaved_program_factory.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create_program_spec(
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

    // ---- Resource names ----
    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};

    // ---- Dataflow buffers (legacy CBs c_0 / c_16) ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = output_cb_data_format,
    };

    // ---- Tensor parameters ----
    TensorParameter src_param{.unique_id = SRC, .spec = input.tensor_spec()};
    TensorParameter dst_param{.unique_id = DST, .spec = output.tensor_spec()};

    // ---- Reader kernel (tilized reader; forked m2 copy of the shared eltwise/unary reader) ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "reader_unary_interleaved_start_id_m2.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ---- Writer kernel (untilized split-rows writer; ported in place) ----
    // The per-core block-rep tuples are variable-length, so they are passed as runtime varargs.
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_stick_layout_split_rows_multicore.cpp",
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
                {"float32_dtype",
                 (std::uint32_t)(input_cb_data_format == tt::DataFormat::Float32 or
                                 input_cb_data_format == tt::DataFormat::UInt32 or
                                 input_cb_data_format == tt::DataFormat::Int32)},
                {"unpadded_X_size", unpadded_row_size_bytes},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"padded_X_size", "start_stick_id", "n_block_reps"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ---- Compute kernel (forked m2 copy of the shared untilize compute) ----
    KernelSpec::CompilerOptions::Defines compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.insert({"DST_ACCUM_MODE", "1"});
    }

    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t nblocks) {
        ComputeHardwareConfig compute_hw_config{.fp32_dest_acc_en = fp32_dest_acc_en};
        if (fp32_dest_acc_en) {
            compute_hw_config.unpack_to_dest_mode.insert({IN, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32});
        }
        return KernelSpec{
            .unique_id = unique_id,
            .source =
                "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/compute/untilize_m2.cpp",
            .compiler_options = {.defines = compute_kernel_defines},
            .dfb_bindings =
                {
                    DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                    DFBBinding{
                        .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
                },
            .compile_time_args =
                {
                    {"per_core_block_cnt", nblocks},
                    {"per_core_block_tile_cnt", num_tiles_per_row},
                },
            .hw_config = std::move(compute_hw_config),
        };
    };

    // ---- Per-core runtime args ----
    uint32_t tile_height = output.tensor_spec().tile().get_height();
    auto core_assignments = ttnn::distribute_work(
        output_shape, input_shape, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff, tile_height);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;

    const auto& cores = corerange_to_cores(available_grid);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    reader_run_args.runtime_arg_values.reserve(ncores);
    writer_run_args.runtime_arg_values.reserve(ncores);

    // Per-core vararg counts (block-rep tuples vary in number per core). Declared via the
    // per-node vararg override so the framework validates each core's exact vararg length.
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // writer block-rep varargs (run-length-compressed 5-tuples)
        AdvancedKernelRunArgs::Varargs writer_varargs;
        uint32_t row_start_id_core = row_start_id;

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
        reader_run_args.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core, .args = {{"num_pages", num_tiles_per_core}, {"start_id", tile_start_id}}});

        // writer runtime args: fixed named RTAs + per-core vararg block-rep tuples
        writer_run_args.runtime_arg_values.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core,
            .args = {
                {"padded_X_size", padded_row_size_bytes},
                {"start_stick_id", row_start_id_core},
                {"n_block_reps", static_cast<uint32_t>(assignment.size())}}});
        writer.advanced_options.num_runtime_varargs_per_node[Nodes{NodeCoord{core}}] =
            static_cast<uint32_t>(writer_varargs.size());
        writer_run_args.advanced_options.runtime_varargs[core] = std::move(writer_varargs);

        tile_start_id += num_tiles_per_core;
    }

    // ---- Work units ----
    // Local DFBs (IN, OUT) require their producer/consumer KernelSpecs to share the SAME
    // WorkUnitSpec(s). Reader/writer run on all_cores (so they are members of both groups' WUs);
    // the compute differs per work-split group only in its per_core_block_cnt CTA (preserving
    // compile-time loop unrolling — no CTA→RTA demotion). Assembled after the RTA loop so the
    // writer's per-node vararg-count schema (populated above) is included when it is moved in.
    Group<KernelSpec> kernels;
    std::vector<WorkUnitSpec> work_units;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    if (!core_range.empty()) {
        kernels.push_back(make_compute(COMPUTE_FULL, nblocks_per_core));
        work_units.push_back(
            WorkUnitSpec{.name = "uwu_full", .kernels = {READER, WRITER, COMPUTE_FULL}, .target_nodes = core_range});
    }
    if (has_cliff) {
        kernels.push_back(make_compute(COMPUTE_CLIFF, nblocks_per_core_cliff));
        work_units.push_back(WorkUnitSpec{
            .name = "uwu_cliff", .kernels = {READER, WRITER, COMPUTE_CLIFF}, .target_nodes = core_range_cliff});
    }

    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_interleaved",
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
