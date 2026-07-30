// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_interleaved_program_factory.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
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

    const bool float32_dtype = input_cb_data_format == tt::DataFormat::Float32 or
                               input_cb_data_format == tt::DataFormat::UInt32 or
                               input_cb_data_format == tt::DataFormat::Int32;

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};

    // ---- DataflowBuffers (legacy c_0 / c_16 CBs) ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_row,
        .data_format_metadata = output_cb_data_format,
    };

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Reader kernel ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/"
            "reader_unary_interleaved_start_id.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config =
            ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // ---- Writer kernel ----
    // The per-BlockRep groups are a runtime-variable count, so they are carried as
    // runtime varargs (one 5-uint group per BlockRep run, count varying per core).
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_stick_layout_split_rows_multicore.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"float32_dtype", static_cast<uint32_t>(float32_dtype)}, {"unpadded_X_size", unpadded_row_size_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"padded_X_size", "start_stick_id", "n_block_reps"}},
    };
    writer.hw_config =
        ttnn::create_writer_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true);

    // ---- Compute kernels (full + cliff) ----
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (float32_dtype) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    auto make_compute_hw = [&]() -> ComputeHardwareConfig {
        ttnn::ComputeKernelConfig cfg{
            .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
        ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(device->arch(), cfg);
        if (fp32_dest_acc_en) {
            std::visit(
                [&](auto& c) { c.unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest); }, compute_hw);
        }
        return compute_hw;
    };
    const std::filesystem::path compute_source(
        "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/compute/"
        "untilize_metal2.cpp");
    auto make_compute = [&](const KernelSpecName& id, uint32_t nblocks) {
        return KernelSpec{
            .unique_id = id,
            .source = compute_source,
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args = {{"per_core_block_cnt", nblocks}, {"per_core_block_tile_cnt", num_tiles_per_row}},
            .hw_config = make_compute_hw(),
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    if (!core_range.empty()) {
        kernels.push_back(make_compute(COMPUTE_FULL, nblocks_per_core));
        work_units.push_back(
            WorkUnitSpec{.name = "wu_full", .kernels = {READER, WRITER, COMPUTE_FULL}, .target_nodes = core_range});
    }
    if (has_cliff) {
        kernels.push_back(make_compute(COMPUTE_CLIFF, nblocks_per_core_cliff));
        work_units.push_back(WorkUnitSpec{
            .name = "wu_cliff", .kernels = {READER, WRITER, COMPUTE_CLIFF}, .target_nodes = core_range_cliff});
    }

    // ---- Per-core runtime args ----
    uint32_t tile_height = output.tensor_spec().tile().get_height();
    auto core_assignments = ttnn::distribute_work(
        output_shape, input_shape, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff, tile_height);

    uint32_t tile_start_id = 0;

    const auto& cores = corerange_to_cores(available_grid);

    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;
    Table<NodeCoord, AdvancedKernelRunArgs::Varargs> writer_varargs;
    uint32_t max_varargs = 0;

    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // Build the per-BlockRep vararg tail (mirrors the legacy run-length encoding).
        AdvancedKernelRunArgs::Varargs writer_tail;

        uint32_t nblocks_per_core_core = 0;
        BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;  // will be incremented in first iteration of the loop
        for (const auto& el : assignment) {
            nblocks_per_core_core += el.block_count();
            if (compare_assignments(ref_el, el)) {
                count_repeated++;
            } else {
                writer_tail.push_back(ref_el.n_data);
                writer_tail.push_back(ref_el.n_mixed);
                writer_tail.push_back(ref_el.n_pads);
                writer_tail.push_back(ref_el.times);
                writer_tail.push_back(count_repeated);
                ref_el = el;
                count_repeated = 1;
            }
        }
        writer_tail.push_back(ref_el.n_data);
        writer_tail.push_back(ref_el.n_mixed);
        writer_tail.push_back(ref_el.n_pads);
        writer_tail.push_back(ref_el.times);
        writer_tail.push_back(count_repeated);

        const uint32_t n_block_reps = static_cast<uint32_t>(assignment.size());
        max_varargs = std::max<uint32_t>(max_varargs, static_cast<uint32_t>(writer_tail.size()));

        // Writer named RTAs (the legacy fixed prefix, minus the dropped buffer address).
        // start_stick_id (legacy row_start_id at the core's start) is filled in by the
        // fixup loop below.
        AddRuntimeArgsForNode(
            writer_node_args,
            core,
            {
                {"padded_X_size", padded_row_size_bytes},
                {"start_stick_id", 0u},
                {"n_block_reps", n_block_reps},
            });

        writer_varargs.emplace(core, std::move(writer_tail));

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core_core;

        AddRuntimeArgsForNode(
            reader_node_args,
            core,
            {
                {"num_pages", num_tiles_per_core},
                {"start_id", tile_start_id},
            });

        tile_start_id += num_tiles_per_core;
    }

    // The legacy writer's start_stick_id is `row_start_id` AS OF the start of the core's
    // assignment (row_start_id accumulates each core's data_row_count over all its blocks).
    // Compute it per core so the run-arg values match the legacy semantics exactly.
    {
        uint32_t rsid = 0;
        for (uint32_t i = 0; i < ncores; ++i) {
            const std::vector<BlockRep>& assignment = core_assignments.at(i);
            writer_node_args["start_stick_id"][cores[i]] = rsid;
            for (const auto& el : assignment) {
                rsid += el.data_row_count();
            }
        }
    }

    // Every node must supply exactly num_runtime_varargs words, so pad each core's
    // vararg vector up to the max with zeros. The padding is never read (the writer's
    // loop is bounded by n_block_reps); this just satisfies the uniform-count contract.
    for (auto& entry : writer_varargs) {
        entry.second.resize(max_varargs, 0u);
    }

    // Declare the writer's vararg count (uniform max over cores). The writer is
    // kernels[1] (reader is [0]).
    kernels[1].advanced_options.num_runtime_varargs = max_varargs;

    // ---- ProgramSpec ----
    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_interleaved",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    // ---- ProgramRunArgs ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_args{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)};
    KernelRunArgs writer_args{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)};
    writer_args.advanced_options.runtime_varargs = std::move(writer_varargs);
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args)};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
