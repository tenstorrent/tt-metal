// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"

#include "ttnn/operations/experimental/kda/select_final_carry/device/select_final_carry_program_factory.hpp"

#include <algorithm>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {

ttnn::device_operation::MeshWorkloadArtifacts SelectFinalCarryProgramFactory::create_mesh_workload_artifacts(
    const SelectFinalCarryParams& attrs,
    const SelectFinalCarryInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    namespace m2 = tt::tt_metal::experimental;
    const auto& rank_finals = in.rank_finals.mesh_tensor();
    const auto& prefix_final = in.prefix_final.mesh_tensor();
    const auto& output = outputs[0].mesh_tensor();
    const auto& device = rank_finals.device();

    const uint32_t state_tiles = in.prefix_final.physical_volume() / tt::constants::TILE_HW;
    // Each worker copies a contiguous run of tiles; a few tiles per core keep the copy latency-bound.
    constexpr uint32_t tiles_per_worker = 8;
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t workers =
        std::min<uint32_t>(grid.x * grid.y, (state_tiles + tiles_per_worker - 1) / tiles_per_worker);
    const uint32_t tiles_per_core = (state_tiles + workers - 1) / workers;
    const auto cores = tt::tt_metal::num_cores_to_corerangeset(workers, grid, /*row_wise=*/true);

    const m2::KernelSpecName dataflow_kernel_name{"dataflow"};
    const m2::DFBSpecName staging_dfb{"staging"};
    const m2::TensorParamName rank_finals_name{"rank_finals"};
    const m2::TensorParamName prefix_final_name{"prefix_final"};
    const m2::TensorParamName output_name{"output"};
    const m2::TensorParamName actual_start_name{"actual_start"};
    constexpr auto fp32 = tt::DataFormat::Float32;

    m2::KernelSpec dataflow{
        .unique_id = dataflow_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/select_final_carry/device/kernels/dataflow/"
            "select_final_carry.cpp",
        .dfb_bindings = {m2::ProducerOf(staging_dfb, "staging"), m2::ConsumerOf(staging_dfb, "staging")},
        .tensor_bindings =
            {
                m2::TensorBinding{rank_finals_name, "rank_finals"},
                m2::TensorBinding{prefix_final_name, "prefix_final"},
                m2::TensorBinding{output_name, "output"},
                m2::TensorBinding{actual_start_name, "actual_start"},
            },
        .compile_time_args = {{"state_tiles", state_tiles}, {"tiles_per_core", tiles_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"first_tile"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    m2::KernelRunArgs dataflow_run{.kernel = dataflow_kernel_name};
    for (uint32_t index = 0; index < workers; ++index) {
        const tt::tt_metal::CoreCoord core{index % grid.x, index / grid.x};
        m2::AddRuntimeArgsForNode(dataflow_run.runtime_arg_values, core, {{"first_tile", index * tiles_per_core}});
    }

    m2::ProgramSpec spec{
        .name = "select_final_carry",
        .dataflow_buffers = {m2::DataflowBufferSpec{
            .unique_id = staging_dfb,
            .entry_size = tt::tile_size(fp32),
            .num_entries = tiles_per_core,
            .data_format_metadata = fp32}},
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = rank_finals_name, .spec = rank_finals.tensor_spec()},
                m2::TensorParameter{.unique_id = prefix_final_name, .spec = prefix_final.tensor_spec()},
                m2::TensorParameter{.unique_id = output_name, .spec = output.tensor_spec()},
                m2::TensorParameter{
                    .unique_id = actual_start_name, .spec = in.actual_start.mesh_tensor().tensor_spec()},
            },
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {dataflow_kernel_name}, .target_nodes = cores}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(dataflow_run)};
    run_args.tensor_args = {
        {rank_finals_name, rank_finals},
        {prefix_final_name, prefix_final},
        {output_name, output},
        {actual_start_name, in.actual_start.mesh_tensor()},
    };
    kda_factory_detail::bind_actual_end(spec, run_args, in.actual_end, dataflow);
    spec.kernels = {std::move(dataflow)};
    return kda_factory_detail::chronology_workload(
        ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)},
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        attrs.local_rows,
        dataflow_kernel_name);
}

}  // namespace ttnn::experimental::prim
