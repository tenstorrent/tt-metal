// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/select_tile_rows/device/select_tile_rows_program_factory.hpp"

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

ttnn::device_operation::ProgramArtifacts SelectTileRowsProgramFactory::create_program_artifacts(
    const SelectTileRowsParams& attrs, const SelectTileRowsInputs& in, std::vector<Tensor>& outputs) {
    namespace m2 = tt::tt_metal::experimental;
    const auto& input = in.input.mesh_tensor();
    const auto& indices = in.indices.mesh_tensor();
    const auto& output = outputs[0].mesh_tensor();
    const auto& device = input.device();

    const uint32_t rows = in.indices.logical_shape()[0];
    const uint32_t input_row_tiles = in.input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t width_tiles = attrs.width / tt::constants::TILE_WIDTH;
    // A few column tiles per worker keep the gather latency-bound.
    constexpr uint32_t tiles_per_worker = 8;
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t workers = std::min<uint32_t>(grid.x * grid.y, tt::div_up(width_tiles, tiles_per_worker));
    const uint32_t tiles_per_core = tt::div_up(width_tiles, workers);
    const auto cores = tt::tt_metal::num_cores_to_corerangeset(workers, grid, /*row_wise=*/true);
    // [indices | two aligned 64-byte spans per column tile | gathered 64-byte tile rows per index]
    const uint32_t staging_bytes = 64 + tiles_per_core * (2 * 64 + rows * 64);

    const m2::KernelSpecName dataflow_kernel_name{"dataflow"};
    const m2::DFBSpecName staging_dfb{"staging"};
    const m2::TensorParamName input_name{"input"};
    const m2::TensorParamName indices_name{"indices"};
    const m2::TensorParamName output_name{"output"};

    m2::KernelSpec dataflow{
        .unique_id = dataflow_kernel_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/select_tile_rows/device/kernels/dataflow/select_tile_rows.cpp",
        .dfb_bindings = {m2::ProducerOf(staging_dfb, "staging"), m2::ConsumerOf(staging_dfb, "staging")},
        .tensor_bindings =
            {
                m2::TensorBinding{input_name, "input"},
                m2::TensorBinding{indices_name, "indices"},
                m2::TensorBinding{output_name, "output"},
            },
        .compile_time_args =
            {{"rows", rows},
             {"input_row_tiles", input_row_tiles},
             {"width_tiles", width_tiles},
             {"tiles_per_core", tiles_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"first_tile"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    m2::KernelRunArgs dataflow_run{.kernel = dataflow_kernel_name};
    for (uint32_t index = 0; index < workers; ++index) {
        const tt::tt_metal::CoreCoord core{index % grid.x, index / grid.x};
        m2::AddRuntimeArgsForNode(dataflow_run.runtime_arg_values, core, {{"first_tile", index * tiles_per_core}});
    }

    m2::ProgramSpec spec{
        .name = "select_tile_rows",
        .kernels = {std::move(dataflow)},
        .dataflow_buffers = {m2::DataflowBufferSpec{
            .unique_id = staging_dfb,
            .entry_size = staging_bytes,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Float16_b}},
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = input_name, .spec = input.tensor_spec()},
                m2::TensorParameter{.unique_id = indices_name, .spec = indices.tensor_spec()},
                m2::TensorParameter{.unique_id = output_name, .spec = output.tensor_spec()},
            },
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {dataflow_kernel_name}, .target_nodes = cores}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(dataflow_run)};
    run_args.tensor_args = {{input_name, input}, {indices_name, indices}, {output_name, output}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
