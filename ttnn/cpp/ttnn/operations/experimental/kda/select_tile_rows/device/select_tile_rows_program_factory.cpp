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
#include "ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"

namespace ttnn::experimental::prim {

uint32_t select_tile_rows_count(const SelectTileRowsParams& attrs, const SelectTileRowsInputs& in) {
    if (!attrs.record.has_value()) {
        return in.indices->logical_shape()[0];
    }
    using namespace kda_chronology::selection;
    return *attrs.record == outgoing_and_local_final_history || *attrs.record == predecessor_and_final_history
               ? packed_history_rows
               : history_rows;
}

ttnn::device_operation::MeshWorkloadArtifacts SelectTileRowsProgramFactory::create_mesh_workload_artifacts(
    const SelectTileRowsParams& attrs,
    const SelectTileRowsInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    namespace m2 = tt::tt_metal::experimental;
    const auto& input = in.input.mesh_tensor();
    const auto& device = input.device();
    const bool tiled = in.input.layout() == tt::tt_metal::Layout::TILE;

    const uint32_t rows = select_tile_rows_count(attrs, in);
    const uint32_t rows_per_output = attrs.rows_per_output == 0 ? rows : attrs.rows_per_output;
    const uint32_t width_tiles = attrs.width / tt::constants::TILE_WIDTH;
    // A few column tiles per worker keep the gather latency-bound.
    constexpr uint32_t tiles_per_worker = 8;
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t workers = std::min<uint32_t>(grid.x * grid.y, tt::div_up(width_tiles, tiles_per_worker));
    const uint32_t tiles_per_core = tt::div_up(width_tiles, workers);
    const auto cores = tt::tt_metal::num_cores_to_corerangeset(workers, grid, /*row_wise=*/true);
    // Tiled: [indices | two aligned 64-byte spans per column tile | gathered 64-byte tile rows per index].
    // Row-major: [indices | each selected row's column slice].
    const uint32_t staging_bytes = tiled ? 64 + tiles_per_core * (2 * 64 + rows * 64) : 64 + rows * tiles_per_core * 64;

    const m2::KernelSpecName dataflow_kernel_name{"dataflow"};
    const m2::DFBSpecName staging_dfb{"staging"};
    const m2::TensorParamName input_name{"input"};
    const m2::TensorParamName indices_name{"indices"};
    const m2::TensorParamName output_name{"output"};
    const m2::TensorParamName output_second_name{"output_second"};
    const m2::TensorParamName actual_start_name{"actual_start"};
    const m2::TensorParamName actual_end_name{"actual_end"};
    constexpr uint32_t no_record = ~0U;

    m2::KernelSpec dataflow{
        .unique_id = dataflow_kernel_name,
        .source = tiled ? "ttnn/cpp/ttnn/operations/experimental/kda/select_tile_rows/device/kernels/dataflow/"
                          "select_tile_rows.cpp"
                        : "ttnn/cpp/ttnn/operations/experimental/kda/select_tile_rows/device/kernels/dataflow/"
                          "select_rows.cpp",
        .dfb_bindings = {m2::ProducerOf(staging_dfb, "staging"), m2::ConsumerOf(staging_dfb, "staging")},
        .tensor_bindings =
            {
                m2::TensorBinding{input_name, "input"},
                m2::TensorBinding{output_name, "output"},
                // A single output aliases the second binding.
                m2::TensorBinding{outputs.size() > 1 ? output_second_name : output_name, "output_second"},
            },
        .compile_time_args =
            {{"rows", rows},
             {"rows_per_output", rows_per_output},
             {"input_row_tiles", static_cast<uint32_t>(in.input.padded_shape()[-1]) / tt::constants::TILE_WIDTH},
             {"width_tiles", width_tiles},
             {"tiles_per_core", tiles_per_core},
             {"record", attrs.record.value_or(no_record)},
             {"has_actual_end", static_cast<uint32_t>(in.actual_end.has_value())}},
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
        .dataflow_buffers = {m2::DataflowBufferSpec{
            .unique_id = staging_dfb,
            .entry_size = staging_bytes,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::Float16_b}},
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = input_name, .spec = input.tensor_spec()},
                m2::TensorParameter{.unique_id = output_name, .spec = outputs[0].mesh_tensor().tensor_spec()},
            },
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {dataflow_kernel_name}, .target_nodes = cores}},
    };
    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(dataflow_run)};
    run_args.tensor_args = {{input_name, input}, {output_name, outputs[0].mesh_tensor()}};
    if (outputs.size() > 1) {
        spec.tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = output_second_name, .spec = outputs[1].mesh_tensor().tensor_spec()});
        run_args.tensor_args.emplace(output_second_name, outputs[1].mesh_tensor());
    }
    // Optional tensors stay unbound when absent; the kernel resolves them with get_token_if_present.
    const auto bind = [&](const std::optional<Tensor>& tensor, const m2::TensorParamName& name) {
        if (tensor.has_value()) {
            spec.tensor_parameters.push_back(
                m2::TensorParameter{.unique_id = name, .spec = tensor->mesh_tensor().tensor_spec()});
            run_args.tensor_args.emplace(name, tensor->mesh_tensor());
            dataflow.tensor_bindings.push_back(m2::TensorBinding{name, *name});
        }
    };
    bind(in.indices, indices_name);
    bind(in.actual_start, actual_start_name);
    bind(in.actual_end, actual_end_name);
    spec.kernels = {std::move(dataflow)};
    return kda_factory_detail::chronology_workload(
        {.spec = std::move(spec), .run_params = std::move(run_args)},
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        attrs.local_rows,
        dataflow_kernel_name);
}

}  // namespace ttnn::experimental::prim
