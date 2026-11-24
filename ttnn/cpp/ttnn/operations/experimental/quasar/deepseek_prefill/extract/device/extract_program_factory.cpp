// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "extract_program_factory.hpp"

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::quasar::deepseek_prefill::extract {

ttnn::device_operation::ProgramArtifacts ExtractProgramFactory::create_program_artifacts(
    const ExtractParams& operation_attributes, const ExtractInputs& tensor_args, Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName TILE{"tile"};
    const ScratchpadSpecName START_SCRATCH{"start_scratch"};
    const ScratchpadSpecName COUNTS_SCRATCH_READER{"counts_scratch_reader"};
    const ScratchpadSpecName COUNTS_SCRATCH_WRITER{"counts_scratch_writer"};
    const ScratchpadSpecName GLOBAL_EXPERT_IDX_SCRATCH_READER{"global_expert_idx_scratch_reader"};
    const ScratchpadSpecName GLOBAL_EXPERT_IDX_SCRATCH_WRITER{"global_expert_idx_scratch_writer"};
    const TensorParamName GLOBAL{"global"};
    const TensorParamName START{"start"};
    const TensorParamName COUNTS{"counts"};
    const TensorParamName GLOBAL_EXPERT_IDX_TABLE{"global_expert_idx_table"};
    const TensorParamName OUTPUT{"output"};

    const auto& global_tensor = tensor_args.global_tensor;
    const auto& start = tensor_args.start;
    const auto& counts = tensor_args.counts;
    const auto& global_expert_idx_table = tensor_args.global_expert_idx_table;
    Tensor& output_tensor = tensor_return_value;

    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    const auto hidden_dim = global_tensor.logical_shape()[-1];
    const auto global_rows = global_tensor.logical_shape()[0];
    const uint32_t tiles_per_row = hidden_dim / tile_width;
    // Total tiles available in global_tensor — passed to the reader so it can
    // assert (at runtime) that start[id] + ceil_tile(counts[id]) stays within
    // the tensor bounds.
    const uint32_t global_num_tiles = (global_rows / tt::constants::TILE_HEIGHT) * tiles_per_row;
    // Maximum tiles we're allowed to emit into the output — passed to both
    // kernels so they can assert ceil_tile(counts[id]) <= max_dispatched_tokens_per_expert.
    const uint32_t max_output_tiles =
        (operation_attributes.max_dispatched_tokens_per_expert / tt::constants::TILE_HEIGHT) * tiles_per_row;

    // NOTE: We do NOT verify the inter-expert layout invariant
    //   start[id] + counts[id] <= start[id + 1]
    // i.e. that one expert's slice doesn't bleed into the next expert's slice
    // in global_tensor. That's the caller's contract to uphold. Enforcing it
    // would require reading start/counts host-side (which we explicitly avoid
    // to keep the op device-local across a mesh) or cross-iteration state in
    // the kernels. The bounds checks below cover only "stay inside global_tensor"
    // and "stay inside the output tensor".

    auto* global_buffer = global_tensor.buffer();
    auto* start_buffer = start.buffer();
    auto* counts_buffer = counts.buffer();
    auto* global_expert_idx_table_buffer = global_expert_idx_table.buffer();
    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(global_buffer != nullptr, "global_tensor buffer must be allocated on device");
    TT_FATAL(start_buffer != nullptr, "start buffer must be allocated on device");
    TT_FATAL(counts_buffer != nullptr, "counts buffer must be allocated on device");
    TT_FATAL(global_expert_idx_table_buffer != nullptr, "global_expert_idx_table buffer must be allocated on device");
    TT_FATAL(output_buffer != nullptr, "output buffer must be allocated on device");

    const tt::DataFormat tile_data_format = tt::tt_metal::datatype_to_dataformat_converter(global_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(tile_data_format);

    const uint32_t start_page_size = start_buffer->aligned_page_size();
    const uint32_t counts_page_size = counts_buffer->aligned_page_size();
    const uint32_t global_expert_idx_table_page_size = global_expert_idx_table_buffer->aligned_page_size();

    // Multi-core implementation: every tensix core in the device's compute
    // grid participates. Cores are assigned a flat core_id in row-major order
    // (core_id = y * grid_x + x) and each handles a contiguous chunk of tile
    // rows computed as
    //   my_row_start = (num_tile_rows * core_id) / num_cores
    //   my_row_end   = (num_tile_rows * (core_id + 1)) / num_cores
    // so the remainder is distributed across the last few cores. The split is
    // computed inside each kernel from counts[global_expert_id], so no
    // host-side knowledge of the slice size is needed. When num_tile_rows <
    // num_cores, tail cores compute my_rows == 0 and exit immediately.
    const auto grid = global_tensor.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    constexpr bool row_wise = true;
    const auto cores = tt::tt_metal::grid_to_cores(num_cores, grid.x, grid.y, row_wise);
    const CoreRange core_range{{0, 0}, {grid.x - 1, grid.y - 1}};
    const CoreRangeSet core_range_set{core_range};

    // Deep pipeline so the reader can race ahead of the writer (and vice versa)
    // without stalling on reserve_back / wait_front. Each tile is small
    // (e.g. ~1 KB for bfp8_b) so even 32 entries is only ~32 KB of L1.
    constexpr uint32_t tile_buffering = 32;
    DataflowBufferSpec tile_dfb{
        .unique_id = TILE,
        .entry_size = single_tile_size,
        .num_entries = tile_buffering,
        .data_format_metadata = tile_data_format,
    };

    // Per-core L1 scratch for the start / counts / global_expert_idx_table pages, one per kernel that reads them.
    // Each is filled and read by a single DM kernel, so it is a scratchpad rather than a DFB.
    Group<ScratchpadSpec> scratchpads = {
        ScratchpadSpec{.unique_id = START_SCRATCH, .size_per_node = start_page_size},
        ScratchpadSpec{.unique_id = COUNTS_SCRATCH_READER, .size_per_node = counts_page_size},
        ScratchpadSpec{.unique_id = COUNTS_SCRATCH_WRITER, .size_per_node = counts_page_size},
        ScratchpadSpec{
            .unique_id = GLOBAL_EXPERT_IDX_SCRATCH_READER, .size_per_node = global_expert_idx_table_page_size},
        ScratchpadSpec{
            .unique_id = GLOBAL_EXPERT_IDX_SCRATCH_WRITER, .size_per_node = global_expert_idx_table_page_size},
    };

    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/extract/device/kernels/dataflow/"
            "reader_extract.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TILE,
                    .accessor_name = "tile",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .scratchpad_bindings =
            {
                ScratchpadBinding{.scratchpad_spec_name = START_SCRATCH, .accessor_name = "start"},
                ScratchpadBinding{.scratchpad_spec_name = COUNTS_SCRATCH_READER, .accessor_name = "counts"},
                ScratchpadBinding{
                    .scratchpad_spec_name = GLOBAL_EXPERT_IDX_SCRATCH_READER,
                    .accessor_name = "global_expert_idx",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = GLOBAL, .accessor_name = "global"},
                TensorBinding{.tensor_parameter_name = START, .accessor_name = "start"},
                TensorBinding{.tensor_parameter_name = COUNTS, .accessor_name = "counts"},
                TensorBinding{
                    .tensor_parameter_name = GLOBAL_EXPERT_IDX_TABLE,
                    .accessor_name = "global_expert_idx_table",
                },
            },
        .compile_time_args =
            {
                {"local_expert_id", operation_attributes.local_expert_id},
                {"tiles_per_row", tiles_per_row},
                {"global_num_tiles", global_num_tiles},
                {"max_output_tiles", max_output_tiles},
                {"num_cores", num_cores},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/extract/device/kernels/dataflow/"
            "writer_extract.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TILE,
                    .accessor_name = "tile",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .scratchpad_bindings =
            {
                ScratchpadBinding{.scratchpad_spec_name = COUNTS_SCRATCH_WRITER, .accessor_name = "counts"},
                ScratchpadBinding{
                    .scratchpad_spec_name = GLOBAL_EXPERT_IDX_SCRATCH_WRITER,
                    .accessor_name = "global_expert_idx",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
                TensorBinding{.tensor_parameter_name = COUNTS, .accessor_name = "counts"},
                TensorBinding{
                    .tensor_parameter_name = GLOBAL_EXPERT_IDX_TABLE,
                    .accessor_name = "global_expert_idx_table",
                },
            },
        .compile_time_args =
            {
                {"local_expert_id", operation_attributes.local_expert_id},
                {"tiles_per_row", tiles_per_row},
                {"max_output_tiles", max_output_tiles},
                {"num_cores", num_cores},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"core_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // Per-core runtime arg: core_id selects which chunk of tile rows this core processes.
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t core_id = 0; core_id < num_cores; ++core_id) {
        const auto& core = cores[core_id];
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"core_id", core_id}});
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"core_id", core_id}});
    }

    ProgramSpec spec{
        .name = "extract",
        .kernels = {std::move(reader_spec), std::move(writer_spec)},
        .dataflow_buffers = {std::move(tile_dfb)},
        .scratchpads = std::move(scratchpads),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = GLOBAL, .spec = global_tensor.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = START, .spec = start.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = COUNTS, .spec = counts.mesh_tensor().tensor_spec()},
                TensorParameter{
                    .unique_id = GLOBAL_EXPERT_IDX_TABLE,
                    .spec = global_expert_idx_table.mesh_tensor().tensor_spec(),
                },
                TensorParameter{.unique_id = OUTPUT, .spec = output_tensor.mesh_tensor().tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER},
                    .target_nodes = core_range_set,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(2);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args.emplace(GLOBAL, global_tensor.mesh_tensor());
    run_args.tensor_args.emplace(START, start.mesh_tensor());
    run_args.tensor_args.emplace(COUNTS, counts.mesh_tensor());
    run_args.tensor_args.emplace(GLOBAL_EXPERT_IDX_TABLE, global_expert_idx_table.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output_tensor.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::operations::experimental::quasar::deepseek_prefill::extract
