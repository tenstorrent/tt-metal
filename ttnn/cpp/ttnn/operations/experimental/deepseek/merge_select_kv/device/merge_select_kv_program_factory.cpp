// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "merge_select_kv_device_operation.hpp"

#include <algorithm>
#include <array>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek::merge_select_kv {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr const char* kMergeSource =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/merge_select_kv/device/kernels/dataflow/merge_select_kv.cpp";

// Output rows each core gathers. Every core runs the (small) merge itself, so cores need no sync.
constexpr uint32_t kRowsPerCore = 32;
// kv rows each core stages in L1, as in fused_lightning_select_kv's gather.
constexpr uint32_t kKvStageRows = 16;
constexpr uint32_t kHistBins = 256;

}  // namespace

ttnn::device_operation::ProgramArtifacts MergeSelectKvDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    auto* device = tensor_args.kv_cache.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_parts = tensor_args.candidates.logical_shape()[-2];
    const uint32_t cand_bytes = tensor_args.candidates.buffer()->page_size();
    const uint32_t num_cores = std::min<uint32_t>(grid.x * grid.y, tt::div_up(args.k, kRowsPerCore));
    const uint32_t rows_per_core = tt::div_up(args.k, num_cores);
    const auto cores =
        corerange_to_cores(CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})), num_cores, /*row_wise=*/true);
    std::vector<CoreRange> core_ranges;
    for (const auto& core : cores) {
        core_ranges.emplace_back(core, core);
    }
    const CoreRangeSet all_cores(core_ranges);

    // ---- Resource names ----
    const DFBSpecName CAND_DFB{"cand"};
    const DFBSpecName HIST_DFB{"hist"};
    const DFBSpecName ROWS_DFB{"rows"};
    const DFBSpecName STAGE_DFB{"stage"};
    const TensorParamName CANDIDATES{"candidates"};
    const TensorParamName KV_CACHE{"kv_cache"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName MERGE{"merge"};

    // ---- Dataflow buffers (all kernel-local scratch, reserved and never pushed) ----
    const auto local_dfb = [](const DFBSpecName& name, const char* accessor) {
        return std::array{
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::CONSUMER},
        };
    };
    const auto cand = local_dfb(CAND_DFB, "cand");
    const auto hist = local_dfb(HIST_DFB, "hist");
    const auto rows = local_dfb(ROWS_DFB, "rows");
    const auto stage = local_dfb(STAGE_DFB, "stage");

    const KernelSpec merge{
        .unique_id = MERGE,
        .source = kMergeSource,
        .dfb_bindings = {cand[0], cand[1], hist[0], hist[1], rows[0], rows[1], stage[0], stage[1]},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = CANDIDATES, .accessor_name = "candidates"},
             TensorBinding{.tensor_parameter_name = KV_CACHE, .accessor_name = "kv_cache"},
             TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"k", args.k},
             {"num_parts", num_parts},
             {"cand_page_bytes", cand_bytes},
             {"num_hist_bins", kHistBins},
             {"rows_per_core", rows_per_core},
             {"kv_stage_rows", kKvStageRows},
             {"output_row_offset", args.output_row_offset}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_index"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    ProgramSpec spec{
        .name = "merge_select_kv",
        .kernels = {merge},
        .dataflow_buffers =
            {DataflowBufferSpec{
                 .unique_id = CAND_DFB,
                 .entry_size = num_parts * cand_bytes,
                 .num_entries = 1,
                 .data_format_metadata = tt::DataFormat::UInt32,
             },
             DataflowBufferSpec{
                 .unique_id = HIST_DFB,
                 .entry_size = kHistBins * static_cast<uint32_t>(sizeof(uint32_t)),
                 .num_entries = 1,
                 .data_format_metadata = tt::DataFormat::UInt32,
             },
             DataflowBufferSpec{
                 .unique_id = ROWS_DFB,
                 .entry_size = tt::round_up(rows_per_core * static_cast<uint32_t>(sizeof(uint32_t)), 16u),
                 .num_entries = 1,
                 .data_format_metadata = tt::DataFormat::UInt32,
             },
             DataflowBufferSpec{
                 .unique_id = STAGE_DFB,
                 .entry_size = tensor_args.kv_cache.buffer()->page_size(),
                 .num_entries = kKvStageRows,
                 .data_format_metadata = datatype_to_dataformat_converter(tensor_args.kv_cache.dtype()),
             }},
        .tensor_parameters =
            {TensorParameter{.unique_id = CANDIDATES, .spec = tensor_args.candidates.tensor_spec()},
             TensorParameter{.unique_id = KV_CACHE, .spec = tensor_args.kv_cache.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {WorkUnitSpec{.name = "merge_select_kv", .kernels = {MERGE}, .target_nodes = all_cores}},
    };

    KernelRunArgs merge_run_args{.kernel = MERGE};
    for (uint32_t i = 0; i < num_cores; ++i) {
        AddRuntimeArgsForNode(merge_run_args.runtime_arg_values, cores[i], {{"core_index", i}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(merge_run_args)};
    run_args.tensor_args = {
        {CANDIDATES, tensor_args.candidates.mesh_tensor()},
        {KV_CACHE, tensor_args.kv_cache.mesh_tensor()},
        {OUTPUT, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::deepseek::merge_select_kv
