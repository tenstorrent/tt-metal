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

// Cores merging the keys and, in candidate source mode, the candidate blocks. Each splits the 256
// histogram bins evenly, so both are powers of two of at most 64.
constexpr uint32_t kKeyCores = 16;
constexpr uint32_t kBlockCores = 32;
// kv rows each core stages in L1, as in fused_lightning_select_kv's gather.
constexpr uint32_t kKvStageRows = 16;
constexpr uint32_t kHistBins = 256;
constexpr uint32_t kHeaderWords = 16;
constexpr uint32_t kCountWords = 4;
// Once at most this many keys are tied with the radix prefix, the cores exchange them and finish
// locally instead of running more passes. Each core's message is 4 header words and the keys.
constexpr uint32_t kMaxTies = 64;
constexpr uint32_t kTieHeaderWords = 4;
// Slices, parts and block-list ranges start on multiples of this many entries (64 B), so every NoC
// transfer of them is aligned.
constexpr uint32_t kAlignEntries = 16;

}  // namespace

ttnn::device_operation::ProgramArtifacts MergeSelectKvDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    auto* device = tensor_args.kv_cache.device();
    const auto grid = device->compute_with_storage_grid_size();
    const bool is_candidate_source = args.candidate_topk_blocks > 0;
    const uint32_t num_parts = tensor_args.candidates.logical_shape()[-2];
    // Each part's page: {k rows, k order keys, 16 header words}, then in candidate source mode
    // {candidate_topk_blocks blocks, their order keys}.
    const uint32_t key_section_bytes = (2 * args.k + kHeaderWords) * static_cast<uint32_t>(sizeof(uint32_t));
    const uint32_t num_key_cores = kKeyCores;
    const uint32_t num_block_cores = is_candidate_source ? kBlockCores : 0;
    const uint32_t num_cores = num_key_cores + num_block_cores;
    TT_FATAL(
        num_cores <= grid.x * grid.y, "merge_select_kv needs {} cores, the grid has {}", num_cores, grid.x * grid.y);
    const uint32_t key_slice_size = tt::round_up(tt::div_up(num_parts * args.k, num_key_cores), kAlignEntries);
    const uint32_t block_slice_size =
        is_candidate_source
            ? tt::round_up(tt::div_up(num_parts * args.candidate_topk_blocks, num_block_cores), kAlignEntries)
            : 0;
    const uint32_t rows_per_core = tt::div_up(args.k, num_key_cores);
    const uint32_t blocks_per_core =
        is_candidate_source ? tt::round_up(tt::div_up(args.candidate_topk_blocks, num_block_cores), kAlignEntries) : 0;
    const uint32_t max_slice = std::max(key_slice_size, block_slice_size);
    const uint32_t max_group = std::max(num_key_cores, num_block_cores);
    const uint32_t max_list = std::max(rows_per_core, blocks_per_core);

    const auto cores =
        corerange_to_cores(CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})), num_cores, /*row_wise=*/true);
    std::vector<CoreRange> core_ranges;
    for (const auto& core : cores) {
        core_ranges.emplace_back(core, core);
    }
    const CoreRangeSet all_cores(core_ranges);

    // ---- Resource names ----
    const DFBSpecName HEADER_DFB{"header"};
    const DFBSpecName SLICE_DFB{"slice"};
    const DFBSpecName HIST_DFB{"hist"};
    const DFBSpecName SCATTER_DFB{"scatter"};
    const DFBSpecName SUMS_DFB{"sums"};
    const DFBSpecName GLOBAL_DFB{"global"};
    const DFBSpecName COUNTS_DFB{"counts"};
    const DFBSpecName COMPACT_DFB{"compact"};
    const DFBSpecName TIES_DFB{"ties"};
    const DFBSpecName REMOTE_DFB{"remote"};
    const DFBSpecName LIST_DFB{"list"};
    const DFBSpecName STAGE_DFB{"stage"};
    const SemaphoreSpecName SEM_HIST_IN{"hist_in"};
    const SemaphoreSpecName SEM_SUM_IN{"sum_in"};
    const SemaphoreSpecName SEM_COUNTS_IN{"counts_in"};
    const SemaphoreSpecName SEM_TIES_IN{"ties_in"};
    const TensorParamName CANDIDATES{"candidates"};
    const TensorParamName KV_CACHE{"kv_cache"};
    const TensorParamName OUTPUT{"output"};
    const TensorParamName CANDIDATE_BLOCKS{"candidate_blocks"};
    const TensorParamName POOL_COUNT{"pool_count"};
    const KernelSpecName MERGE{"merge"};

    // ---- Dataflow buffers (all kernel-local scratch, reserved and never pushed) ----
    // Buffers other cores write into or read from must sit at the same L1 address on every core, which
    // holds because every core allocates the same buffers.
    const auto local_dfb = [](const DFBSpecName& name, const char* accessor) {
        return std::array{
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::CONSUMER},
        };
    };
    const auto scratch_dfb = [](const DFBSpecName& name, uint32_t bytes) {
        return DataflowBufferSpec{
            .unique_id = name,
            .entry_size = tt::round_up(bytes, 64u),
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32,
        };
    };
    constexpr auto word = static_cast<uint32_t>(sizeof(uint32_t));
    std::vector<DFBBinding> dfb_bindings;
    for (const auto& [name, accessor] :
         {std::pair{HEADER_DFB, "header"},
          std::pair{SLICE_DFB, "slice"},
          std::pair{HIST_DFB, "hist"},
          std::pair{SCATTER_DFB, "scatter"},
          std::pair{SUMS_DFB, "sums"},
          std::pair{GLOBAL_DFB, "global"},
          std::pair{COUNTS_DFB, "counts"},
          std::pair{COMPACT_DFB, "compact"},
          std::pair{TIES_DFB, "ties"},
          std::pair{REMOTE_DFB, "remote"},
          std::pair{LIST_DFB, "list"},
          std::pair{STAGE_DFB, "stage"}}) {
        const auto binding = local_dfb(name, accessor);
        dfb_bindings.insert(dfb_bindings.end(), binding.begin(), binding.end());
    }
    std::vector<DataflowBufferSpec> dataflow_buffers = {
        scratch_dfb(HEADER_DFB, num_parts * kHeaderWords * word),
        // entries, then their order keys
        scratch_dfb(SLICE_DFB, 2 * max_slice * word),
        scratch_dfb(HIST_DFB, kHistBins * word),
        scratch_dfb(SCATTER_DFB, kHistBins * word),
        scratch_dfb(SUMS_DFB, kHistBins * word),
        scratch_dfb(GLOBAL_DFB, kHistBins * word),
        scratch_dfb(COUNTS_DFB, max_group * kCountWords * word),
        scratch_dfb(COMPACT_DFB, max_slice * word),
        scratch_dfb(TIES_DFB, max_group * (kTieHeaderWords + kMaxTies) * word),
        scratch_dfb(REMOTE_DFB, max_group * 2 * max_slice * word),
        // pool_count 64 B past the longest block list
        scratch_dfb(LIST_DFB, tt::round_up(max_list * word, 64u) + 64),
        DataflowBufferSpec{
            .unique_id = STAGE_DFB,
            .entry_size = tensor_args.kv_cache.buffer()->page_size(),
            .num_entries = kKvStageRows,
            .data_format_metadata = datatype_to_dataformat_converter(tensor_args.kv_cache.dtype()),
        }};
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = CANDIDATES, .spec = tensor_args.candidates.tensor_spec()},
        TensorParameter{.unique_id = KV_CACHE, .spec = tensor_args.kv_cache.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}};
    Group<TensorBinding> tensor_bindings = {
        TensorBinding{.tensor_parameter_name = CANDIDATES, .accessor_name = "candidates"},
        TensorBinding{.tensor_parameter_name = KV_CACHE, .accessor_name = "kv_cache"},
        TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}};
    KernelSpec::CompilerOptions compiler_options;
    if (is_candidate_source) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = CANDIDATE_BLOCKS, .spec = tensor_args.candidate_blocks_output->tensor_spec()});
        tensor_parameters.push_back(
            TensorParameter{.unique_id = POOL_COUNT, .spec = tensor_args.pool_count_output->tensor_spec()});
        tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = CANDIDATE_BLOCKS, .accessor_name = "candidate_blocks"});
        tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = POOL_COUNT, .accessor_name = "pool_count"});
        compiler_options.defines.emplace("CANDIDATE_SOURCE", "1");
    }

    const KernelSpec merge{
        .unique_id = MERGE,
        .source = kMergeSource,
        .compiler_options = std::move(compiler_options),
        .dfb_bindings = std::move(dfb_bindings),
        .semaphore_bindings =
            {
                SemaphoreBinding{.semaphore_spec_name = SEM_HIST_IN, .accessor_name = "hist_in"},
                SemaphoreBinding{.semaphore_spec_name = SEM_SUM_IN, .accessor_name = "sum_in"},
                SemaphoreBinding{.semaphore_spec_name = SEM_COUNTS_IN, .accessor_name = "counts_in"},
                SemaphoreBinding{.semaphore_spec_name = SEM_TIES_IN, .accessor_name = "ties_in"},
            },
        .tensor_bindings = std::move(tensor_bindings),
        .compile_time_args =
            {{"k", args.k},
             {"num_parts", num_parts},
             {"key_section_bytes", key_section_bytes},
             {"num_hist_bins", kHistBins},
             {"max_ties", kMaxTies},
             {"num_key_cores", num_key_cores},
             {"num_block_cores", num_block_cores},
             {"key_slice_size", key_slice_size},
             {"block_slice_size", block_slice_size},
             {"rows_per_core", rows_per_core},
             {"blocks_per_core", blocks_per_core},
             {"kv_stage_rows", kKvStageRows},
             {"output_row_offset", args.output_row_offset},
             {"candidate_topk_blocks", args.candidate_topk_blocks},
             {"candidate_block_size", args.candidate_block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_index"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
        // NoC (x, y) of every core, in core_index order.
        .advanced_options = {.num_common_runtime_varargs = 2 * num_cores},
    };

    ProgramSpec spec{
        .name = "merge_select_kv",
        .kernels = {merge},
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores =
            {SemaphoreSpec{.unique_id = SEM_HIST_IN, .target_nodes = all_cores},
             SemaphoreSpec{.unique_id = SEM_SUM_IN, .target_nodes = all_cores},
             SemaphoreSpec{.unique_id = SEM_COUNTS_IN, .target_nodes = all_cores},
             SemaphoreSpec{.unique_id = SEM_TIES_IN, .target_nodes = all_cores}},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {WorkUnitSpec{.name = "merge_select_kv", .kernels = {MERGE}, .target_nodes = all_cores}},
    };

    KernelRunArgs merge_run_args{.kernel = MERGE};
    for (uint32_t i = 0; i < num_cores; ++i) {
        AddRuntimeArgsForNode(merge_run_args.runtime_arg_values, cores[i], {{"core_index", i}});
        const auto physical = device->worker_core_from_logical_core(cores[i]);
        merge_run_args.advanced_options.common_runtime_varargs.push_back(physical.x);
        merge_run_args.advanced_options.common_runtime_varargs.push_back(physical.y);
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(merge_run_args)};
    run_args.tensor_args = {
        {CANDIDATES, tensor_args.candidates.mesh_tensor()},
        {KV_CACHE, tensor_args.kv_cache.mesh_tensor()},
        {OUTPUT, output.mesh_tensor()},
    };
    if (is_candidate_source) {
        run_args.tensor_args.emplace(CANDIDATE_BLOCKS, std::cref(tensor_args.candidate_blocks_output->mesh_tensor()));
        run_args.tensor_args.emplace(POOL_COUNT, std::cref(tensor_args.pool_count_output->mesh_tensor()));
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::deepseek::merge_select_kv
