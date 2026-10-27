// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "build_candidate_pool_device_operation.hpp"

#include <algorithm>
#include <array>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek::build_candidate_pool {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr const char* kKeysSource =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/build_candidate_pool/device/kernels/dataflow/"
    "copy_pool_keys.cpp";

// Key chunks each kernel stages in L1; half are read while the other half drain.
constexpr uint32_t kKeyStageChunks = 64;

}  // namespace

ttnn::device_operation::ProgramArtifacts BuildCandidatePoolDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const Tensor& pool_keys = output.at(0);
    const Tensor& pool_rows = output.at(1);
    auto* device = tensor_args.key_cache.device();
    const auto grid = device->compute_with_storage_grid_size();
    const CoreRangeSet all_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));

    const uint32_t cbs = args.candidate_block_size;
    const uint32_t topk_blocks = tensor_args.candidate_blocks.logical_shape()[-1];
    const uint32_t page_block_size = tensor_args.key_cache.logical_shape()[-2];
    const uint32_t tiles_per_row = tensor_args.key_cache.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t num_cores = all_cores.num_cores();
    // A candidate block's keys are cbs rows of 2 faces in each tile along the row.
    const uint32_t key_chunk_bytes = cbs * 16 * static_cast<uint32_t>(sizeof(uint16_t));
    const uint32_t chunks_per_block = 2 * tiles_per_row;
    const uint32_t key_stage_chunks =
        std::max(kKeyStageChunks, 2 * chunks_per_block) / (2 * chunks_per_block) * (2 * chunks_per_block);

    // Each core takes a multiple of align_blocks candidate blocks, so its run of pool_rows starts on a
    // 64 B boundary.
    const uint32_t align_blocks = std::max(1u, 16 / cbs);
    const uint32_t max_per_core = tt::round_up(tt::div_up(topk_blocks, num_cores), align_blocks);
    // Lookup scratch: pool_count, one page-table chunk, the list slice (with up to 64 B in front of
    // it), the physical blocks, then the rows, each 64 B aligned.
    const uint32_t chunk_offset = 64;
    const uint32_t list_offset = 128;
    const uint32_t phys_offset =
        list_offset + tt::round_up(max_per_core * static_cast<uint32_t>(sizeof(uint32_t)), 64u) + 64;
    const uint32_t rows_offset =
        phys_offset + tt::round_up(max_per_core * static_cast<uint32_t>(sizeof(uint32_t)), 64u);
    const uint32_t lookup_bytes =
        rows_offset + tt::round_up(max_per_core * cbs * static_cast<uint32_t>(sizeof(uint32_t)), 64u);

    // ---- Resource names ----
    const DFBSpecName LOOKUP0_DFB{"lookup0"};
    const DFBSpecName STAGE0_DFB{"stage0"};
    const DFBSpecName LOOKUP1_DFB{"lookup1"};
    const DFBSpecName STAGE1_DFB{"stage1"};

    const TensorParamName CANDIDATE_BLOCKS{"candidate_blocks"};
    const TensorParamName POOL_COUNT{"pool_count"};
    const TensorParamName PAGE_TABLE{"page_table"};
    const TensorParamName KEY_CACHE{"key_cache"};
    const TensorParamName POOL_KEYS{"pool_keys"};
    const TensorParamName POOL_ROWS{"pool_rows"};
    const TensorParamName PAGE_TABLE_ROW{"page_table_row"};
    const bool has_page_table_row = tensor_args.page_table_row_tensor.has_value();

    const KernelSpecName KEYS0{"keys0"};
    const KernelSpecName KEYS1{"keys1"};

    // ---- Dataflow buffers (all kernel-local scratch, reserved and never pushed) ----
    const auto make_lookup_dfb = [lookup_bytes](const DFBSpecName& name) {
        return DataflowBufferSpec{
            .unique_id = name,
            .entry_size = lookup_bytes,
            .num_entries = 1,
            .data_format_metadata = tt::DataFormat::UInt32,
        };
    };
    const auto make_stage_dfb = [key_chunk_bytes, key_stage_chunks](const DFBSpecName& name) {
        return DataflowBufferSpec{
            .unique_id = name,
            .entry_size = key_chunk_bytes,
            .num_entries = key_stage_chunks,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
    };
    const auto local_dfb = [](const DFBSpecName& name, const char* accessor) {
        return std::array{
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::CONSUMER},
        };
    };
    const auto lookup0 = local_dfb(LOOKUP0_DFB, "lookup");
    const auto stage0 = local_dfb(STAGE0_DFB, "stage");
    const auto lookup1 = local_dfb(LOOKUP1_DFB, "lookup");
    const auto stage1 = local_dfb(STAGE1_DFB, "stage");

    // ---- Tensor parameters ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = CANDIDATE_BLOCKS, .spec = tensor_args.candidate_blocks.tensor_spec()},
        TensorParameter{.unique_id = POOL_COUNT, .spec = tensor_args.pool_count.tensor_spec()},
        TensorParameter{.unique_id = PAGE_TABLE, .spec = tensor_args.page_table_tensor.tensor_spec()},
        TensorParameter{.unique_id = KEY_CACHE, .spec = tensor_args.key_cache.tensor_spec()},
        TensorParameter{.unique_id = POOL_KEYS, .spec = pool_keys.tensor_spec()},
        TensorParameter{.unique_id = POOL_ROWS, .spec = pool_rows.tensor_spec()},
    };
    Group<TensorBinding> key_bindings = {
        TensorBinding{.tensor_parameter_name = CANDIDATE_BLOCKS, .accessor_name = "candidate_blocks"},
        TensorBinding{.tensor_parameter_name = POOL_COUNT, .accessor_name = "pool_count"},
        TensorBinding{.tensor_parameter_name = PAGE_TABLE, .accessor_name = "page_table"},
        TensorBinding{.tensor_parameter_name = KEY_CACHE, .accessor_name = "key_cache"},
        TensorBinding{.tensor_parameter_name = POOL_KEYS, .accessor_name = "pool_keys"},
    };
    KernelSpec::CompilerOptions keys_options;
    if (has_page_table_row) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = PAGE_TABLE_ROW, .spec = tensor_args.page_table_row_tensor->tensor_spec()});
        key_bindings.push_back(
            TensorBinding{.tensor_parameter_name = PAGE_TABLE_ROW, .accessor_name = "page_table_row"});
        keys_options.defines.emplace("PAGE_TABLE_ROW", "1");
    }
    Group<TensorBinding> rows_bindings = key_bindings;
    rows_bindings.push_back(TensorBinding{.tensor_parameter_name = POOL_ROWS, .accessor_name = "pool_rows"});

    // ---- Kernels ----
    // Both data-movement RISCs of every core copy keys, each half of the core's blocks on its own NoC;
    // the second also writes the core's rows.
    const auto compile_time_args = [&](uint32_t half) -> KernelSpec::CompileTimeArgs {
        return {
            {"candidate_block_size", cbs},
            {"page_block_size", page_block_size},
            {"tiles_per_row", tiles_per_row},
            {"stage_chunks", key_stage_chunks},
            {"half", half},
            {"align_blocks", align_blocks},
            {"chunk_offset", chunk_offset},
            {"list_offset", list_offset},
            {"phys_offset", phys_offset},
            {"rows_offset", rows_offset},
        };
    };
    KernelSpec::CompilerOptions rows_options = keys_options;
    rows_options.defines.emplace("WRITE_ROWS", "1");
    const KernelSpec keys0{
        .unique_id = KEYS0,
        .source = kKeysSource,
        .compiler_options = std::move(keys_options),
        .dfb_bindings = {lookup0[0], lookup0[1], stage0[0], stage0[1]},
        .tensor_bindings = key_bindings,
        .compile_time_args = compile_time_args(0),
        .runtime_arg_schema = {.runtime_arg_names = {"core_index", "num_cores"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    const KernelSpec keys1{
        .unique_id = KEYS1,
        .source = kKeysSource,
        .compiler_options = std::move(rows_options),
        .dfb_bindings = {lookup1[0], lookup1[1], stage1[0], stage1[1]},
        .tensor_bindings = std::move(rows_bindings),
        .compile_time_args = compile_time_args(1),
        .runtime_arg_schema = {.runtime_arg_names = {"core_index", "num_cores"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    ProgramSpec spec{
        .name = "build_candidate_pool",
        .kernels = {keys0, keys1},
        .dataflow_buffers =
            {make_lookup_dfb(LOOKUP0_DFB),
             make_stage_dfb(STAGE0_DFB),
             make_lookup_dfb(LOOKUP1_DFB),
             make_stage_dfb(STAGE1_DFB)},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {WorkUnitSpec{
            .name = "build_candidate_pool", .kernels = {KEYS0, KEYS1}, .target_nodes = all_cores}},
    };

    const auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    KernelRunArgs keys0_run_args{.kernel = KEYS0};
    KernelRunArgs keys1_run_args{.kernel = KEYS1};
    for (uint32_t i = 0; i < num_cores; ++i) {
        AddRuntimeArgsForNode(
            keys0_run_args.runtime_arg_values, cores[i], {{"core_index", i}, {"num_cores", num_cores}});
        AddRuntimeArgsForNode(
            keys1_run_args.runtime_arg_values, cores[i], {{"core_index", i}, {"num_cores", num_cores}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(keys0_run_args), std::move(keys1_run_args)};
    run_args.tensor_args = {
        {CANDIDATE_BLOCKS, tensor_args.candidate_blocks.mesh_tensor()},
        {POOL_COUNT, tensor_args.pool_count.mesh_tensor()},
        {PAGE_TABLE, tensor_args.page_table_tensor.mesh_tensor()},
        {KEY_CACHE, tensor_args.key_cache.mesh_tensor()},
        {POOL_KEYS, pool_keys.mesh_tensor()},
        {POOL_ROWS, pool_rows.mesh_tensor()},
    };
    if (has_page_table_row) {
        run_args.tensor_args.emplace(PAGE_TABLE_ROW, tensor_args.page_table_row_tensor->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::deepseek::build_candidate_pool
