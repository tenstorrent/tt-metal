// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_lightning_select_kv_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr const char* kReaderSource =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/fused_lightning_select_kv/device/kernels/dataflow/"
    "reader_fused_lightning_select_kv.cpp";
constexpr const char* kWriterSource =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/fused_lightning_select_kv/device/kernels/dataflow/"
    "writer_fused_lightning_select_kv.cpp";
constexpr const char* kComputeSource =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/fused_lightning_select_kv/device/kernels/compute/"
    "fused_lightning_select_kv.cpp";

constexpr uint32_t kNumEntriesPerDfb = 2;
constexpr uint32_t kHeadsPerQueryTile = 8;

DataflowBufferSpec make_dfb(const DFBSpecName& name, const Tensor& tensor, uint32_t num_entries = kNumEntriesPerDfb) {
    return DataflowBufferSpec{
        .unique_id = name,
        .entry_size = tensor.buffer()->page_size(),
        .num_entries = num_entries,
        .data_format_metadata = datatype_to_dataformat_converter(tensor.dtype()),
    };
}

}  // namespace

ttnn::device_operation::ProgramArtifacts
FusedLightningSelectKvDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const Tensor& kv_output = output.at(0);
    const Tensor& scores_output = output.at(1);
    const auto* device = kv_output.device();
    const auto& query_shard = tensor_args.query.memory_config().shard_spec().value();
    const CoreRangeSet& all_cores = query_shard.grid;
    // B == 1 and Sq == 1: each core holds one query row per head and a single weights row.
    const uint32_t num_heads = query_shard.shape[0];
    const uint32_t d_tiles = query_shard.shape[1] / tt::constants::TILE_WIDTH;
    const uint32_t num_query_tiles = num_heads * d_tiles;
    const uint32_t num_weight_tiles = num_heads / tt::constants::TILE_WIDTH;
    const bool has_valid_length = tensor_args.valid_length_tensor.has_value();
    const uint32_t page_block_size = tensor_args.key_cache.logical_shape()[-2];
    const uint32_t chunks_per_block = page_block_size / tt::constants::TILE_HEIGHT;

    const auto& paged_key_shape = tensor_args.key_cache.padded_shape();
    const uint32_t num_tiles_per_block_of_key = paged_key_shape[-1] * paged_key_shape[-2] / tt::constants::TILE_HW;

    // ---- Resource names ----
    const DFBSpecName QUERY_RM_DFB{"query_rm"};
    const DFBSpecName QUERY_TILED_DFB{"query_tiled"};
    const DFBSpecName KEY_DFB{"key"};
    const DFBSpecName WEIGHTS_DFB{"weights"};
    const DFBSpecName INDICES_DFB{"indices"};
    const DFBSpecName KV_DFB{"kv"};
    const DFBSpecName CUR_POS_DFB{"cur_pos"};
    const DFBSpecName CTRL_DFB{"ctrl"};
    const DFBSpecName SCORES_DFB{"scores"};

    const TensorParamName QUERY{"query"};
    const TensorParamName KEY_CACHE{"key_cache"};
    const TensorParamName HEAD_WEIGHTS{"head_weights"};
    const TensorParamName KV_CACHE{"kv_cache"};
    const TensorParamName PAGE_TABLE{"page_table"};
    const TensorParamName CUR_POS{"cur_pos"};
    const TensorParamName VALID_LENGTH{"valid_length"};
    const TensorParamName OUTPUT{"output"};
    const TensorParamName SCORES{"scores"};

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- Dataflow buffers ----
    // reader -> compute: query, key, weights
    // compute -> reader: indices (top-k positions used to gather kv rows)
    // reader -> writer: kv rows
    //
    // query and weights are borrowed from their replicated shards: every core already holds all
    // Hi query rows and the one weights row. A row-major row of 32 values is exactly one 1x32 tile,
    // so each weights entry is one 1x32 tile.
    const tt::tt_metal::Tile row_tile({1, tt::constants::TILE_WIDTH});
    const auto make_replicated_dfb =
        [&row_tile](const DFBSpecName& name, const TensorParamName& param, const Tensor& tensor, uint32_t num_tiles) {
            const tt::DataFormat format = datatype_to_dataformat_converter(tensor.dtype());
            return DataflowBufferSpec{
                .unique_id = name,
                .entry_size = row_tile.get_tile_size(format),
                .num_entries = num_tiles,
                .data_format_metadata = format,
                .tile_format_metadata = row_tile,
                .borrowed_from = param,
            };
        };
    const DataflowBufferSpec weights_dfb =
        make_replicated_dfb(WEIGHTS_DFB, HEAD_WEIGHTS, tensor_args.head_weights, num_weight_tiles);

    // The same query shard viewed as row-major 8-row strips: entry i of strip g covers rows
    // [8g, 8g + 8) and columns [32i, 32i + 32). Compute tilizes each strip into query_tiled, one
    // 8x32 tile per (8-head group, D/32 slice), heads along rows.
    const tt::tt_metal::Tile head_group_tile({kHeadsPerQueryTile, tt::constants::TILE_WIDTH});
    const tt::DataFormat query_format = datatype_to_dataformat_converter(tensor_args.query.dtype());
    const uint32_t num_query_group_tiles = num_query_tiles / kHeadsPerQueryTile;
    const DataflowBufferSpec query_rm_dfb{
        .unique_id = QUERY_RM_DFB,
        .entry_size = head_group_tile.get_tile_size(query_format),
        .num_entries = num_query_group_tiles,
        .data_format_metadata = query_format,
        .tile_format_metadata = head_group_tile,
        .borrowed_from = QUERY,
    };
    const DataflowBufferSpec query_tiled_dfb{
        .unique_id = QUERY_TILED_DFB,
        .entry_size = head_group_tile.get_tile_size(query_format),
        .num_entries = num_query_group_tiles,
        .data_format_metadata = query_format,
        .tile_format_metadata = head_group_tile,
    };

    // One fp32 1x32 score tile per 32 keys, double-buffered across a block.
    constexpr tt::DataFormat scores_format = tt::DataFormat::Float32;
    const DataflowBufferSpec scores_dfb{
        .unique_id = SCORES_DFB,
        .entry_size = row_tile.get_tile_size(scores_format),
        .num_entries = 2 * chunks_per_block,
        .data_format_metadata = scores_format,
        .tile_format_metadata = row_tile,
    };

    const DataflowBufferSpec key_dfb =
        make_dfb(KEY_DFB, tensor_args.key_cache, num_tiles_per_block_of_key * 2);  // 2 for Double Buffering
    constexpr tt::DataFormat indices_format = tt::DataFormat::UInt32;
    constexpr tt::DataFormat cur_pos_format = tt::DataFormat::UInt32;

    const DataflowBufferSpec indices_dfb{
        .unique_id = INDICES_DFB,
        .entry_size = tt::tile_size(indices_format),
        .num_entries = kNumEntriesPerDfb,
        .data_format_metadata = indices_format,
    };

    const DataflowBufferSpec cur_pos_dfb{
        .unique_id = CUR_POS_DFB,
        .entry_size = tt::tile_size(cur_pos_format),
        .num_entries = 1,
        .data_format_metadata = cur_pos_format,
    };

    // Entry 0 carries the per-core block count to compute; entry 1 is the reader's page-table scratch.
    const DataflowBufferSpec ctrl_dfb{
        .unique_id = CTRL_DFB,
        .entry_size = tt::tile_size(cur_pos_format),
        .num_entries = 2,
        .data_format_metadata = cur_pos_format,
    };

    // ---- Tensor parameters ----
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = QUERY, .spec = tensor_args.query.tensor_spec()},
        TensorParameter{.unique_id = KEY_CACHE, .spec = tensor_args.key_cache.tensor_spec()},
        TensorParameter{.unique_id = HEAD_WEIGHTS, .spec = tensor_args.head_weights.tensor_spec()},
        TensorParameter{.unique_id = KV_CACHE, .spec = tensor_args.kv_cache.tensor_spec()},
        TensorParameter{.unique_id = PAGE_TABLE, .spec = tensor_args.page_table_tensor.tensor_spec()},
        TensorParameter{.unique_id = CUR_POS, .spec = tensor_args.cur_pos_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = kv_output.tensor_spec()},
        TensorParameter{.unique_id = SCORES, .spec = scores_output.tensor_spec()},
    };

    Group<TensorBinding> reader_tensor_bindings = {
        TensorBinding{.tensor_parameter_name = KEY_CACHE, .accessor_name = "key_cache"},
        TensorBinding{.tensor_parameter_name = KV_CACHE, .accessor_name = "kv_cache"},
        TensorBinding{.tensor_parameter_name = PAGE_TABLE, .accessor_name = "page_table"},
        TensorBinding{.tensor_parameter_name = CUR_POS, .accessor_name = "cur_pos"},
    };
    if (has_valid_length) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = VALID_LENGTH, .spec = tensor_args.valid_length_tensor->tensor_spec()});
        reader_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = VALID_LENGTH, .accessor_name = "valid_length"});
    }

    // ---- Kernels ----
    KernelSpec::CompilerOptions reader_compiler_options;
    if (has_valid_length) {
        reader_compiler_options.defines.emplace("HAS_VALID_LENGTH", "1");
    }

    const KernelSpec reader{
        .unique_id = READER,
        .source = kReaderSource,
        .compiler_options = std::move(reader_compiler_options),
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = QUERY_RM_DFB,
                 .accessor_name = "query_rm",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = KEY_DFB, .accessor_name = "key", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = WEIGHTS_DFB, .accessor_name = "weights", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = INDICES_DFB, .accessor_name = "indices", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = KV_DFB, .accessor_name = "kv", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = CUR_POS_DFB, .accessor_name = "cur_pos", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = CTRL_DFB, .accessor_name = "ctrl", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = std::move(reader_tensor_bindings),
        .compile_time_args =
            {{"k", args.k},
             {"num_query_group_tiles", num_query_group_tiles},
             {"num_weight_tiles", num_weight_tiles},
             {"page_block_size", page_block_size},
             {"num_tiles_per_block_of_key", num_tiles_per_block_of_key}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_index", "num_cores"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    const KernelSpec writer{
        .unique_id = WRITER,
        .source = kWriterSource,
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = KV_DFB, .accessor_name = "kv", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = CUR_POS_DFB,
                    .accessor_name = "cur_pos",
                    .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = SCORES_DFB, .accessor_name = "scores", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
             TensorBinding{.tensor_parameter_name = SCORES, .accessor_name = "scores"}},
        .compile_time_args = {{"k", args.k}, {"page_block_size", page_block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_index", "num_cores"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);
    (void)packer_l1_acc;
    (void)math_fidelity;

    const KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kComputeSource,
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = QUERY_RM_DFB,
                 .accessor_name = "query_rm",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = QUERY_TILED_DFB,
                 .accessor_name = "query_tiled",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = QUERY_TILED_DFB,
                 .accessor_name = "query_tiled",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = KEY_DFB, .accessor_name = "key", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = WEIGHTS_DFB, .accessor_name = "weights", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = INDICES_DFB, .accessor_name = "indices", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = CTRL_DFB, .accessor_name = "ctrl", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = SCORES_DFB, .accessor_name = "scores", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"k", args.k},
             {"num_heads", num_heads},
             {"d_tiles", d_tiles},
             {"num_weight_tiles", num_weight_tiles},
             {"chunks_per_block", chunks_per_block}},
        .hw_config = ComputeHardwareConfig{ComputeGen1Config{
            // The score matmul uses custom_mm, which only supports LoFi.
            .fpu_math_fidelity = MathFidelity::LoFi,
            .sfpu_precision_mode = math_approx_mode ? Precision::Approximate : Precision::Precise,
            .enable_32_bit_dest = fp32_dest_acc_en,
            .double_buffer_dest = !dst_full_sync_en,
        }},
    };

    ProgramSpec spec{
        .name = "fused_lightning_select_kv",
        .kernels = {reader, writer, compute},
        .dataflow_buffers =
            {query_rm_dfb,
             query_tiled_dfb,
             key_dfb,
             weights_dfb,
             indices_dfb,
             make_dfb(KV_DFB, tensor_args.kv_cache),
             cur_pos_dfb,
             ctrl_dfb,
             scores_dfb},
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = {WorkUnitSpec{
            .name = "fused_lightning_select_kv", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };

    const auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    const uint32_t num_cores = cores.size();
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (uint32_t i = 0; i < num_cores; ++i) {
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, cores[i], {{"core_index", i}, {"num_cores", num_cores}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, cores[i], {{"core_index", i}, {"num_cores", num_cores}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {QUERY, tensor_args.query.mesh_tensor()},
        {KEY_CACHE, tensor_args.key_cache.mesh_tensor()},
        {HEAD_WEIGHTS, tensor_args.head_weights.mesh_tensor()},
        {KV_CACHE, tensor_args.kv_cache.mesh_tensor()},
        {PAGE_TABLE, tensor_args.page_table_tensor.mesh_tensor()},
        {CUR_POS, tensor_args.cur_pos_tensor.mesh_tensor()},
        {OUTPUT, kv_output.mesh_tensor()},
        {SCORES, scores_output.mesh_tensor()},
    };
    if (has_valid_length) {
        run_args.tensor_args.emplace(VALID_LENGTH, std::cref(tensor_args.valid_length_tensor->mesh_tensor()));
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::experimental::deepseek::fused_lightning_select_kv
