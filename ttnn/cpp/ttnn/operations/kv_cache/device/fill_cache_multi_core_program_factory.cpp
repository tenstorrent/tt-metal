// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "fill_cache_multi_core_program_factory.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

using namespace tt::constants;

// Kernel sources and spec resource names are declared function-locally at their use sites (below) to
// avoid unity-build symbol collisions with the update factory's identically-named constants.

std::vector<std::pair<CoreCoord, std::uint32_t>> compute_fill_cache_start_ids(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args) {
    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    const auto batch_idx = operation_attributes.batch_idx;
    const auto update_idx = operation_attributes.update_idx;

    // Mirror the shape-derived geometry and work-split from create_program_artifacts exactly so the
    // per-core cache_start_id values (and the core ordering) are identical on cache miss and hit.
    std::uint32_t num_blocks_of_work = input_tensor.padded_shape()[1] * input_tensor.padded_shape()[-2] / TILE_HEIGHT;

    std::uint32_t Wt = cache_tensor.padded_shape()[-1] / TILE_WIDTH;
    std::uint32_t input_Ht = input_tensor.padded_shape()[-2] / TILE_HEIGHT;  // seq_len
    std::uint32_t cache_HtWt = cache_tensor.padded_shape()[-2] * Wt / TILE_HEIGHT;
    std::uint32_t cache_CHtWt = cache_tensor.padded_shape()[1] * cache_HtWt;
    std::uint32_t update_idxt = update_idx / TILE_HEIGHT;
    std::uint32_t start_idx = (batch_idx * cache_CHtWt) + (update_idxt * Wt);
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    std::uint32_t num_cores_x = compute_with_storage_grid_size.x;
    std::uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    std::uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();

    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_blocks_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_blocks_per_core_group_2 = 0;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end_coord.x + 1;
        num_cores_y = bbox.end_coord.y + 1;
    } else {
        row_major = true;
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major);
    }

    std::uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    std::vector<std::pair<CoreCoord, std::uint32_t>> start_ids;
    start_ids.reserve(num_cores);
    for (std::uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        const CoreCoord& core = cores.at(i);
        std::uint32_t num_blocks_per_core = 0;
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else {
            num_blocks_per_core = num_blocks_per_core_group_2;
        }

        const std::uint32_t cache_start_id = start_idx                                       // user batch start
                                             + (num_blocks_written / input_Ht * cache_HtWt)  // cache head offset
                                             + ((num_blocks_written % input_Ht) * Wt);       // seq_len offset
        start_ids.emplace_back(core, cache_start_id);
        num_blocks_written += num_blocks_per_core;
    }
    return start_ids;
}

ttnn::device_operation::ProgramArtifacts FillCacheMultiCoreProgramFactory::create_program_artifacts(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& tensor_return_value) {
    // Spec resource names (function-local to avoid unity-build collisions with the update factory).
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName SRC0_DFB{"src0"};  // legacy c_0 (input, reused as pass-through output)
    const TensorParamName INPUT{"input"};
    const TensorParamName DST{"dst"};  // the cache tensor (donor writer's tensor::dst)

    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    std::uint32_t single_tile_size = tt::tile_size(data_format);

    // TODO: For interleaved and kv_heads > 1, we assert that each core only gets 1 tile along seq_len
    // For sharded, each core gets shard_shape[0] number of tiles along seq_len.
    // For either case, assume that work doesn't spill over to next head, so we just increment by Wt within
    // reader/writer
    std::uint32_t num_blocks_of_work = input_tensor.padded_shape()[1] * input_tensor.padded_shape()[-2] / TILE_HEIGHT;

    // Wt is the only shape-derived geometry create_program_artifacts still needs directly; the
    // batch_idx/update_idx-dependent cache_start_id lives in compute_fill_cache_start_ids, shared
    // with override_runtime_arguments.
    std::uint32_t Wt = cache_tensor.padded_shape()[-1] / TILE_WIDTH;
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    std::uint32_t num_cores_x = compute_with_storage_grid_size.x;
    std::uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    std::uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();

    std::uint32_t num_input_tiles;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_blocks_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_blocks_per_core_group_2 = 0;
        num_input_tiles = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end_coord.x + 1;
        num_cores_y = bbox.end_coord.y + 1;
    } else {
        row_major = true;
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_of_work, row_major);
        num_input_tiles = 2;  // double buffered
    }

    const bool input_sharded = shard_spec.has_value();

    // c_0: the input DFB, reused as the pass-through output (reader produces, donor writer consumes).
    // For sharded inputs, borrow the input buffer's L1 memory (equivalent to the old
    // set_globally_allocated_address + UpdateDynamicCircularBufferAddress pair).
    const DataflowBufferSpec src0_dfb{
        .unique_id = SRC0_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = data_format,
        .borrowed_from = input_sharded ? std::optional<TensorParamName>(INPUT) : std::nullopt,
    };

    const TensorParameter input_param{.unique_id = INPUT, .spec = input_tensor.tensor_spec()};
    const TensorParameter dst_param{.unique_id = DST, .spec = cache_tensor.tensor_spec()};

    // ---- Reader ----
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (input_sharded) {
        reader_defines.emplace("INPUT_SHARDED", "1");
    }
    // The reader's TensorAccessor on the input tensor exists only on the interleaved path; the sharded
    // path reads through the borrowed input DFB. Bind tensor::input only when the kernel constructs it.
    Group<TensorBinding> reader_tensor_bindings;
    if (!input_sharded) {
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"});
    }
    const KernelSpec reader{
        .unique_id = READER,
        // Converted in place — not shared: deepseek_prefill carries its own fork, not this file.
        .source =
            "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/"
            "reader_fill_cache_interleaved_start_id.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = reader_tensor_bindings,
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // ---- Writer (donor Metal 2.0 fork; its interface: dfb::out CONSUMER, tensor::dst, num_pages/start_id) ----
    const KernelSpec writer{
        .unique_id = WRITER,
        // Reuse the existing cross-family donor Metal 2.0 fork (shared-kernel rung 1): bind it, adopt
        // its interface (dfb::out CONSUMER, tensor::dst, named RTAs num_pages + start_id).
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    // ---- Per-core runtime args ----
    std::uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
    const auto start_ids = compute_fill_cache_start_ids(operation_attributes, tensor_args);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (std::uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        const CoreCoord& core = cores.at(i);
        std::uint32_t num_blocks_per_core = 0;
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else {
            num_blocks_per_core = num_blocks_per_core_group_2;
        }
        const std::uint32_t cache_start_id = start_ids.at(i).second;
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_blocks_per_core * Wt}, {"start_id", num_blocks_written * Wt}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_blocks_per_core * Wt}, {"start_id", cache_start_id}});
        num_blocks_written += num_blocks_per_core;
    }

    ProgramSpec spec{
        .name = "fill_cache_multi_core",
        .kernels = {reader, writer},
        .dataflow_buffers = {src0_dfb},
        .tensor_parameters = {input_param, dst_param},
        .work_units = {WorkUnitSpec{.name = "fill", .kernels = {READER, WRITER}, .target_nodes = all_cores}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    // The cache tensor is the in-place output; bind it (tensor::dst) from tensor_return_value.
    run_args.tensor_args = {{INPUT, input_tensor.mesh_tensor()}, {DST, tensor_return_value.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs FillCacheMultiCoreProgramFactory::override_runtime_arguments(
    const KvCacheParams& operation_attributes,
    const KvCacheInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Spec resource names — must match create_program_artifacts (function-local, per-factory).
    const KernelSpecName WRITER{"writer"};
    const TensorParamName INPUT{"input"};
    const TensorParamName DST{"dst"};

    // Runs on every program-cache hit. compute_program_hash excludes batch_idx / update_idx, so the
    // writer's start_id (== cache_start_id) is NOT stable and must be re-applied. Buffer addresses
    // refresh through the typed tensor channel (the borrowed input DFB re-resolves from the INPUT
    // TensorArgument). The reader's num_tiles / start_id are shape-derived (in the hash) — stable.
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (const auto& [core, cache_start_id] : compute_fill_cache_start_ids(operation_attributes, tensor_args)) {
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"start_id", cache_start_id}});
    }

    ProgramRunArgs params;
    params.kernel_run_args = {std::move(writer_run_args)};
    params.tensor_args = {{INPUT, tensor_args.input.mesh_tensor()}, {DST, tensor_return_value.mesh_tensor()}};
    return params;
}

}  // namespace ttnn::prim
