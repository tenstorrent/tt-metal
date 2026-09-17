// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "update_cache_multi_core_program_factory.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

using namespace tt::constants;

// Kernel sources (converted in place — not shared: paged_cache carries its own private copies).
// Declared function-locally at their use sites to avoid unity-build symbol collisions; the spec
// resource names (KernelSpecName / DFBSpecName / TensorParamName) are likewise function-local below.

UpdateCacheDynamicArgs compute_update_cache_dynamic_args(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args) {
    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    const auto update_idx = operation_attributes.update_idx;
    const auto batch_offset = operation_attributes.batch_offset;
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config is required");
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();

    tt::tt_metal::IDevice* device = input_tensor.device();

    // Mirror the shape/dtype-derived geometry and work-split from create_program_artifacts exactly so
    // the per-core cache_start_id values, the two op-wide offsets, and the core ordering are identical
    // on cache miss and hit.
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    std::uint32_t Wt = cache_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    // Width size after untilize
    std::uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor.padded_shape()[-1] * sizeof(float)
                                            : cache_tensor.padded_shape()[-1] * sizeof(::bfloat16);

    std::uint32_t cache_total_num_tiles = cache_tensor.physical_volume() / TILE_HW;
    std::uint32_t cache_batch_num_tiles = cache_total_num_tiles / cache_tensor.padded_shape()[0];
    std::uint32_t cache_head_num_tiles = cache_batch_num_tiles / cache_tensor.padded_shape()[1];

    std::uint32_t B = input_tensor.padded_shape()[-2];
    std::uint32_t num_batched_heads = input_tensor.padded_shape()[1] * B / tt::constants::TILE_HEIGHT;

    UpdateCacheDynamicArgs result;
    result.tile_update_offset = update_idx % tt::constants::TILE_HEIGHT * Wbytes;
    result.batch_read_offset = batch_offset * Wbytes;  // Offset to read from input tensor
    result.Wbytes = Wbytes;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    std::uint32_t num_cores_x = compute_with_storage_grid_size.x;
    std::uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    std::uint32_t num_cores, num_batched_heads_per_core_group_1, num_batched_heads_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();

    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_batched_heads_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_batched_heads_per_core_group_2 = 0;
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
            num_batched_heads_per_core_group_1,
            num_batched_heads_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_batched_heads, row_major);
    }

    std::uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    std::uint32_t cache_tile_idx = update_idx / tt::constants::TILE_HEIGHT * Wt;
    std::uint32_t total_batched_heads = 0;
    result.cache_start_ids.reserve(num_cores);
    for (std::uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        std::uint32_t num_batched_heads_per_core;
        if (i < g1_numcores) {
            num_batched_heads_per_core = num_batched_heads_per_core_group_1;
        } else {
            num_batched_heads_per_core = num_batched_heads_per_core_group_2;
        }
        std::uint32_t batch_start_id = (total_batched_heads * TILE_HEIGHT) % B;
        // Batch Offset + Head Offset + Index Offset
        std::uint32_t cache_start_id = batch_start_id * cache_batch_num_tiles +
                                       ((total_batched_heads * tt::constants::TILE_HEIGHT) / B) * cache_head_num_tiles;
        cache_start_id += cache_tile_idx;
        result.cache_start_ids.emplace_back(core, cache_start_id);
        total_batched_heads += num_batched_heads_per_core;
    }
    return result;
}

ttnn::device_operation::ProgramArtifacts UpdateCacheMultiCoreProgramFactory::create_program_artifacts(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& tensor_return_value) {
    // Spec resource names (function-local to avoid unity-build collisions with the fill factory).
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_group_1"};
    const KernelSpecName COMPUTE_G2{"compute_group_2"};
    const DFBSpecName CACHE_DFB{"cache"};      // legacy c_0 (src0)
    const DFBSpecName INPUT_DFB{"input"};      // legacy c_1 (src1)
    const DFBSpecName INTERM0_DFB{"interm0"};  // legacy c_24 (aliases c_25's L1)
    const DFBSpecName INTERM1_DFB{"interm1"};  // legacy c_25 (aliases c_24's L1)
    const DFBSpecName INTERM2_DFB{"interm2"};  // legacy c_26 (untilized_input)
    const DFBSpecName OUTPUT_DFB{"output"};    // legacy c_16
    const TensorParamName CACHE{"cache"};
    const TensorParamName INPUT{"input"};

    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    TT_FATAL(operation_attributes.compute_kernel_config.has_value(), "Compute kernel config is required");
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();

    tt::DataFormat cache_data_format = tt::tt_metal::datatype_to_dataformat_converter(cache_tensor.dtype());
    std::uint32_t cache_single_tile_size = tt::tile_size(cache_data_format);

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    std::uint32_t input_single_tile_size = tt::tile_size(input_data_format);

    tt::tt_metal::IDevice* device = input_tensor.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    tt::DataFormat interm_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    std::uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);

    std::uint32_t Wt = cache_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    // Width size after untilize
    std::uint32_t Wbytes = fp32_dest_acc_en ? cache_tensor.padded_shape()[-1] * sizeof(float)
                                            : cache_tensor.padded_shape()[-1] * sizeof(::bfloat16);

    log_debug(tt::LogOp, "cache_data_format: {}", cache_data_format);
    log_debug(tt::LogOp, "input_data_format: {}", input_data_format);
    log_debug(tt::LogOp, "interm_data_format: {}", interm_data_format);
    log_debug(tt::LogOp, "Wbytes: {}", Wbytes);
    log_debug(tt::LogOp, "Wt: {}", Wt);

    std::uint32_t cache_total_num_tiles = cache_tensor.physical_volume() / TILE_HW;
    std::uint32_t cache_batch_num_tiles = cache_total_num_tiles / cache_tensor.padded_shape()[0];
    std::uint32_t cache_head_num_tiles = cache_batch_num_tiles / cache_tensor.padded_shape()[1];

    std::uint32_t B = input_tensor.padded_shape()[-2];
    std::uint32_t Bcache = cache_tensor.padded_shape()[0];
    const std::uint32_t granularity =
        std::min(static_cast<std::uint32_t>(2), Bcache);  // granularity = 2 best for performance
    std::uint32_t num_batched_heads = input_tensor.padded_shape()[1] * B / tt::constants::TILE_HEIGHT;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    std::uint32_t num_cores_x = compute_with_storage_grid_size.x;
    std::uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    std::uint32_t num_cores, num_batched_heads_per_core_group_1, num_batched_heads_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();

    std::uint32_t num_input_tiles;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_batched_heads_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_batched_heads_per_core_group_2 = 0;
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
            num_batched_heads_per_core_group_1,
            num_batched_heads_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_batched_heads, row_major);
        num_input_tiles = 2 * Wt;  // double buffered
    }

    const bool input_sharded = shard_spec.has_value();

    // ---- Dataflow buffers ----
    std::uint32_t num_cache_tiles = 2 * granularity * Wt;   // double buffered
    std::uint32_t num_interm_tiles = 2 * granularity * Wt;  // double buffered
    std::uint32_t num_output_tiles = B * Wt;                // must buffer all tiles for a single head

    const DataflowBufferSpec cache_dfb{
        .unique_id = CACHE_DFB,
        .entry_size = cache_single_tile_size,
        .num_entries = num_cache_tiles,
        .data_format_metadata = cache_data_format,
    };
    // For sharded inputs, borrow the input buffer's L1 memory (equivalent to the old
    // set_globally_allocated_address + UpdateDynamicCircularBufferAddress pair); the backing address
    // resolves at runtime from the INPUT TensorArgument.
    const DataflowBufferSpec input_dfb{
        .unique_id = INPUT_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = input_data_format,
        .borrowed_from = input_sharded ? std::optional<TensorParamName>(INPUT) : std::nullopt,
    };
    // interm0 (c_24) and interm1 (c_25) shared a single legacy circular-buffer descriptor's L1 region
    // via two format descriptors: two distinct FIFOs (compute produces c_24 / consumes c_25; writer
    // consumes c_24 / produces c_25) over one backing region. Modeled as two aliased DFBs.
    const DataflowBufferSpec interm0_dfb{
        .unique_id = INTERM0_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = interm_data_format,
        .advanced_options = {.alias_with = {INTERM1_DFB}},
    };
    const DataflowBufferSpec interm1_dfb{
        .unique_id = INTERM1_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = interm_data_format,
        .advanced_options = {.alias_with = {INTERM0_DFB}},
    };
    const DataflowBufferSpec interm2_dfb{
        .unique_id = INTERM2_DFB,
        .entry_size = interm_single_tile_size,
        .num_entries = num_interm_tiles,
        .data_format_metadata = interm_data_format,
    };
    const DataflowBufferSpec output_dfb{
        .unique_id = OUTPUT_DFB,
        .entry_size = cache_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = cache_data_format,
    };

    const TensorParameter cache_param{.unique_id = CACHE, .spec = cache_tensor.tensor_spec()};
    const TensorParameter input_param{.unique_id = INPUT, .spec = input_tensor.tensor_spec()};

    const std::uint32_t u_range = std::min(static_cast<std::uint32_t>(32), Bcache);
    const std::uint32_t u_count = u_range / granularity;

    // ---- Reader ----
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (input_sharded) {
        reader_defines.emplace("INPUT_SHARDED", "1");
    }
    // The reader's TensorAccessor on the input tensor exists only on the interleaved path; the sharded
    // path reads the input through the borrowed input DFB instead. Bind tensor::input only when the
    // kernel constructs it (gated by the same INPUT_SHARDED define).
    Group<TensorBinding> reader_tensor_bindings = {
        TensorBinding{.tensor_parameter_name = CACHE, .accessor_name = "cache"},
    };
    if (!input_sharded) {
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"});
    }
    const KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/"
            "reader_update_cache_interleaved_start_id.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = CACHE_DFB, .accessor_name = "cache", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = {{"granularity", granularity}, {"u_count", u_count}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"Wt",
                  "B",
                  "num_batched_heads",
                  "cache_total_num_tiles",
                  "cache_batch_num_tiles",
                  "cache_head_num_tiles",
                  "cache_start_id",
                  "input_start_id",
                  "batch_start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- Writer ----
    const KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/"
            "writer_update_cache_interleaved_start_id.cpp",
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = OUTPUT_DFB, .accessor_name = "cache", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = INTERM0_DFB,
                 .accessor_name = "untilized_cache",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = INTERM1_DFB,
                 .accessor_name = "untilized_cache2",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = INTERM2_DFB,
                 .accessor_name = "untilized_input",
                 .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = CACHE, .accessor_name = "cache"}},
        .compile_time_args = {{"granularity", granularity}, {"u_count", u_count}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"Wt",
                  "B",
                  "num_batched_heads",
                  "cache_total_num_tiles",
                  "cache_batch_num_tiles",
                  "cache_head_num_tiles",
                  "cache_start_id",
                  "batch_start_id",
                  "Wbytes",
                  "offset",
                  "batch_read_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Compute (one KernelSpec per core group; per-group head count stays a CTA) ----
    const auto make_compute = [&](const KernelSpecName& id, std::uint32_t num_batched_heads_per_core_group) {
        auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config);
        // Metal 2.0 requires an explicit unpack_modes entry for a consumed Float32 DFB when
        // enable_32_bit_dest (== fp32_dest_acc_en) is set, where legacy silently defaulted. The legacy
        // op set no unpack_to_dest_mode at all (all Default), so mirror that as UnpackToSrc. The compute
        // kernel consumes cache (c_0), input (c_1) and interm1 (c_25); interm carries Float32 exactly
        // when fp32_dest_acc_en, cache/input only if their tensor dtype is FLOAT32.
        if (fp32_dest_acc_en) {
            auto& um = unpack_modes(compute_hw);
            if (cache_data_format == tt::DataFormat::Float32) {
                um.emplace(CACHE_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            }
            if (input_data_format == tt::DataFormat::Float32) {
                um.emplace(INPUT_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
            }
            um.emplace(INTERM1_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        return KernelSpec{
            .unique_id = id,
            .source = "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/compute/update_cache.cpp",
            // Legacy compute defaulted opt_level to O3; Metal 2.0 defaults to O2, so set it explicitly.
            .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = CACHE_DFB, .accessor_name = "cache", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = INTERM0_DFB,
                     .accessor_name = "untilized_cache",
                     .endpoint_type = DFBEndpointType::PRODUCER},
                 DFBBinding{
                     .dfb_spec_name = INTERM1_DFB,
                     .accessor_name = "untilized_cache2",
                     .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = INTERM2_DFB,
                     .accessor_name = "untilized_input",
                     .endpoint_type = DFBEndpointType::PRODUCER},
                 DFBBinding{
                     .dfb_spec_name = OUTPUT_DFB,
                     .accessor_name = "output",
                     .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"num_batched_heads", num_batched_heads_per_core_group},
                 {"Wt", Wt},
                 {"granularity", granularity},
                 {"u_count", u_count}},
            .hw_config = compute_hw,
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    kernels.push_back(make_compute(COMPUTE_G1, num_batched_heads_per_core_group_1));
    work_units.push_back(
        WorkUnitSpec{.name = "update_group_1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
    const bool has_group_2 = !core_group_2.ranges().empty();
    if (has_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_batched_heads_per_core_group_2));
        work_units.push_back(WorkUnitSpec{
            .name = "update_group_2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    // ---- Per-core runtime args (name-first tables built from the legacy node-first loop) ----
    // cache_start_id comes from the shared helper; input_start_id / batch_start_id / the per-core head
    // count are shape-only and computed here, mirroring the legacy loop order exactly.
    // NOTE: the kernels' "B" arg is the per-head cache-user count used as the next-head wrap threshold,
    // so it must carry Bcache (cache batch), NOT the input's padded height B. B is used only for the
    // input-side work-split math (num_batched_heads, batch_start_id).
    std::uint32_t g1_numcores = core_group_1.num_cores();
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);
    const auto dyn = compute_update_cache_dynamic_args(operation_attributes, tensor_args);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    std::uint32_t total_batched_heads = 0;
    for (std::uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        std::uint32_t num_batched_heads_per_core =
            (i < g1_numcores) ? num_batched_heads_per_core_group_1 : num_batched_heads_per_core_group_2;
        std::uint32_t input_start_id = total_batched_heads * Wt;
        std::uint32_t batch_start_id = (total_batched_heads * TILE_HEIGHT) % B;
        const std::uint32_t cache_start_id = dyn.cache_start_ids.at(i).second;
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"Wt", Wt},
             {"B", Bcache},
             {"num_batched_heads", num_batched_heads_per_core},
             {"cache_total_num_tiles", cache_total_num_tiles},
             {"cache_batch_num_tiles", cache_batch_num_tiles},
             {"cache_head_num_tiles", cache_head_num_tiles},
             {"cache_start_id", cache_start_id},
             {"input_start_id", input_start_id},
             {"batch_start_id", batch_start_id}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"Wt", Wt},
             {"B", Bcache},
             {"num_batched_heads", num_batched_heads_per_core},
             {"cache_total_num_tiles", cache_total_num_tiles},
             {"cache_batch_num_tiles", cache_batch_num_tiles},
             {"cache_head_num_tiles", cache_head_num_tiles},
             {"cache_start_id", cache_start_id},
             {"batch_start_id", batch_start_id},
             {"Wbytes", dyn.Wbytes},
             {"offset", dyn.tile_update_offset},
             {"batch_read_offset", dyn.batch_read_offset}});
        total_batched_heads += num_batched_heads_per_core;
    }

    ProgramSpec spec{
        .name = "update_cache_multi_core",
        .kernels = std::move(kernels),
        .dataflow_buffers = {cache_dfb, input_dfb, interm0_dfb, interm1_dfb, interm2_dfb, output_dfb},
        .tensor_parameters = {cache_param, input_param},
        .work_units = std::move(work_units),
    };

    // The compute kernels have no runtime args, so they need no KernelRunArgs entry.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    // The cache tensor is the in-place output; bind it from tensor_return_value. Input is the source.
    run_args.tensor_args = {{CACHE, tensor_return_value.mesh_tensor()}, {INPUT, input_tensor.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs UpdateCacheMultiCoreProgramFactory::override_runtime_arguments(
    const KvCacheParams& operation_attributes,
    const KvCacheInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Spec resource names — must match create_program_artifacts (function-local, per-factory).
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName CACHE{"cache"};
    const TensorParamName INPUT{"input"};

    // Runs on every program-cache hit. compute_program_hash excludes update_idx / batch_offset /
    // compute_kernel_config, so the args they drive are NOT stable across hits and must be re-applied:
    //   - reader/writer cache_start_id (per core), and writer Wbytes / offset / batch_read_offset.
    // Buffer addresses refresh through the typed tensor channel (the borrowed input DFB re-resolves
    // from the INPUT TensorArgument). Everything else (Wt, B, per-core head count, cache_*_num_tiles,
    // input_start_id, batch_start_id) is shape-derived — covered by the hash, so a hit means it matches.
    const auto dyn = compute_update_cache_dynamic_args(operation_attributes, tensor_args);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (const auto& [core, cache_start_id] : dyn.cache_start_ids) {
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"cache_start_id", cache_start_id}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"cache_start_id", cache_start_id},
             {"Wbytes", dyn.Wbytes},
             {"offset", dyn.tile_update_offset},
             {"batch_read_offset", dyn.batch_read_offset}});
    }

    ProgramRunArgs params;
    params.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    params.tensor_args = {{CACHE, tensor_return_value.mesh_tensor()}, {INPUT, tensor_args.input.mesh_tensor()}};
    return params;
}

}  // namespace ttnn::prim
