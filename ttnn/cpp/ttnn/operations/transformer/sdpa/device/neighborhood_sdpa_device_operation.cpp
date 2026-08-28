// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/neighborhood_sdpa_device_operation.hpp"

#include "ttnn/operations/transformer/sdpa/device/kernels/neighborhood_kernel_args.hpp"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string_view>

#include <tt-metalium/constants.hpp>

#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

namespace neighborhood = ttnn::transformer::neighborhood;

namespace {

uint32_t probe_from_name(std::string_view name) {
    if (name == "full" || name == "0") {
        return 0;
    }
    if (name == "skip_kv" || name == "1") {
        return 1;
    }
    if (name == "mask_memset" || name == "2") {
        return 2;
    }
    if (name == "drain" || name == "3") {
        return 3;
    }
    if (name == "qk" || name == "4") {
        return 4;
    }
    if (name == "qk_softmax" || name == "5") {
        return 5;
    }
    if (name == "qk_pv" || name == "6") {
        return 6;
    }
    if (name == "skip_slots" || name == "7") {
        return 7;
    }
    if (name == "skip_slots_drain" || name == "8") {
        return 8;
    }
    return 0;
}

uint32_t resolve_probe(std::optional<uint32_t> probe) {
    if (probe.has_value()) {
        return *probe;
    }
    if (const char* env = std::getenv("DIFFVAE_NA_SDPA_PROBE")) {
        return probe_from_name(env);
    }
    // Older decode knobs: same ablations, now hashed through `probe` so they actually recompile.
    if (const char* env = std::getenv("DIFFVAE_NA_SKIP_KV"); env != nullptr && env[0] == '1') {
        return 1;
    }
    if (const char* env = std::getenv("DIFFVAE_NA_MASK_MEMSET_ONLY"); env != nullptr && env[0] == '1') {
        return 2;
    }
    return 0;
}

}  // namespace

void NeighborhoodSDPAOperation::validate_on_program_cache_miss(
    const NeighborhoodSDPAParams& attributes, const NeighborhoodSDPAInputs& tensors) {
    // Geometry first: a bad config produces a confusing tensor-shape error otherwise.
    try {
        neighborhood::validate_config(attributes.config);
    } catch (const std::invalid_argument& error) {
        TT_THROW("neighborhood_sdpa: {}", error.what());
    }

    const auto& query_tensor = tensors.query_tensor;
    const auto& key_tensor = tensors.key_tensor;
    const auto& value_tensor = tensors.value_tensor;
    const auto& gather_origin_table = tensors.gather_origin_table;

    TT_FATAL(
        query_tensor.device() == key_tensor.device() && query_tensor.device() == value_tensor.device() &&
            query_tensor.device() == gather_origin_table.device(),
        "neighborhood_sdpa: all inputs must be on the same device");

    // K and V must match each other, but Q need NOT match them: with a query sub-region Q spans
    // only the sites this device produces output for, while K and V span those plus the halo the
    // windows reach into. The site counts are checked against the plan below, which is what
    // actually pins each tensor to its own brick grid.
    TT_FATAL(
        key_tensor.logical_shape() == value_tensor.logical_shape(),
        "neighborhood_sdpa: key and value must have the same shape, got {} and {}",
        key_tensor.logical_shape(),
        value_tensor.logical_shape());
    TT_FATAL(
        query_tensor.logical_shape()[0] == key_tensor.logical_shape()[0] &&
            query_tensor.logical_shape()[3] == key_tensor.logical_shape()[3],
        "neighborhood_sdpa: query and key must agree on batch and width, got {} and {}",
        query_tensor.logical_shape(),
        key_tensor.logical_shape());

    for (const Tensor* tensor : {&query_tensor, &key_tensor, &value_tensor}) {
        TT_FATAL(tensor->layout() == Layout::TILE, "neighborhood_sdpa: query/key/value must be TILE layout");
        TT_FATAL(
            tensor->dtype() == DataType::BFLOAT16 || tensor->dtype() == DataType::BFLOAT8_B,
            "neighborhood_sdpa: query/key/value must be bfloat16 or bfloat8_b, got {}",
            tensor->dtype());
    }

    // Sites arrive BRICKED, so the site axis is a whole number of bricks -- one brick per tile
    // row. A volume that does not divide into whole bricks is padded by the permute, which is
    // why this compares against the plan's brick count and not against volume.sites().
    const neighborhood::NeighborhoodPlan plan = neighborhood::build_plan(attributes.config);
    const auto query_shape = query_tensor.logical_shape();
    const uint32_t site_count = query_shape[2];
    // Q is sized by the QUERY region, K and V by the resident one. They differ exactly when the
    // caller asked for a query sub-region; otherwise both reduce to the resident brick count.
    const uint32_t expected_key_site_count = plan.brick_count * neighborhood::SITES_PER_BRICK;
    const uint32_t key_site_count = key_tensor.logical_shape()[2];
    TT_FATAL(
        key_site_count == expected_key_site_count,
        "neighborhood_sdpa: key/value have {} sites but the resident region implies {} bricked sites",
        key_site_count,
        expected_key_site_count);
    const uint32_t expected_site_count = plan.query_brick_count * neighborhood::SITES_PER_BRICK;
    TT_FATAL(
        site_count == expected_site_count,
        "neighborhood_sdpa: query has {} sites but volume {}x{}x{} with brick {}x{}x{} implies {} bricked sites",
        site_count,
        attributes.config.volume.time(),
        attributes.config.volume.height(),
        attributes.config.volume.width(),
        attributes.config.brick.time(),
        attributes.config.brick.height(),
        attributes.config.brick.width(),
        expected_site_count);

    const auto table_shape = gather_origin_table.logical_shape();
    TT_FATAL(
        table_shape[2] == plan.chunk_count && table_shape[3] == neighborhood::kernel_args::GATHER_ORIGIN_COLUMNS,
        "neighborhood_sdpa: gather_origin_table must be [1, 1, {}, {}], got {} -- one row per "
        "query CHUNK, not per brick",
        plan.chunk_count,
        neighborhood::kernel_args::GATHER_ORIGIN_COLUMNS,
        table_shape);
    TT_FATAL(
        gather_origin_table.dtype() == DataType::UINT32 && gather_origin_table.layout() == Layout::ROW_MAJOR,
        "neighborhood_sdpa: gather_origin_table must be uint32 ROW_MAJOR");

    // Two independent bounds.
    //
    // Whole BRICKS, not the site-exact tile count: the reader walks the gather brick by brick,
    // since one brick is one tile row.
    //
    // And DST capacity: a chunk's score tiles are live in the destination registers through the
    // row-max and the exp, so a chunk wider than DST silently returns wrong numbers -- it does
    // not fault. This is the same reason the rest of the SDPA family uses k_chunk_size = 256,
    // which is exactly 8 tiles. fp32 accumulation halves DST, so it halves this too.
    const uint32_t dst_capacity_tiles = get_fp32_dest_acc_en(attributes.compute_kernel_config) ? 4u : 8u;
    const uint32_t largest_chunk = std::min(plan.gather_brick_count, dst_capacity_tiles);
    TT_FATAL(
        attributes.tiles_per_kv_chunk > 0 && attributes.tiles_per_kv_chunk <= largest_chunk,
        "neighborhood_sdpa: tiles_per_kv_chunk must be in [1, {}] (gather is {} bricks, DST holds {} "
        "tiles), got {}",
        largest_chunk,
        plan.gather_brick_count,
        dst_capacity_tiles,
        attributes.tiles_per_kv_chunk);

    TT_FATAL(
        attributes.head_count > 0 && query_shape[3] % attributes.head_count == 0,
        "neighborhood_sdpa: {} channels do not divide into {} heads",
        query_shape[3],
        attributes.head_count);
    const uint32_t head_dim = query_shape[3] / attributes.head_count;
    TT_FATAL(
        head_dim % tt::constants::TILE_WIDTH == 0,
        "neighborhood_sdpa: head_dim {} must be a multiple of {}",
        head_dim,
        tt::constants::TILE_WIDTH);
    if (attributes.k_norm_bound.has_value()) {
        TT_FATAL(
            *attributes.k_norm_bound > 0.f,
            "neighborhood_sdpa: k_norm_bound must be positive (sqrt(head_dim)*max|k_norm_weight|); got {}",
            *attributes.k_norm_bound);
    }
    TT_FATAL(
        attributes.path_mode <= 2,
        "neighborhood_sdpa: path_mode must be 0 (auto), 1 (interior), or 2 (edge); got {}",
        attributes.path_mode);
}

NeighborhoodSDPAOperation::spec_return_value_t NeighborhoodSDPAOperation::compute_output_specs(
    const NeighborhoodSDPAParams& attributes, const NeighborhoodSDPAInputs& tensors) {
    if (tensors.output_tensor.has_value()) {
        return tensors.output_tensor->tensor_spec();
    }
    // Attention is shape-preserving, and the output stays in bricked order so the next block
    // can consume it without a round trip through natural order.
    const auto& query_tensor = tensors.query_tensor;
    return tt::tt_metal::TensorSpec(
        query_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            query_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), attributes.output_memory_config));
}

NeighborhoodSDPAOperation::tensor_return_value_t NeighborhoodSDPAOperation::create_output_tensors(
    const NeighborhoodSDPAParams& attributes, const NeighborhoodSDPAInputs& tensors) {
    if (tensors.output_tensor.has_value()) {
        return *tensors.output_tensor;
    }
    return create_device_tensor(compute_output_specs(attributes, tensors), tensors.query_tensor.device());
}

ttsl::hash::hash_t NeighborhoodSDPAOperation::compute_program_hash(
    const NeighborhoodSDPAParams& attributes, const NeighborhoodSDPAInputs& tensors) {
    // The eight stage-5 blocks share one geometry, so they must share one compiled program.
    // The gather origin table's CONTENTS are a runtime buffer; only its shape matters here.
    const auto& config = attributes.config;
    return tt::tt_metal::operation::hash_operation<NeighborhoodSDPAOperation>(
        config.volume.by_axis,
        config.context_window.by_axis,
        config.stride.by_axis,
        config.brick.by_axis,
        config.shard_extent.by_axis,
        // shard_origin is deliberately NOT hashed: it rides the gather origin table as runtime
        // data, so one compiled program serves every shard of the mesh.
        //
        // query_extent and query_origin ARE hashed, unlike shard_origin: both reach the reader as
        // COMPILE-TIME arguments (query_bricks, query_origin_bricks), and query_extent also sets
        // the chunk count. They are uniform across the mesh -- every shard owns the same-shaped
        // region at the same offset in its resident box -- so hashing them costs no sharing.
        config.query_extent.by_axis,
        config.query_origin.by_axis,
        attributes.head_count,
        attributes.scale,
        attributes.k_norm_bound.has_value(),
        attributes.tiles_per_kv_chunk,
        attributes.probe,
        attributes.path_mode,
        tensors.interior_mask.has_value(),
        tensors.query_tensor.logical_shape(),
        tensors.query_tensor.dtype(),
        tensors.query_tensor.memory_config(),
        tensors.key_tensor.dtype(),
        tensors.value_tensor.dtype(),
        attributes.output_memory_config);
}

Tensor neighborhood_sdpa(
    const Tensor& query_tensor,
    const Tensor& key_tensor,
    const Tensor& value_tensor,
    const Tensor& gather_origin_table,
    const std::optional<Tensor>& interior_mask,
    const neighborhood::NeighborhoodConfig& config,
    uint32_t head_count,
    float scale,
    uint32_t tiles_per_kv_chunk,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    DeviceComputeKernelConfig compute_kernel_config,
    const std::optional<Tensor>& output_tensor,
    std::optional<float> k_norm_bound,
    std::optional<uint32_t> probe,
    uint32_t path_mode) {
    using OperationType = ttnn::prim::NeighborhoodSDPAOperation;
    const uint32_t resolved_probe = resolve_probe(probe);

    // Classify and the tight gather cannot share a kernel (64 ns/slot). Stride-1 with an
    // uploaded relative table therefore launches two programs: interior (tight, skip edges)
    // then edge (classify, skip interiors). Writer of each launch only DRAM-writes its set.
    const bool relative_mask = config.stride.time() == 1 && config.stride.height() == 1 && config.stride.width() == 1;
    const uint32_t query_tile_rows = config.bricks_per_query_chunk();
    const bool chunk_exceeds_stride = query_tile_rows > 1 && !(config.query_chunk_sites() == config.stride);
    const char* per_brick_env = std::getenv("DIFFVAE_NA_PER_BRICK_MASK");
    const bool per_brick_mask = per_brick_env != nullptr ? per_brick_env[0] == '1' : chunk_exceeds_stride;
    const char* always_env = std::getenv("DIFFVAE_NA_TABLE_ALWAYS");
    const bool table_always = always_env != nullptr && always_env[0] == '1';
    const bool split = path_mode == 0 && resolved_probe == 0 && relative_mask && interior_mask.has_value() &&
                       !per_brick_mask && !table_always;

    auto launch_with_mode = [&](uint32_t mode, const std::optional<Tensor>& out) {
        return ttnn::device_operation::launch<OperationType>(
            OperationType::operation_attributes_t{
                .config = config,
                .head_count = head_count,
                .scale = scale,
                .k_norm_bound = k_norm_bound,
                .tiles_per_kv_chunk = tiles_per_kv_chunk,
                .probe = resolved_probe,
                .path_mode = mode,
                .compute_kernel_config = compute_kernel_config,
                .output_memory_config = output_memory_config,
            },
            OperationType::tensor_args_t{
                .query_tensor = query_tensor,
                .key_tensor = key_tensor,
                .value_tensor = value_tensor,
                .gather_origin_table = gather_origin_table,
                .interior_mask = interior_mask,
                .output_tensor = out,
            });
    };

    if (split) {
        Tensor interior = launch_with_mode(1, output_tensor);
        return launch_with_mode(2, interior);
    }
    return launch_with_mode(path_mode, output_tensor);
}

}  // namespace ttnn::prim
