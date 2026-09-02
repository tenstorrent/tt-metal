// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "synthesize_output_shard_spec.hpp"

#include <algorithm>

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::data_movement::common {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::ShardSpec;
using tt::tt_metal::TensorMemoryLayout;

namespace {

ShardOrientation resolve_orientation(const SynthesizeOutputShardSpecOpts& opts) {
    if (opts.orientation_hint.has_value()) {
        return *opts.orientation_hint;
    }
    if (opts.input_orientation.has_value()) {
        return *opts.input_orientation;
    }
    return ShardOrientation::ROW_MAJOR;
}

}  // namespace

ShardSpec synthesize_output_shard_spec(
    const CoreCoord& compute_grid_size,
    uint64_t tensor_height,
    uint64_t tensor_width,
    TensorMemoryLayout memory_layout,
    const SynthesizeOutputShardSpecOpts& opts) {
    // Sharded-only contract — callers filter INTERLEAVED at their wrapper (see repeat_utils.cpp:186-190).
    TT_FATAL(
        memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
            memory_layout == TensorMemoryLayout::BLOCK_SHARDED,
        "{}: unsupported memory_layout; only HEIGHT/WIDTH/BLOCK sharded.",
        opts.caller_tag);
    const CoreRangeSet all_cores(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}));
    const uint32_t num_cores = all_cores.num_cores();
    TT_FATAL(num_cores > 0, "{}: empty compute grid.", opts.caller_tag);
    // Guards `div_up(_, shard_shape[i])` below; repeat's soft-reject path never reaches here (see
    // repeat_utils.cpp:202).
    TT_FATAL(
        tensor_height > 0 && tensor_width > 0,
        "{}: tensor dims must be > 0; got ({}, {}).",
        opts.caller_tag,
        tensor_height,
        tensor_width);

    const ShardOrientation orientation = resolve_orientation(opts);
    const bool row_wise = (orientation == ShardOrientation::ROW_MAJOR);
    const uint32_t h_align = opts.is_tile ? tt::constants::TILE_HEIGHT : 1u;
    const uint32_t w_align = opts.is_tile ? tt::constants::TILE_WIDTH : 1u;

    std::array<uint32_t, 2> shard_shape = {0, 0};
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto height_padded = tt::round_up(tensor_height, static_cast<uint64_t>(num_cores) * h_align);
        const auto shard_height = tt::round_up(tt::div_up(height_padded, static_cast<uint64_t>(num_cores)), h_align);
        shard_shape = {static_cast<uint32_t>(shard_height), static_cast<uint32_t>(tensor_width)};
    } else if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        const auto shard_width = tt::round_up(tt::div_up(tensor_width, static_cast<uint64_t>(num_cores)), w_align);
        shard_shape = {static_cast<uint32_t>(tensor_height), static_cast<uint32_t>(shard_width)};
    } else {
        // BLOCK: COL_MAJOR swaps h↔grid.x, w↔grid.y (matches conv2d_utils::determine_parallel_config).
        const uint32_t h_div = row_wise ? compute_grid_size.y : compute_grid_size.x;
        const uint32_t w_div = row_wise ? compute_grid_size.x : compute_grid_size.y;
        const auto height_padded = tt::round_up(tensor_height, static_cast<uint64_t>(h_div) * h_align);
        const auto shard_height = tt::round_up(tt::div_up(height_padded, static_cast<uint64_t>(h_div)), h_align);
        const auto shard_width = tt::round_up(tt::div_up(tensor_width, static_cast<uint64_t>(w_div)), w_align);
        shard_shape = {static_cast<uint32_t>(shard_height), static_cast<uint32_t>(shard_width)};
    }

    CoreRangeSet used_cores;
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t n_used = static_cast<uint32_t>(tt::div_up(tensor_height, static_cast<uint64_t>(shard_shape[0])));
        n_used = std::min(std::max(n_used, 1u), num_cores);
        used_cores = (n_used == num_cores)
                         ? all_cores
                         : tt::tt_metal::num_cores_to_corerangeset(n_used, compute_grid_size, row_wise);
    } else if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t n_used = static_cast<uint32_t>(tt::div_up(tensor_width, static_cast<uint64_t>(shard_shape[1])));
        n_used = std::min(std::max(n_used, 1u), num_cores);
        used_cores = (n_used == num_cores)
                         ? all_cores
                         : tt::tt_metal::num_cores_to_corerangeset(n_used, compute_grid_size, row_wise);
    } else {
        const uint32_t n_h = static_cast<uint32_t>(tt::div_up(tensor_height, static_cast<uint64_t>(shard_shape[0])));
        const uint32_t n_w = static_cast<uint32_t>(tt::div_up(tensor_width, static_cast<uint64_t>(shard_shape[1])));
        const uint32_t n_along_x = row_wise ? n_w : n_h;
        const uint32_t n_along_y = row_wise ? n_h : n_w;
        TT_FATAL(
            n_along_x <= static_cast<uint32_t>(compute_grid_size.x) &&
                n_along_y <= static_cast<uint32_t>(compute_grid_size.y),
            "{}: BLOCK shard-grid ({}x{} along x/y) exceeds compute grid ({}x{}); shard=({},{}) orientation={}",
            opts.caller_tag,
            n_along_x,
            n_along_y,
            compute_grid_size.x,
            compute_grid_size.y,
            shard_shape[0],
            shard_shape[1],
            row_wise ? "ROW_MAJOR" : "COL_MAJOR");
        const uint32_t phys_x = std::max(n_along_x, 1u);
        const uint32_t phys_y = std::max(n_along_y, 1u);
        used_cores = (phys_x == static_cast<uint32_t>(compute_grid_size.x) &&
                      phys_y == static_cast<uint32_t>(compute_grid_size.y))
                         ? all_cores
                         : CoreRangeSet(CoreRange({0, 0}, {phys_x - 1, phys_y - 1}));
    }

    return ShardSpec(used_cores, shard_shape, orientation);
}

ShardSpec synthesize_output_shard_spec(
    const CoreCoord& compute_grid_size,
    const ttnn::Shape& padded_out_shape,
    TensorMemoryLayout memory_layout,
    const SynthesizeOutputShardSpecOpts& opts) {
    uint64_t tensor_height = 1;
    for (int32_t i = 0; i < static_cast<int32_t>(padded_out_shape.rank()) - 1; ++i) {
        tensor_height *= static_cast<uint64_t>(padded_out_shape[i]);
    }
    const uint64_t tensor_width = padded_out_shape[-1];
    return synthesize_output_shard_spec(compute_grid_size, tensor_height, tensor_width, memory_layout, opts);
}

}  // namespace ttnn::operations::data_movement::common
