// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "synthesize_output_shard_spec.hpp"

#include <algorithm>

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
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
    // Non-{H/W} layouts (ND_SHARDED, INTERLEAVED) fall through to BLOCK in the non-zero path,
    // but FATAL in the zero-vol representable check at line 55 — asymmetry is intentional.
    const CoreRangeSet all_cores(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}));
    const uint32_t num_cores = all_cores.num_cores();
    TT_FATAL(num_cores > 0, "{}: empty compute grid.", opts.caller_tag);

    const ShardOrientation orientation = resolve_orientation(opts);
    const bool row_wise = (orientation == ShardOrientation::ROW_MAJOR);
    const uint32_t h_align = opts.is_tile ? tt::constants::TILE_HEIGHT : 1u;
    const uint32_t w_align = opts.is_tile ? tt::constants::TILE_WIDTH : 1u;

    // Zero-volume: TensorSpec (tensor_spec.cpp:41-49) pins shard extent to physical extent on the non-sharded
    // axis, so only HEIGHT+zero-h and WIDTH+zero-w are representable; crossover / BLOCK+zero → FATAL.
    if (tensor_height == 0 || tensor_width == 0) {
        const bool representable = (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED && tensor_width > 0) ||
                                   (memory_layout == TensorMemoryLayout::WIDTH_SHARDED && tensor_height > 0);
        TT_FATAL(
            representable,
            "{}: zero-volume specless-sharded is only representable for HEIGHT_SHARDED + non-zero width or "
            "WIDTH_SHARDED + non-zero height; got layout={}, h={}, w={}.",
            opts.caller_tag,
            static_cast<int>(memory_layout),
            tensor_height,
            tensor_width);
        uint32_t sh = h_align;
        uint32_t sw = w_align;
        if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            sw = static_cast<uint32_t>(tt::round_up(tensor_width, w_align));
        } else {
            sh = static_cast<uint32_t>(tt::round_up(tensor_height, h_align));
        }
        return ShardSpec(CoreRangeSet(CoreRange({0, 0}, {0, 0})), {sh, sw}, orientation);
    }

    // Non-sharded axis aligned to output layout (TILE=32, RM=1); TensorSpec pins physical to align-rounded logical.
    std::array<uint32_t, 2> shard_shape = {0, 0};
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto height_padded = tt::round_up(tensor_height, static_cast<uint64_t>(num_cores) * h_align);
        const auto shard_height = tt::round_up(tt::div_up(height_padded, static_cast<uint64_t>(num_cores)), h_align);
        const auto shard_width = tt::round_up(tensor_width, w_align);
        shard_shape = {static_cast<uint32_t>(shard_height), static_cast<uint32_t>(shard_width)};
    } else if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        const auto shard_width = tt::round_up(tt::div_up(tensor_width, static_cast<uint64_t>(num_cores)), w_align);
        const auto shard_height = tt::round_up(tensor_height, h_align);
        shard_shape = {static_cast<uint32_t>(shard_height), static_cast<uint32_t>(shard_width)};
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

std::optional<ShardSpec> shrink_shard_for_rm_page_alignment(
    const ShardSpec& spec,
    tt::tt_metal::Layout input_layout,
    uint32_t element_size_bytes,
    uint64_t tensor_width,
    const CoreCoord& compute_grid_size,
    TensorMemoryLayout memory_layout,
    RmPageAlignmentMode mode) {
    // TILE inputs land tile-aligned pages by construction; RM alignment coupling is the only concern.
    if (input_layout != tt::tt_metal::Layout::ROW_MAJOR) {
        return spec;
    }
    // Zero-width WIDTH_SHARDED is representable in the synth; nc-shrink is a no-op (no page to align).
    if (tensor_width == 0) {
        return (mode == RmPageAlignmentMode::Strict) ? std::optional<ShardSpec>{std::nullopt} : std::optional{spec};
    }
    const uint64_t l1_page_align = static_cast<uint64_t>(tt::tt_metal::hal::get_l1_alignment());
    auto rm_page_valid = [&](uint64_t shard_width) {
        const uint64_t page = shard_width * static_cast<uint64_t>(element_size_bytes);
        return page != 0 && (page % l1_page_align) == 0 && (tensor_width % shard_width) == 0;
    };
    if (rm_page_valid(spec.shape[1])) {
        return spec;
    }
    // Only WIDTH_SHARDED can be repaired by shrinking nc; {H,B}_SHARDED pin shard_width to tensor_width.
    // Lenient returns `spec` unchanged (downstream FATAL surfaces later); Strict rejects so caller falls back.
    if (memory_layout != TensorMemoryLayout::WIDTH_SHARDED) {
        return (mode == RmPageAlignmentMode::Strict) ? std::optional<ShardSpec>{std::nullopt} : std::optional{spec};
    }
    const uint32_t max_cores = static_cast<uint32_t>(compute_grid_size.x) * static_cast<uint32_t>(compute_grid_size.y);
    const bool row_wise = (spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    uint32_t chosen_cores = 0;
    uint32_t new_shard_width = 0;
    for (uint32_t nc = max_cores; nc >= 1; --nc) {
        if (tensor_width % nc != 0) {
            continue;
        }
        if (rm_page_valid(tensor_width / nc)) {
            chosen_cores = nc;
            new_shard_width = static_cast<uint32_t>(tensor_width / nc);
            break;
        }
    }
    if (chosen_cores == 0) {
        if (mode == RmPageAlignmentMode::Strict) {
            return std::nullopt;
        }
        // Lenient: tile-pad shard_width (page = TILE_WIDTH * elem covers every L1 alignment); shape[0] stays physical.
        const uint32_t tile_w = tt::constants::TILE_WIDTH;
        new_shard_width = static_cast<uint32_t>(
            tt::round_up(tt::div_up(tensor_width, static_cast<uint64_t>(max_cores)), static_cast<uint64_t>(tile_w)));
        chosen_cores = static_cast<uint32_t>(tt::div_up(tensor_width, static_cast<uint64_t>(new_shard_width)));
        // TILE_WIDTH=32 * elem_size >= 32B covers every L1 alignment (<= 16B) this repo ships; unreachable otherwise.
        TT_ASSERT(
            (static_cast<uint64_t>(new_shard_width) * element_size_bytes) % l1_page_align == 0,
            "shrink_shard_for_rm_page_alignment: tile-padded page ({}B) not L1-aligned ({}B); unexpected (elem={}, arch).",
            new_shard_width * element_size_bytes,
            l1_page_align,
            element_size_bytes);
    }
    auto new_grid = tt::tt_metal::num_cores_to_corerangeset(chosen_cores, compute_grid_size, row_wise);
    return ShardSpec(new_grid, {spec.shape[0], new_shard_width}, spec.orientation);
}

}  // namespace ttnn::operations::data_movement::common
