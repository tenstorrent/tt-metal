// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side helpers shared by BOTH indexer_score program factories (the classic Program-model factory and the
// ring-fused ProgramDescriptor-model factory): per-device causal geometry, the SP-ring index lookup, and the
// persistent-cache runtime args. Kept in one place so the two factories can never drift (they produce
// byte-identical values by construction). Pure host math -- no kernel/program-model dependency.

#include <cstdint>

#include <tt_stl/assert.hpp>          // TT_FATAL
#include <tt-metalium/constants.hpp>  // tt::constants::TILE_WIDTH

#include "ttnn/distributed/types.hpp"                // ttnn::MeshCoordinate
#include "ttnn/operations/ccl/ccl_common.hpp"        // get_linearized_index_from_physical_coord
#include "indexer_score_device_operation_types.hpp"  // operation_attributes_t, Tensor

namespace ttnn::operations::experimental::indexer_score::program {

// Per-device causal geometry for the block-cyclic slab layout, all in tiles. The global chunk
// [chunk_start_idx, chunk_start_idx + chunk_global) is written round-robin across the sp chips by
// update_padded_kv_cache, so chip c's Sq queries are a CONTIGUOUS logical block whose start follows the
// writer's rotation -- NOT the linear chunk_start_idx + c*Sq. Two effects of a mid-slab chunk_start_idx:
//   (a) block rotation: the starting block index (chunk_start_idx / chunk_local) can land on a chip != 0
//       (boundary_chip), rotating which chip owns which block -- so chip c's logical start is the writer's
//       update_idxt, mirroring rotated_chip_positions[c][0], not chunk_start_idx + c*chunk_local; and
//   (b) straddle: the boundary chip's Sq queries cross a slab boundary, so its causal diagonal JUMPS by
//       (chunk_global - chunk_local) tiles at q-row (chunk_local - offset).
// The linear form only misses (a) when boundary_chip != 0 -- exactly the mid-slab, non-chip-0-start case
// (e.g. the multi-turn rotated prefill). Chunk-aligned (offset == 0, boundary_chip == 0) reduces to linear.
// No block_cyclic -> plain linear. The both-axes case (cluster_axis unset, block_cyclic_chunk_local == tp*Sq)
// keeps the prior linear+straddle form. Shared by create_at (device_index from the coordinate) and override
// (stored device_index).
struct DeviceCausalGeometry {
    uint32_t chunk_start_tiles;    // global position of this device's q-row 0 (tiles)
    uint32_t straddle_q_tile;      // q-tile-row at/after which the diagonal jumps (only when this device straddles)
    uint32_t straddle_jump_tiles;  // diagonal jump in tiles (0 unless this device straddles)
};
inline DeviceCausalGeometry device_causal_geometry(
    const operation_attributes_t& args, uint32_t device_index, uint32_t Sq) {
    const uint32_t TW = tt::constants::TILE_WIDTH;
    if (!args.block_cyclic.has_value()) {
        return {(args.chunk_start_idx + device_index * Sq) / TW, 0u, 0u};  // contiguous K -> linear diagonal
    }
    const uint32_t sp = args.block_cyclic->sp;
    const uint32_t chunk_local = args.block_cyclic->chunk_local;  // cache per-shard slab width (elements)
    const uint32_t chunk_global = sp * chunk_local;

    if (args.cluster_axis.has_value()) {
        TT_FATAL(
            device_index < sp,
            "indexer_score: device_index {} out of range for block-cyclic sp={} (check cluster_axis vs "
            "block_cyclic_sp_axis)",
            device_index,
            sp);
        // SP-only block-cyclic: device_index is the SP-ring index and owns ONE block (Sq == chunk_local).
        // Mirror the update_padded_kv_cache writer's update_idxt (== rotated_chip_positions[device_index][0])
        // so the diagonal starts at this chip's TRUE logical block -- handling the boundary_chip rotation that
        // the linear chunk_start_idx + c*Sq misses. Only the boundary chip is mid-slab, so only it straddles.
        const uint32_t boundary_slab = args.chunk_start_idx / chunk_global;
        const uint32_t boundary_chip = (args.chunk_start_idx / chunk_local) % sp;
        const uint32_t offset = args.chunk_start_idx % chunk_local;
        const uint32_t update_idxt = device_index < boundary_chip    ? (boundary_slab + 1) * chunk_local
                                     : device_index == boundary_chip ? boundary_slab * chunk_local + offset
                                                                     : boundary_slab * chunk_local;
        const uint32_t logical_start =
            (update_idxt / chunk_local) * chunk_global + device_index * chunk_local + (update_idxt % chunk_local);
        uint32_t straddle_q_tile = 0, straddle_jump_tiles = 0;
        if (device_index == boundary_chip && offset != 0 && offset + Sq > chunk_local) {
            straddle_q_tile = (chunk_local - offset) / TW;
            straddle_jump_tiles = (chunk_global - chunk_local) / TW;
        }
        return {logical_start / TW, straddle_q_tile, straddle_jump_tiles};
    }

    // Both-axes (cluster_axis unset): prior linear + within-block straddle geometry.
    const uint32_t chunk_start = args.chunk_start_idx + device_index * Sq;
    const uint32_t offset = chunk_start % chunk_local;
    uint32_t straddle_q_tile = 0, straddle_jump_tiles = 0;
    if (offset != 0 && offset + Sq > chunk_local) {
        straddle_q_tile = (chunk_local - offset) / TW;
        straddle_jump_tiles = (chunk_global - chunk_local) / TW;
    }
    return {chunk_start / TW, straddle_q_tile, straddle_jump_tiles};
}

// This device's linearized SP-ring index; 0 on a single device (no coordinate lookup needed).
inline uint32_t device_index_for(
    const operation_attributes_t& args, const ttnn::MeshCoordinate& coord, const Tensor& q) {
    if (q.device_storage().get_coords().size() <= 1) {
        return 0;
    }
    return ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, args.cluster_axis);
}

// The two non-hashed runtime args derived from k's shape + the optionals. Single source for both create()
// (bakes at miss) and override_runtime_arguments() (re-patches on a hit).
struct PersistentCacheArgs {
    uint32_t k_batch_page_offset;  // cache_batch_idx * Tt * Dt; 0 when not indexed
    uint32_t kv_len_tiles;         // valid key prefix in tiles; full Tt when kv_len unset
};
inline PersistentCacheArgs persistent_cache_args(const operation_attributes_t& attrs, const Tensor& k) {
    const auto& shape = k.logical_shape();
    const uint32_t Tt = shape[2] / tt::constants::TILE_WIDTH;
    const uint32_t Dt = shape[3] / tt::constants::TILE_WIDTH;
    return {
        .k_batch_page_offset = attrs.cache_batch_idx.value_or(0) * Tt * Dt,
        .kv_len_tiles = attrs.kv_len.value_or(shape[2]) / tt::constants::TILE_WIDTH};
}

}  // namespace ttnn::operations::experimental::indexer_score::program
