// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-device causal geometry for indexer_score, as PURE integer math over plain uint32_t.
//
// This header is included by BOTH the host (indexer_score_host_common.hpp delegates to it) and the
// device kernels (reader/writer, which recompute the same geometry on-device from a metadata-read
// chunk_start_idx on the trace-safe path). Single-sourcing it is the point: host and kernel must agree
// bit-for-bit, and a "keep in sync" comment across two copies is exactly the kind of drift that shows
// up later as a wrong causal offset on one code path only.
//
// Therefore: NO std::, NO tt-metalium, NO tt_stl includes -- kernels cannot take those.

#include <cstdint>

namespace ttnn::operations::experimental::indexer_score {

// Tile width in elements. Local constant rather than tt::constants::TILE_WIDTH so this header stays
// kernel-includable; the host header static_asserts that the two agree.
inline constexpr uint32_t kCausalTileWidth = 32;

struct CausalGeometryTiles {
    uint32_t chunk_start_tiles;    // global position of this device's q-row 0 (tiles)
    uint32_t straddle_q_tile;      // q-tile-row at/after which the diagonal jumps (0 unless straddling)
    uint32_t straddle_jump_tiles;  // diagonal jump in tiles (0 unless straddling)
};

// The global chunk [chunk_start_idx, chunk_start_idx + chunk_global) is written round-robin across the
// sp chips by update_padded_kv_cache, so chip c's Sq queries are a CONTIGUOUS logical block whose start
// follows the writer's rotation -- NOT the linear chunk_start_idx + c*Sq. Two effects of a mid-slab
// chunk_start_idx:
//   (a) block rotation: the starting block index (chunk_start_idx / chunk_local) can land on a chip != 0
//       (boundary_chip), rotating which chip owns which block -- so chip c's logical start is the
//       writer's update_idxt, mirroring rotated_chip_positions[c][0]; and
//   (b) straddle: the boundary chip's Sq queries cross a slab boundary, so its causal diagonal JUMPS by
//       (chunk_global - chunk_local) tiles at q-row (chunk_local - offset).
// The linear form only misses (a) when boundary_chip != 0 -- exactly the mid-slab, non-chip-0-start case
// (e.g. the multi-turn rotated prefill). Chunk-aligned (offset == 0, boundary_chip == 0) reduces to
// linear. No block_cyclic -> plain linear. The both-axes case (SP axis unset) keeps the prior
// linear+straddle form.
//
// All arguments other than chunk_start_idx are structural (hashed / compile-time constant per program),
// which is what makes the trace-safe path possible: only chunk_start_idx varies per chunk.
inline CausalGeometryTiles causal_geometry_tiles(
    uint32_t chunk_start_idx,
    bool has_block_cyclic,
    bool has_sp_axis,
    uint32_t sp,
    uint32_t chunk_local,
    uint32_t device_index,
    uint32_t tp_index,
    uint32_t Sq) {
    constexpr uint32_t TW = kCausalTileWidth;

    if (!has_block_cyclic) {
        // Contiguous K -> linear diagonal at chunk_start + (seq-shard rank)*Sq. The rank is device_index
        // for an SP-only seq shard; but a 2D SP x TP sub-shard whose SP axis is size-1 (e.g. QuietBox
        // sp=1) is stored as no-block-cyclic (identity permutation), and there the query is seq-sharded
        // over the TP axis, so the rank is tp_index. The two are mutually exclusive nonzero here, so
        // their sum is the rank.
        return {(chunk_start_idx + (device_index + tp_index) * Sq) / TW, 0u, 0u};
    }

    const uint32_t chunk_global = sp * chunk_local;

    if (has_sp_axis) {
        // Block-cyclic, named SP axis. device_index is the SP-ring index; its slab starts at the writer's
        // update_idxt, handling the boundary_chip rotation the linear form misses. tp_index (SP x TP 2D
        // sub-shard) selects this device's Sq-row sub-range within that slab: it owns local rows
        // [tp_index*Sq, (tp_index+1)*Sq). lr0 is its first slab-local row; the mapping and straddle below
        // are EXACT for both the SP-only case (tp_index==0, Sq==chunk_local) and the 2D case.
        //
        // Caller contract: device_index < sp. The host validates it with a TT_FATAL; kernels cannot
        // raise, and the value is a compile-time constant there, so it is not re-checked.
        const uint32_t boundary_slab = chunk_start_idx / chunk_global;
        const uint32_t boundary_chip = (chunk_start_idx / chunk_local) % sp;
        const uint32_t offset = chunk_start_idx % chunk_local;
        const uint32_t update_idxt = device_index < boundary_chip    ? (boundary_slab + 1) * chunk_local
                                     : device_index == boundary_chip ? boundary_slab * chunk_local + offset
                                                                     : boundary_slab * chunk_local;
        const uint32_t lr0 = update_idxt + tp_index * Sq;  // this device's first slab-local row
        const uint32_t loff = lr0 % chunk_local;           // its offset within the current slab
        const uint32_t logical_start = (lr0 / chunk_local) * chunk_global + device_index * chunk_local + loff;
        uint32_t straddle_q_tile = 0, straddle_jump_tiles = 0;
        if (loff != 0 && loff + Sq > chunk_local) {  // this device's Sq rows cross a slab boundary
            straddle_q_tile = (chunk_local - loff) / TW;
            straddle_jump_tiles = (chunk_global - chunk_local) / TW;
        }
        return {logical_start / TW, straddle_q_tile, straddle_jump_tiles};
    }

    // Both-axes (SP axis unset): prior linear + within-block straddle geometry.
    const uint32_t chunk_start = chunk_start_idx + device_index * Sq;
    const uint32_t offset = chunk_start % chunk_local;
    uint32_t straddle_q_tile = 0, straddle_jump_tiles = 0;
    if (offset != 0 && offset + Sq > chunk_local) {
        straddle_q_tile = (chunk_local - offset) / TW;
        straddle_jump_tiles = (chunk_global - chunk_local) / TW;
    }
    return {chunk_start / TW, straddle_q_tile, straddle_jump_tiles};
}

}  // namespace ttnn::operations::experimental::indexer_score
