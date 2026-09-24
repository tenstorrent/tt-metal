// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared host/device integer math for per-device causal geometry. Keep this header free of host-only
// dependencies so device kernels can include it.

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

// Block-cyclic cache writes rotate ownership when a chunk starts mid-slab. The boundary chip may also
// cross a slab boundary, which moves its causal diagonal by chunk_global - chunk_local.
//
// `rotation_exact` selects the rotation-exact SP mapping over the flat both-axes approximation. It is NOT
// simply "has a named SP axis": a fused FULL-MESH ring has no named SP axis, yet every canonical tensor
// rank is an SP rank and ownership follows the same rotation. Sending that case down the flat path would
// assign the causal diagonals to the wrong tensor ranks and could mark every rank as straddling. The
// caller computes the predicate once -- host in device_causal_geometry(), device from a compile-time arg
// the factory sets from the same expression -- so the two cannot disagree.
inline CausalGeometryTiles causal_geometry_tiles(
    uint32_t chunk_start_idx,
    bool has_block_cyclic,
    bool rotation_exact,
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

    if (rotation_exact) {
        // device_index selects the SP slab and tp_index selects this device's row range within it.
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
