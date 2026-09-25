// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared host/device integer math for per-device causal geometry. Keep this header free of host-only
// dependencies so device kernels can include it.

#include <cstdint>

#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/block_cyclic_causal_geometry.hpp"

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
// cross a slab boundary, which moves its causal diagonal by chunk_global - chunk_local. The closed form is
// shared with sparse_sdpa_msa (token granularity) in block_cyclic_causal_geometry.hpp; this is its tile view.
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
    const auto g = tt::block_cyclic::causal_geometry(
        chunk_start_idx, has_block_cyclic, rotation_exact, sp, chunk_local, device_index, tp_index, Sq);
    return {g.chunk_start / TW, g.straddle_q / TW, g.straddle_jump / TW};
}

}  // namespace ttnn::operations::experimental::indexer_score
