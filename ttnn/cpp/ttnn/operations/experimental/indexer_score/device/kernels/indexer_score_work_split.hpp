// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Work-split arithmetic for indexer_score, shared by the host factory (deals units across
// cores) and the kernels (WorkUnitSpan inverts a flat index to group/unit).
// Single-sourcing stops host/device desync (else cores overlap or miss output tiles).
// Pure uint32_t math (no std::, no CT args) so it includes cleanly on both sides.

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::indexer_score {

// Schedule: every q-row-group is dealt the full [0, k_len_tiles) k rectangle (not just its causal
// prefix), so the flat deal is perfectly uniform across groups -- which is what lets the grid-aligned
// multicast path apply. Future/diagonal tiles are computed and masked to -inf in-band (see
// valid_prefix_tiles); cost is negligible at high sp_rank (~2x only near sp~0).

/** Unmasked prefix k-tiles of q-tile-row q_row_abs in a unit (start k_tile_start, k_tiles_in_unit
 *  valid). Tiles [0, this) are below the diagonal (no mask); diagonal and beyond are masked. */
constexpr uint32_t valid_prefix_tiles(
    uint32_t q_row_abs, uint32_t k_tile_start, uint32_t k_tiles_in_unit, uint32_t chunk_start_tiles) {
    const uint32_t diag_tile = chunk_start_tiles + q_row_abs;
    const uint32_t v = diag_tile > k_tile_start ? diag_tile - k_tile_start : 0;
    return v < k_tiles_in_unit ? v : k_tiles_in_unit;
}

/** Work units per q-row-group: ceil of the full k rectangle over the k chunk. Uniform across groups
 *  under the dense schedule. */
constexpr uint32_t units_in_group(uint32_t k_tiles_per_unit, uint32_t k_len_tiles) {
    return (k_len_tiles + k_tiles_per_unit - 1) / k_tiles_per_unit;
}

}  // namespace ttnn::operations::experimental::indexer_score
