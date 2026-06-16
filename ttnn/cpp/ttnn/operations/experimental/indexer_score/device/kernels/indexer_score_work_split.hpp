// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Causal work-split arithmetic for indexer_score, shared by the host factory (deals
// units across cores) and the kernels (WorkUnitSpan inverts a flat index to group/unit).
// Single-sourcing stops host/device desync (else cores overlap or miss output tiles).
// Pure uint32_t math (no std::, no CT args) so it includes cleanly on both sides.

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::indexer_score {

// Dense schedule: deal the full [0, k_len_tiles) rectangle for every group (not just its causal
// prefix), so every group has the same unit count and the flat deal is perfectly uniform. Future
// tiles are computed + masked to -inf in-band. Negligible cost at high sp_rank, ~2x at sp~0 -- so
// the causal split (false) is kept for early chunks. Shared constexpr (not a CT arg) so host and
// device read the SAME value and can never desync the unit count.
constexpr bool dense_schedule = true;

/** Causal-valid output k-tiles in q-tile-row q_row: columns [0, this) are causally visible. */
constexpr uint32_t valid_k_tiles_in_row(uint32_t q_row, uint32_t chunk_start_tiles, uint32_t k_len_tiles) {
    const uint32_t v = chunk_start_tiles + q_row + 1;
    return v < k_len_tiles ? v : k_len_tiles;
}

/** k-tiles scheduled for a q-row-group. Dense: full rectangle; causal: valid k-tiles of its last
 *  (widest) row. The ONE place the schedule shape is decided; host and device derive the rest. */
constexpr uint32_t valid_k_tiles_in_group(
    uint32_t group, uint32_t q_tiles_per_unit, uint32_t chunk_start_tiles, uint32_t k_len_tiles) {
    if (dense_schedule) {
        return k_len_tiles;
    }
    return valid_k_tiles_in_row((group + 1) * q_tiles_per_unit - 1, chunk_start_tiles, k_len_tiles);
}

/** Unmasked prefix k-tiles of q-tile-row q_row_abs in a unit (start k_tile_start, k_tiles_in_unit
 *  valid). Tiles [0, this) are below the diagonal (no mask); diagonal and beyond are masked. */
constexpr uint32_t valid_prefix_tiles(
    uint32_t q_row_abs, uint32_t k_tile_start, uint32_t k_tiles_in_unit, uint32_t chunk_start_tiles) {
    const uint32_t diag_tile = chunk_start_tiles + q_row_abs;
    const uint32_t v = diag_tile > k_tile_start ? diag_tile - k_tile_start : 0;
    return v < k_tiles_in_unit ? v : k_tiles_in_unit;
}

/** Number of k-chunk work units in a q-row-group (ceil of its valid width over the k chunk). */
constexpr uint32_t units_in_group(
    uint32_t group,
    uint32_t q_tiles_per_unit,
    uint32_t k_tiles_per_unit,
    uint32_t chunk_start_tiles,
    uint32_t k_len_tiles) {
    const uint32_t valid = valid_k_tiles_in_group(group, q_tiles_per_unit, chunk_start_tiles, k_len_tiles);
    return (valid + k_tiles_per_unit - 1) / k_tiles_per_unit;
}

}  // namespace ttnn::operations::experimental::indexer_score
