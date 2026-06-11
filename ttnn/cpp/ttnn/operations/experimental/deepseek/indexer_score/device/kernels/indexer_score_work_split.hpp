// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Causal work-split arithmetic for indexer_score, shared by the host program
// factory (which deals the flat count of valid work units across cores) and the
// device kernels (whose WorkUnitSpan inverts the flat index back to a group/unit).
// Single-sourcing the formula keeps the two from drifting: if the host counted
// units differently than the kernels walk them, cores would compute overlapping
// or missing output tiles. Pure uint32_t math (no std::, no compile-time args) so
// it includes cleanly on both sides - same idea as SDPA's chunked_prefill_utils.hpp.

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::deepseek::indexer {

/** Causal-valid output k-tiles in q-tile-row q_row: columns [0, this) are computed. */
constexpr uint32_t valid_k_tiles_in_row(uint32_t q_row, uint32_t chunk_start_tiles, uint32_t k_len_tiles) {
    const uint32_t v = chunk_start_tiles + q_row + 1;
    return v < k_len_tiles ? v : k_len_tiles;
}

/** Valid k-tiles of a q-row-group = those of its last row (the widest in the group). */
constexpr uint32_t valid_k_tiles_in_group(
    uint32_t group, uint32_t q_tiles_per_unit, uint32_t chunk_start_tiles, uint32_t k_len_tiles) {
    return valid_k_tiles_in_row((group + 1) * q_tiles_per_unit - 1, chunk_start_tiles, k_len_tiles);
}

/** Unmasked prefix k-tiles for absolute q-tile-row q_row_abs within a unit starting at k tile
 *  k_tile_start with k_tiles_in_unit causal-valid tiles. Tiles [0, this) are < the diagonal tile
 *  (fully valid, no mask); the diagonal tile and beyond are masked. Shared by compute (which
 *  produces the prefix then masks the suffix) and the writer (which picks the full-strip vs
 *  per-tile output path), so the two never disagree on which tiles are a clean full-width strip. */
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

}  // namespace ttnn::operations::experimental::deepseek::indexer
