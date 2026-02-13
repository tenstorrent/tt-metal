// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

/**
 * Common utility functions for ring reduce scatter kernels.
 */

FORCE_INLINE uint32_t wrap_slice_idx(const int32_t slice_idx, const bool direction, const uint32_t ring_size) {
    if (direction) {
        return slice_idx < 0 ? slice_idx + ring_size : slice_idx;
    } else {
        return slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
    }
}

FORCE_INLINE uint32_t get_effective_chunk_width_in_tiles(
    const uint32_t chunk_idx, const uint32_t chunk_width_in_tiles, const uint32_t mm_N_block_wt) {
    const uint32_t start_col = chunk_idx * chunk_width_in_tiles;
    const uint32_t remaining_width = mm_N_block_wt - start_col;

    return std::min(remaining_width, chunk_width_in_tiles);
}

FORCE_INLINE void get_next_tile_coordinates(
    uint32_t& tile_row_in_mm_M_block,
    uint32_t& chunk_col_in_tiles,
    uint32_t& mm_core_idx,
    uint32_t advance_by_tiles,
    const uint32_t chunk_piece_size,
    const uint32_t chunk_width_in_tiles,
    const uint32_t mm_block_unit_ht) {
    // Optimized to avoid modulo operations.
    // Note: chunk_piece_size == mm_block_unit_ht * chunk_width_in_tiles

    if (advance_by_tiles >= chunk_piece_size) [[unlikely]] {
        uint32_t move_by_pieces = advance_by_tiles / chunk_piece_size;
        advance_by_tiles -= move_by_pieces * chunk_piece_size;
        mm_core_idx += move_by_pieces;
    }

    if (advance_by_tiles >= chunk_width_in_tiles) {
        uint32_t move_by_rows = advance_by_tiles / chunk_width_in_tiles;
        uint32_t new_row = tile_row_in_mm_M_block + move_by_rows;
        advance_by_tiles -= move_by_rows * chunk_width_in_tiles;

        if (new_row >= mm_block_unit_ht) {
            mm_core_idx += 1;
            tile_row_in_mm_M_block = new_row - mm_block_unit_ht;
        } else {
            tile_row_in_mm_M_block = new_row;
        }
    }

    // 3. Handle Remaining Columns
    uint32_t new_col = chunk_col_in_tiles + advance_by_tiles;

    if (new_col >= chunk_width_in_tiles) {
        tile_row_in_mm_M_block += 1;
        new_col -= chunk_width_in_tiles;
        if (tile_row_in_mm_M_block >= mm_block_unit_ht) {
            mm_core_idx += 1;
            tile_row_in_mm_M_block = 0;
        }
    }

    chunk_col_in_tiles = new_col;
}

FORCE_INLINE uint32_t how_many_tiles_to_read_formula(
    uint32_t tile_row_in_mm_M_block,
    uint32_t chunk_col_in_tiles,
    uint32_t mm_core_idx,
    uint32_t advance_by_tiles,
    uint32_t last_mm_core_idx,
    uint32_t chunk_piece_size,
    uint32_t chunk_width_in_tiles) {
    if (mm_core_idx > last_mm_core_idx) {
        return 0;
    }
    uint32_t current_tile_offset = tile_row_in_mm_M_block * chunk_width_in_tiles + chunk_col_in_tiles;
    uint32_t current_block_tiles_remaining = chunk_piece_size - current_tile_offset - 1;
    uint32_t future_blocks_tiles = (last_mm_core_idx - mm_core_idx) * chunk_piece_size;
    uint32_t all_tiles = current_block_tiles_remaining + future_blocks_tiles;
    return 1 + all_tiles / advance_by_tiles;
}

FORCE_INLINE std::pair<uint32_t, uint32_t> coordinates_to_slice_coordinates(
    uint32_t tile_row_in_mm_M_block,
    uint32_t chunk_col_in_tiles,
    uint32_t mm_core_idx,
    uint32_t N_block_idx,
    uint32_t M_block_idx,
    uint32_t chunk_idx,
    uint32_t N_block_wt,
    uint32_t tiles_ht_per_core,
    uint32_t mm_block_unit_ht,
    uint32_t chunk_width_in_tiles) {
    uint32_t rows_before_this_core = mm_core_idx * tiles_ht_per_core;
    uint32_t rows_before_piece = M_block_idx * mm_block_unit_ht;
    uint32_t slice_row = rows_before_this_core + rows_before_piece + tile_row_in_mm_M_block;
    uint32_t slice_col = N_block_idx * N_block_wt + chunk_idx * chunk_width_in_tiles + chunk_col_in_tiles;

    return {slice_row, slice_col};
}

FORCE_INLINE uint32_t slice_coordinates_to_slice_tile_index(uint32_t slice_row, uint32_t slice_col, uint32_t slice_Wt) {
    return slice_row * slice_Wt + slice_col;
}

FORCE_INLINE uint32_t slice_coordinates_to_global_tile_index(
    uint32_t slice_row, uint32_t slice_col, uint32_t slice_idx, uint32_t slice_Wt, uint32_t global_Wt) {
    uint32_t global_col = slice_col + slice_idx * slice_Wt;
    return slice_row * global_Wt + global_col;
}

struct TileIndices {
    uint32_t slice;
    uint32_t global;
};

FORCE_INLINE TileIndices coordinates_to_tile_indices(
    uint32_t tile_row_in_mm_M_block,
    uint32_t chunk_col_in_tiles,
    uint32_t mm_core_idx,
    uint32_t N_block_idx,
    uint32_t M_block_idx,
    uint32_t chunk_idx,
    uint32_t N_block_wt,
    uint32_t tiles_ht_per_core,
    uint32_t mm_block_unit_ht,
    uint32_t chunk_width_in_tiles,
    uint32_t actual_slice_idx,
    uint32_t slice_Wt,
    uint32_t input_tensor_Wt) {
    auto [slice_row, slice_col] = coordinates_to_slice_coordinates(
        tile_row_in_mm_M_block,
        chunk_col_in_tiles,
        mm_core_idx,
        N_block_idx,
        M_block_idx,
        chunk_idx,
        N_block_wt,
        tiles_ht_per_core,
        mm_block_unit_ht,
        chunk_width_in_tiles);
    return {
        .slice = slice_coordinates_to_slice_tile_index(slice_row, slice_col, slice_Wt),
        .global =
            slice_coordinates_to_global_tile_index(slice_row, slice_col, actual_slice_idx, slice_Wt, input_tensor_Wt)};
}
