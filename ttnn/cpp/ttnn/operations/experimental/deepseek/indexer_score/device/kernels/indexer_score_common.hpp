// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared between the indexer_score reader / compute / writer kernels:
// compile-time dims and knobs (args 0..7 are common to all three) and the
// walk over this core's flat span of causal-valid work units in row-major
// order (the host's flat deal, INDEXER_OP.md). One unit = q_tiles_per_unit
// q-tile-rows x up-to-k_tiles_per_unit k-tiles.

#pragma once

#include <cstdint>

constexpr uint32_t num_heads = get_compile_time_arg_val(0);          // indexer heads
constexpr uint32_t q_len_tiles = get_compile_time_arg_val(1);        // q chunk rows, in tiles
constexpr uint32_t k_len_tiles = get_compile_time_arg_val(2);        // total k positions, in tiles
constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(3);     // head dim, in tiles
constexpr uint32_t chunk_start_tiles = get_compile_time_arg_val(4);  // q chunk start offset, in tiles
constexpr uint32_t q_tiles_per_unit = get_compile_time_arg_val(5);   // q-tile-rows per work unit (q_chunk knob)
constexpr uint32_t k_tiles_per_unit = get_compile_time_arg_val(6);   // k tiles per work unit (k_chunk knob)
constexpr uint32_t heads_per_group = get_compile_time_arg_val(7);    // heads resident at once (head_group knob)

constexpr uint32_t num_head_groups = num_heads / heads_per_group;

/** Number of causal-valid output tiles in q-tile-row q_row. */
inline uint32_t num_valid_k_tiles(uint32_t q_row) {
    uint32_t v = chunk_start_tiles + q_row + 1;
    return v < k_len_tiles ? v : k_len_tiles;
}

/** Valid k-tiles of q-row-group `group` = num_valid_k_tiles() of its last row. */
inline uint32_t group_valid_k_tiles(uint32_t group) { return num_valid_k_tiles((group + 1) * q_tiles_per_unit - 1); }

/** Number of k-chunk units in q-row-group `group`. */
inline uint32_t units_in_group(uint32_t group) {
    return (group_valid_k_tiles(group) + k_tiles_per_unit - 1) / k_tiles_per_unit;
}

/** (group, unit) cursor over causal-valid work units, starting at a flat index. */
struct WorkUnitSpan {
    uint32_t group = 0;
    uint32_t unit = 0;

    void start(uint32_t flat) {
        uint32_t rowsum = 0;
        while (flat >= rowsum + units_in_group(group)) {
            rowsum += units_in_group(group);
            ++group;
        }
        unit = flat - rowsum;
    }

    /** Advance one unit; true when a new q-row-group begins. */
    bool advance() {
        if (++unit == units_in_group(group)) {
            ++group;
            unit = 0;
            return true;
        }
        return false;
    }

    uint32_t q_tile_start() const { return group * q_tiles_per_unit; }  // first q-tile-row of this unit
    uint32_t k_tile_start() const { return unit * k_tiles_per_unit; }   // first k-tile of this unit
    uint32_t k_tiles() const {                                          // valid k-tiles in this unit (< full on edge)
        uint32_t left = group_valid_k_tiles(group) - k_tile_start();
        return left < k_tiles_per_unit ? left : k_tiles_per_unit;
    }
    bool last_in_group() const { return unit == units_in_group(group) - 1; }
};
