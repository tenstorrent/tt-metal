// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared between the indexer_score reader / compute / writer kernels:
// compile-time dims and knobs (args 0..7 are common to all three) and the
// walk over this core's flat span of causal-valid work units in row-major
// order (the host's flat deal, INDEXER_OP.md). One unit = QC q-tile-rows x
// up-to-KC k-tiles.

#pragma once

#include <cstdint>

constexpr uint32_t Hi = get_compile_time_arg_val(0);       // indexer heads
constexpr uint32_t Sqt = get_compile_time_arg_val(1);      // q chunk rows, in tiles
constexpr uint32_t Tt = get_compile_time_arg_val(2);       // total k positions, in tiles
constexpr uint32_t Dt = get_compile_time_arg_val(3);       // head dim, in tiles
constexpr uint32_t chunk_t = get_compile_time_arg_val(4);  // q chunk start offset, in tiles
constexpr uint32_t QC = get_compile_time_arg_val(5);       // q tiles per unit (q_chunk)
constexpr uint32_t KC = get_compile_time_arg_val(6);       // k tiles per unit (k_chunk)
constexpr uint32_t HB = get_compile_time_arg_val(7);       // heads per group (head_group)

constexpr uint32_t num_head_groups = Hi / HB;

/** Number of causal-valid output tiles in q-tile-row s. */
inline uint32_t valid(uint32_t s) {
    uint32_t v = chunk_t + s + 1;
    return v < Tt ? v : Tt;
}

/** Valid k-tiles of q-row-group g = valid() of its last row. */
inline uint32_t valid_max(uint32_t g) { return valid((g + 1) * QC - 1); }

/** Number of k-chunk units in q-row-group g. */
inline uint32_t units(uint32_t g) { return (valid_max(g) + KC - 1) / KC; }

/** (g, u) cursor over causal-valid work units, starting at a flat index. */
struct WorkUnitSpan {
    uint32_t g = 0;
    uint32_t u = 0;

    void start(uint32_t flat) {
        uint32_t rowsum = 0;
        while (flat >= rowsum + units(g)) {
            rowsum += units(g);
            ++g;
        }
        u = flat - rowsum;
    }

    /** Advance one unit; true when a new q-row-group begins. */
    bool advance() {
        if (++u == units(g)) {
            ++g;
            u = 0;
            return true;
        }
        return false;
    }

    uint32_t s0() const { return g * QC; }                  // first q-tile-row
    uint32_t c0() const { return u * KC; }                  // first k-tile
    uint32_t kw() const {                                   // k-tiles in this unit
        uint32_t left = valid_max(g) - c0();
        return left < KC ? left : KC;
    }
    bool last_in_group() const { return u == units(g) - 1; }
};
