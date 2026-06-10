// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared between the indexer_score reader / compute / writer kernels:
// compile-time dims (args 0..4 are common to all three) and the walk over
// this core's flat span of causal-valid output tiles in row-major order
// (the host's flat deal, INDEXER_OP.md).

#pragma once

#include <cstdint>

constexpr uint32_t Hi = get_compile_time_arg_val(0);       // indexer heads
constexpr uint32_t Sqt = get_compile_time_arg_val(1);      // q chunk rows, in tiles
constexpr uint32_t Tt = get_compile_time_arg_val(2);       // total k positions, in tiles
constexpr uint32_t Dt = get_compile_time_arg_val(3);       // head dim, in tiles
constexpr uint32_t chunk_t = get_compile_time_arg_val(4);  // q chunk start offset, in tiles

/** Number of causal-valid output tiles in q-tile-row s. */
inline uint32_t valid(uint32_t s) {
    uint32_t v = chunk_t + s + 1;
    return v < Tt ? v : Tt;
}

/** (s, t) cursor over causal-valid output tiles, starting at a flat index. */
struct ValidTileSpan {
    uint32_t s = 0;
    uint32_t t = 0;

    void start(uint32_t flat) {
        uint32_t rowsum = 0;
        while (flat >= rowsum + valid(s)) {
            rowsum += valid(s);
            ++s;
        }
        t = flat - rowsum;
    }

    /** Advance one tile; true when a new q-tile-row begins. */
    bool advance() {
        if (++t == valid(s)) {
            ++s;
            t = 0;
            return true;
        }
        return false;
    }

    bool on_diagonal() const { return t == chunk_t + s; }
    bool last_in_row() const { return t == valid(s) - 1; }
};
