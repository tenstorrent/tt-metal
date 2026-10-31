// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A worker core's units, derived on the core from its own position.
//
// A unit is a run of `chunk_tiles` along one token row: chunk `c` of row `g` covers the
// row's tiles [c * chunk_tiles, (c + 1) * chunk_tiles). Both passes use the one split, so
// the live stream a core reduces in pass one is the stream it folds in pass two, and it
// stays in L1 between them rather than being read twice. Units are handed out in order,
// `units_per_core` to a core, so a row's chunks sit on consecutive cores and its chunk 0
// core is the row's leader, which sums the row's partial statistics.
//
// The split is a function of the shapes alone, so as runtime args it would be the
// same few words shipped to every core at every launch: on Galaxy, ~110 cores times
// three kernels times 32 chips, most of what dispatching a read cost the host. Cores
// are handed out row-wise from the origin, which makes a core's run a closed form in
// its linear index; the host checks the closed form against the cores it chose.
//
// Include after the kernel's API header, which supplies the logical coordinates.

#pragma once

#include <cstdint>

struct AttnResUnit {
    uint32_t row;    // token row-tile
    uint32_t chunk;  // which run of the row; chunk 0's core leads the row
};

struct AttnResCoreUnits {
    uint32_t first;
    uint32_t count;
};

// The split's compile-time args start at `Base`, in the order the program factory
// appends them: grid width, units per core, units, chunks per row, tiles per chunk.
template <uint32_t Base>
struct AttnResSplit {
    static constexpr uint32_t grid_x = get_compile_time_arg_val(Base);
    static constexpr uint32_t units_per_core = get_compile_time_arg_val(Base + 1);
    static constexpr uint32_t num_units = get_compile_time_arg_val(Base + 2);
    static constexpr uint32_t chunks_per_row = get_compile_time_arg_val(Base + 3);
    static constexpr uint32_t chunk_tiles = get_compile_time_arg_val(Base + 4);

    static AttnResCoreUnits core_units() {
        const uint32_t i = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
        const uint32_t first = i * units_per_core;
        const uint32_t end = first + units_per_core < num_units ? first + units_per_core : num_units;
        return {.first = first, .count = end - first};
    }

    static constexpr AttnResUnit unit(uint32_t u) { return {.row = u / chunks_per_row, .chunk = u % chunks_per_row}; }

    // A unit's first tile in a [.., N, d] stream's page order.
    template <uint32_t Wt>
    static constexpr uint32_t first_tile(uint32_t u) {
        return unit(u).row * Wt + unit(u).chunk * chunk_tiles;
    }
};
