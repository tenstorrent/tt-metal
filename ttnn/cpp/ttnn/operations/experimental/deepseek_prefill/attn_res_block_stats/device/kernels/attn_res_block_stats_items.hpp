// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The work split, derived on the core from its own position so that a launch ships no
// per-core runtime args.
//
// A unit is a run of one token row-tile, across every candidate: chunk `c` of a row covers
// its tiles [c * Wt / chunks, (c + 1) * Wt / chunks). The op is bound by how fast the grid
// pulls the sealed set in, so the split spreads it evenly — a row is cut into as many runs
// as the grid seats — rather than handing out whole (candidate, row) items, of which a
// grid seats a whole number. Units go out in order, `units_per_core` to a core, so a row's
// runs sit on consecutive cores; where a row is cut, its chunk 0 core leads it and sums
// the row's partial statistics.
//
// Include after the kernel's API header, which supplies the logical coordinates.

#pragma once

#include <cstdint>

struct AttnResBlockStatsUnit {
    uint32_t row;    // token row-tile
    uint32_t chunk;  // which run of the row; chunk 0's core leads the row
    uint32_t begin;  // first tile of the run along the row
    uint32_t end;    // one past its last

    uint32_t num_tiles() const { return end - begin; }
};

// The split's compile-time args start at `Base`: grid width, units per core, units, chunks
// per row, tiles per row.
template <uint32_t Base>
struct AttnResBlockStatsSplit {
    static constexpr uint32_t grid_x = get_compile_time_arg_val(Base);
    static constexpr uint32_t units_per_core = get_compile_time_arg_val(Base + 1);
    static constexpr uint32_t num_units = get_compile_time_arg_val(Base + 2);
    static constexpr uint32_t chunks_per_row = get_compile_time_arg_val(Base + 3);
    static constexpr uint32_t Wt = get_compile_time_arg_val(Base + 4);
    // The longest run, which is what a run's buffers are sized for.
    static constexpr uint32_t max_run = (Wt + chunks_per_row - 1) / chunks_per_row;

    static AttnResBlockStatsUnit unit(uint32_t u) {
        const uint32_t chunk = u % chunks_per_row;
        return {
            .row = u / chunks_per_row,
            .chunk = chunk,
            .begin = chunk * Wt / chunks_per_row,
            .end = (chunk + 1) * Wt / chunks_per_row};
    }

    // This core's units, [first, first + count).
    static void core_units(uint32_t& first, uint32_t& count) {
        const uint32_t i = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
        first = i * units_per_core;
        const uint32_t end = first + units_per_core < num_units ? first + units_per_core : num_units;
        count = end - first;
    }

    // Whether this core leads its row: every core where rows are whole.
    static bool leads(uint32_t first) { return chunks_per_row == 1 || unit(first).chunk == 0; }
};

// Progress counts packed into bit fields of a semaphore word, so one atomic increment
// advances one count and several counts share a semaphore: a field never carries into the
// next as long as its count stays below 2^width.
//
// A row leader counts, per other run of its row, the candidates that run has handed over;
// the gather cores count, per row, the candidates whose sum is parked.
struct AttnResProgressFields {
    uint32_t width;  // bits per field

    constexpr uint32_t per_word() const { return 32 / width; }
    constexpr uint32_t word(uint32_t field) const { return field / per_word(); }
    constexpr uint32_t increment(uint32_t field) const { return 1u << (width * (field % per_word())); }
    constexpr uint32_t count(uint32_t value, uint32_t field) const {
        return (value >> (width * (field % per_word()))) & ((1u << width) - 1);
    }
};
