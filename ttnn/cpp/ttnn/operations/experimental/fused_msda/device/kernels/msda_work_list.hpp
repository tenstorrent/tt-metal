// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The output tiles of one call and how they are cut across cores, as plain
// arithmetic over per-batch valid query counts. The program factory builds the
// host tile list from it; with device valid counts, the reader and the writer
// each rebuild their core's run from it on device. Sharing this one definition
// is what keeps the two lists identical.
//
// Tiles go batch-major, then query block, heads innermost, so most of a core's
// consecutive tiles share (b, q_start) and the reader stages a block's
// head-invariant reference points once. Each core takes a contiguous run; run
// lengths differ by at most one, the longer runs on the lower core indices.
//
// `valid(b)` returns batch b's valid query count, already clamped to Q.
// No device or host dependencies: included by the factory and by the kernels.

#pragma once

#include <cstdint>

namespace fused_msda_work {

constexpr uint32_t TILE_ROWS = 32;

struct Unit {
    uint32_t b;
    uint32_t head;
    uint32_t q_start;
    uint32_t v_rows;
};

struct Run {
    uint32_t first;  // the core's first tile, in the global order
    uint32_t count;  // tiles the core runs
};

inline uint32_t tiles_of(uint32_t valid_rows, uint32_t num_heads) {
    return (valid_rows + TILE_ROWS - 1) / TILE_ROWS * num_heads;
}

template <typename Valid>
uint32_t active_tiles(const Valid& valid, uint32_t batch, uint32_t num_heads) {
    uint32_t active = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        active += tiles_of(valid(b), num_heads);
    }
    return active;
}

inline Run core_run(uint32_t active, uint32_t core, uint32_t num_cores) {
    const uint32_t base = active / num_cores;
    const uint32_t extra = active % num_cores;
    return Run{core * base + (core < extra ? core : extra), base + (core < extra ? 1u : 0u)};
}

// Tile g of the global order; g must be below active_tiles().
template <typename Valid>
Unit unit(const Valid& valid, uint32_t batch, uint32_t num_heads, uint32_t g) {
    for (uint32_t b = 0; b < batch; ++b) {
        const uint32_t n = valid(b);
        const uint32_t tiles = tiles_of(n, num_heads);
        if (g < tiles) {
            const uint32_t block = g / num_heads;
            const uint32_t q_start = block * TILE_ROWS;
            const uint32_t rows = n - q_start;
            return Unit{b, g - block * num_heads, q_start, rows < TILE_ROWS ? rows : TILE_ROWS};
        }
        g -= tiles;
    }
    return Unit{0, 0, 0, 0};
}

}  // namespace fused_msda_work
