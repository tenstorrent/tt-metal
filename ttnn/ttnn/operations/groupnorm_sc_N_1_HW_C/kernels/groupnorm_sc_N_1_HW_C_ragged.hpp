// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — ragged block accounting shared by the reader, compute and writer kernels.
//
// op_design.md -> Work Distribution: a core owns `Ht_core x Ct_core` tiles cut into `chunk_rows x cols` blocks;
// "ragged last group/chunk keeps nominal push/pop counts and narrows only the work". Every CB quantum stays
// the nominal block (`chunk`, `cols`, `cols*Kg`, ...) so the rings never straddle their wrap point; the valid
// extents below decide how many tiles are actually read / reduced / applied / stored inside each quantum.
// Valid tiles of a block are laid out DENSELY (row-major `valid_rows x valid_cols`) from the block's first page.

#pragma once

#include <stdint.h>

namespace groupnorm_ragged {

struct Axis {
    uint32_t count;  // blocks along the axis (ceil(extent / block))
    uint32_t last;   // valid units in the last block (1..block)

    // Valid units of block `i` along this axis.
    FORCE_INLINE uint32_t valid(uint32_t i, uint32_t block) const { return (i + 1 == count) ? last : block; }
};

FORCE_INLINE Axis split(uint32_t extent, uint32_t block) {
    const uint32_t count = (extent + block - 1) / block;
    return Axis{count, extent - (count - 1) * block};
}

// Pad the CB accounting by `n` pages without writing data (ring pointers only). Split into pieces of at most
// `piece` pages so a pad larger than the ring's free space cannot deadlock against a consumer that drains
// `piece` pages at a time.
FORCE_INLINE void pad_push(uint32_t cb, uint32_t n, uint32_t piece) {
    while (n > 0) {
        const uint32_t k = (n < piece) ? n : piece;
        cb_reserve_back(cb, k);
        cb_push_back(cb, k);
        n -= k;
    }
}

FORCE_INLINE void pad_pop(uint32_t cb, uint32_t n, uint32_t piece) {
    while (n > 0) {
        const uint32_t k = (n < piece) ? n : piece;
        cb_wait_front(cb, k);
        cb_pop_front(cb, k);
        n -= k;
    }
}

}  // namespace groupnorm_ragged
