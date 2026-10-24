// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pack_untilize over a tile row, shared by the compute kernels that untilize.
#pragma once

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"

// Blocks per tile row of tile_cnt tiles: the widest block of at most max_bct tiles that divides the row.
constexpr uint32_t compute_num_blocks_per_column(uint32_t tile_cnt, uint32_t max_bct) {
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (tile_cnt % bct == 0) {
            return tile_cnt / bct;
        }
    }
    return 1;
}

// Untilizes one tile row of full_ct_dim tiles from src into full_ct_dim pages of dst, block_ct_dim tiles per pack.
// Expects pack_untilize_init<block_ct_dim, full_ct_dim>(src, dst).
template <uint32_t block_ct_dim, uint32_t full_ct_dim>
ALWI void pack_untilize_tile_row(CircularBuffer& src, CircularBuffer& dst) {
    constexpr uint32_t num_blocks_per_col = full_ct_dim / block_ct_dim;
    dst.reserve_back(full_ct_dim);
    for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
        src.wait_front(block_ct_dim);
        pack_untilize_block<block_ct_dim, full_ct_dim>(src.get_cb_id(), 1, dst.get_cb_id(), b);
        src.pop_front(block_ct_dim);
    }
    dst.push_back(full_ct_dim);
}
