// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack_untilize.h"

namespace compute_kernel_lib {

// Untilizes silu(x * scale + shift) from in_cb, block_size tile rows per block, even blocks into out_cb0 and odd
// blocks into out_cb1. scale_cb and shift_cb each hold one row of tiles_per_row [1, C] tiles (row 0 is broadcast).
// One tile at a time: the row-broadcast multiply (FPU) and the shift broadcast into dest need different unpack setups,
// so both inits run per tile.
template <
    uint32_t tiles_per_row,
    uint32_t block_size,
    uint32_t in_cb,
    uint32_t scale_cb,
    uint32_t shift_cb,
    uint32_t out_cb0,
    uint32_t out_cb1>
ALWI void untilize_affine_silu(uint32_t total_blocks) {
    cb_wait_front(scale_cb, tiles_per_row);
    cb_wait_front(shift_cb, tiles_per_row);
    pack_untilize_init<1, tiles_per_row>(in_cb, out_cb0);
    add_binary_tile_init();
    silu_tile_init();
    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        const uint32_t out_cb = (block_idx % 2 == 0) ? out_cb0 : out_cb1;
        for (uint32_t r = 0; r < block_size; r++) {
            cb_wait_front(in_cb, tiles_per_row);
            cb_reserve_back(out_cb, tiles_per_row);
            for (uint32_t c = 0; c < tiles_per_row; c++) {
                tile_regs_acquire();
                mul_bcast_rows_init(in_cb, scale_cb);
                mul_tiles_bcast<BroadcastType::ROW>(in_cb, scale_cb, c, c, 0);
                unary_bcast_init<BroadcastType::ROW>(shift_cb);
                unary_bcast<BroadcastType::ROW>(shift_cb, c, 1);
                add_binary_tile(0, 1, 0);
                silu_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<1, tiles_per_row>(out_cb, 1, c);
                tile_regs_release();
            }
            cb_pop_front(in_cb, tiles_per_row);
            cb_push_back(out_cb, tiles_per_row);
        }
    }
    pack_untilize_uninit(out_cb0);
}

}  // namespace compute_kernel_lib
