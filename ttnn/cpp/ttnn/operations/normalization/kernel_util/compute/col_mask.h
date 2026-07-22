// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file col_mask.h
 * @brief In-place column masking of a width block for sharded normalization compute kernels.
 */

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"

namespace norm::kernel_util::compute {

// Zero the padding columns of a width block in place, so a following reduce (mean or mean-of-squares)
// excludes them. The block's num_tiles_per_block tiles (block_h rows of block_w tiles) are re-circulated
// through their own CB one tile at a time, each multiplied by the per-tile column mask; tile t uses mask
// tile (t % block_w). The mask CB is read by tile index and never popped. On return the masked tiles
// have been pushed back to cb; the caller waits on them before the reduce. The caller configures the
// SrcB (mask) data format before calling.
inline void mask_block_in_place(
    CircularBuffer& cb, uint32_t cb_col_mask_id, uint32_t num_tiles_per_block, uint32_t block_w) {
    const uint32_t cb_id = cb.get_cb_id();
    mul_init(cb_id, cb_col_mask_id);
    for (uint32_t t = 0; t < num_tiles_per_block; t++) {
        const uint32_t wt = t % block_w;
        cb.wait_front(1);
        tile_regs_acquire();
        mul_tiles(cb_id, cb_col_mask_id, 0, wt, 0);
        tile_regs_commit();
        cb.pop_front(1);
        cb.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_id);
        cb.push_back(1);
        tile_regs_release();
    }
}

}  // namespace norm::kernel_util::compute
