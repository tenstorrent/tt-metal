// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file pre_add.h
 * @brief Helpers for fused pre-add (cb_in0 + cb_res -> cb_inp) in layernorm/rmsnorm
 *        distributed pre-allgather compute kernels.
 */

#pragma once

#include "api/compute/eltwise_binary.h"

namespace norm::kernel_util::compute::pre_add {

/**
 * Perform fused pre-add for one H row: cb_inp = cb_in0 + cb_res for Wt tiles,
 * processed in blocks of blk tiles. Compile-time no-op when !fuse_pre_add.
 */
template <bool fuse_pre_add>
ALWI void one_row(uint32_t cb_in0, uint32_t cb_res, uint32_t cb_inp, uint32_t Wt, uint32_t blk) {
    if constexpr (!fuse_pre_add) {
        return;
    }
    reconfig_data_format(cb_in0, cb_res);
    pack_reconfig_data_format(cb_inp);
    add_tiles_init(cb_in0, cb_res);
    for (uint32_t wt = 0; wt < Wt; wt += blk) {
        cb_wait_front(cb_in0, blk);
        cb_wait_front(cb_res, blk);
        cb_reserve_back(cb_inp, blk);
        tile_regs_acquire();
        tile_regs_wait();
        for (uint32_t wtr = 0; wtr < blk; wtr++) {
            add_tiles(cb_in0, cb_res, wtr, wtr, wtr);
            pack_tile(wtr, cb_inp);
        }
        tile_regs_commit();
        tile_regs_release();
        cb_push_back(cb_inp, blk);
        cb_pop_front(cb_in0, blk);
        cb_pop_front(cb_res, blk);
    }
}

}  // namespace norm::kernel_util::compute::pre_add
