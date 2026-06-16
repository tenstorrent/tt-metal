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
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

namespace norm::kernel_util::compute::pre_add {

/**
 * Perform fused pre-add for one H row: cb_inp = cb_in0 + cb_res for Wt tiles,
 * processed in blocks of blk tiles. Compile-time no-op when !fuse_pre_add.
 *
 * When float32_dtype is true, uses copy_tile + add_binary_tile (SFPU) to preserve
 * full FP32 precision. Otherwise uses add_tiles (FPU / TF32).
 */
template <bool fuse_pre_add, bool float32_dtype = false>
ALWI void one_row(uint32_t cb_in0, uint32_t cb_res, uint32_t cb_inp, uint32_t Wt, uint32_t blk) {
    if constexpr (!fuse_pre_add) {
        return;
    }
    reconfig_data_format(cb_in0, cb_res);
    pack_reconfig_data_format(cb_inp);
    if constexpr (float32_dtype) {
        copy_tile_to_dst_init_short(cb_in0);
    } else {
        add_tiles_init(cb_in0, cb_res);
    }
    for (uint32_t wt = 0; wt < Wt; wt += blk) {
        cb_wait_front(cb_in0, blk);
        cb_wait_front(cb_res, blk);
        cb_reserve_back(cb_inp, blk);
        if constexpr (float32_dtype) {
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                tile_regs_acquire();
                copy_tile(cb_in0, wtr, 0);
                copy_tile_to_dst_init_short_with_dt(cb_in0, cb_res);
                copy_tile(cb_res, wtr, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_inp);
                tile_regs_release();
                copy_tile_to_dst_init_short_with_dt(cb_res, cb_in0);
            }
        } else {
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                add_tiles(cb_in0, cb_res, wtr, wtr, wtr);
                pack_tile(wtr, cb_inp);
            }
            tile_regs_commit();
            tile_regs_release();
        }
        cb_push_back(cb_inp, blk);
        cb_pop_front(cb_in0, blk);
        cb_pop_front(cb_res, blk);
    }
}

}  // namespace norm::kernel_util::compute::pre_add
