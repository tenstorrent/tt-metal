// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file pre_add.h
 * @brief Helpers for fused pre-add (dfb_in0 + dfb_res -> dfb_inp) in layernorm/rmsnorm
 *        distributed pre-allgather compute kernels.
 */

#pragma once

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"

namespace norm::kernel_util::compute::pre_add {

/**
 * Perform fused pre-add for one H row: dfb_inp = dfb_in0 + dfb_res for Wt tiles,
 * processed in blocks of blk tiles. Compile-time no-op when !fuse_pre_add.
 *
 * When unpack_fp32_active, adds on the SFPU: copy_tile brings both operands into DEST at full
 * fp32, and add_binary_tile adds them there. FPU add_tiles reads its operands from SrcA/SrcB
 * instead, and the unpacker rounds fp32 down to tf32 when it loads those registers.
 */
template <bool fuse_pre_add, bool unpack_fp32_active = false>
ALWI void one_row(
    DataflowBuffer& dfb_in0, DataflowBuffer& dfb_res, DataflowBuffer& dfb_inp, uint32_t Wt, uint32_t blk) {
    if constexpr (!fuse_pre_add) {
        return;
    }
    reconfig_data_format(dfb_in0.get_id(), dfb_res.get_id());
    pack_reconfig_data_format(dfb_inp.get_id());
    if constexpr (unpack_fp32_active) {
        copy_tile_to_dst_init_short(dfb_in0.get_id());
        add_binary_tile_init();
    } else {
        add_init(dfb_in0.get_id(), dfb_res.get_id());
    }
    for (uint32_t wt = 0; wt < Wt; wt += blk) {
        dfb_in0.wait_front(blk);
        dfb_res.wait_front(blk);
        dfb_inp.reserve_back(blk);
        if constexpr (unpack_fp32_active) {
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                tile_regs_acquire();
                copy_tile(dfb_in0.get_id(), wtr, 0);
                copy_tile_to_dst_init_short_with_dt(dfb_in0.get_id(), dfb_res.get_id());
                copy_tile(dfb_res.get_id(), wtr, 1);
                add_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb_inp.get_id());
                tile_regs_release();
                // Restore SrcA to dfb_in0's format for the next tile's first copy_tile; the
                // residual CB may carry a different dtype than the input.
                copy_tile_to_dst_init_short_with_dt(dfb_res.get_id(), dfb_in0.get_id());
            }
        } else {
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32_t wtr = 0; wtr < blk; wtr++) {
                add_tiles(dfb_in0.get_id(), dfb_res.get_id(), wtr, wtr, wtr);
                pack_tile(wtr, dfb_inp.get_id());
            }
            tile_regs_commit();
            tile_regs_release();
        }
        dfb_inp.push_back(blk);
        dfb_in0.pop_front(blk);
        dfb_res.pop_front(blk);
    }
}

}  // namespace norm::kernel_util::compute::pre_add
