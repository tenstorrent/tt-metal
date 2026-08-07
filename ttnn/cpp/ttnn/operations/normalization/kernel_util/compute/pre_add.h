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
#include "api/dataflow/dataflow_buffer.h"

namespace norm::kernel_util::compute::pre_add {

/**
 * Perform fused pre-add for one H row: dfb_inp = dfb_in0 + dfb_res for Wt tiles,
 * processed in blocks of blk tiles. Compile-time no-op when !fuse_pre_add.
 */
template <bool fuse_pre_add>
ALWI void one_row(
    DataflowBuffer& dfb_in0, DataflowBuffer& dfb_res, DataflowBuffer& dfb_inp, uint32_t Wt, uint32_t blk) {
    if constexpr (!fuse_pre_add) {
        return;
    }
    reconfig_data_format(dfb_in0.get_id(), dfb_res.get_id());
    pack_reconfig_data_format(dfb_inp.get_id());
    add_tiles_init(dfb_in0.get_id(), dfb_res.get_id());
    for (uint32_t wt = 0; wt < Wt; wt += blk) {
        dfb_in0.wait_front(blk);
        dfb_res.wait_front(blk);
        dfb_inp.reserve_back(blk);
        tile_regs_acquire();
        tile_regs_wait();
        for (uint32_t wtr = 0; wtr < blk; wtr++) {
            add_tiles(dfb_in0.get_id(), dfb_res.get_id(), wtr, wtr, wtr);
            pack_tile(wtr, dfb_inp.get_id());
        }
        tile_regs_commit();
        tile_regs_release();
        dfb_inp.push_back(blk);
        dfb_in0.pop_front(blk);
        dfb_res.pop_front(blk);
    }
}

}  // namespace norm::kernel_util::compute::pre_add
