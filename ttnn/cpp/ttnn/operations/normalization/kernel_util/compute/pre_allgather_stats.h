// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file pre_allgather_stats.h
 * @brief Shared pieces of the layernorm/rmsnorm distributed pre-allgather compute kernels.
 */

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"

namespace norm::kernel_util::compute::pre_allgather {

/**
 * Square cw tiles from dfb_inp into dfb_x2. The input tiles stay in dfb_inp (cumulative wait,
 * indexed access); the caller pops them once it no longer needs them.
 *
 * When unpack_fp32_active, squares on the SFPU at full fp32; otherwise mul_tiles on the FPU.
 */
template <bool unpack_fp32_active>
ALWI void square_row(DataflowBuffer& dfb_inp, DataflowBuffer& dfb_x2, std::uint32_t cw, std::uint32_t blk) {
    const std::uint32_t inp_id = dfb_inp.get_id();
    const std::uint32_t x2_id = dfb_x2.get_id();
    reconfig_data_format(inp_id, inp_id);
    pack_reconfig_data_format(x2_id);
    if constexpr (unpack_fp32_active) {
        copy_init(inp_id);
        square_tile_init();
    } else {
        mul_init(inp_id, inp_id);
    }
    for (std::uint32_t wt = 0; wt < cw; wt += blk) {
        dfb_inp.wait_front(wt + blk);  // cumulative wait
        dfb_x2.reserve_back(blk);

        if constexpr (unpack_fp32_active) {
            for (std::uint32_t wtr = 0; wtr < blk; wtr++) {
                tile_regs_acquire();
                copy_tile(inp_id, wt + wtr, 0);
                square_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, x2_id, wt + wtr);
                tile_regs_release();
            }
        } else {
            tile_regs_acquire();
            for (std::uint32_t wtr = 0; wtr < blk; wtr++) {
                mul_tiles(inp_id, inp_id, wt + wtr, wt + wtr, wtr);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (std::uint32_t wtr = 0; wtr < blk; wtr++) {
                pack_tile(wtr, x2_id, wt + wtr);
            }
            tile_regs_release();
        }
        dfb_x2.push_back(blk);
    }
}

/**
 * Move the final running partial (one tile) from the accumulator to the stats output; the pack
 * converts to the output's data format, the single rounding the unchunked path also has.
 */
ALWI void partial_to_out(DataflowBuffer& dfb_acc, DataflowBuffer& dfb_out) {
    const std::uint32_t acc_id = dfb_acc.get_id();
    const std::uint32_t out_id = dfb_out.get_id();
    dfb_acc.wait_front(1);
    reconfig_data_format(acc_id, acc_id);
    pack_reconfig_data_format(out_id);
    copy_init(acc_id);
    dfb_out.reserve_back(1);
    tile_regs_acquire();
    copy_tile(acc_id, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_id);
    tile_regs_release();
    dfb_out.push_back(1);
    dfb_acc.pop_front(1);
}

/**
 * Sum num_chunks per-chunk partial tiles (each masked to column 0 by the reduce's pack mask) on
 * the SFPU at full fp32 and emit the total to the stats output.
 *
 * Used on the accurate-SFPU path instead of the reduce's Accumulate: the copy_tile reload there
 * measurably over-counts a SUM running partial (by ~1/32 of it per reload on Blackhole), while
 * per-call partials are exact.
 */
ALWI void fold_partials_to_out(DataflowBuffer& dfb_acc, DataflowBuffer& dfb_out, std::uint32_t num_chunks) {
#ifndef ARCH_QUASAR  // add_binary_tile is unavailable on Quasar; the SFPU path never runs there
    const std::uint32_t acc_id = dfb_acc.get_id();
    const std::uint32_t out_id = dfb_out.get_id();
    dfb_acc.wait_front(num_chunks);
    reconfig_data_format(acc_id, acc_id);
    pack_reconfig_data_format(out_id);
    copy_init(acc_id);
    add_binary_tile_init();
    dfb_out.reserve_back(1);
    tile_regs_acquire();
    copy_tile(acc_id, 0, 0);
    for (std::uint32_t i = 1; i < num_chunks; i++) {
        copy_tile(acc_id, i, 1);
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_id);
    tile_regs_release();
    dfb_out.push_back(1);
    dfb_acc.pop_front(num_chunks);
#endif
}

}  // namespace norm::kernel_util::compute::pre_allgather
