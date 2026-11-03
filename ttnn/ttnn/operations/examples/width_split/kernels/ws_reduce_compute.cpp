// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// width_split reduce-case compute (unpack/math/pack TRISCs): per-row centering,
//   y = x - mean_W(x)        (mean over the full width W of each row)
// i.e. one per-row reduction followed by one per-row apply back onto every tile of the row.
//
// Per tile-row:
//   1. partial = REDUCE_ROW-sum(slice) * (1/W)        (column 0 of the tile; the reduce packer
//                                                      mask zeroes every other datum)
//   2. k == 1: partial IS the mean -> cb_stat.
//      k  > 1: partial -> cb_send (the writer exchanges it with the k-1 other Tensix cores of
//              the group); wait for all k partials in cb_gather (slot q = member q's partial),
//              sum them in member order -> cb_stat. Every member sums the same k tiles in the
//              same order, so every slice of a row subtracts the bit-identical mean.
//   3. y = x - bcast_cols(cb_stat) for every tile of the slice -> cb_out.
// The slice stays resident in cb_in between steps 1 and 3 (no DRAM re-read).
//
// `k` is a runtime arg: row_split_reduce (k = 1) and width_split_reduce (k > 1) run the same
// binary; k == 1 simply skips step 2's exchange.
//
// WS_ROW_MAJOR (Layout::ROW_MAJOR input and output): step 0 tilizes the slice's 32 stick pieces
// (cb_rm_in, S tile-sized pages of row-major data) into cb_in, step 3 packs into cb_y instead of
// cb_out, and step 4 untilizes cb_y back to 32 stick pieces in cb_out for the writer. S = WS_RM_SLICE
// is a compile-time define (the tilize / untilize helpers take the block width as a template arg).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#ifdef WS_ROW_MAJOR
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#endif

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t cb_send = 2;
    constexpr uint32_t cb_gather = 3;
    constexpr uint32_t cb_stat = 4;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(0);
#ifdef WS_ROW_MAJOR
    constexpr uint32_t cb_rm_in = 5;
    constexpr uint32_t cb_y = 6;
    constexpr uint32_t rm_slice = WS_RM_SLICE;
    constexpr uint32_t cb_apply = cb_y;  // apply -> cb_y -> untilize -> cb_out
#else
    constexpr uint32_t cb_apply = cb_out;
#endif

    const uint32_t slice = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t k = get_arg_val<uint32_t>(2);
    const uint32_t blk = get_arg_val<uint32_t>(3);  // <= 4 tiles: one half-sync DST bank (fp32 DST)

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);
    cb_wait_front(cb_scaler, 1);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r = 0; r < num_rows; ++r) {
#ifdef WS_ROW_MAJOR
            // ---- 0. tilize the slice's 32 stick pieces ----
            compute_kernel_lib::tilize<rm_slice, cb_rm_in, cb_in>(1);
#endif
            // ---- 1. partial row-sum of this slice (scaled by 1/W) ----
            const uint32_t part_cb = (k > 1) ? cb_send : cb_stat;
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, part_cb);
            tile_regs_acquire();
            for (uint32_t i = 0; i < slice; i += blk) {
                cb_wait_front(cb_in, i + blk);
                for (uint32_t j = 0; j < blk; ++j) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, i + j, 0, 0);
                }
            }
            tile_regs_commit();
            cb_reserve_back(part_cb, 1);
            tile_regs_wait();
            pack_tile(0, part_cb);
            tile_regs_release();
            cb_push_back(part_cb, 1);
            reduce_uninit();

            // ---- 2. combine the k partials of the row (k > 1 only) ----
            if (k > 1) {
                cb_wait_front(cb_gather, k);
                cb_reserve_back(cb_stat, 1);
                // Sum in pairs straight from the CB with the FPU accumulating into the fp32 DST
                // (DST += a + b). A dest-reuse add would move the running sum DST -> SrcA every
                // step, truncating it to SrcA's 19-bit format: a bias of ~1 SrcA ulp per add that
                // grew to 0.07 at k = 32 partials.
                tile_regs_acquire();
                uint32_t q = 0;
                if (k & 1) {
                    copy_init(cb_gather);
                    copy_tile(cb_gather, 0, 0);
                    q = 1;
                } else {
                    add_init(cb_gather, cb_gather, false);
                    add_tiles(cb_gather, cb_gather, 0, 1, 0);
                    q = 2;
                }
                if (q < k) {
                    add_init(cb_gather, cb_gather, true /* acc_to_dest */);
                    for (; q < k; q += 2) {
                        add_tiles(cb_gather, cb_gather, q, q + 1, 0);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_stat);
                tile_regs_release();
                cb_push_back(cb_stat, 1);
                cb_pop_front(cb_gather, k);
            }

            // ---- 3. apply: y = x - mean, broadcast down the columns ----
            cb_wait_front(cb_stat, 1);
            sub_bcast_cols_init(cb_in, cb_stat);
            for (uint32_t i = 0; i < slice; i += blk) {
                cb_reserve_back(cb_apply, blk);
                tile_regs_acquire();
                for (uint32_t j = 0; j < blk; ++j) {
                    sub_tiles_bcast_cols(cb_in, cb_stat, i + j, 0, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < blk; ++j) {
                    pack_tile(j, cb_apply);
                }
                tile_regs_release();
                cb_push_back(cb_apply, blk);
            }
            cb_pop_front(cb_stat, 1);
            cb_pop_front(cb_in, slice);
#ifdef WS_ROW_MAJOR
            // ---- 4. untilize back to 32 stick pieces for the writer ----
            compute_kernel_lib::untilize<rm_slice, cb_y, cb_out>(1);
#endif
        }
    }
}
