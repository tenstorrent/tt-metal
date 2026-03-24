// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Welford HW-dimension reduction kernel (compute side).
// For each NC slice, H-reduces each of Wt columns using the Welford LLK,
// then finalizes to row format (welford_finalize_to_row) and packs
// the mean+var tile pair to cb_partial for the writer kernel to combine
// across W using the parallel Welford merge formula.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/welford.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    // Runtime arg: number of NC slices this core must process.
    uint32_t NC_per_core = get_arg_val<uint32_t>(0);

    // Compile-time args:
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr bool do_scale = get_compile_time_arg_val(4) != 0;

    constexpr uint32_t onetile = 1;

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    // Intermediate CB for mean+var tile pairs, consumed by writer kernel.
    constexpr auto cb_partial = tt::CBIndex::c_21;
    // 1-tile intermediate for scaled input (same as welford_reduce_h.cpp).
    constexpr auto cb_scaled = tt::CBIndex::c_20;

    experimental::CircularBuffer cb_in_obj(cb_in);
    experimental::CircularBuffer cb_scaler_obj(cb_scaler);
    experimental::CircularBuffer cb_partial_obj(cb_partial);
    experimental::CircularBuffer cb_scaled_obj(cb_scaled);

    constexpr uint32_t input_dst = 0;
    constexpr uint32_t mean_dst = 1;

    // Valid rows in the last H tile (for padding exclusion).
    constexpr uint32_t last_tile_rows = (H % tile_height) == 0 ? tile_height : H % tile_height;

    // Population variance: scale_idx = H-1 gives reciprocal 1/H.
    // Bessel's correction is applied later by the writer kernel.
    constexpr uint32_t scale_idx = H - 1;

    compute_kernel_hw_startup(cb_in, cb_partial);
    pack_reconfig_data_format(cb_partial);

    if constexpr (do_scale) {
        cb_scaler_obj.wait_front(onetile);
    }

    for (uint32_t nc = 0; nc < NC_per_core; ++nc) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // H-reduce one column of Ht tiles.
            uint32_t start_N = 0;
            welford_init();

            for (uint32_t ht = 0; ht < Ht; ++ht) {
                if constexpr (do_scale) {
                    // Scale step in its own DST cycle (same pattern as welford_reduce_h.cpp).
                    cb_in_obj.wait_front(onetile);
                    tile_regs_acquire();
                    mul_tiles_bcast_scalar_init_short(cb_in, cb_scaler);
                    mul_tiles_bcast_scalar(cb_in, cb_scaler, 0, 0, input_dst);
                    tile_regs_commit();
                    cb_in_obj.pop_front(1);
                    cb_scaled_obj.reserve_back(onetile);
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_scaled);
                    pack_tile(input_dst, cb_scaled);
                    tile_regs_release();
                    cb_scaled_obj.push_back(onetile);

                    // Read scaled tile back into DST for Welford.
                    cb_scaled_obj.wait_front(onetile);
                    tile_regs_acquire();
                    copy_tile_to_dst_init_short(cb_scaled);
                    copy_tile(cb_scaled, 0, input_dst);
                    cb_scaled_obj.pop_front(onetile);
                } else {
                    cb_in_obj.wait_front(onetile);
                    tile_regs_acquire();
                    copy_tile_to_dst_init_short(cb_in);
                    copy_tile(cb_in, 0, input_dst);
                    cb_in_obj.pop_front(1);
                }

                if (ht < (Ht - 1)) {
                    welford_update<0>(input_dst, start_N, {});
                    tile_regs_commit();
                    tile_regs_wait();
                    tile_regs_release();
                } else {
                    // Last tile: process only valid rows, then finalize.
                    welford_update_rows<0>(input_dst, start_N, 0, last_tile_rows, {});
                    // Finalize to row format: 32 per-column (mean, var) values
                    // stored in tile row 0 (across Face 0 and Face 1).
                    // welford_finalize_to_row applies SFPTRANSP to convert from
                    // SFPU lane order to tile column order; the "raw face" variant
                    // (welford_finalize_to_face) skips this and stores in lane
                    // order, which is NOT the same as tile column order.
                    // Population variance (scale_idx = H-1); Bessel's correction
                    // is applied by the writer kernel after W-combine.
                    welford_finalize_to_row<0>(mean_dst, scale_idx, {});
                    tile_regs_commit();
                }
                start_N += tile_height;
            }

            // Pack mean (DST[1]) and var (DST[2]) tiles to cb_partial.
            cb_partial_obj.reserve_back(2);
            tile_regs_wait();
            pack_reconfig_data_format(cb_partial);
            pack_tile_block(mean_dst, cb_partial, 2);
            tile_regs_release();
            cb_partial_obj.push_back(2);
        }
    }
}
