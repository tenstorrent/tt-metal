// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"

#include "llk_math_eltwise_binary.h"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    // Circular buffers:
    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_acc = tt::CBIndex::c_4;
    constexpr uint32_t cb_ineg = tt::CBIndex::c_5;

    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);

    cb_wait_front(cb_scaler, 1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(cb_input, onetile);
                tile_regs_acquire();
                copy_tile_init(cb_input);
                copy_tile(cb_input, 0, dst_idx);
                negative_tile_init();
                negative_tile(dst_idx);
                tile_regs_wait();
                cb_pop_front(cb_input, onetile);
                cb_reserve_back(cb_ineg, onetile);
                tile_regs_commit();
                pack_tile(dst_idx, cb_ineg);
                tile_regs_release();
                cb_push_back(cb_ineg, onetile);

                tile_regs_acquire();
                if (wt > 0) {
                    cb_wait_front(cb_acc, onetile);
                    copy_tile_init(cb_acc);
                    copy_tile(cb_acc, 0, dst_idx);
                }

                cb_wait_front(cb_ineg, onetile);
                reduce_init(cb_ineg, cb_scaler, cb_acc);
                reduce_tile(cb_ineg, cb_scaler, 0, 0, dst_idx);
                reduce_uninit();
                tile_regs_wait();
                cb_pop_front(cb_ineg, onetile);
                if (wt > 0) {
                    cb_pop_front(cb_acc, onetile);
                }
                cb_reserve_back(cb_acc, onetile);
                tile_regs_commit();
                pack_tile(dst_idx, cb_acc);
                tile_regs_release();
                cb_push_back(cb_acc, onetile);
            }  // wt

            cb_wait_front(cb_acc, onetile);
            tile_regs_acquire();
            copy_tile_init(cb_acc);
            copy_tile(cb_acc, 0, dst_idx);
            negative_tile_init();
            negative_tile(dst_idx);
            tile_regs_wait();
            cb_pop_front(cb_acc, onetile);
            cb_reserve_back(cb_output, onetile);
            tile_regs_commit();
            pack_tile(dst_idx, cb_output);
            tile_regs_release();
            cb_push_back(cb_output, onetile);
        }  // ht
    }  // nc
}
