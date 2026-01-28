// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/reduce.h"

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "api/debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_acc = tt::CBIndex::c_4;
    constexpr uint32_t cb_ineg = tt::CBIndex::c_5;

    compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);
    reduce_init(tt::CBIndex::c_0, cb_scaler, cb_output);

    DPRINT << "Starting reduce_hw_neg kernel" << ENDL();

    cb_wait_front(cb_scaler, 1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        DPRINT << "LOOP1: nc=" << nc << ENDL();
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            DPRINT << "LOOP2: ht=" << ht << ENDL();
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                DPRINT << "LOOP3: wt=" << wt << ", ht=" << ht << ENDL();
                acquire_dst();
                DPRINT << "cb_wait_front(cb_input, onetile)" << ENDL();
                cb_wait_front(cb_input, onetile);
                copy_tile_init(cb_input);
                copy_tile(cb_input, 0, reduce_dst_idx);
                negative_tile_init();
                negative_tile(reduce_dst_idx);
                cb_pop_front(cb_input, onetile);
                DPRINT << "cb_reserve_back(cb_ineg, onetile)" << ENDL();
                cb_reserve_back(cb_ineg, onetile);
                pack_tile(reduce_dst_idx, cb_ineg);
                cb_push_back(cb_ineg, onetile);
                DPRINT << "release_dst(1)" << ENDL();
                release_dst();

                acquire_dst();
                if (wt > 0 || ht > 0) {
                    DPRINT << "cb_wait_front(cb_acc, onetile)" << ENDL();
                    cb_wait_front(cb_acc, onetile);
                    copy_tile_init(cb_acc);
                    copy_tile(cb_acc, 0, reduce_dst_idx);
                }

                DPRINT << "cb_wait_front(cb_input, onetile)" << ENDL();
                cb_wait_front(cb_ineg, onetile);
                reduce_init(cb_ineg, cb_scaler, cb_acc);
                reduce_tile(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
                reduce_uninit();
                cb_pop_front(cb_ineg, onetile);
                if (wt > 0 || ht > 0) {
                    cb_pop_front(cb_acc, onetile);
                }
                DPRINT << "cb_reserve_back(cb_acc, onetile)" << ENDL();
                cb_reserve_back(cb_acc, onetile);
                pack_tile(dst_idx, cb_acc);
                cb_push_back(cb_acc, onetile);
                DPRINT << "release_dst(2)" << ENDL();
                release_dst();
            }  // wt
        }  // ht

        acquire_dst();
        DPRINT << "cb_wait_front(cb_acc, onetile)" << ENDL();
        cb_reserve_back(cb_output, onetile);
        cb_wait_front(cb_acc, onetile);
        copy_tile_init(cb_acc);
        copy_tile(cb_acc, 0, dst_idx);
        negative_tile_init();
        negative_tile(dst_idx);
        cb_pop_front(cb_acc, onetile);
        DPRINT << "cb_reserve_back(cb_output, onetile)" << ENDL();
        cb_reserve_back(cb_output, onetile);
        pack_tile(reduce_dst_idx, cb_output);
        cb_push_back(cb_output, onetile);
        DPRINT << "release_dst(3)" << ENDL();
        release_dst();
    }  // nc
}
}  // namespace NAMESPACE
