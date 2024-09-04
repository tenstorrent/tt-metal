// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/reduce.h"

namespace NAMESPACE {
void MAIN {

    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    reduce_init<true>(tt::CB::c_in0, tt::CB::c_in2);
    cb_wait_front(tt::CB::c_in2, 1); // scaler tile from the reader

    for (uint32_t nc = 0; nc < NC; nc++) {

        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for(uint32_t wt = 0; wt < Wt; ++wt) {
            // tiles are expected to be coming in in NCWH order (H-contiguous)
            // reducing in W means out[0][w] = sum(h=0..H-1, in[h][w])
            // in this case we just sequentially add to accumulator all the H-tiles in a column
            acquire_dst(tt::DstMode::Half);
            for(uint32_t ht = 0; ht < Ht; ++ht) {
                cb_wait_front(tt::CB::c_in0, onetile);
                // REDUCE_OP is expected to come from add_define
                reduce_tile(tt::CB::c_in0, tt::CB::c_in2, 0, 0, reduce_dst_idx);
                cb_pop_front(tt::CB::c_in0, onetile);
            }

            cb_reserve_back(tt::CB::c_out0, onetile);
            pack_tile(reduce_dst_idx, tt::CB::c_out0);
            cb_push_back(tt::CB::c_out0, onetile);
            release_dst(tt::DstMode::Half);
        }
    }
}
}
