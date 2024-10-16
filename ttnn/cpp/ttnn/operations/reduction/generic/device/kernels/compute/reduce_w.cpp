// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifndef REDUCE_ROW_SUM_VIA_MM
#include "compute_kernel_api/reduce.h"
#else
#include "compute_kernel_api/matmul.h"
#endif

namespace NAMESPACE {

void MAIN {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

#ifndef REDUCE_ROW_SUM_VIA_MM
    reduce_init<true>(tt::CB::c_in0, tt::CB::c_in2);
#else
    mm_init(tt::CB::c_in0, tt::CB::c_in2);
#endif

    cb_wait_front(tt::CB::c_in2, 1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            acquire_dst();
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(tt::CB::c_in0, onetile);
                // REDUCE_OP is expected to come from add_define
#ifndef REDUCE_ROW_SUM_VIA_MM
                reduce_tile(tt::CB::c_in0, tt::CB::c_in2, 0, 0, reduce_dst_idx);
#else
                matmul_tiles(tt::CB::c_in0, tt::CB::c_in2, 0, 0, 0, false);
#endif
                cb_pop_front(tt::CB::c_in0, onetile);
            }

            cb_reserve_back(tt::CB::c_out0, onetile);
            pack_tile(reduce_dst_idx, tt::CB::c_out0);
            cb_push_back(tt::CB::c_out0, onetile);
            release_dst();
        }
    }
}
}  // namespace NAMESPACE
