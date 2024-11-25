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

    constexpr int onetile = 1;
    for (uint32_t nc = 0; nc < NC; ++nc) {
        uint32_t row_chunk = 8;
        for(uint32_t wt = 0; wt < Wt; wt += row_chunk) {
            uint32_t chunk_end = std::min(wt + row_chunk, Wt);
            uint32_t tile_num = std::min(row_chunk, Wt - wt);
            int reduce_dst_idx = 0;

            //reduce a chunk of columns(max 8)
            acquire_dst();
            for(uint32_t ht = 0; ht < Ht; ++ht) {
                reduce_dst_idx = 0;
                for(uint32_t i = wt; i < chunk_end; ++i) {
                    cb_wait_front(tt::CB::c_in0, onetile);
                    reduce_tile(tt::CB::c_in0, tt::CB::c_in2, 0, 0, reduce_dst_idx);
                    cb_pop_front(tt::CB::c_in0, onetile);
                    ++reduce_dst_idx;
                }
            }
            for(uint32_t i = 0; i < tile_num; i++) {
                cb_reserve_back(tt::CB::c_out0, onetile);
                pack_tile(i, tt::CB::c_out0);
                cb_push_back(tt::CB::c_out0, onetile);
            }
            release_dst();
        }
    }
}
}
