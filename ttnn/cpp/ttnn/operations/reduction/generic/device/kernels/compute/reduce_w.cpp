// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifndef REDUCE_ROW_SUM_VIA_MM
#include "api/compute/reduce.h"
#else
#include "api/compute/matmul_op.h"
#endif
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb2(tt::CBIndex::c_2);
    experimental::CircularBuffer cb3(tt::CBIndex::c_3);

#ifndef REDUCE_ROW_SUM_VIA_MM
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
    reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
#else
    ckernel::MatmulOpConfig mm_cfg{};
    mm_cfg.in0_cb_id = tt::CBIndex::c_0;
    mm_cfg.in1_cb_id = tt::CBIndex::c_2;
    mm_cfg.out_cb_id = tt::CBIndex::c_3;
    ckernel::TileMatmulOp mm(mm_cfg);
    mm.init();
#endif

    cb2.wait_front(1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            acquire_dst();
#ifndef REDUCE_ROW_SUM_VIA_MM
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb0.wait_front(onetile);
                reduce_tile(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, reduce_dst_idx);
                cb0.pop_front(onetile);
            }
#else
            mm.reduce_w_tiles(Wt, 0);
#endif

            cb3.reserve_back(onetile);
            pack_tile(reduce_dst_idx, tt::CBIndex::c_3);
            cb3.push_back(onetile);
            release_dst();
        }
    }
}
