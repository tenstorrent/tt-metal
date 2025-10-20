// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifndef REDUCE_ROW_SUM_VIA_MM
#include "compute_kernel_api/reduce.h"
#else
#include "compute_kernel_api/matmul.h"
#endif

#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

namespace NAMESPACE {

void MAIN {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    // Circular buffers:
    // 0: input
    // 2: scaler
    // 3: output
    // 4: temporary for output accumulation and negation

#ifndef REDUCE_ROW_SUM_VIA_MM
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
    // reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_4);
#else
    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_4);
#endif

    // negative_tile_init();

    cb_wait_front(tt::CBIndex::c_2, 1);  // scaler tile from the reader
    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                tile_regs_acquire();
                DPRINT << "inside wt loop: wt=" << wt << " ht=" << ht << ENDL();
                cb_wait_front(tt::CBIndex::c_0, onetile);
                if (wt > 0) {
                    cb_wait_front(tt::CBIndex::c_4, onetile);
                    DPRINT << "cb_tile_init wt=" << wt << " ht=" << ht << ENDL();
                    copy_tile_init(tt::CBIndex::c_4);
                    copy_tile(tt::CBIndex::c_4, 0, 0);
                }
                // REDUCE_OP is expected to come from add_define

#ifndef REDUCE_ROW_SUM_VIA_MM
                DPRINT << "reduce_tile wt=" << wt << " ht=" << ht << ENDL();
                reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_4);
                reduce_tile(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, reduce_dst_idx);
#else
                matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, 0, false);
#endif
                DPRINT << "cb_pop_front wt=" << wt << " ht=" << ht << ENDL();
                cb_pop_front(tt::CBIndex::c_0, onetile);
                cb_pop_front(tt::CBIndex::c_4, onetile);
                tile_regs_commit();
                DPRINT << "cb_reserve_back wt=" << wt << " ht=" << ht << ENDL();
                cb_reserve_back(tt::CBIndex::c_4, onetile);
                tile_regs_wait();
                DPRINT << "pack_tile wt=" << wt << " ht=" << ht << ENDL();
                pack_tile(reduce_dst_idx, tt::CBIndex::c_4);
                cb_push_back(tt::CBIndex::c_4, onetile);
                tile_regs_release();
            }  // wt

            // DPRINT << "back in ht loop " << ht << ENDL(); // uncommenting this DPRINT will cause another deadlock

            tile_regs_acquire();
            cb_wait_front(tt::CBIndex::c_4, onetile);
            copy_tile_init(tt::CBIndex::c_4);
            copy_tile(tt::CBIndex::c_4, 0, 0);
            // negative_tile_init();
            // negative_tile(reduce_dst_idx);
            cb_pop_front(tt::CBIndex::c_4, onetile);
            tile_regs_commit();
            cb_reserve_back(tt::CBIndex::c_3, onetile);
            tile_regs_wait();
            pack_tile(reduce_dst_idx, tt::CBIndex::c_3);
            cb_push_back(tt::CBIndex::c_3, onetile);
            tile_regs_release();
        }  // ht
    }
}
}  // namespace NAMESPACE
