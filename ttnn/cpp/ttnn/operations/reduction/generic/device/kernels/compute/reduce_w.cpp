// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifndef REDUCE_ROW_SUM_VIA_MM
#include "api/compute/reduce.h"
#else
#include "api/compute/matmul.h"
#endif
#include "experimental/circular_buffer.h"

#ifdef REDUCE_MINMAX_TWO_TILE_SCALER
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#endif

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
    mm_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
#endif

#ifdef REDUCE_MINMAX_TWO_TILE_SCALER
    // Reader: two pages on c_2 - tile0 = 1.0 for reduce_tile, tile1 = user scale for post-mul
    cb2.wait_front(2);
#else
    cb2.wait_front(1);  // scaler tile from the reader
#endif

    for (uint32_t nc = 0; nc < NC; nc++) {
        constexpr int onetile = 1;
        int reduce_dst_idx = 0;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            acquire_dst();
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb0.wait_front(onetile);
                // REDUCE_OP is expected to come from add_define
#ifndef REDUCE_ROW_SUM_VIA_MM
                reduce_tile(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, reduce_dst_idx);
#else
                matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, 0);
#endif
                cb0.pop_front(onetile);
            }

#ifdef REDUCE_MINMAX_TWO_TILE_SCALER
            /* Apply user-provided scaling factor to the reduced output.
             * In the two-tile scaler configuration, reduction uses unity scaling,
             * then the final reduced result is divided by (1/scalar) == multiplied by scalar.
             */
            reduce_uninit();
            copy_tile_init(tt::CBIndex::c_2);
            copy_tile(tt::CBIndex::c_2, 1, 1);
            div_binary_tile_init();
            div_binary_tile(reduce_dst_idx, 1, reduce_dst_idx);
            // Prepare for the next row's reduce_tile calls.
            reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
#endif
            cb3.reserve_back(onetile);
            pack_tile(reduce_dst_idx, tt::CBIndex::c_3);
            cb3.push_back(onetile);
            release_dst();
        }
    }
}
