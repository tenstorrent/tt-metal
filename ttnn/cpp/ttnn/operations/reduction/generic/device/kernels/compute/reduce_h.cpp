// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"
#include "experimental/circular_buffer.h"

#ifdef REDUCE_MINMAX_TWO_TILE_SCALER
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#endif

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    uint32_t row_chunk = get_compile_time_arg_val(3);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb2(tt::CBIndex::c_2);
    experimental::CircularBuffer cb3(tt::CBIndex::c_3);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
    reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

#ifdef REDUCE_MINMAX_TWO_TILE_SCALER
    // Reader: two pages on c_2 - tile0 = 1.0 for reduce_tile, tile1 = user scale for post-mul
    cb2.wait_front(2);
#else
    cb2.wait_front(1);  // scaler tile from the reader
#endif

    constexpr int onetile = 1;

    // tiles are expected to come in the N C W_skip H W_chunk order
    // W_skip(chunk size) represents the number of tile columns whose reduction will be intertwined
    // H W_chunk represent tiles of the chunk in row major order
    // each column in the chunk will have its intermediate result in a separate tile of DST
    // chunk size is calculated based on the number of available tiles in DST
    // exmpl. Ht = 3; Wt = 4; row_chunk = 2;
    //        tile order (H, W):
    //        1. chunk: (0, 0); (0, 1); (1, 0); (1, 1); (2, 0); (2, 1);
    //        2. chunk: (0, 2); (0, 3); (1, 2); (1, 3); (2, 2); (2, 3);
    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t wt = 0; wt < Wt; wt += row_chunk) {
            uint32_t chunk_end = std::min(wt + row_chunk, Wt);
            int reduce_dst_idx = 0;

            // reduction for one chunk
            // accumulation of Ht results in separate DST indexes
            acquire_dst();
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                reduce_dst_idx = 0;
                for (uint32_t i = wt; i < chunk_end; ++i) {
                    cb0.wait_front(onetile);
                    reduce_tile(tt::CB::c_in0, tt::CB::c_in2, 0, 0, reduce_dst_idx);
                    cb0.pop_front(onetile);
                    ++reduce_dst_idx;
                }
            }

#ifdef REDUCE_MINMAX_TWO_TILE_SCALER
            /* Apply user-provided scaling factor to the reduced output.
             * In the two-tile scaler configuration, reduction uses unity scaling,
             * then the final reduced result is divided by (1/scalar) == multiplied by scalar.
             */
            uint32_t ntiles = chunk_end - wt;
            // Switch from reduce op to SFPU binary op.
            reduce_uninit();
            copy_tile_init(tt::CBIndex::c_2);
            copy_tile(tt::CBIndex::c_2, 1, ntiles);
            div_binary_tile_init();
            for (uint32_t i = 0; i < ntiles; ++i) {
                div_binary_tile(i, ntiles, i);
            }
            // Prepare for the next chunk's reduce_tile calls.
            reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
#endif
            for (uint32_t i = wt; i < chunk_end; ++i) {
                cb3.reserve_back(onetile);
                pack_tile((i - wt), tt::CBIndex::c_3);
                cb3.push_back(onetile);
            }
            release_dst();
        }
    }
}
