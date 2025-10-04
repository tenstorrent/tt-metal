// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "universal_common.h"

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/compute_kernel_hw_startup.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

KERNEL_MAIN {
    INIT_ARGUMENTS

    compute_kernel_hw_startup(src0_cb, tt::CBIndex::c_4, out_cb);
    reduce_init(src0_cb, tt::CBIndex::c_4, out_cb);

#ifdef COMPILE_FOR_BRISC
    generate_reduce_scaler(tt::CBIndex::c_4, packed_scaler_value);
#endif

#ifdef COMPILE_FOR_TRISC
    cb_wait_front(tt::CBIndex::c_4, 1);  // scaler tile from the reader
#endif

    // tiles are expected to come in the N C W_skip H W_chunk order
    // W_skip(chunk size) represents the number of tile columns whose reduction will be intertwined
    // H W_chunk represent tiles of the chunk in row major order
    // each column in the chunk will have its intermediate result in a separate tile of DST
    // chunk size is calculated based on the number of available tiles in DST
    // exmpl. Ht = 3; Wt = 4; row_chunk = 2;
    //        tile order (H, W):
    //        1. chunk: (0, 0); (0, 1); (1, 0); (1, 1); (2, 0); (2, 1);
    //        2. chunk: (0, 2); (0, 3); (1, 2); (1, 3); (2, 2); (2, 3);

    uint32_t write_page_id = start_write_page_id;
    uint32_t col_start_chunk = col_start_tile_id;
    uint32_t w_chunk = curr_col_in_batch;
    for (uint32_t wt = 0; wt < num_cols; wt += row_chunk) {
        uint32_t chunk_end = std::min(wt + row_chunk, num_cols);
        int reduce_dst_idx = 0;

        // reduction for one chunk
        // accumulation of Ht results in separate DST indexes
        acquire_dst();
        uint32_t col_start_after_chunk = col_start_chunk;
        uint32_t w_after_chunk = w_chunk;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            uint32_t w_row = w_chunk;
            uint32_t col_start_row = col_start_chunk;
            uint32_t curr_id_row = col_start_chunk + ht * Wt;

            reduce_dst_idx = 0;
            for (uint32_t i = wt; i < chunk_end; ++i) {
                auto src_tile = read_tile(src0, curr_id_row);
                reduce_tile(src0_cb, tt::CBIndex::c_4, 0, 0, reduce_dst_idx);
                ++reduce_dst_idx;

                ++w_row;
                if (w_row == Wt) {
                    col_start_row = curr_id_row + (Ht - ht - 1) * Wt + 1;
                    curr_id_row = col_start_row + ht * Wt;
                    w_row = 0;
                } else {
                    ++curr_id_row;
                    ++col_start_row;
                }
            }
            if (ht == (Ht - 1)) {
                col_start_after_chunk = col_start_row;
                w_after_chunk = w_row;
            }
        }
        col_start_chunk = col_start_after_chunk;
        w_chunk = w_after_chunk;

        for (uint32_t i = wt; i < chunk_end; ++i) {
            write_tile((i - wt), out, write_page_id++);
        }
        release_dst();
    }
}
