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

    compute_kernel_hw_startup(src0_cb, scaler_tile_cb, out_cb);
    reduce_init(src0_cb, scaler_tile_cb, out_cb);

    uint32_t write_page_id = start_write_page_id;
    uint32_t col_start_chunk = col_start_tile_id;
    uint32_t w_chunk = curr_col_in_batch;
    for (uint32_t wt = 0; wt < num_cols; wt += row_chunk) {
        uint32_t chunk_end = std::min(wt + row_chunk, num_cols);

        // reduction for one chunk
        // accumulation of Ht results in separate DST indexes
        acquire_dst();
        uint32_t final_col_start = col_start_chunk;
        uint32_t final_w = w_chunk;
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            uint32_t w_row = w_chunk;
            uint32_t col_start_row = col_start_chunk;
            uint32_t curr_id_row = col_start_chunk + ht * Wt;

            for (uint32_t i = wt; i < chunk_end; ++i) {
                auto src_tile = read_tile(src0, curr_id_row);
                reduce_tile(src_tile, scaler_tile, i - wt);

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
                final_col_start = col_start_row;
                final_w = w_row;
            }
        }
        col_start_chunk = final_col_start;
        w_chunk = final_w;

        for (uint32_t i = wt; i < chunk_end; ++i) {
            write_tile((i - wt), out, write_page_id++);
        }
        release_dst();
    }
}
