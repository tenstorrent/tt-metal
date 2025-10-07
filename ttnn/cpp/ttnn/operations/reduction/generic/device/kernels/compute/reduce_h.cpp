// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/reduce.h"
#include "unified_common.h"
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
        const uint32_t tiles_in_chunk = std::min(wt + row_chunk, num_cols) - wt;
        const uint32_t row_wrap_increment = (Ht - 1) * Wt + 1;
        uint32_t w_row = w_chunk;
        uint32_t col_start_row = col_start_chunk;

        // reduction for one chunk
        // accumulation of Ht results in separate DST indexes
        acquire_dst();
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            w_row = w_chunk;
            col_start_row = col_start_chunk;
            uint32_t curr_id_row = col_start_row + ht * Wt;

            for (uint32_t k = 0; k < tiles_in_chunk; ++k) {
                auto src_tile = read_tile(src0, curr_id_row);
                reduce_tile(src_tile, scaler_tile, k);

                ++w_row;
                if (w_row == Wt) {
                    col_start_row += row_wrap_increment;
                    curr_id_row = col_start_row + ht * Wt;
                    w_row = 0;
                } else {
                    ++curr_id_row;
                    ++col_start_row;
                }
            }
        }
        col_start_chunk = col_start_row;
        w_chunk = w_row;

        for (uint32_t k = 0; k < tiles_in_chunk; ++k) {
            write_tile(k, out, write_page_id++);
        }
        release_dst();
    }
}
