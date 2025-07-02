// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t col_start_tile_id =
        get_arg_val<uint32_t>(1);  // Start id in column major order. This should be the start of a column
    uint32_t curr_col_in_batch = get_arg_val<uint32_t>(2);
    uint32_t num_cols = get_arg_val<uint32_t>(3);  // number of cols to read

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t HtWt = get_compile_time_arg_val(2);
    constexpr uint32_t row_chunk = get_compile_time_arg_val(3);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);

    constexpr uint32_t cb_id_in2 = tt::CBIndex::c_2;
    constexpr uint32_t scalar = get_compile_time_arg_val(4);
    generate_reduce_scaler(cb_id_in2, scalar);

    constexpr auto tensor_args = make_tensor_accessor_args<5>();
    auto tensor_accessor = make_tensor_accessor_from_args(tensor_args, src_addr, tile_bytes);

    uint32_t w = curr_col_in_batch;

    // tiles are read in the N W_skip H W_chunk order
    // W_skip(chunk size) represents the number of tile columns whose reading will be intertwined
    // H W_chunk represent tiles of the chunk read in row major order
    // exmpl. Ht = 3; Wt = 4; row_chunk = 2;
    //        read order (H, W):
    //        1. chunk:  1:(0, 0)  2:(0, 1)  3:(1, 0)   4:(1, 1)   5:(2, 0)   6:(2, 1)
    //        2. chunk:  7:(0, 2)  8:(0, 3)  9:(1, 2)  10:(1, 3)  11:(2, 2)  12:(2, 3)

    // for [N, C, W, H] tensor shape, where N != 1 or C != 1
    // chunk can contain elements with different N or C values
    // in each row we possibly need to move the col_start_tile_id to the first column of the next batch
    // reset variables are used to correctly return to the start column + repeat the process for each row
    // reset_col_start - resets col_start_tile_id to the starting column
    // reset_w - resets w to the column number in the batch of the starting column
    // reset_curr_id - resets curr_id to the next tile in the starting column
    for (uint32_t i = 0; i < num_cols; i += row_chunk) {
        uint32_t chunk_end = std::min(i + row_chunk, num_cols);
        uint32_t curr_id = col_start_tile_id;
        uint32_t reset_curr_id = curr_id;
        uint32_t reset_w = w;
        uint32_t reset_col_start = col_start_tile_id;

        // row wise read for one chunk
        for (uint32_t j = 0; j < Ht; ++j) {
            w = reset_w;
            col_start_tile_id = reset_col_start;
            for (uint32_t k = i; k < chunk_end; ++k) {
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read_tile(curr_id, tensor_accessor, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);

                ++w;

                if (w == Wt) {
                    col_start_tile_id = curr_id + (Ht - j - 1) * Wt + 1;
                    curr_id = col_start_tile_id + j * Wt;
                    w = 0;
                } else {
                    ++curr_id;
                    ++col_start_tile_id;
                }
            }
            curr_id = reset_curr_id + (j + 1) * Wt;  // stride in H
        }
    }
}
