// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tilize compute kernel: tilize_block per tile-row chunk.
//
// When num_col_chunks == 1: processes chunk_Wt == Wt tiles per tile-row.
// When num_col_chunks >  1: processes chunk_Wt tiles per chunk, repeated
//   num_col_chunks times per tile-row. Total tiles = chunk_Wt * num_col_chunks.
#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t num_col_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t chunk_Wt = get_compile_time_arg_val(3);

    uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    CircularBuffer cb_in_exp(cb_in);
    CircularBuffer cb_out_exp(cb_out);

    unary_op_init_common(cb_in, cb_out);
    tilize_init(cb_in, chunk_Wt, cb_out);

    for (uint32_t b = 0; b < num_tile_rows; ++b) {
        for (uint32_t c = 0; c < num_col_chunks; ++c) {
            cb_in_exp.wait_front(chunk_Wt);
            cb_out_exp.reserve_back(chunk_Wt);

            tilize_block(cb_in, chunk_Wt, cb_out);

            cb_out_exp.push_back(chunk_Wt);
            cb_in_exp.pop_front(chunk_Wt);
        }
    }
}
