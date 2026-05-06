// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 column-partitioned reader for the multi-core H reduction primitive
// (also used by welford H/HW once that factory is migrated).
//
// Migration notes:
//   - Compile-time arguments are bound by name (args::Ht, args::Wt, args::HtWt,
//     args::scaler_bits, args::use_welford, args::is_dram, args::aligned_page_size).
//   - Runtime arguments are bound by name (args::src_addr, args::col_start_tile_id,
//     args::curr_col_in_batch, args::num_cols).
//   - The input dataflow buffer is bound by name (dfb::input); placement is
//     derived from kernel bindings, not specified per-CB on the host.
//
// Arch coverage caveat: address generation goes through InterleavedAddrGenFast and
// noc_async_read_tile (Gen1-only). See the header of
// reader_unary_reduce_universal_start_id.cpp for the framework-level discussion.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Per-node runtime arguments.
    const uint32_t src_addr = get_arg(args::src_addr);
    uint32_t col_start_tile_id = get_arg(args::col_start_tile_id);  // start id in column-major order
    uint32_t curr_col_in_batch = get_arg(args::curr_col_in_batch);
    const uint32_t num_cols = get_arg(args::num_cols);

    // Compile-time arguments.
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t HtWt = get_arg(args::HtWt);
    constexpr uint32_t scaler_bits = get_arg(args::scaler_bits);
    constexpr bool use_welford = get_arg(args::use_welford) != 0;
    constexpr uint32_t aligned_page_size = get_arg(args::aligned_page_size);
    constexpr bool is_dram = get_arg(args::is_dram) != 0;

    // Welford must process one column at a time because the SFPU can only maintain
    // a single running mean/M2 state. DEST_AUTO_LIMIT interleaves multiple columns
    // per chunk, which would feed the Welford kernel tiles from the wrong columns.
    constexpr uint32_t row_chunk = use_welford ? 1 : compute_kernel_lib::DEST_AUTO_LIMIT;

    experimental::DataflowBuffer input_buf(dfb::input);

    // Fill the scaler tile via the kernel-side helper (templated on the scaler
    // buffer id; on Gen1 the id is the CB id, on Gen2 the DFB id — both compile-
    // time through dfb::scaler.id).
    constexpr uint32_t scaler_buf_id = dfb::scaler.id;
    const float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<scaler_buf_id, REDUCE_OP, REDUCE_DIM>(scaler_f);

    const InterleavedAddrGenFast<is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = aligned_page_size,
        .data_format = get_dataformat(input_buf.get_id()),
    };

    constexpr uint32_t onetile = 1;

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

        for (uint32_t j = 0; j < Ht; ++j) {
            w = reset_w;
            col_start_tile_id = reset_col_start;
            for (uint32_t k = i; k < chunk_end; ++k) {
                input_buf.reserve_back(onetile);
                noc_async_read_tile(curr_id, s, input_buf.get_write_ptr());
                noc_async_read_barrier();
                input_buf.push_back(onetile);

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
