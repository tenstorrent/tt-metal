// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Reads LayerNorm inputs from interleaved DRAM for Welford pre-allgather.
 * LayerNorm only; non-sharded; no 2D variants.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);     // Source address in dram
    const uint32_t NCHt = get_arg_val<uint32_t>(1);         // Number of NCH tiles
    const uint32_t Wt = get_arg_val<uint32_t>(2);           // Width in tiles
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);  // Tile offset for this core

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;

    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);

    constexpr uint32_t input_block_size = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    // c_1 belongs to compute scratch; the Welford path needs no scaler.
    const auto src_a = TensorAccessor(src_args, src_addr);

    uint32_t inp_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (uint32_t wt = 0; wt < Wt; wt += input_block_size) {
            cb_reserve_back(cb_inp, input_block_size);
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);
            for (uint32_t r = 0; r < input_block_size && wt + r < Wt; r++) {
                noc_async_read_page(inp_tile_idx, src_a, inp_wr_ptr);
                inp_wr_ptr += src0_tile_bytes;
                inp_tile_idx++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_inp, input_block_size);
        }
    }
}
