// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs from interleaved dram.
 */

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "debug/assert.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);     // Source address in dram
    const uint32_t NCHt = get_arg_val<uint32_t>(1);         // Number of NCH tiles
    const uint32_t Wt = get_arg_val<uint32_t>(2);           // Width in tiles
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);  // Tile offset for this core

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);
    const DataFormat src0_data_format = get_dataformat(cb_inp);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t blk = get_compile_time_arg_val(1);

    const InterleavedAddrGenFast<src0_is_dram> src_a = {
        .bank_base_address = src_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    // Generate constant tiles for reduce scalar
    uint32_t scaler = get_arg_val<uint32_t>(4);
    generate_reduce_scaler(cb_reduce, scaler);

    uint32_t inp_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // read input tiles
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_reserve_back(cb_inp, blk);
            uint32_t inp_wr_ptr = get_write_ptr(cb_inp);

            for (uint32_t r = 0; r < blk; r++) {
                noc_async_read_tile(inp_tile_idx, src_a, inp_wr_ptr);
                inp_wr_ptr += src0_tile_bytes;
                inp_tile_idx++;
            }
            noc_async_read_barrier();
            cb_push_back(cb_inp, blk);

        }  // wt loop

    }  // ncht loop
}
