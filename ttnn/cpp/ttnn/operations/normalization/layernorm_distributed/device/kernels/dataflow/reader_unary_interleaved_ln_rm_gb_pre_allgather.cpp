// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs from interleaved dram.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/debug/assert.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);     // Source address in dram
    const uint32_t NCHt = get_arg_val<uint32_t>(1);         // Number of NCH tiles
    const uint32_t Wt = get_arg_val<uint32_t>(2);           // Width in tiles
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);  // Tile offset for this core

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);

    constexpr uint32_t blk = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();
    uint32_t scaler = get_arg_val<uint32_t>(4);
    generate_reduce_scaler(cb_reduce, scaler);

    const auto src_a = TensorAccessor(src_args, src_addr, src0_tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_inp_buf(cb_inp);

    uint32_t inp_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            for (uint32_t r = 0; r < blk; r++) {
                cb_inp_buf.reserve_back(1);
                noc.async_read(src_a, cb_inp_buf, src0_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
                inp_tile_idx++;
                noc.async_read_barrier();
                cb_inp_buf.push_back(1);
            }
        }  // wt loop
    }  // ncht loop
}
