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
#include "experimental/noc_semaphore.h"
#include "experimental/tensor.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);     // Source address in dram
    const uint32_t NCHt = get_arg_val<uint32_t>(1);         // Number of NCH tiles
    const uint32_t Wt = get_arg_val<uint32_t>(2);           // Width in tiles
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);  // Tile offset for this core
    const bool is_merge_core = get_arg_val<uint32_t>(4);
    const uint32_t reduce_core_noc_x = get_arg_val<uint32_t>(5);
    const uint32_t reduce_core_noc_y = get_arg_val<uint32_t>(6);
    const uint32_t y = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_x2_merge = tt::CBIndex::c_15;
    constexpr uint32_t cb_zero = tt::CBIndex::c_13;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);
    const uint32_t TILE_SIZE = 32 * 32;
    const uint32_t BF16_TILE_BYTES = 2 * TILE_SIZE;
    const uint32_t onetile = 1;

    constexpr uint32_t blk = get_compile_time_arg_val(0);
    constexpr uint32_t reducer_semaphore_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores_to_wait = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    const auto src_a = TensorAccessor(src_args, src_addr, src0_tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_inp_buf(cb_inp);
    experimental::CircularBuffer cb_out_buf(cb_out);
    experimental::CircularBuffer cb_x2_merge_buf(cb_x2_merge);
    experimental::Semaphore<> reducer_sem(reducer_semaphore_id);

    // Generate constant tiles for reduce scalar
    uint32_t scaler = get_arg_val<uint32_t>(8);

    generate_reduce_scaler(cb_reduce, scaler);
    if (is_merge_core) {
        generate_reduce_scaler(cb_zero, 0);
    }

    uint32_t inp_tile_idx = tile_offset;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // read input tiles
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            cb_inp_buf.reserve_back(blk);

            for (uint32_t r = 0; r < blk; r++) {
                noc.async_read(
                    src_a,
                    cb_inp_buf,
                    src0_tile_bytes,
                    {.page_id = inp_tile_idx},
                    {.offset_bytes = r * src0_tile_bytes});
                inp_tile_idx++;
            }
            noc.async_read_barrier();

            cb_inp_buf.push_back(blk);

        }  // wt loop

    }  // ncht loop

    // wait on cb_out and then write to merge core over noc
    cb_out_buf.wait_front(onetile);

    uint32_t o_write_size = BF16_TILE_BYTES;
    uint32_t worker_offset = o_write_size * y;

    experimental::UnicastEndpoint reduce_ep;
    noc.async_write(
        experimental::use<experimental::CircularBuffer::AddrSelector::READ_PTR>(cb_out_buf),
        reduce_ep,
        o_write_size,
        {.offset_bytes = 0},
        {.noc_x = reduce_core_noc_x,
         .noc_y = reduce_core_noc_y,
         .addr = cb_x2_merge_buf.get_write_ptr() + worker_offset});
    noc.async_write_barrier();
    cb_out_buf.pop_front(onetile);

    // increase semaphore
    reducer_sem.up(noc, reduce_core_noc_x, reduce_core_noc_y, 1);
    noc.async_atomic_barrier();

    if (is_merge_core) {
        reducer_sem.wait(num_cores_to_wait);
        cb_x2_merge_buf.push_back(num_cores_to_wait);
        reducer_sem.set(0);
    }
}
