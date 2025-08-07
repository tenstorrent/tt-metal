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
#include "debug/dprint.h"

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
    const DataFormat src0_data_format = get_dataformat(cb_inp);
    const uint32_t TILE_SIZE = 32 * 32;
    const uint32_t BF16_TILE_BYTES = 2 * TILE_SIZE;
    const uint32_t onetile = 1;

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    uint32_t reducer_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));  // semaphore for reducer
    constexpr uint32_t num_cores_to_wait = get_compile_time_arg_val(3);

    const uint64_t in0_sender_semaphore_noc_addr =
        get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, reducer_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_semaphore_addr);

    const InterleavedAddrGenFast<src0_is_dram> src_a = {
        .bank_base_address = src_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

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

    // wait on cb_out and then write to merge core over noc
    cb_wait_front(cb_out, onetile);

    uint32_t o_write_size = BF16_TILE_BYTES;
    uint32_t worker_offset = o_write_size * y;
    uint64_t output_write_addr =
        get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, get_write_ptr(cb_x2_merge)) + worker_offset;

    noc_async_write(get_read_ptr(cb_out), output_write_addr, o_write_size);
    noc_async_write_barrier();
    cb_pop_front(cb_out, onetile);

    // increase semaphore
    noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
    noc_async_atomic_barrier();

    if (is_merge_core) {
        noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, num_cores_to_wait);
        cb_push_back(cb_x2_merge, num_cores_to_wait);
        noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);
    }
}
