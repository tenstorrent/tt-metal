// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streamed-matmul weight forwarder (NCRISC, NOC1) on a reader core. For every chunk the reader pushed, sends each
// receiver's K-block (contiguous, BLK_TILES tiles) into that receiver's in1 landing ring (SLOTS blocks deep) and bumps
// the receiver's data counter. A receiver grants one credit per free slot by incrementing credit semaphore j on this
// core; block c may only be written once credit_j >= c + 1.
//
// CT: 0 CB, 1 R, 2 BLK_TILES, 3 TILE_BYTES, 4 NUM_CHUNKS, 5 SLOTS, 6 CREDIT_SEM0 (R consecutive ids), 7 DATA_SEM
// RT: 0 landing_base (receiver in1 ring L1 address, same on every receiver), 1.. R packed (x << 16 | y) NoC coords
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t blk_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t credit_sem0 = get_compile_time_arg_val(6);
    constexpr uint32_t data_sem = get_compile_time_arg_val(7);
    constexpr uint32_t blk_bytes = blk_tiles * tile_bytes;

    const uint32_t landing_base = get_arg_val<uint32_t>(0);
    uint64_t recv_noc[R];
    uint64_t recv_data_sem[R];
    volatile tt_l1_ptr uint32_t* credit[R];
    for (uint32_t j = 0; j < R; ++j) {
        const uint32_t xy = get_arg_val<uint32_t>(1 + j);
        recv_noc[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, 0);
        recv_data_sem[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(data_sem));
        credit[j] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_sem0 + j));
    }

    for (uint32_t c = 0; c < num_chunks; ++c) {
        cb_wait_front(cb, R * blk_tiles);
        const uint32_t l1 = get_read_ptr(cb);
        const uint32_t dst = landing_base + (c % slots) * blk_bytes;
        for (uint32_t j = 0; j < R; ++j) {
            noc_semaphore_wait_min(credit[j], c + 1);
            noc_async_write(l1 + j * blk_bytes, recv_noc[j] | dst, blk_bytes);
        }
        // The data counter may only move once the blocks have landed (an atomic is not ordered behind the
        // write's data packets).
        noc_async_write_barrier();
        for (uint32_t j = 0; j < R; ++j) {
            noc_semaphore_inc(recv_data_sem[j], 1);
        }
        cb_pop_front(cb, R * blk_tiles);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
