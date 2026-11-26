// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 worker writer (BRISC, NoC 1) — publish_block.
//
// cb_out holds every block of this core's heads until the program ends (fabric_tx pulls remote
// blocks from it by address), so blocks are waited cumulatively and nothing is popped before the
// end. Per block:
//   local head (destination == this chip): write its tiles into the local L1 output, one barrier
//   every block                          : bump this core's sem_blocks_done (fabric_tx polls it)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t mt_block = get_compile_time_arg_val(3);
    constexpr uint32_t heads_per_chip = get_compile_time_arg_val(4);
    constexpr uint32_t ring_size = get_compile_time_arg_val(5);
    constexpr uint32_t sem_blocks_done_id = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();

    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;
    constexpr uint32_t tiles_per_source_head = ring_size * Mt * Nt;  // one destination head, all sources

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t head_start = get_arg_val<uint32_t>(1);
    const uint32_t num_heads = get_arg_val<uint32_t>(2);
    const uint32_t my_pos = get_arg_val<uint32_t>(3);  // this chip's ring position = source index j

    const auto out = TensorAccessor(out_args, out_addr, out_tile_bytes);
    const uint64_t sem_noc_addr = get_noc_addr(get_semaphore(sem_blocks_done_id));
    const uint32_t cb_base = get_read_ptr(cb_out);

    uint32_t tiles_published = 0;
    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        const uint32_t h = head_start + head_idx;
        const bool local = (h / heads_per_chip) == my_pos;
        const uint32_t h_local = h % heads_per_chip;
        for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
            const uint32_t mt0 = mb * mt_block;
            const uint32_t mt_rows = (Mt - mt0) < mt_block ? (Mt - mt0) : mt_block;
            const uint32_t block_tiles = mt_rows * Nt;
            cb_wait_front(cb_out, tiles_published + block_tiles);
            if (local) {
                uint32_t l1 = cb_base + tiles_published * out_tile_bytes;
                const uint32_t tile0 = h_local * tiles_per_source_head + (my_pos * Mt + mt0) * Nt;
                for (uint32_t t = 0; t < block_tiles; ++t) {
                    noc_async_write(l1, out.get_noc_addr(tile0 + t), out_tile_bytes);
                    l1 += out_tile_bytes;
                }
                noc_async_write_barrier();
            }
            tiles_published += block_tiles;
            noc_semaphore_inc(sem_noc_addr, 1);
        }
    }
    noc_async_atomic_barrier();
    cb_pop_front(cb_out, tiles_published);
}
