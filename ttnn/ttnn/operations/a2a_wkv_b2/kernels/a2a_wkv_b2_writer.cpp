// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 worker writer (BRISC; NoC 1, or NoC 0 on cores whose reader takes NoC 1) — publish_block.
//
// cb_out holds every block of this core's heads until the program ends (fabric_tx pulls remote
// blocks from it by address), so blocks are waited cumulatively and nothing is popped before the
// end. Per block:
//   local head (destination == this chip): write its tiles into the local L1 output, one barrier
//   remote head (sent by a fabric_tx core): bump that head's block counter (a per-head-slot
//                                           GlobalSemaphore; its fabric_tx reader polls it and resets it
//                                           to 0 after the head's last block)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t mt_block = get_compile_time_arg_val(3);
    constexpr uint32_t heads_per_chip = get_compile_time_arg_val(4);
    constexpr uint32_t ring_size = get_compile_time_arg_val(5);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();

    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;
    constexpr uint32_t tiles_per_source_head = ring_size * Mt * Nt;  // one destination head, all sources

    // Common runtime args only (no per-core args: launch gap, Perf 1):
    //   [out_addr, my_pos, grid_x, table_base, block-counter address per head slot...,
    //    from table_base, per logical core index y * grid_x + x: head_start | num_heads << 8 | noc << 16]
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t my_pos = get_common_arg_val<uint32_t>(1);  // this chip's ring position = source index j
    const uint32_t core_word = get_common_arg_val<uint32_t>(
        get_common_arg_val<uint32_t>(3) + get_relative_logical_y() * get_common_arg_val<uint32_t>(2) +
        get_relative_logical_x());
    const uint32_t head_start = core_word & 0xFF;
    const uint32_t num_heads = (core_word >> 8) & 0xFF;
    // One writer kernel over every worker core (Perf 2, launch_gap2): this core's NoC is bit 16 of its table word
    // (NoC 1, or NoC 0 on the cores whose reads default to NoC 1); every write, barrier and inc names it.
    const uint8_t noc = (core_word >> 16) & 1;
    constexpr uint32_t counter_args_base = 4;

    const auto out = TensorAccessor(out_args, out_addr, out_tile_bytes);
    const uint32_t cb_base = get_read_ptr(cb_out);

    uint32_t tiles_published = 0;
    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        const uint32_t h = head_start + head_idx;
        const bool local = (h / heads_per_chip) == my_pos;
        const uint32_t h_local = h % heads_per_chip;
        const uint64_t counter_noc_addr = get_noc_addr(get_common_arg_val<uint32_t>(counter_args_base + head_idx), noc);
        for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
            const uint32_t mt0 = mb * mt_block;
            const uint32_t mt_rows = (Mt - mt0) < mt_block ? (Mt - mt0) : mt_block;
            const uint32_t block_tiles = mt_rows * Nt;
            {
                MaybeDeviceZoneScope("writer_wait_block");  // starved on compute
                cb_wait_front(cb_out, tiles_published + block_tiles);
            }
            if (local) {
                MaybeDeviceZoneScope("writer_local_write");  // issue + barrier into the local L1 output
                uint32_t l1 = cb_base + tiles_published * out_tile_bytes;
                const uint32_t tile0 = h_local * tiles_per_source_head + (my_pos * Mt + mt0) * Nt;
                for (uint32_t t = 0; t < block_tiles; ++t) {
                    noc_async_write(l1, out.get_noc_addr(tile0 + t, 0, noc), out_tile_bytes, noc);
                    l1 += out_tile_bytes;
                }
                noc_async_write_barrier(noc);
            }
            tiles_published += block_tiles;
            if (!local) {
                noc_semaphore_inc(counter_noc_addr, 1, noc);
            }
        }
    }
    noc_async_atomic_barrier(noc);
    cb_pop_front(cb_out, tiles_published);
}
