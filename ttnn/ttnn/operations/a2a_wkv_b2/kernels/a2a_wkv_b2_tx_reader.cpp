// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 fabric_tx reader (NCRISC, NoC 0) — pull_block.
//
// One stream = the heads this chip sends to one destination over one (direction, link). Order is
// mt-major, then the stream's heads in head order (identical loop in the tx writer). Per block:
//   poll the owning worker's sem_blocks_done (remote 4 B NoC read) until the block is published,
//   then pull its tiles from the worker's cb_out (same L1 address on every program core) into
//   cb_tx_stage, one packet slot (tiles_per_packet pages) at a time.
//
// No helper: this is a one-to-one pull of one worker's block by address (the mcast helper is a
// sender-driven multicast; the ccl helpers own fabric egress only) — op_design.md.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_tx_stage = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t sem_blocks_done_id = get_compile_time_arg_val(2);
    constexpr uint32_t Mt = get_compile_time_arg_val(3);
    constexpr uint32_t Nt = get_compile_time_arg_val(4);
    constexpr uint32_t mt_block = get_compile_time_arg_val(5);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);

    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;

    const uint32_t num_heads = get_arg_val<uint32_t>(0);
    constexpr uint32_t head_args_base = 1;  // then (worker noc x, worker noc y, head idx in worker) per head

    // Every CB spans all program cores, so the workers' cb_out starts at this core's cb_out base.
    const uint32_t remote_cb_out_base = get_read_ptr(cb_out);
    const uint32_t sem_addr = get_semaphore(sem_blocks_done_id);

    for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
        const uint32_t mt0 = mb * mt_block;
        const uint32_t mt_rows = (Mt - mt0) < mt_block ? (Mt - mt0) : mt_block;
        const uint32_t block_tiles = mt_rows * Nt;
        for (uint32_t s = 0; s < num_heads; ++s) {
            const uint32_t wx = get_arg_val<uint32_t>(head_args_base + 3 * s);
            const uint32_t wy = get_arg_val<uint32_t>(head_args_base + 3 * s + 1);
            const uint32_t head_idx = get_arg_val<uint32_t>(head_args_base + 3 * s + 2);
            const uint32_t blocks_needed = head_idx * num_mt_blocks + mb + 1;
            const uint32_t first_tile = (head_idx * Mt + mt0) * Nt;

            for (uint32_t p = 0; p < block_tiles; p += tiles_per_packet) {
                const uint32_t chunks = (block_tiles - p) < tiles_per_packet ? (block_tiles - p) : tiles_per_packet;
                cb_reserve_back(cb_tx_stage, tiles_per_packet);
                const uint32_t l1 = get_write_ptr(cb_tx_stage);
                if (p == 0) {
                    // Poll the worker's publish counter, using the reserved slot as scratch.
                    volatile tt_l1_ptr uint32_t* poll = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
                    const uint64_t remote_sem = get_noc_addr(wx, wy, sem_addr);
                    do {
                        noc_async_read(remote_sem, l1, sizeof(uint32_t));
                        noc_async_read_barrier();
                    } while (*poll < blocks_needed);
                }
                noc_async_read(
                    get_noc_addr(wx, wy, remote_cb_out_base + (first_tile + p) * tile_bytes), l1, chunks * tile_bytes);
                noc_async_read_barrier();
                cb_push_back(cb_tx_stage, tiles_per_packet);
            }
        }
    }
}
