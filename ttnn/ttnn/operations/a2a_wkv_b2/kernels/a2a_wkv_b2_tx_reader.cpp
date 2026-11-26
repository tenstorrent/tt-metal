// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 fabric_tx reader (NCRISC, NoC 0) — pull_block.
//
// One stream = the heads this chip sends to one destination over one (direction, link). Order is
// mt-major, then the stream's heads in head order, then the packets of each head's block (identical
// loop in the tx writer). The packets of one mt round are processed in batches of up to tx_batch
// packets; each batch occupies one tx_batch-packet region of cb_tx_stage (always reserved and pushed
// whole, so batch regions stay contiguous; a short last batch of a round leaves its tail slots unused).
// Per batch:
//   1. publish check: for every packet whose head's cached block count is short, issue one remote 4 B
//      read of that head's block counter on its worker (all polls of the batch behind one barrier);
//      re-poll individually only a head that is still short. The counter is monotonic within a call,
//      so a cached value that already covers the block skips the poll entirely.
//   2. pull: issue every packet's read from the worker's cb_out (same L1 address on every program
//      core) behind one barrier, then push the whole batch region.
// Block counters are per-head-slot GlobalSemaphores (persistent address, zero at creation), not program
// semaphores: a program semaphore is re-initialised only when its worker core launches, which can be after
// this core starts polling, so a stale count from the previous call would pass. Instead this reader, the
// head's only consumer, resets each head's counter to 0 after the head's last block (write barrier before
// exit), so every call starts from 0 whatever the launch order.
//
// No helper: this is a one-to-one pull of one worker's block by address (the mcast helper is a
// sender-driven multicast; the ccl helpers own fabric egress only) — op_design.md.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_tx_stage = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t Mt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t mt_block = get_compile_time_arg_val(4);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t tx_batch = get_compile_time_arg_val(7);
    constexpr uint32_t max_stream_heads = get_compile_time_arg_val(8);

    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;
    constexpr uint32_t packet_bytes = tiles_per_packet * tile_bytes;
    constexpr uint32_t batch_pages = tx_batch * tiles_per_packet;

    const uint32_t num_heads = get_arg_val<uint32_t>(0);
    constexpr uint32_t head_args_base = 1;
    constexpr uint32_t head_args = 4;  // per head: worker noc x, worker noc y, head idx in worker, counter address

    // Every CB spans all program cores, so the workers' cb_out starts at this core's cb_out base.
    const uint32_t remote_cb_out_base = get_read_ptr(cb_out);

    uint32_t seen[max_stream_heads];  // last block count read per head
    for (uint32_t s = 0; s < max_stream_heads; ++s) {
        seen[s] = 0;
    }

    for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
        const uint32_t mt0 = mb * mt_block;
        const uint32_t mt_rows = (Mt - mt0) < mt_block ? (Mt - mt0) : mt_block;
        const uint32_t block_tiles = mt_rows * Nt;
        const uint32_t packets_per_block = (block_tiles + tiles_per_packet - 1) / tiles_per_packet;
        const uint32_t round_packets = num_heads * packets_per_block;

        for (uint32_t q0 = 0; q0 < round_packets; q0 += tx_batch) {
            const uint32_t k = (round_packets - q0) < tx_batch ? (round_packets - q0) : tx_batch;
            {
                DeviceZoneScopedN("TX_RSV");
                cb_reserve_back(cb_tx_stage, batch_pages);
            }
            const uint32_t base = get_write_ptr(cb_tx_stage);

            // 1. publish check (slot i's first word is the poll scratch for packet i).
            bool polled = false;
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t s = (q0 + i) / packets_per_block;
                if (seen[s] < mb + 1) {
                    const uint32_t wx = get_arg_val<uint32_t>(head_args_base + head_args * s);
                    const uint32_t wy = get_arg_val<uint32_t>(head_args_base + head_args * s + 1);
                    const uint32_t sem_addr = get_arg_val<uint32_t>(head_args_base + head_args * s + 3);
                    noc_async_read(get_noc_addr(wx, wy, sem_addr), base + i * packet_bytes, sizeof(uint32_t));
                    polled = true;
                }
            }
            if (polled) {
                DeviceZoneScopedN("TX_POLL");
                noc_async_read_barrier();
                for (uint32_t i = 0; i < k; ++i) {
                    const uint32_t s = (q0 + i) / packets_per_block;
                    const uint32_t blocks_needed = mb + 1;
                    if (seen[s] >= blocks_needed) {
                        continue;  // covered before this batch (or by an earlier packet's poll)
                    }
                    const uint32_t scratch = base + i * packet_bytes;
                    volatile tt_l1_ptr uint32_t* poll = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
                    seen[s] = *poll;
                    if (seen[s] < blocks_needed) {
                        const uint32_t wx = get_arg_val<uint32_t>(head_args_base + head_args * s);
                        const uint32_t wy = get_arg_val<uint32_t>(head_args_base + head_args * s + 1);
                        const uint32_t sem_addr = get_arg_val<uint32_t>(head_args_base + head_args * s + 3);
                        const uint64_t remote_sem = get_noc_addr(wx, wy, sem_addr);
                        do {
                            noc_async_read(remote_sem, scratch, sizeof(uint32_t));
                            noc_async_read_barrier();
                            seen[s] = *poll;
                        } while (seen[s] < blocks_needed);
                    }
                }
            }

            // 2. pull every packet of the batch behind one barrier.
            for (uint32_t i = 0; i < k; ++i) {
                const uint32_t q = q0 + i;
                const uint32_t s = q / packets_per_block;
                const uint32_t p = (q % packets_per_block) * tiles_per_packet;
                const uint32_t chunks = (block_tiles - p) < tiles_per_packet ? (block_tiles - p) : tiles_per_packet;
                const uint32_t wx = get_arg_val<uint32_t>(head_args_base + head_args * s);
                const uint32_t wy = get_arg_val<uint32_t>(head_args_base + head_args * s + 1);
                const uint32_t head_idx = get_arg_val<uint32_t>(head_args_base + head_args * s + 2);
                const uint32_t first_tile = (head_idx * Mt + mt0) * Nt;
                noc_async_read(
                    get_noc_addr(wx, wy, remote_cb_out_base + (first_tile + p) * tile_bytes),
                    base + i * packet_bytes,
                    chunks * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_tx_stage, batch_pages);
        }
    }

    // Every head's last block was observed published, so its counter gets no further increments this
    // call: return it to 0 for the next call.
    for (uint32_t s = 0; s < num_heads; ++s) {
        const uint32_t wx = get_arg_val<uint32_t>(head_args_base + head_args * s);
        const uint32_t wy = get_arg_val<uint32_t>(head_args_base + head_args * s + 1);
        const uint32_t sem_addr = get_arg_val<uint32_t>(head_args_base + head_args * s + 3);
        noc_inline_dw_write(get_noc_addr(wx, wy, sem_addr), 0);
    }
    noc_async_write_barrier();
}
