// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 fabric_tx reader (NCRISC, NoC 0) — pull_block.
//
// One stream = the heads this chip sends to one destination over one (direction, link). Its packets are
// pulled from the workers' cb_out into cb_tx_stage in batches of up to tx_batch packets; each batch
// occupies one whole tx_batch-packet region of cb_tx_stage (always reserved and pushed whole, so batch
// regions stay contiguous). Every batch also carries one cb_tx_meta page: [k, code_0 .. code_{k-1}], the
// packet count and, per packet, which (mt block, head, packet) it is — the tx writer looks the packet's
// destination addresses up from that code, so it does not care about the order.
//
// Send order (TX_ORDER, compile-time):
//   strict (0): mt-major, then the stream's heads in head order, then the packets of each head's block
//               (Phase 0 / Refinement 1-2 order). A batch waits for its packets in that sequence.
//   ready  (1): whatever is published, round-robin over the heads from a cursor (Refinement 3). Worker
//               cores progress at very different rates (DRAM/NoC locality: a core's last block lands
//               anywhere from ~24 to ~60 us), so a strict mt-major stream is gated by its slowest head
//               every round and leaves the link idle; ready order keeps the link busy with the fast
//               heads' blocks. A batch is sent as soon as a poll finds nothing more ready (partial batch).
//
// Publish check: one remote 4 B read per candidate head of its block counter on its worker (all polls of
// one scan behind one barrier). The counter is monotonic within a call, so a cached value that already
// covers the block skips the poll. Block counters are per-head-slot GlobalSemaphores (persistent
// address, zero at creation), not program semaphores (a program semaphore is re-initialised only when
// its worker core launches, which can be after this core starts polling). This reader, the head's only
// consumer, resets each head's counter to 0 after the head's last block (write barrier before exit).
//
// No helper: this is a one-to-one pull of one worker's block by address (the mcast helper is a
// sender-driven multicast; the ccl helpers own fabric egress only) — op_design.md.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
#ifdef A2A_TX_COMMON
    // launch_gap2: no per-core runtime args. This core's args sit in the kernel's common table:
    //   [grid_x, first logical core index, counters_base, offset per tx core slot..., per-core arg blocks...]
    // get_arg_val is pointed at this core's block, so the rest of the kernel (and the fabric connection
    // helper, which consumes its block through get_arg_val) is unchanged.
    {
        const uint32_t slot = get_relative_logical_y() * get_common_arg_val<uint32_t>(0) + get_relative_logical_x() -
                              get_common_arg_val<uint32_t>(1);
        const uint32_t off = get_common_arg_val<uint32_t>(3 + slot);
        rta_l1_base = crta_l1_base + off;
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)
        rta_count = crta_count - off;
#endif
    }
#endif
    constexpr uint32_t cb_tx_stage = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t Mt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t mt_block = get_compile_time_arg_val(4);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t tx_batch = get_compile_time_arg_val(7);
    constexpr uint32_t max_stream_heads = get_compile_time_arg_val(8);
    constexpr uint32_t cb_tx_meta = get_compile_time_arg_val(9);
    constexpr uint32_t poll_stride = get_compile_time_arg_val(10);  // L1 alignment of one poll scratch word
    constexpr bool ready_order = get_compile_time_arg_val(11) != 0;

    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;
    constexpr uint32_t packet_bytes = tiles_per_packet * tile_bytes;
    constexpr uint32_t batch_pages = tx_batch * tiles_per_packet;
    // Poll scratch lives in a free packet slot of the batch being filled (a poll only runs with k < tx_batch).
    static_assert(max_stream_heads * poll_stride <= packet_bytes, "poll scratch must fit one packet slot");
    static_assert(max_stream_heads <= 256, "meta code packs the head in 8 bits");

    const uint32_t num_heads = get_arg_val<uint32_t>(0);
    constexpr uint32_t head_args_base = 1;
#ifdef A2A_TX_COMMON
    // per head one packed word x | y << 8 | head idx in worker << 16; the block-counter addresses (one per worker
    // head slot) are shared, in the common table from counters_base (keeps the 4-core table under the cap).
    const uint32_t counters_base = get_common_arg_val<uint32_t>(2);
    auto head_arg = [&](uint32_t s, uint32_t field) -> uint32_t {
        const uint32_t w = get_arg_val<uint32_t>(head_args_base + s);
        if (field == 3) {
            return get_common_arg_val<uint32_t>(counters_base + (w >> 16));
        }
        return (w >> (8 * field)) & 0xFF;
    };
#else
    constexpr uint32_t head_args = 4;  // per head: worker noc x, worker noc y, head idx in worker, counter address
    auto head_arg = [&](uint32_t s, uint32_t field) {
        return get_arg_val<uint32_t>(head_args_base + head_args * s + field);
    };
#endif

    // Every CB spans all program cores, so the workers' cb_out starts at this core's cb_out base.
    const uint32_t remote_cb_out_base = get_read_ptr(cb_out);

    uint32_t seen[max_stream_heads];     // last block count read per head
    uint32_t next_mb[max_stream_heads];  // next mt block to pull, per head
    uint32_t next_p[max_stream_heads];   // next tile offset within that block
    for (uint32_t s = 0; s < max_stream_heads; ++s) {
        seen[s] = 0;
        next_mb[s] = 0;
        next_p[s] = 0;
    }
    uint32_t heads_left = num_heads;
    uint32_t cursor = 0;     // ready: round-robin start of the next scan; strict: current head
    uint32_t strict_mb = 0;  // strict: current mt round

    auto published = [&](uint32_t s) { return seen[s] >= next_mb[s] + 1; };

    // Batches can be as small as one packet (ready order), so no per-batch zone: the marker budget
    // (~125 zone executions per RISC) would run out silently mid-stream.
    {
        MaybeDeviceZoneScope("tx_reader_stream");  // polls + pulls + back-pressure from the tx writer
        while (heads_left > 0) {
            cb_reserve_back(cb_tx_stage, batch_pages);
            cb_reserve_back(cb_tx_meta, 1);
            const uint32_t base = get_write_ptr(cb_tx_stage);
            volatile tt_l1_ptr uint32_t* meta =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tx_meta));
            uint32_t k = 0;

            // Take head s's next packet: issue its pull (one barrier for the batch), describe it in the meta
            // page, advance the head. Returns true when that completes the head's block.
            auto take = [&](uint32_t s) {
                const uint32_t mb = next_mb[s];
                const uint32_t mt0 = mb * mt_block;
                const uint32_t mt_rows = (Mt - mt0) < mt_block ? (Mt - mt0) : mt_block;
                const uint32_t block_tiles = mt_rows * Nt;
                const uint32_t p = next_p[s];
                const uint32_t chunks = (block_tiles - p) < tiles_per_packet ? (block_tiles - p) : tiles_per_packet;
                const uint32_t first_tile = (head_arg(s, 2) * Mt + mt0) * Nt;
#ifndef A2A_ABL_NO_PULL
                noc_async_read(
                    get_noc_addr(head_arg(s, 0), head_arg(s, 1), remote_cb_out_base + (first_tile + p) * tile_bytes),
                    base + k * packet_bytes,
                    chunks * tile_bytes);
#endif
                meta[1 + k] = (mb << 16) | (s << 8) | (p / tiles_per_packet);
                ++k;
                if (p + chunks < block_tiles) {
                    next_p[s] = p + chunks;
                    return false;
                }
                next_p[s] = 0;
                if (++next_mb[s] == num_mt_blocks) {
                    --heads_left;
                }
                return true;
            };

            for (;;) {
                if constexpr (ready_order) {
                    for (uint32_t n = 0; n < num_heads && k < tx_batch; ++n) {
                        const uint32_t s = cursor;
                        cursor = cursor + 1 == num_heads ? 0 : cursor + 1;
                        while (k < tx_batch && next_mb[s] < num_mt_blocks && published(s)) {
                            take(s);
                        }
                    }
                } else {
                    while (k < tx_batch && strict_mb < num_mt_blocks && published(cursor)) {
                        if (take(cursor) && ++cursor == num_heads) {
                            cursor = 0;
                            ++strict_mb;
                        }
                    }
                }
                if (k == tx_batch || heads_left == 0) {
                    break;
                }
                // poll the counter of every candidate head that is short (ready: every unfinished head;
                // strict: the heads of the rest of this batch in sequence) into scratch past the pulled slots.
                const uint32_t scratch_base = base + k * packet_bytes;
                uint32_t first = 0, count = num_heads;
                if constexpr (!ready_order) {
                    first = cursor;
                    const uint32_t rest = tx_batch - k;
                    count = rest < num_heads ? rest : num_heads;
                }
                uint32_t found = 0;
                for (uint32_t n = 0, s = first; n < count; ++n, s = s + 1 == num_heads ? 0 : s + 1) {
                    if (next_mb[s] < num_mt_blocks && !published(s)) {
                        noc_async_read(
                            get_noc_addr(head_arg(s, 0), head_arg(s, 1), head_arg(s, 3)),
                            scratch_base + s * poll_stride,
                            sizeof(uint32_t));
                    }
                }
                noc_async_read_barrier();
                for (uint32_t n = 0, s = first; n < count; ++n, s = s + 1 == num_heads ? 0 : s + 1) {
                    if (next_mb[s] < num_mt_blocks && !published(s)) {
                        seen[s] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_base + s * poll_stride);
                        found += published(s) ? 1 : 0;
                    }
                }
                if constexpr (ready_order) {
                    if (found == 0 && k > 0) {
                        break;  // nothing more ready: send the partial batch now rather than idle the link
                    }
                }
            }

            noc_async_read_barrier();
            meta[0] = k;
            cb_push_back(cb_tx_stage, batch_pages);
            cb_push_back(cb_tx_meta, 1);
        }
    }

    // Every head's last block was observed published, so its counter gets no further increments this
    // call: return it to 0 for the next call.
    MaybeDeviceZoneScope("tx_reader_reset");
    for (uint32_t s = 0; s < num_heads; ++s) {
        noc_inline_dw_write(get_noc_addr(head_arg(s, 0), head_arg(s, 1), head_arg(s, 3)), 0);
    }
    noc_async_write_barrier();
}
