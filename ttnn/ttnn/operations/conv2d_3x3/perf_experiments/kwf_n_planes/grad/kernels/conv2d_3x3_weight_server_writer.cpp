// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 weight server, writer half (BRISC / NoC1). Per round: wait until every receiver of the rectangle can take
// it (a monotonic ack counter: one ack per receiver at its kernel start, then one per round that reuses a cb_weights
// slot), then multicast the chunks the server reader hands over into the same slot of every receiver, and free the
// slot once the multicast has read it. Every chunk that has landed goes out in one linked chain, one path set-up per
// chain (NoC estimator, Blackhole: 16 linked 8 KB multicasts to 64 cores at 43 B/cycle, unlinked at 15; one chain per
// chunk measured ~0.3 us per 8.7 KB chunk on res3.x). The chain is closed by a 4 B write to the receivers' flag of
// this slot when the round is complete (the flag is set) or when the next chunk has not landed yet (a write of zero:
// the flag stays INVALID until this round's end, the receiver cleared it before its ack let this round start), so no
// path stays reserved across a wait. One multicast into the rectangle at a time: concurrent multicasts from several
// senders into one rectangle measured 1.6x to 3.7x slower.
//
// With a bias the server also reads it once (NoC1, at kernel start) and multicasts it into every receiver's cb_bias
// (same index and size, so the same L1 address) before round 0, then sets the receivers' bias flag (the semaphore
// after the slot flags). HEIGHT: every receiver takes all of C_out. ~100 receivers reading the same bias pages from
// DRAM themselves landed it at ~7 us, behind kw-fold's ~6 us bias decode in front of the final pass. A round 0 of at
// most one NoC burst gates compute's first matmul on a short multicast, so the bias goes out after it instead (res2
// unzoned, leave-one-out: 10.87 us with the bias first, 10.27 us after); larger rounds keep it first (res3.x: a later
// bias pushed the reader's ~1.3 us-per-tile decode past the final pass, +1.9%).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t ack_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t flag_sem_id = get_compile_time_arg_val(3);  // one per weight slot, then the bias flag
    constexpr uint32_t weight_depth = get_compile_time_arg_val(4);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(5);
    constexpr bool has_bias = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t cb_bias = get_compile_time_arg_val(7);
    constexpr uint32_t bias_tiles = get_compile_time_arg_val(8);
    constexpr auto bias_args = TensorAccessorArgs<9>();
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;
    constexpr uint32_t bias_flag_sem_id = flag_sem_id + weight_depth;

    const uint32_t num_rounds = get_arg_val<uint32_t>(0);
    const uint32_t num_receivers = get_arg_val<uint32_t>(1);
    // The rectangle in NoC1 multicast order: start (high corner) x, y, end (low corner) x, y.
    const uint32_t sx = get_arg_val<uint32_t>(2);
    const uint32_t sy = get_arg_val<uint32_t>(3);
    const uint32_t ex = get_arg_val<uint32_t>(4);
    const uint32_t ey = get_arg_val<uint32_t>(5);
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ack_sem_id));
    // The round whose ack wait the bias multicast follows. Round 1 only with two or more slots: its ack wait then needs
    // only the kernel-start acks, while a non-kw-fold receiver blocks on the bias right after round 0 (with one slot,
    // round 1 would wait for that receiver's reuse ack).
    const uint32_t bias_round =
        round_tiles * tile_bytes <= NOC_MAX_BURST_SIZE && num_rounds > 1 && weight_depth >= 2 ? 1 : 0;

    [[maybe_unused]] uint32_t bias_l1 = 0;
    if constexpr (has_bias) {
        const uint32_t bias_tile_bytes = get_tile_size(cb_bias);
        const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(6), bias_tile_bytes);
        bias_l1 = get_write_ptr(cb_bias);
        for (uint32_t n = 0; n < bias_tiles; ++n) {
            noc_async_read_page(n, bias, bias_l1 + n * bias_tile_bytes);
        }
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(bias_flag_sem_id)) = VALID;
    }
    // The multicast source of every slot's flag.
    for (uint32_t slot = 0; slot < weight_depth; ++slot) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(flag_sem_id + slot)) = VALID;
    }

    for (uint32_t r = 0; r < num_rounds; ++r) {
        {
            MaybeDeviceZoneScope("server_wait_ack");
#ifdef SERVER_SLOT_ACKS
            // One counter per slot (the receivers' SERVER_SLOT_ACKS): slot 0's also holds the kernel-start acks, so a
            // receiver's acks for later slots never stand in for another receiver's ack of this one.
            const uint32_t slot = r < weight_depth ? 0 : r % weight_depth;
            const uint32_t uses = r < weight_depth ? 0 : r / weight_depth;
            noc_semaphore_wait_min(
                slot == 0 ? ack
                          : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SERVER_SLOT_ACKS + slot - 1)),
                (uses + (slot == 0 ? 1 : 0)) * num_receivers);
#else
            const uint32_t acks = 1 + (r + 1 > weight_depth ? r + 1 - weight_depth : 0);
            noc_semaphore_wait_min(ack, acks * num_receivers);
#endif
        }
        if constexpr (has_bias) {
            if (r == bias_round) {
                MaybeDeviceZoneScope("server_send_bias");
                noc_async_read_barrier();
                const uint32_t bias_bytes = bias_tiles * get_tile_size(cb_bias);
                noc_async_write_multicast(
                    bias_l1, get_noc_multicast_addr(sx, sy, ex, ey, bias_l1), bias_bytes, num_receivers, true);
                const uint32_t bias_flag = get_semaphore(bias_flag_sem_id);
                noc_semaphore_set_multicast(
                    bias_flag, get_noc_multicast_addr(sx, sy, ex, ey, bias_flag), num_receivers, false);
            }
        }
        const uint32_t flag = get_semaphore(flag_sem_id + r % weight_depth);
        const uint64_t flag_mcast = get_noc_multicast_addr(sx, sy, ex, ey, flag);
        const uint32_t base = get_read_ptr(cb_weights);
        auto landed = [&](uint32_t c) -> bool {
            return cb_pages_available_at_front(cb_weights, (c + 1) * chunk_tiles);
        };
        for (uint32_t c = 0; c < num_chunks;) {
            {
                MaybeDeviceZoneScope("server_wait_read");
                cb_wait_front(cb_weights, (c + 1) * chunk_tiles);
            }
            MaybeDeviceZoneScope("server_send");
            uint32_t n = 1;
            while (c + n < num_chunks && landed(c + n)) {
                ++n;
            }
            const uint32_t src = base + c * chunk_tiles * tile_bytes;
            noc_async_write_multicast(
                src, get_noc_multicast_addr(sx, sy, ex, ey, src), n * chunk_tiles * tile_bytes, num_receivers, true);
            c += n;
            // Keep the chain open only into a chunk that already landed.
            if (c == num_chunks || !landed(c)) {
                noc_semaphore_set_multicast(c == num_chunks ? flag : MEM_ZEROS_BASE, flag_mcast, num_receivers, false);
            }
        }
        {
            // The reader refills this slot weight_depth rounds on: the multicast must have read it.
            MaybeDeviceZoneScope("server_flush");
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_weights, round_tiles);
    }
    noc_async_write_barrier();
}
