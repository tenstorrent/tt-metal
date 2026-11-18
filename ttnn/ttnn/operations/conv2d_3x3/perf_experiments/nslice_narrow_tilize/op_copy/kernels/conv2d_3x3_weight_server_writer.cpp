// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 weight server, writer half (BRISC / NoC1). Per round: wait until every receiver of the rectangle can take
// it (a monotonic ack counter: one ack per receiver at its kernel start, then one per round that reuses a cb_weights
// slot), then multicast each chunk the server reader hands over into the same slot of every receiver, and free the
// slot once the multicast has read it. Each chunk's linked multicast is closed by a 4 B write of zero to the receivers'
// flag of this slot (INVALID there until this round's end: the receiver cleared it before its ack let this round
// start); the last chunk's closing write sets the flag. One multicast into the rectangle at a time: concurrent
// multicasts from several senders into one rectangle measured 1.6x to 3.7x slower.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
#ifdef CONV2D_3X3_ABLATE_WEIGHTS
    return;
#endif
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t round_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t ack_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t flag_sem_id = get_compile_time_arg_val(3);  // one per weight slot
    constexpr uint32_t weight_depth = get_compile_time_arg_val(4);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;

    const uint32_t num_rounds = get_arg_val<uint32_t>(0);
    const uint32_t num_receivers = get_arg_val<uint32_t>(1);
    // The rectangle in NoC1 multicast order: start (high corner) x, y, end (low corner) x, y.
    const uint32_t sx = get_arg_val<uint32_t>(2);
    const uint32_t sy = get_arg_val<uint32_t>(3);
    const uint32_t ex = get_arg_val<uint32_t>(4);
    const uint32_t ey = get_arg_val<uint32_t>(5);
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ack_sem_id));
    // The multicast source of every slot's flag.
    for (uint32_t slot = 0; slot < weight_depth; ++slot) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(flag_sem_id + slot)) = VALID;
    }

    for (uint32_t r = 0; r < num_rounds; ++r) {
        {
            MaybeDeviceZoneScope("server_wait_ack");
            const uint32_t acks = 1 + (r + 1 > weight_depth ? r + 1 - weight_depth : 0);
            noc_semaphore_wait_min(ack, acks * num_receivers);
        }
        const uint32_t flag = get_semaphore(flag_sem_id + r % weight_depth);
        const uint64_t flag_mcast = get_noc_multicast_addr(sx, sy, ex, ey, flag);
        const uint32_t base = get_read_ptr(cb_weights);
        for (uint32_t c = 0; c < num_chunks; ++c) {
            {
                MaybeDeviceZoneScope("server_wait_read");
                cb_wait_front(cb_weights, (c + 1) * chunk_tiles);
            }
            MaybeDeviceZoneScope("server_send");
            const uint32_t src = base + c * chunk_tiles * tile_bytes;
            noc_async_write_multicast(
                src, get_noc_multicast_addr(sx, sy, ex, ey, src), chunk_tiles * tile_bytes, num_receivers, true);
            noc_semaphore_set_multicast(c + 1 == num_chunks ? flag : MEM_ZEROS_BASE, flag_mcast, num_receivers, false);
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
