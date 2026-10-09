// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x-download probe, rotating multicast (NCRISC on a bank reader). Per physical column run (one rectangle) the readers
// pass a token around a ring: the holder multicasts its current round (ROUND chunks its BRISC has read, sm_reader.cpp)
// into the rectangle as one linked burst, then hands the token on. So exactly one multicast sender per rectangle at a
// time (overlapping multicasts from many senders collapse), linked bursts (~2x unlinked), and the multicast source is
// the reader's own L1 (no relay whose L1 would carry every byte twice). Two rings (rectangles) run concurrently.
// CT: 0 CB, 1 CHUNK_TILES, 2 TILE_BYTES, 3 NUM_CHUNKS, 4 ROUND (chunks), 5 SLOTS, 6 TOKEN_SEM0 (TOKEN_SEM0 + k for
//     rectangle k), 7 N_RECTS, 8 DONE_SEM, 9 NOC_MC (unused: multicasts go on this kernel's NoC)
// RT: 0 ring address, 1 my index in the ring, 2 N_READERS, 3 next reader xy, 4 first slot, then per rectangle: start
//     xy, end xy (this NoC's order), dests; then N_DONE and the compute cores to signal (reader 0 only)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t round = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t token0 = get_compile_time_arg_val(6);
    constexpr uint32_t n_rects = get_compile_time_arg_val(7);
    constexpr uint32_t done_sem = get_compile_time_arg_val(8);
    constexpr uint32_t bytes = chunk_tiles * tile_bytes;
    constexpr uint32_t rounds = num_chunks / round;
    static_assert(num_chunks % round == 0 && n_rects <= 12);
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t me = get_arg_val<uint32_t>(1);
    const uint32_t nxt = get_arg_val<uint32_t>(3);
    const uint32_t slot0 = get_arg_val<uint32_t>(4);
    uint64_t rect[n_rects];
    uint32_t dests[n_rects];
    for (uint32_t r = 0; r < n_rects; ++r) {
        const uint32_t a0 = get_arg_val<uint32_t>(5 + 3 * r), a1 = get_arg_val<uint32_t>(6 + 3 * r);
        rect[r] = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
        dests[r] = get_arg_val<uint32_t>(7 + 3 * r);
    }
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    // Reader 0 opens every round; reader r > 0 waits for its predecessor's hand-off of that round. The CB holds two
    // rounds; round n sits (n - popped) rounds behind the read pointer and is popped once both rectangles have it.
    constexpr uint32_t round_tiles = round * chunk_tiles;
    uint32_t sent[n_rects], popped = 0;
    for (uint32_t r = 0; r < n_rects; ++r) {
        sent[r] = 0;
    }
    uint32_t low = 0;  // rounds every rectangle has
    while (low < rounds) {
        invalidate_l1_cache();
        for (uint32_t r = 0; r < n_rects; ++r) {
            const uint32_t n = sent[r];
            if (n == rounds || n - popped >= 2 || *sem(token0 + r) < n + (me == 0 ? 0 : 1) ||
                !cb_pages_available_at_front(cb, (n - popped + 1) * round_tiles)) {
                continue;
            }
            const uint32_t src = get_read_ptr(cb) + (n - popped) * round * bytes;
            for (uint32_t i = 0; i < round; ++i) {
                const uint32_t dst = ring + ((slot0 + n * round + i) % slots) * bytes;
                noc_async_write_multicast(src + i * bytes, rect[r] | dst, bytes, dests[r], i + 1 < round);
            }
            noc_semaphore_inc(get_noc_addr(nxt >> 16, nxt & 0xFFFF, get_semaphore(token0 + r)), 1);  // hand on
            ++sent[r];
        }
        low = sent[0];
        for (uint32_t r = 1; r < n_rects; ++r) {
            low = sent[r] < low ? sent[r] : low;
        }
        if (popped < low) {
            noc_async_writes_flushed();  // the round has left L1
            cb_pop_front(cb, round_tiles);
            ++popped;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    const uint32_t base = 5 + 3 * n_rects;
    const uint32_t n_done = get_arg_val<uint32_t>(base);
    for (uint32_t i = 0; i < n_done; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(base + 1 + i);
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(done_sem)), 1);
    }
    noc_async_atomic_barrier();
}
