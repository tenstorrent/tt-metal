// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x-download probe, sender (NCRISC, NOC1) on a bank reader: sends every chunk its BRISC reads (sm_reader.cpp) to all
// compute cores, either as MODE 1 multicasts to the grid rectangles or MODE 0 one unicast per compute core, into a
// SLOTS-chunk ring there (bandwidth probe: nothing is consumed, slots are simply overwritten). When done, bumps every
// compute core's DONE semaphore once. MODE 2: one unicast per column head (the N_RECV list) followed, once the
// write is acknowledged, by an ARR increment there; the heads multicast down their columns (xdl_head.cpp).
//
// CT: 0 CB, 1 CHUNK_TILES, 2 TILE_BYTES, 3 NUM_CHUNKS, 4 MODE, 5 SLOTS, 6 DONE_SEM
// RT: 0 ring address, 1 first slot of this reader, 2 N_RECV, 3.. N_RECV xy, then N_RECTS and per rectangle start xy,
//     end xy (already in NOC1 order), destinations
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t mode = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t done_sem = get_compile_time_arg_val(6);
    constexpr uint32_t arr_sem = get_compile_time_arg_val(7);
    constexpr uint32_t per_group =
        get_compile_time_arg_val(8);  // MODE 2: chunk c goes to target (c % PER_GROUP) of each group
    constexpr bool by_reader = get_compile_time_arg_val(9) != 0;  // ... or target (reader index % PER_GROUP)
    constexpr uint32_t bytes = chunk_tiles * tile_bytes;
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t slot0 = get_arg_val<uint32_t>(1);
    const uint32_t n_recv = get_arg_val<uint32_t>(2);
    const uint32_t r0 = 3 + n_recv;
    const uint32_t n_rects = get_arg_val<uint32_t>(r0);
    if constexpr (mode == 2) {
        // Pipelined: up to DEPTH chunks in flight (each on its own NoC transaction id), completed in order.
        constexpr uint32_t depth = 4;
        auto write_trid = [](uint32_t src, uint64_t dst, uint32_t trid) {
            for (uint32_t o = 0; o < bytes; o += NOC_MAX_BURST_SIZE) {
                const uint32_t n = bytes - o < NOC_MAX_BURST_SIZE ? bytes - o : NOC_MAX_BURST_SIZE;
                noc_async_write_one_packet_with_trid(src + o, dst + o, n, trid);
            }
        };
        uint32_t issued = 0, done = 0;
        while (done < num_chunks) {
            while (issued < num_chunks && issued - done < depth &&
                   cb_pages_available_at_front(cb, (issued - done + 1) * chunk_tiles)) {
                const uint32_t src = get_read_ptr(cb) + (issued - done) * bytes;
                const uint32_t dst = ring + ((slot0 + issued) % slots) * bytes;
                for (uint32_t i = (by_reader ? slot0 / num_chunks : slot0 + issued) % per_group; i < n_recv;
                     i += per_group) {
                    const uint32_t xy = get_arg_val<uint32_t>(3 + i);
                    write_trid(src, get_noc_addr(xy >> 16, xy & 0xFFFF, dst), 1 + issued % depth);
                }
                ++issued;
            }
            if (done < issued && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + done % depth)) {
                for (uint32_t i = (by_reader ? slot0 / num_chunks : slot0 + done) % per_group; i < n_recv;
                     i += per_group) {
                    const uint32_t xy = get_arg_val<uint32_t>(3 + i);
                    noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(arr_sem)), 1);
                }
                cb_pop_front(cb, chunk_tiles);
                ++done;
            }
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        return;
    }
    for (uint32_t c = 0; c < num_chunks; ++c) {
        cb_wait_front(cb, chunk_tiles);
        const uint32_t src = get_read_ptr(cb);
        const uint32_t dst = ring + ((slot0 + c) % slots) * bytes;
        if constexpr (mode == 1) {
            for (uint32_t r = 0; r < n_rects; ++r) {
                const uint32_t a0 = get_arg_val<uint32_t>(r0 + 1 + 3 * r), a1 = get_arg_val<uint32_t>(r0 + 2 + 3 * r);
                noc_async_write_multicast(
                    src,
                    get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, dst),
                    bytes,
                    get_arg_val<uint32_t>(r0 + 3 + 3 * r));
            }
        } else {
            for (uint32_t i = 0; i < n_recv; ++i) {
                const uint32_t xy = get_arg_val<uint32_t>(3 + i);
                noc_async_write(src, get_noc_addr(xy >> 16, xy & 0xFFFF, dst), bytes);
            }
            if constexpr (mode == 2) {
                noc_async_write_barrier();
                for (uint32_t i = 0; i < n_recv; ++i) {
                    const uint32_t xy = get_arg_val<uint32_t>(3 + i);
                    noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(arr_sem)), 1);
                }
            }
        }
        noc_async_writes_flushed();  // the chunk has left L1: its CB slot can be refilled
        cb_pop_front(cb, chunk_tiles);
    }
    noc_async_write_barrier();
    if constexpr (mode == 2) {
        noc_async_atomic_barrier();
        return;  // the heads report DONE
    }
    for (uint32_t i = 0; i < n_recv; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(3 + i);
        noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(done_sem)), 1);
    }
    noc_async_atomic_barrier();
}
