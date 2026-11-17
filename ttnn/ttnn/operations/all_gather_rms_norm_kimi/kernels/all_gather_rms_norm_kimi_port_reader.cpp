// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Port reader (RISCV_1, NoC0), one per (link, direction). Adapted from examples/fabric_all_gather's port reader: the
// chunk walk is over compute-core `landing` blocks (L1), not DRAM pages. Its stream: entry 0 = the own slice's row
// blocks of this link (read once every own core of the link has loaded: own_loaded, a fan-in counter waited for its
// total), then the relay entries (each block read once the receiver says it landed: relay_ready, one ordered sender).
// Chunks go into cb_port_chunks in batches of `batch` (2 batches in the CB); a batch ends at a multiple of `batch`
// in the CB (a reservation never wraps), at the stream end, and before any relay wait that would block (an
// unpushed batch would otherwise sit behind data that has not landed yet).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb = get_compile_time_arg_val(0);
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t batch = get_compile_time_arg_val(3);
constexpr uint32_t sem_own_loaded = get_compile_time_arg_val(5);
constexpr uint32_t sem_relay_ready = get_compile_time_arg_val(6);
constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;

void kernel_main() {
    size_t a = 0;
    const uint32_t landing_addr = get_arg_val<uint32_t>(a++);
    const uint32_t own_total = get_arg_val<uint32_t>(a++);  // own cores of my link (each increments me once)
    uint32_t chunks_left = get_arg_val<uint32_t>(a++);
    const uint32_t n_blocks = get_arg_val<uint32_t>(a++);  // then per block: core xy, n_chunks | relay << 16

    volatile tt_l1_ptr uint32_t* own_loaded =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own_loaded));
    volatile tt_l1_ptr uint32_t* relay_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_relay_ready));

    uint32_t nbat = 0, need = 0, pos = 0, wptr = 0, relays = 0;
    bool own_waited = false;
    auto push_batch = [&]() {
        if (nbat > 0) {
            noc_async_read_barrier();
            cb_push_back(cb, chunk_tiles * nbat);
            pos += nbat;
            nbat = 0;
        }
    };
    for (uint32_t i = 0; i < n_blocks; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(a++);
        const uint32_t info = get_arg_val<uint32_t>(a++);
        const uint32_t n = info & 0xFFFF;
        if ((info >> 16) & 1) {
            ++relays;
            if (*relay_ready < relays) {
                push_batch();
                noc_semaphore_wait_min(relay_ready, relays);
            }
        } else if (!own_waited) {
            noc_semaphore_wait_min(own_loaded, own_total);
            own_waited = true;
        }
        const uint64_t src_base = get_noc_addr(xy >> 8, xy & 0xFF, landing_addr);
        for (uint32_t k = 0; k < n; ++k) {
            if (nbat == 0) {
                need = batch - pos % batch;
                cb_reserve_back(cb, chunk_tiles * need);
                wptr = get_write_ptr(cb);
            }
            noc_async_read(src_base + k * chunk_bytes, wptr + nbat * chunk_bytes, chunk_bytes);
            --chunks_left;
            if (++nbat == need || chunks_left == 0) {
                push_batch();
            }
        }
    }
    push_batch();
}
