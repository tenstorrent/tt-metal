// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Port reader (RISCV_1, NoC0), one per (link, direction). Adapted from examples/fabric_all_gather's port reader: the
// chunk walk is over compute-core `landing` blocks (L1), not DRAM pages. Its stream: entry 0 = the own slice's row
// blocks of this link (read once every own core of the link has loaded: own_loaded, a fan-in counter waited for its
// total), then the relay entries (each block read once the receiver says it landed: relay_ready, one ordered sender).
// The trailing n_tail blocks are read chunk-major (chunk 0 of every tail block, then chunk 1, ...), the sender's order
// (Perf 2, late_tail_interleave).
// The supply is a rolling pipeline (Perf 2, port_supply_pipeline): every chunk read carries its own NoC transaction
// id (1..15 by stream position), up to `depth` reads stay in flight, and each chunk is pushed to the sender on its
// own, in stream order, as soon as ITS read has completed, so the sender starts on chunk j while chunks j+1.. are
// still reading and the reader never idles in a whole-batch barrier. Chunk j sits in CB slot j % cb_chunks (the
// sender pops in batches of `batch`, a divisor of cb_chunks, so its batch never wraps). While the CB has no room for
// every in-flight chunk plus the next, the oldest in-flight chunk is pushed (the sender may be waiting for it to
// complete its batch). Before a relay wait that would block, every issued read is pushed (an unpushed chunk would
// otherwise sit behind data that has not landed yet).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_gather_rms_norm_kimi_perf.hpp"

constexpr uint32_t cb = get_compile_time_arg_val(0);
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t sem_own_loaded = get_compile_time_arg_val(5);
constexpr uint32_t sem_relay_ready = get_compile_time_arg_val(6);
constexpr uint32_t sem_landed = get_compile_time_arg_val(11);
// own-store gate: 0 off, 1 release each own block once its reads are issued, 2 release every own block of the
// link once the whole own entry's reads are issued
constexpr uint32_t own_store_gate = get_compile_time_arg_val(12);
constexpr uint32_t cb_chunks = get_compile_time_arg_val(13);  // cb_port_chunks capacity in chunks
constexpr uint32_t depth = get_compile_time_arg_val(14);      // chunk reads in flight (one trid each)
constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
constexpr uint32_t NUM_TRIDS = 15;  // trids 1..15; trid 0 stays the untagged default
static_assert(depth >= 1 && depth <= NUM_TRIDS && depth <= cb_chunks, "port reader: reads in flight");

void kernel_main() {
    size_t a = 0;
    const uint32_t landing_addr = get_arg_val<uint32_t>(a++);
    const uint32_t own_total = get_arg_val<uint32_t>(a++);  // own cores of my link (each increments me once)
    uint32_t chunks_left = get_arg_val<uint32_t>(a++);
    const uint32_t n_blocks = get_arg_val<uint32_t>(a++);
    const uint32_t n_tail = get_arg_val<uint32_t>(a++);  // trailing blocks read chunk-major (the sender's order)
    const uint32_t blocks_idx = a;                       // then per block: core xy, n_chunks | relay << 16

    volatile tt_l1_ptr uint32_t* own_loaded =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own_loaded));
    volatile tt_l1_ptr uint32_t* relay_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_relay_ready));

    const uint32_t cb_base = get_write_ptr(cb);  // fifo start: the CB interface is re-initialized every launch
    uint32_t issued = 0, pushed = 0, relays = 0;
    uint32_t trid_issue = 1, trid_retire = 1;
    bool own_waited = false;
    // push the oldest in-flight chunk once its read has completed
    auto retire_one = [&]() __attribute__((always_inline)) {
        noc_async_read_barrier_with_trid(trid_retire);
        cb_push_back(cb, chunk_tiles);
        ++pushed;
        trid_retire = (trid_retire == NUM_TRIDS) ? 1 : trid_retire + 1;
    };
    // push every already-completed chunk at the front of the window (never blocks)
    auto retire_done = [&]() __attribute__((always_inline)) {
        while (pushed < issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid_retire)) {
            cb_push_back(cb, chunk_tiles);
            ++pushed;
            trid_retire = (trid_retire == NUM_TRIDS) ? 1 : trid_retire + 1;
        }
    };
    auto drain = [&]() {
        while (pushed < issued) {
            retire_one();
        }
    };
    // a block's first visit: wait until it is readable (own: every own core of the link loaded; relay: landed)
    auto open_block = [&](uint32_t info) __attribute__((always_inline)) {
        if ((info >> 16) & 1) {
            ++relays;
            if (*relay_ready < relays) {
                drain();
                noc_semaphore_wait_min(relay_ready, relays);
            }
        } else if (!own_waited) {
            MaybeDeviceZoneScope("preader_own_wait");
            noc_semaphore_wait_min(own_loaded, own_total);
            own_waited = true;
        }
    };
    auto read_chunk = [&](uint32_t xy, uint32_t k) __attribute__((always_inline)) {
        retire_done();
        if (issued - pushed == depth) {
            retire_one();
        }
        // room for every in-flight chunk + this one; while there is none, push the oldest in-flight chunk (the sender
        // pops in batches: holding finished chunks back here would deadlock against a sender waiting for the rest of
        // its batch)
        while (!cb_pages_reservable_at_back(cb, chunk_tiles * (issued - pushed + 1))) {
            if (pushed < issued) {
                retire_one();
            }
        }
        cb_reserve_back(cb, chunk_tiles * (issued - pushed + 1));
        noc_async_read_set_trid(trid_issue);
        noc_async_read(
            get_noc_addr(xy >> 8, xy & 0xFF, landing_addr) + k * chunk_bytes,
            cb_base + (issued % cb_chunks) * chunk_bytes,
            chunk_bytes);
        // the NIU has taken the request once the cmd buf is free again: from here on the trid's outstanding count
        // covers this read, so a trid poll can never pass before it was counted
        while (!noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
        }
        ++issued;
        trid_issue = (trid_issue == NUM_TRIDS) ? 1 : trid_issue + 1;
        --chunks_left;
    };
    // the own core may start its DRAM store once this port has issued the reads of its block (gate 1) or of the
    // whole own entry (gate 2). No data hazard (the store reads cb_out); the gate only keeps the store burst off the
    // bulk. A tail block's reads are complete only after the last round.
    auto own_done = [&](uint32_t i, bool own_end) __attribute__((always_inline)) {
        const uint32_t xy = get_arg_val<uint32_t>(blocks_idx + 2 * i);
        const bool relay = (get_arg_val<uint32_t>(blocks_idx + 2 * i + 1) >> 16) & 1;
        if constexpr (own_store_gate == 1) {
            if (!relay) {
                noc_semaphore_inc(get_noc_addr(xy >> 8, xy & 0xFF, get_semaphore(sem_landed)), 1);
            }
        } else if constexpr (own_store_gate == 2) {
            if (!relay && own_end) {
                for (uint32_t j = 0; j <= i; ++j) {
                    const uint32_t oxy = get_arg_val<uint32_t>(blocks_idx + 2 * j);
                    noc_semaphore_inc(get_noc_addr(oxy >> 8, oxy & 0xFF, get_semaphore(sem_landed)), 1);
                }
            }
        }
    };
    const uint32_t n_head = n_blocks - n_tail;
    for (uint32_t i = 0; i < n_head; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(blocks_idx + 2 * i);
        const uint32_t info = get_arg_val<uint32_t>(blocks_idx + 2 * i + 1);
        open_block(info);
        for (uint32_t k = 0; k < (info & 0xFFFF); ++k) {
            read_chunk(xy, k);
        }
        own_done(i, i + 1 == n_blocks || ((get_arg_val<uint32_t>(blocks_idx + 2 * i + 3) >> 16) & 1));
    }
    for (uint32_t c = 0; chunks_left > 0; ++c) {
        for (uint32_t i = n_head; i < n_blocks; ++i) {
            const uint32_t xy = get_arg_val<uint32_t>(blocks_idx + 2 * i);
            const uint32_t info = get_arg_val<uint32_t>(blocks_idx + 2 * i + 1);
            if (c == 0) {
                open_block(info);
            }
            if ((info & 0xFFFF) > c) {
                read_chunk(xy, c);
            }
        }
    }
    for (uint32_t i = n_head; i < n_blocks; ++i) {
        own_done(i, i + 1 == n_blocks);
    }
    {
        MaybeDeviceZoneScope("preader_tail_drain");
        drain();
    }
    noc_async_read_set_trid(0);
    noc_async_atomic_barrier();
}
