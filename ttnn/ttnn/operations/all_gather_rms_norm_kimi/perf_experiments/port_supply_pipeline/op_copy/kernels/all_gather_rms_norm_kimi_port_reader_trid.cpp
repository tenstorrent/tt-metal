// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Port reader (RISCV_1, NoC0), one per (link, direction) -- port_supply_pipeline variant "trid".
// Same stream as the baseline (entry 0 = own row blocks of this link after own_loaded, then the relay entries, each
// block once relay_ready covers it), but the supply is a rolling pipeline instead of read-batch / barrier / push:
// every chunk read carries its own NoC transaction id (1..15, by stream position), up to `depth` reads stay in
// flight, and each chunk is pushed to the sender (one chunk at a time, in stream order) as soon as ITS read has
// completed (noc_async_read_barrier_with_trid), so the sender starts on chunk j while chunks j+1.. are still reading
// and the reader never idles in a whole-batch barrier. Chunk j sits in CB slot j % cb_chunks (the CB is a whole
// number of chunks; the sender still pops in batches of `batch`, a divisor of cb_chunks). Before a relay wait that
// would block, every issued read is retired and pushed (an unpushed chunk would otherwise sit behind data that has
// not landed yet).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_gather_rms_norm_kimi_perf.hpp"

constexpr uint32_t cb = get_compile_time_arg_val(0);
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t sem_own_loaded = get_compile_time_arg_val(5);
constexpr uint32_t sem_relay_ready = get_compile_time_arg_val(6);
constexpr uint32_t sem_landed = get_compile_time_arg_val(11);
constexpr uint32_t own_store_gate = get_compile_time_arg_val(12);
constexpr uint32_t cb_chunks = get_compile_time_arg_val(13);
constexpr uint32_t depth = get_compile_time_arg_val(14);  // reads in flight, <= 15 (one trid each), <= cb_chunks
constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
constexpr uint32_t NUM_TRIDS = 15;  // trids 1..15; trid 0 stays the untagged default
static_assert(depth >= 1 && depth <= NUM_TRIDS && depth <= cb_chunks, "trid reader depth");

void kernel_main() {
    size_t a = 0;
    const uint32_t landing_addr = get_arg_val<uint32_t>(a++);
    const uint32_t own_total = get_arg_val<uint32_t>(a++);
    const uint32_t chunks_total = get_arg_val<uint32_t>(a++);
    const uint32_t n_blocks = get_arg_val<uint32_t>(a++);

    volatile tt_l1_ptr uint32_t* own_loaded =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own_loaded));
    volatile tt_l1_ptr uint32_t* relay_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_relay_ready));

    const uint32_t cb_base = get_write_ptr(cb);  // fifo start: the CB interface is re-initialized every launch
    uint32_t issued = 0, pushed = 0, relays = 0;
    uint32_t trid_issue = 1, trid_retire = 1;
    bool own_waited = false;
#ifdef PSP_DIAG
    uint32_t t_first = 0;
#endif
    uint32_t d_bar = 0, d_res = 0, d_relay = 0, d_iss = 0, n_relaywait = 0;
    (void)d_bar, (void)d_res, (void)d_relay, (void)d_iss, (void)n_relaywait;

    // retire the oldest in-flight chunk: wait for its read, push it
    auto retire_one = [&]() {
        PSP_T0(t);
        noc_async_read_barrier_with_trid(trid_retire);
        PSP_ACC(d_bar, t);
        cb_push_back(cb, chunk_tiles);
        ++pushed;
        trid_retire = (trid_retire == NUM_TRIDS) ? 1 : trid_retire + 1;
    };
    // push every already-completed chunk at the front of the window (never blocks)
    auto retire_done = [&]() {
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

    for (uint32_t i = 0; i < n_blocks; ++i) {
        const uint32_t xy = get_arg_val<uint32_t>(a++);
        const uint32_t info = get_arg_val<uint32_t>(a++);
        const uint32_t n = info & 0xFFFF;
        const bool relay = (info >> 16) & 1;
        if (relay) {
            ++relays;
            if (*relay_ready < relays) {
                drain();
                PSP_T0(t);
                noc_semaphore_wait_min(relay_ready, relays);
                PSP_ACC(d_relay, t);
#ifdef PSP_DIAG
                ++n_relaywait;
#endif
            }
        } else if (!own_waited) {
            MaybeDeviceZoneScope("preader_own_wait");
            noc_semaphore_wait_min(own_loaded, own_total);
            own_waited = true;
        }
#ifdef PSP_DIAG
        if (i == 0) {
            t_first = get_timestamp_32b();
        }
#endif
        const uint64_t src_base = get_noc_addr(xy >> 8, xy & 0xFF, landing_addr);
        for (uint32_t k = 0; k < n; ++k) {
            retire_done();
            if (issued - pushed == depth) {
                retire_one();
            }
            {
                // room for every in-flight chunk + this one. While there is none, push the oldest in-flight chunk
                // (its read completes on its own): the sender pops in batches, so holding finished chunks back while
                // waiting for space would deadlock against a sender waiting for the rest of its batch.
                PSP_T0(t);
                while (!cb_pages_reservable_at_back(cb, chunk_tiles * (issued - pushed + 1))) {
                    if (pushed < issued) {
                        retire_one();
                    }
                }
                cb_reserve_back(cb, chunk_tiles * (issued - pushed + 1));
                PSP_ACC(d_res, t);
            }
            const uint32_t slot = issued % cb_chunks;
            {
                PSP_T0(t);
                noc_async_read_set_trid(trid_issue);
                noc_async_read(src_base + k * chunk_bytes, cb_base + slot * chunk_bytes, chunk_bytes);
                // the NIU has taken the (last) request once the cmd buf is free again: from here on the trid's
                // outstanding count covers this read, so a trid poll can never pass before it was counted
                while (!noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
                }
                PSP_ACC(d_iss, t);
            }
            ++issued;
            trid_issue = (trid_issue == NUM_TRIDS) ? 1 : trid_issue + 1;
        }
        if constexpr (own_store_gate == 1) {
            if (!relay) {
                noc_semaphore_inc(get_noc_addr(xy >> 8, xy & 0xFF, get_semaphore(sem_landed)), 1);
            }
        } else if constexpr (own_store_gate == 2) {
            const bool own_end = !relay && (i + 1 == n_blocks || ((get_arg_val<uint32_t>(a + 1) >> 16) & 1));
            if (own_end) {
                for (uint32_t j = 0, aa = 4; j <= i; ++j, aa += 2) {
                    const uint32_t oxy = get_arg_val<uint32_t>(aa);
                    noc_semaphore_inc(get_noc_addr(oxy >> 8, oxy & 0xFF, get_semaphore(sem_landed)), 1);
                }
            }
        }
    }
    drain();
    noc_async_read_set_trid(0);
    (void)chunks_total;
#ifdef PSP_DIAG
    {
        uint32_t d_tot = 0;
        PSP_ACC(d_tot, t_first);
        PSP_EMIT("psp_r_total", d_tot);
        PSP_EMIT("psp_r_bar", d_bar);
        PSP_EMIT("psp_r_res", d_res);
        PSP_EMIT("psp_r_relay", d_relay);
        PSP_EMIT("psp_r_iss", d_iss);
        PSP_EMIT("psp_r_nrelaywait", n_relaywait);
    }
#endif
    noc_async_atomic_barrier();
}
