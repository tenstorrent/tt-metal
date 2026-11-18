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
#include "all_gather_rms_norm_kimi_perf.hpp"

constexpr uint32_t cb = get_compile_time_arg_val(0);
constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t batch = get_compile_time_arg_val(3);
constexpr uint32_t sem_own_loaded = get_compile_time_arg_val(5);
constexpr uint32_t sem_relay_ready = get_compile_time_arg_val(6);
constexpr uint32_t sem_landed = get_compile_time_arg_val(11);
// own-store gate: 0 off, 1 release each own block once its reads are issued, 2 release every own block of the
// link once the whole own entry's reads are issued
constexpr uint32_t own_store_gate = get_compile_time_arg_val(12);
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
#ifdef PSP_DIAG
    uint32_t t_first = 0;
#endif
    uint32_t d_bar = 0, d_res = 0, d_relay = 0, d_iss = 0, n_relaywait = 0;
    (void)d_bar, (void)d_res, (void)d_relay, (void)d_iss, (void)n_relaywait;
    auto push_batch = [&]() {
        if (nbat > 0) {
            PSP_T0(t);
            noc_async_read_barrier();
            PSP_ACC(d_bar, t);
            cb_push_back(cb, chunk_tiles * nbat);
            pos += nbat;
            nbat = 0;
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
                push_batch();
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
            if (nbat == 0) {
                need = batch - pos % batch;
                PSP_T0(t);
                cb_reserve_back(cb, chunk_tiles * need);
                PSP_ACC(d_res, t);
                wptr = get_write_ptr(cb);
            }
            {
                PSP_T0(t);
                noc_async_read(src_base + k * chunk_bytes, wptr + nbat * chunk_bytes, chunk_bytes);
                PSP_ACC(d_iss, t);
            }
            --chunks_left;
            if (++nbat == need || chunks_left == 0) {
                push_batch();
            }
        }
        // the own core may start its DRAM store once this port has issued the reads of its block (gate 1) or of the
        // whole own entry (gate 2). No data hazard (the store reads cb_out); the gate only keeps the store burst
        // off the bulk.
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
    push_batch();
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
