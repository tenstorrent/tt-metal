// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm port reader (NCRISC / NoC0), one per (direction, link) per chip — TRANSPORT + OVERLAP.
//
// send_entry (per outgoing entry, in send order): own entries wait until every own member that this port reads
//   from has loaded its block (sem_own_loaded); relay entries wait until the upstream entry they forward has
//   landed (arrival counter >= threshold). The entry's runs are read from the compute cores' cb_x (mirror layout:
//   the same core / offset on every chip) into cb_port_stage, one slot per packet (<= chunk_pages tiles), in
//   batches of `group` slots (one read barrier + push per batch; a batch also ends at the entry end).
// notify_landed (serviced before every blocking point, so a final-hop landing is never starved by a relay wait or
//   by stage back-pressure): incoming entry e has landed once the arrival counter reaches e + 1 (one fused
//   increment on the entry's last packet); every member it covers gets one local increment of sem_landed.
// drain_arrivals: after every incoming entry is notified, the arrival counter is re-armed by decrementing it by
//   exactly what was consumed (global semaphore, 0 at every call boundary).
//
// Raw dataflow API (documented helper bypass): the kernel_lib ccl stream helpers are unmeasured for this pattern
// (op_design.md: "port sender ... copy the measured raw pattern"), and none of them polls two event sources.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_gather_rms_norm_common.hpp"

using namespace agrms_cb;

constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
constexpr uint32_t chunk_pages = get_compile_time_arg_val(1);  // tiles per packet slot
constexpr uint32_t group = get_compile_time_arg_val(2);        // packet slots per batch; the CB holds 2 batches
constexpr uint32_t sem_own_loaded_id = get_compile_time_arg_val(3);
constexpr uint32_t sem_landed_id = get_compile_time_arg_val(4);
constexpr uint32_t chunk_bytes = chunk_pages * tile_bytes;

FORCE_INLINE uint32_t arg(uint32_t i) { return get_arg_val<uint32_t>(i); }

void kernel_main() {
    const uint32_t arrival_addr = arg(0);
    const uint32_t own_expected = arg(1);
    const uint32_t num_out = arg(2);
    const uint32_t num_in = arg(3);
    uint32_t a = 4;
    // locate the incoming entries (after the outgoing ones)
    uint32_t in_ptr = a;
    for (uint32_t e = 0; e < num_out; ++e) {
        in_ptr += 2 + 2 * arg(in_ptr + 1);
    }

    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
    volatile tt_l1_ptr uint32_t* own_loaded =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own_loaded_id));
    const uint32_t landed_addr = get_semaphore(sem_landed_id);
    const uint32_t x_base = get_write_ptr(cb_x);  // identical on every core of every chip (cb_x is the first CB)

    uint32_t next_in = 0;
    auto service = [&]() {
        while (next_in < num_in) {
            invalidate_l1_cache();
            if (*arrived < next_in + 1) {
                return;
            }
            const uint32_t n = arg(in_ptr);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t xy = arg(in_ptr + 1 + i);
                noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, landed_addr), 1);
            }
            in_ptr += 1 + n;
            ++next_in;
        }
    };

    AGZ("pr_all");
    bool own_ready = own_expected == 0;
    uint32_t batch = 0, need = 0, pos = 0, wptr = 0;
    for (uint32_t e = 0; e < num_out; ++e) {
        const uint32_t wait = arg(a);
        const uint32_t n_runs = arg(a + 1);
        a += 2;
        if (wait == 0) {
            while (!own_ready) {
                service();
                invalidate_l1_cache();
                own_ready = *own_loaded >= own_expected;
            }
        } else {
            while (true) {
                service();
                invalidate_l1_cache();
                if (*arrived >= wait) {
                    break;
                }
            }
        }
        for (uint32_t r = 0; r < n_runs; ++r) {
            const uint32_t xy = arg(a + 2 * r);
            const uint32_t offn = arg(a + 2 * r + 1);
            const uint32_t off = offn & 0xFFFF;
            const uint32_t n = offn >> 16;
            for (uint32_t t = 0; t < n; t += chunk_pages) {
                const uint32_t pieces = (n - t) < chunk_pages ? (n - t) : chunk_pages;
                if (batch == 0) {
                    need = group - pos % group;
                    while (!cb_pages_reservable_at_back(cb_stage, chunk_pages * need)) {
                        service();
                    }
                    cb_reserve_back(cb_stage, chunk_pages * need);
                    wptr = get_write_ptr(cb_stage);
                }
                noc_async_read(
                    get_noc_addr(xy & 0xFFFF, xy >> 16, x_base + (off + t) * tile_bytes),
                    wptr + batch * chunk_bytes,
                    pieces * tile_bytes);
                if (++batch == need) {
                    noc_async_read_barrier();
                    cb_push_back(cb_stage, chunk_pages * batch);
                    pos += batch;
                    batch = 0;
                }
            }
        }
        a += 2 * n_runs;
        if (batch > 0) {  // an entry's tail is pushed at once: its fused increment must not wait for later entries
            noc_async_read_barrier();
            cb_push_back(cb_stage, chunk_pages * batch);
            pos += batch;
            batch = 0;
        }
    }
    while (next_in < num_in) {
        service();
    }
    if (num_in > 0) {
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - num_in);
    }
    noc_async_atomic_barrier();
}
