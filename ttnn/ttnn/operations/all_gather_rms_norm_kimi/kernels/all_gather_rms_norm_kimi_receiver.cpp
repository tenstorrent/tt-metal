// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Receiver (RISCV_0, NoC1), one per incoming (link, direction).                                     [OVERLAP]
// Walks the upstream port's static chunk stream (host-counted): when the arrival counter reaches chunk k's threshold
// (the upstream increments every INC_EVERY chunks and at each entry end), the chunk has landed in its compute core's
// `landing`: landed++ on that core (each core's counter has exactly this one sender, so it counts its chunks in
// order). For a block the local port relays onward, relay_ready++ on that port after the block's last chunk. Then it
// waits for every increment of the call and re-arms the counter by what it consumed (never by setting 0).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_gather_rms_norm_kimi_perf.hpp"

constexpr uint32_t sem_landed = get_compile_time_arg_val(0);
constexpr uint32_t sem_relay_ready = get_compile_time_arg_val(1);

void kernel_main() {
    size_t a = 0;
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);
    const uint32_t expect_in = get_arg_val<uint32_t>(a++);
    const uint32_t port_xy = get_arg_val<uint32_t>(a++);
    const uint32_t n_events = get_arg_val<uint32_t>(a++);  // per chunk: threshold, core xy | relay << 16

    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
    const uint32_t landed_addr = get_semaphore(sem_landed);
    const uint64_t relay_noc = get_noc_addr(port_xy >> 8, port_xy & 0xFF, get_semaphore(sem_relay_ready));
    {
        MaybeDeviceZoneScope("recv_notify");
        for (uint32_t e = 0; e < n_events; ++e) {
            const uint32_t threshold = get_arg_val<uint32_t>(a++);
            const uint32_t info = get_arg_val<uint32_t>(a++);
            noc_semaphore_wait_min(arrived, threshold);
            noc_semaphore_inc(get_noc_addr((info >> 8) & 0xFF, info & 0xFF, landed_addr), 1);
            if ((info >> 16) & 1) {
                noc_semaphore_inc(relay_noc, 1);
            }
        }
    }  // recv_notify
    if (expect_in > 0) {
        MaybeDeviceZoneScope("recv_drain");
        noc_semaphore_wait_min(arrived, expect_in);
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - expect_in);
    }
    noc_async_atomic_barrier();
}
