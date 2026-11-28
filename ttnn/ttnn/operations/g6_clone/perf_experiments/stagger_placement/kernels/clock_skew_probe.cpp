// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Clock-skew causal probe (g6_clone perf1, stagger_placement STEP A).
//
// Question: the profiler shows each core's KERNEL zone starting at a fixed offset set by its physical
// column (up to ~600 cycles). Is that a real late start, or an offset of the per-tile wall clock?
// Every timestamp here is read on the core's OWN clock (get_timestamp, the counter the profiler reads).
//
// One reference core R (role 0, index 0) and N-1 others B_i (role 1, index i). Semaphores, all 0 at launch:
//   SEM_START  on R: each B increments it at its kernel entry
//   SEM_RSTART on B: R increments it on every B at its kernel entry
//   SEM_PING   on B: R's ping in the ping-pong phase
//   SEM_PONG   on R: B's answer
// Records (64-bit timestamps, two words each) in the uint32 output, 16 words per 64 B page:
//   page i (written by core i)  R: [t0, t_rstart_issued, n_arrival_records]
//                               B: [t0, t_rstart_seen, t_ping_seen, t_pong_sent]
//   page N+i (written by R)     [t_ping_sent_i, t_pong_seen_i, t_arrival_i, value_arrival_i]
//     t_arrival_k / value_arrival_k: R's clock when SEM_START first read value_arrival_k (k = i-1).
// The host derives, per B: the NTP offset theta = ((t_ping_seen - t_ping_sent) + (t_pong_sent - t_pong_seen))/2
// (B clock minus R clock, exact up to half the route asymmetry) and the true start delay
// (t0_B - theta) - t0_R, to compare with the raw (profiler-style) difference t0_B - t0_R.
// The handshake waits test the same thing without the symmetric-latency assumption:
//   real delay D of B   -> R waits ~D + latency for B's start; B finds R's increment already there
//   clock offset only   -> both wait ~ one latency.
// Raw dataflow by design: a timing probe, not an op; no helper exists for timestamped semaphore spins.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t SEM_START = 0;
constexpr uint32_t SEM_RSTART = 1;
constexpr uint32_t SEM_PING = 2;
constexpr uint32_t SEM_PONG = 3;
constexpr uint32_t PAGE_WORDS = 16;
constexpr uint32_t PAGE_BYTES = 64;
// Cycles R idles after the last start arrival before the ping-pong, so every B is polling.
constexpr uint64_t SETTLE_CYCLES = 4000;

FORCE_INLINE volatile tt_l1_ptr uint32_t* sem_ptr(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

FORCE_INLINE void put64(volatile tt_l1_ptr uint32_t* row, uint32_t slot, uint64_t value) {
    row[2 * slot] = static_cast<uint32_t>(value);
    row[2 * slot + 1] = static_cast<uint32_t>(value >> 32);
}
}  // namespace

void kernel_main() {
    const uint64_t t0 = get_timestamp();
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();

    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t my_idx = get_arg_val<uint32_t>(1);
    const uint32_t num_cores = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t ref_x = get_arg_val<uint32_t>(4);
    const uint32_t ref_y = get_arg_val<uint32_t>(5);
    // 1: the start handshake runs both ways (R increments SEM_RSTART on every B); 0: B -> R only, so R's
    // arrival records are not delayed by R issuing N-1 increments.
    const uint32_t both_ways = get_arg_val<uint32_t>(6);
    // R only: (x, y) of B_1 .. B_{N-1} from arg ARG_CORES.
    constexpr uint32_t ARG_CORES = 7;

    const auto out = TensorAccessor(out_args, out_addr, PAGE_BYTES);
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scratch));
    const uint32_t scratch_addr = get_write_ptr(cb_scratch);

    if (role == 1) {
        noc_semaphore_inc(get_noc_addr(ref_x, ref_y, get_semaphore(SEM_START)), 1);
        uint64_t t_rstart_seen = 0;
        if (both_ways) {
            volatile tt_l1_ptr uint32_t* rstart = sem_ptr(SEM_RSTART);
            while (*rstart < 1) {
            }
            t_rstart_seen = get_timestamp();
        }
        volatile tt_l1_ptr uint32_t* ping = sem_ptr(SEM_PING);
        while (*ping < 1) {
        }
        const uint64_t t_ping_seen = get_timestamp();
        noc_semaphore_inc(get_noc_addr(ref_x, ref_y, get_semaphore(SEM_PONG)), 1);
        const uint64_t t_pong_sent = get_timestamp();
        noc_async_atomic_barrier();
        for (uint32_t w = 0; w < PAGE_WORDS; ++w) {
            scratch[w] = 0;
        }
        put64(scratch, 0, t0);
        put64(scratch, 1, t_rstart_seen);
        put64(scratch, 2, t_ping_seen);
        put64(scratch, 3, t_pong_sent);
        noc_async_write(scratch_addr, out.get_noc_addr(my_idx), PAGE_BYTES);
        noc_async_write_barrier();
        return;
    }

    // Reference core.
    if (both_ways) {
        for (uint32_t i = 1; i < num_cores; ++i) {
            const uint32_t x = get_arg_val<uint32_t>(ARG_CORES + 2 * (i - 1));
            const uint32_t y = get_arg_val<uint32_t>(ARG_CORES + 1 + 2 * (i - 1));
            noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(SEM_RSTART)), 1);
        }
    }
    const uint64_t t_rstart_issued = get_timestamp();
    // Scratch is not cleared (that would delay the arrival records); the host reads only written words.
    volatile tt_l1_ptr uint32_t* start = sem_ptr(SEM_START);
    uint32_t seen = 0;
    uint32_t records = 0;
    uint64_t t_last = t0;
    while (seen < num_cores - 1) {
        const uint32_t value = *start;
        if (value != seen) {
            t_last = get_timestamp();
            volatile tt_l1_ptr uint32_t* row = scratch + (num_cores + 1 + records) * PAGE_WORDS;
            put64(row, 2, t_last);
            row[6] = value;
            seen = value;
            ++records;
        }
    }
    while (get_timestamp() < t_last + SETTLE_CYCLES) {
    }

    volatile tt_l1_ptr uint32_t* pong = sem_ptr(SEM_PONG);
    for (uint32_t i = 1; i < num_cores; ++i) {
        const uint32_t x = get_arg_val<uint32_t>(ARG_CORES + 2 * (i - 1));
        const uint32_t y = get_arg_val<uint32_t>(ARG_CORES + 1 + 2 * (i - 1));
        const uint64_t t_ping_sent = get_timestamp();
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(SEM_PING)), 1);
        while (*pong < i) {
        }
        const uint64_t t_pong_seen = get_timestamp();
        volatile tt_l1_ptr uint32_t* row = scratch + (num_cores + i) * PAGE_WORDS;
        put64(row, 0, t_ping_sent);
        put64(row, 1, t_pong_seen);
    }
    noc_async_atomic_barrier();

    put64(scratch, 0, t0);
    put64(scratch, 1, t_rstart_issued);
    scratch[4] = records;
    scratch[5] = 0;
    noc_async_write(scratch_addr, out.get_noc_addr(0), PAGE_BYTES);
    for (uint32_t i = 1; i < num_cores; ++i) {
        noc_async_write(scratch_addr + (num_cores + i) * PAGE_BYTES, out.get_noc_addr(num_cores + i), PAGE_BYTES);
    }
    noc_async_write_barrier();
}
