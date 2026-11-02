// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm flat writer (BRISC / NoC1) — pipelined owner combine of the per-unit partials + output tail.
//
// Per unit u (in order), interleaved with compute's pass 1:
//   send     unit u's partial (cb_partial) -> slot (u / k) * k + m of owner (u % k)'s cb_gathered; flush, then one
//            increment of the owner's partials semaphore (same NoC + VC + destination => in-order delivery).
//            Throttle: unit u is sent only after unit u - k's inv arrived here. Each owner owns units o, o + k, ...,
//            so this keeps every member's sends to one owner at most one owned unit ahead of that owner's return,
//            which makes the owner's monotonic count exact: owned unit j is complete at (j + 1) * k arrivals.
//   return   (owner, unit o = u - lag): wait for o's k partials -> cb_gathered; compute combines it (pipelined into
//            its pass 1 at the same point); the inv tile -> slot o of EVERY member's cb_inv (k unicasts), flush, then
//            one increment of each member's flag[o].
// Then per unit u: wait flag[u] -> push one cb_inv page (compute's pass 2 waits per unit).
// flag[u] lives in this member's cb_flags (16 B per unit, not a host-initialized semaphore): zeroed here before the
// first send. Safe: flag[u] is incremented by u's owner only after it received this member's partial of u.
// wr_store: cb_out IS the output shard (compute packed it); the last partial flat tile is copied from cb_outtail into
// the shard (tail_bytes only — never past the shard end).
// Every member declares the identical CB list and reserves cb_gathered / cb_inv / cb_flags once at kernel start, so
// their write pointers (the slot bases peers address) are the same on all members.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "rms_norm_flat_common.hpp"

using namespace rms_norm_flat_cb;

constexpr uint32_t F_full = get_compile_time_arg_val(0);
constexpr uint32_t tail_bytes = get_compile_time_arg_val(1);
constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t U = get_compile_time_arg_val(3);
constexpr uint32_t K = get_compile_time_arg_val(4);
constexpr uint32_t o_max = get_compile_time_arg_val(5);
constexpr uint32_t acc_tile_bytes = get_compile_time_arg_val(6);
constexpr uint32_t sem_part_id = get_compile_time_arg_val(7);
constexpr uint32_t flag_bytes = get_compile_time_arg_val(8);
constexpr uint32_t NOC_XY_ARG_BASE = get_compile_time_arg_val(9);
constexpr uint32_t LAG = get_compile_time_arg_val(10);

FORCE_INLINE uint64_t peer_addr(uint32_t peer, uint32_t l1_addr) {
    return get_noc_addr(
        get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * peer),
        get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * peer + 1),
        l1_addr);
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* flag_ptr(uint32_t flags_base, uint32_t u) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags_base + u * flag_bytes);
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t member_idx = get_arg_val<uint32_t>(1);
    [[maybe_unused]] const uint32_t owned = get_arg_val<uint32_t>(2);
    constexpr uint32_t T = acc_tile_bytes;

    const uint32_t sem_part = get_semaphore(sem_part_id);
    cb_reserve_back(cb_flags, U);
    const uint32_t flags_base = get_write_ptr(cb_flags);
    for (uint32_t u = 0; u < U; ++u) {
        *flag_ptr(flags_base, u) = 0;
    }
    const uint32_t g_base = get_write_ptr(cb_gathered);  // initial write pointer (the reader produces cb_gathered)
    cb_reserve_back(cb_inv, U);
    const uint32_t inv_base = get_write_ptr(cb_inv);

    {
        MaybeDeviceZoneScope("wr_xchg_send");  // partial sends (incl. waits on compute's pass 1 + the throttle)
        for (uint32_t u = 0; u < U; ++u) {
            if (u >= K) {
                noc_semaphore_wait_min(flag_ptr(flags_base, u - K), 1);
            }
            cb_wait_front(cb_partial, 1);
            const uint32_t owner = u % K;
            noc_async_write(get_read_ptr(cb_partial), peer_addr(owner, g_base + ((u / K) * K + member_idx) * T), T);
            noc_async_writes_flushed();
            noc_semaphore_inc(peer_addr(owner, sem_part), 1);
            cb_pop_front(cb_partial, 1);
        }
    }

    {
        MaybeDeviceZoneScope("wr_wait_inv");
        for (uint32_t u = 0; u < U; ++u) {
            noc_semaphore_wait_min(flag_ptr(flags_base, u), 1);
            cb_push_back(cb_inv, 1);
        }
    }

    MaybeDeviceZoneScope("wr_store");
    if constexpr (F_full > 0) {
        cb_wait_front(cb_out, F_full);
        cb_pop_front(cb_out, F_full);
    }
    if constexpr (tail_bytes > 0) {
        cb_wait_front(cb_outtail, 1);
        noc_async_write(get_read_ptr(cb_outtail), get_noc_addr(out_addr + F_full * out_tile_bytes), tail_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_outtail, 1);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
