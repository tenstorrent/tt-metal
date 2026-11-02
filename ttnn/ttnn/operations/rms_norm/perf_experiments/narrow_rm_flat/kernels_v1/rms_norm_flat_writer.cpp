// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm flat writer (BRISC / NoC1) — owner combine of the per-unit partials + output tail.
//
//   wr_xchg_send   unit u's partial (cb_partial) -> slot (u / k) * k + m of owner (u % k)'s cb_gathered; flush, then
//                  one semaphore increment on the owner (same NoC + VC + destination => in-order delivery).
//   wr_wait_peers  owner: all k partials of its owned units arrived -> cb_gathered (compute combines).
//   wr_inv_return  owner: each combined inv tile -> slot u of EVERY member's cb_inv (k unicasts), then one
//                  semaphore increment per member (count = owned units).
//   wr_wait_inv    every unit's inv arrived (U increments) -> cb_inv.
//   wr_store       cb_out IS the output shard (compute packed it); the last partial flat tile is copied from
//                  cb_outtail into the shard (tail_bytes only — never past the shard end).
// Every member declares the identical CB list and reserves cb_gathered / cb_inv once at kernel start, so their
// write pointers (the slot bases peers address) are the same on all members.

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
constexpr uint32_t sem_inv_id = get_compile_time_arg_val(8);
constexpr uint32_t NOC_XY_ARG_BASE = get_compile_time_arg_val(9);

FORCE_INLINE uint64_t peer_addr(uint32_t peer, uint32_t l1_addr) {
    return get_noc_addr(
        get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * peer),
        get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * peer + 1),
        l1_addr);
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t member_idx = get_arg_val<uint32_t>(1);
    const uint32_t owned = get_arg_val<uint32_t>(2);
    constexpr uint32_t T = acc_tile_bytes;

    const uint32_t sem_part = get_semaphore(sem_part_id);
    const uint32_t sem_inv = get_semaphore(sem_inv_id);
    cb_reserve_back(cb_gathered, o_max * K);
    const uint32_t g_base = get_write_ptr(cb_gathered);
    cb_reserve_back(cb_inv, U);
    const uint32_t inv_base = get_write_ptr(cb_inv);

    {
        MaybeDeviceZoneScope("wr_xchg_send");  // incl. waits on compute's pass 1
        for (uint32_t u = 0; u < U; ++u) {
            const uint32_t owner = u % K;
            const uint32_t j = u / K;
            cb_wait_front(cb_partial, 1);
            noc_async_write(get_read_ptr(cb_partial), peer_addr(owner, g_base + (j * K + member_idx) * T), T);
            noc_async_writes_flushed();
            noc_semaphore_inc(peer_addr(owner, sem_part), 1);
            cb_pop_front(cb_partial, 1);
        }
    }

    if (owned > 0) {
        {
            MaybeDeviceZoneScope("wr_wait_peers");
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_part), owned * K);
        }
        cb_push_back(cb_gathered, owned * K);
        MaybeDeviceZoneScope("wr_inv_return");  // incl. waits on compute's combine
        for (uint32_t j = 0; j < owned; ++j) {
            const uint32_t u = member_idx + j * K;
            cb_wait_front(cb_inv_owned, 1);
            const uint32_t src = get_read_ptr(cb_inv_owned);
            for (uint32_t p = 0; p < K; ++p) {
                noc_async_write(src, peer_addr(p, inv_base + u * T), T);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_inv_owned, 1);
        }
        for (uint32_t p = 0; p < K; ++p) {
            noc_semaphore_inc(peer_addr(p, sem_inv), owned);
        }
    }

    {
        MaybeDeviceZoneScope("wr_wait_inv");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_inv), U);
    }
    cb_push_back(cb_inv, U);

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
