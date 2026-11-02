// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm flat writer (BRISC / NoC1) — partial sends of the owner combine + per-unit inv arrival + output tail.
//
// Exchange sub-units: each unit's partial tile (cb_partial) is cut into Q sub-blocks of T / Q bytes (1024 / Q
// consecutive datums); sub-unit v = u * Q + s is owned by member v % k (its j = v / k -th owned sub-unit).
//   send      sub-block s of unit u -> owner's cb_gathered tile j * KT + m / Q, sub-block m % Q (KT = ceil(k / Q)
//             gathered tiles per owned sub-unit: the k contributions packed densely); flush, then one increment of the
//             owner's partials semaphore (same NoC + VC + destination => in-order delivery).
//             Throttle: sub-unit v is sent only after sub-unit v - k's inv arrived here. The owner of v owns v - k
//             too, so every member's sends to one owner stay at most one owned sub-unit ahead of that owner's
//             return: the owner's monotonic count is exact (owned sub-unit j complete at (j + 1) * k arrivals).
//   inv       the owners (their READER) write each combined sub-block into this member's cb_inv tile u, sub-block s,
//             then set / increment flag[v] (cb_flags, 16 B per sub-unit). Per unit u: wait the Q flags -> push one
//             cb_inv page (compute's pass 2 waits per unit).
// flag[v] is not a host-initialized semaphore: zeroed here before the first send. Safe: flag[v] is written by v's
// owner only after it received this member's contribution to v.
// wr_store: cb_out IS the output shard (compute packed it); the last partial flat tile is copied from cb_outtail into
// the shard (tail_bytes only — never past the shard end).
// cb_gathered / cb_inv / cb_flags are addressed by their initial write pointers (identical CB lists on all members).

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
constexpr uint32_t Q = get_compile_time_arg_val(5);
constexpr uint32_t acc_tile_bytes = get_compile_time_arg_val(6);
constexpr uint32_t sem_part_id = get_compile_time_arg_val(7);
constexpr uint32_t flag_bytes = get_compile_time_arg_val(8);
constexpr uint32_t NOC_XY_ARG_BASE = get_compile_time_arg_val(9);
constexpr uint32_t KT = get_compile_time_arg_val(10);
constexpr uint32_t RET_MODE = get_compile_time_arg_val(11);  // 2: inv arrives per unit (flag[u * Q]), else per sub-unit
constexpr uint32_t V = U * Q;
constexpr uint32_t T = acc_tile_bytes;
constexpr uint32_t SB = T / Q;  // sub-block bytes

FORCE_INLINE uint64_t peer_addr(uint32_t peer, uint32_t l1_addr) {
    return get_noc_addr(
        get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * peer),
        get_arg_val<uint32_t>(NOC_XY_ARG_BASE + 2 * peer + 1),
        l1_addr);
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* flag_ptr(uint32_t flags_base, uint32_t v) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags_base + v * flag_bytes);
}

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t member_idx = get_arg_val<uint32_t>(1);

    const uint32_t sem_part = get_semaphore(sem_part_id);
    const uint32_t flags_base = get_write_ptr(cb_flags);
    for (uint32_t v = 0; v < V + U; ++v) {  // sub-unit flags + the leaders' per-unit collect counters
        *flag_ptr(flags_base, v) = 0;
    }
    const uint32_t g_base = get_write_ptr(cb_gathered);  // initial write pointer (the reader produces cb_gathered)
    cb_reserve_back(cb_inv, U);
    // my contribution's offset inside an owner's gathered slot
    const uint32_t my_ofs = (member_idx / Q) * T + (member_idx % Q) * SB;

    // cb_inv is pushed in unit order as each unit's inv arrives — also WHILE the send loop spins on its throttle:
    // compute defers its last owned combines into pass 2, so the throttle of a late sub-unit can depend on compute
    // having consumed earlier units' inv (a push held back until the sends finish would deadlock).
    uint32_t inv_pushed = 0;
    auto unit_ready = [&](uint32_t u) {
        for (uint32_t s = 0; s < (RET_MODE == 2 ? 1 : Q); ++s) {
            if (*flag_ptr(flags_base, u * Q + s) == 0) {
                return false;
            }
        }
        return true;
    };
    auto push_ready_inv = [&]() {
        while (inv_pushed < U && unit_ready(inv_pushed)) {
            cb_push_back(cb_inv, 1);
            ++inv_pushed;
        }
    };
    {
        MaybeDeviceZoneScope("wr_xchg_send");  // partial sends (incl. waits on compute's pass 1 + the throttle)
        for (uint32_t u = 0; u < U; ++u) {
            cb_wait_front(cb_partial, 1);
            const uint32_t src = get_read_ptr(cb_partial);
            // throttle: the unit's last sub-unit has the latest predecessor (flags are set in unit order per owner)
            for (uint32_t s = 0; s < Q; ++s) {
                const uint32_t v = u * Q + s;
                if (v >= K) {
                    // RET_MODE 2: sub-unit v - K's inv is known here only as part of its unit (flag[unit * Q])
                    const uint32_t w = RET_MODE == 2 ? ((v - K) / Q) * Q : v - K;
                    while (*flag_ptr(flags_base, w) == 0) {
                        push_ready_inv();
                        invalidate_l1_cache();
                    }
                }
            }
            // the unit's Q contributions, ONE flush, then the Q owner increments (each after its own data)
            for (uint32_t s = 0; s < Q; ++s) {
                const uint32_t v = u * Q + s;
                noc_async_write(src + s * SB, peer_addr(v % K, g_base + (v / K) * KT * T + my_ofs), SB);
            }
            noc_async_writes_flushed();
            for (uint32_t s = 0; s < Q; ++s) {
                noc_semaphore_inc(peer_addr((u * Q + s) % K, sem_part), 1);
            }
            cb_pop_front(cb_partial, 1);
        }
    }

    {
        MaybeDeviceZoneScope("wr_wait_inv");
        while (inv_pushed < U) {
            push_ready_inv();
            invalidate_l1_cache();
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
