// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — the operand K-block walk shared by the A (NCRISC) and W (BRISC) line kernels.
//
// A walk is `steps` = units x num_k_blocks K-block steps, one CB slot of `kblock_pages` each. Step s is `fresh` when
// its K-block must be delivered (DRAM read + multicast on the line injector, a receive on the other line cores);
// otherwise it only replays CB credits of a resident ring (reserve + push of the same pages).
//
// Injector read-ahead (READ_AHEAD): the DRAM reads of the next fresh step are issued into the slot after the current
// one *before* the current step is multicast, so they are in flight while the multicast runs; the next step's read
// barrier then waits only for what is left of them. A step is read ahead only when the CB already has room for both
// slots (non-blocking check), so the current step's multicast is never delayed behind the local consumer. The
// mcast_pipe send fences writes only (source-L1 guard), never reads, so the in-flight reads are untouched by it.
// READ_AHEAD = false is the serial read -> barrier -> multicast walk.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (permanent; opt-in via KERNEL_PERF_ZONES), per K-block step: *_reserve = back-pressure from the compute
// consumer (operand CB full), inj_read = DRAM issue + read barrier, inj_mcast / recv_mcast = multicast send (incl. the
// receivers' ready handshake) / receive wait. MMRS_ABLATE_OPERANDS (perf ablation only, wrong results) drops the DRAM
// reads and multicasts and keeps every CB credit.

namespace mmrs {

// Read n pages page0, page0 + stride, ... of an interleaved tensor into dst, dst + dst_stride, ... (no barrier).
// Pages whose indices differ by a multiple of the bank count live in the same bank at linearly increasing offsets, so
// when `same_bank` (stride % num_banks == 0) the run is one stateful read setup (the bank's NoC coordinate) plus n
// one-packet reads by local address -- several times cheaper to issue than a full read per page, which is what the
// line injector is bound by. Falls back to per-page reads if the run would cross the 32-bit local-address window.
template <typename Accessor>
FORCE_INLINE void read_pages_strided(
    const Accessor& acc,
    uint32_t page0,
    uint32_t stride,
    uint32_t n,
    uint32_t dst,
    uint32_t dst_stride,
    uint32_t page_bytes,
    bool same_bank) {
    if (n == 0) {
        return;
    }
    if (same_bank && n > 1) {
        const uint64_t first = acc.get_noc_addr(page0);
        const uint64_t last = acc.get_noc_addr(page0 + (n - 1) * stride);
        if ((first >> 32) == (last >> 32)) {
            const uint32_t lo = static_cast<uint32_t>(first);
            const uint32_t step = n > 1 ? (static_cast<uint32_t>(last) - lo) / (n - 1) : 0;
            noc_async_read_one_packet_set_state(first, page_bytes);
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_read_one_packet_with_state(lo + i * step, dst + i * dst_stride);
            }
            return;
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read(acc.get_noc_addr(page0 + i * stride), dst + i * dst_stride, page_bytes);
    }
}

// slot after `slot` in `cb`'s ring (the capacity is a whole number of K-block slots, so a wrap lands on the base)
FORCE_INLINE uint32_t next_slot(uint32_t cb, uint32_t slot, uint32_t slot_bytes) {
    const uint32_t next = slot + slot_bytes;
    return next == get_local_cb_interface(cb).fifo_limit ? next - get_local_cb_interface(cb).fifo_size : next;
}

// injector: issue(s, dst) issues step s's DRAM reads into dst (no barrier); send(dst) multicasts the slot;
// reserve(pages) blocks for CB room (it may poll other work); poll() runs after every push.
template <bool READ_AHEAD, typename Fresh, typename Issue, typename Reserve, typename Send, typename Poll>
FORCE_INLINE void inject_operand(
    uint32_t cb,
    uint32_t steps,
    uint32_t kblock_pages,
    uint32_t kblock_bytes,
    Fresh&& fresh,
    Issue&& issue,
    Reserve&& reserve,
    Send&& send,
    Poll&& poll) {
    bool ahead = false;  // step s's reads were issued during step s - 1
    for (uint32_t s = 0; s < steps; ++s) {
        {
            MaybeDeviceZoneScope("inj_reserve");
            reserve(kblock_pages);
        }
        const uint32_t dst = get_write_ptr(cb);
#ifdef MMRS_ABLATE_OPERANDS
        const bool f = false;
        (void)fresh;
        (void)issue;
        (void)send;
#else
        const bool f = fresh(s);
#endif
        if (f) {
            MaybeDeviceZoneScope("inj_read");
            if (!ahead) {
                issue(s, dst);
            }
            noc_async_read_barrier();
        }
        ahead = false;
        if constexpr (READ_AHEAD) {
            if (s + 1 < steps && fresh(s + 1) && cb_pages_reservable_at_back(cb, 2 * kblock_pages)) {
                cb_reserve_back(cb, 2 * kblock_pages);
                issue(s + 1, next_slot(cb, dst, kblock_bytes));
                ahead = true;
            }
        }
        if (f) {
            MaybeDeviceZoneScope("inj_mcast");
            send(dst);
        }
        cb_push_back(cb, kblock_pages);
        poll();
    }
}

// line receiver: the same walk, each fresh step is one multicast receive into the reserved slot.
template <typename Fresh, typename Reserve, typename Receive, typename Poll>
FORCE_INLINE void receive_operand(
    uint32_t cb,
    uint32_t steps,
    uint32_t kblock_pages,
    Fresh&& fresh,
    Reserve&& reserve,
    Receive&& receive,
    Poll&& poll) {
    for (uint32_t s = 0; s < steps; ++s) {
        {
            MaybeDeviceZoneScope("recv_reserve");
            reserve(kblock_pages);
        }
#ifdef MMRS_ABLATE_OPERANDS
        (void)fresh;
        (void)receive;
#else
        if (fresh(s)) {
            MaybeDeviceZoneScope("recv_mcast");
            receive();
        }
#endif
        cb_push_back(cb, kblock_pages);
        poll();
    }
}

}  // namespace mmrs
