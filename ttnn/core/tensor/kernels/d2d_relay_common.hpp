// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Helpers shared by the exit and entry relay kernels. Same streaming as persistent_d2d_sender.cpp,
// with sizes as template parameters because a relay copies many tensors.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

namespace d2d_relay {

// go_slot values that are not slots: at the start and end of a lease window the master tells
// the sub to open or close its fabric connections.
constexpr uint32_t kGoOpenConns = 0xFFFFFFFEu;
constexpr uint32_t kGoCloseConns = 0xFFFFFFFFu;

// Write `size` bytes from L1 to a remote NoC address over fabric, in MaxPayload packets.
// A nonzero inc_addr fuses a flushed atomic increment into the last packet ("all data landed").
template <uint32_t MaxPayload>
FORCE_INLINE void fabric_write_bytes(
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    uint64_t dst_addr,
    uint32_t l1_src_addr,
    uint32_t size,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
    uint64_t inc_addr = 0) {
    uint32_t remaining = size;
    while (remaining > 0) {
        const uint32_t packet_size = remaining > MaxPayload ? MaxPayload : remaining;
        if (inc_addr != 0 && packet_size == remaining) {
            header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_addr, inc_addr, 1, true}, packet_size);
        } else {
            header->to_noc_unicast_write(NocUnicastCommandHeader{dst_addr}, packet_size);
        }
        conn.wait_for_empty_write_slot();
        conn.send_payload_without_header_non_blocking_from_address(l1_src_addr, packet_size);
        conn.send_payload_flush_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
        dst_addr += packet_size;
        l1_src_addr += packet_size;
        remaining -= packet_size;
    }
}

// Atomic-increment a remote L1 word by 1 over fabric. flush = true makes the far router
// write every earlier packet on this connection before it applies the increment, so the
// increment means "all my data has landed".
FORCE_INLINE void fabric_atomic_inc(
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    uint64_t dst_noc_addr,
    bool flush,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header) {
    header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{dst_noc_addr, 1, flush});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
}

// Point a packet header at chip `chip_id` of mesh `mesh_id`. 2D fabric only.
FORCE_INLINE void set_route(volatile tt_l1_ptr PACKET_HEADER_TYPE* header, uint32_t chip_id, uint32_t mesh_id) {
    static_assert(
        std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader>,
        "D2D relay mode needs a 2D fabric (HybridMeshPacketHeader)");
    fabric_set_unicast_route(header, chip_id, mesh_id);
}

constexpr uint32_t floor_pow2(uint32_t n) { return n == 0 ? 0 : (1u << (31 - __builtin_clz(n))); }

// Wrap-safe "counter advanced by at least n since last".
FORCE_INLINE bool advanced(uint32_t now, uint32_t last, uint32_t n) { return (now - last) >= n; }

FORCE_INLINE bool terminated(uint32_t termination_addr, uint32_t terminate_value) {
    invalidate_l1_cache();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_addr)[0] == terminate_value;
}

// Copy pages [page_start, page_end) to another chip over fabric, reading up to Depth pages
// ahead (one NoC transaction id per ring slot). Same loop as stream_half in the sender kernel.
// A nonzero inc_addr is fused into the last packet (see fabric_write_bytes).
template <uint32_t PageSize, uint32_t MaxPayload, uint32_t Depth, uint32_t TridBase, typename InAcc, typename OutAcc>
FORCE_INLINE void stream_pages(
    const Noc& noc,
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    const InAcc& in_acc,
    const OutAcc& out_acc,
    uint32_t cb_l1_addr,
    uint32_t page_start,
    uint32_t page_end,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header,
    uint64_t inc_addr = 0) {
    static_assert(Depth > 0 && (Depth & (Depth - 1)) == 0, "ring depth must be a power of 2");
    const uint32_t n = page_end - page_start;
    const uint32_t prime = Depth < n ? Depth : n;
    CoreLocalMem<uint32_t> scratch(cb_l1_addr);
    for (uint32_t i = 0; i < prime; ++i) {
        noc.async_read<NocOptions::TXN_ID>(
            in_acc,
            scratch,
            PageSize,
            {.page_id = page_start + i},
            {.offset_bytes = i * PageSize},
            NocOptVals{.trid = TridBase + i});
    }
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t slot = k & (Depth - 1);
        const uint32_t trid = TridBase + slot;
        noc.async_read_barrier<NocOptions::TXN_ID>(NocOptVals{.trid = trid});
        fabric_write_bytes<MaxPayload>(
            conn,
            out_acc.get_noc_addr(page_start + k),
            cb_l1_addr + slot * PageSize,
            PageSize,
            header,
            k + 1 == n ? inc_addr : 0);
        const uint32_t next = k + Depth;
        if (next < n) {
            noc.async_read<NocOptions::TXN_ID>(
                in_acc,
                scratch,
                PageSize,
                {.page_id = page_start + next},
                {.offset_bytes = slot * PageSize},
                NocOptVals{.trid = trid});
        }
    }
}

// Copy pages [page_start, page_end), then md_size bytes of metadata from md_l1_addr to md_dst
// (none when md_size is 0), and increment inc_addr by 1 in the same packet as the last write.
// With nothing to write, the increment goes alone with the same flush.
template <uint32_t PageSize, uint32_t MaxPayload, uint32_t Depth, uint32_t TridBase, typename InAcc, typename OutAcc>
FORCE_INLINE void stream_pages_then_inc(
    const Noc& noc,
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    const InAcc& in_acc,
    const OutAcc& out_acc,
    uint32_t cb_l1_addr,
    uint32_t page_start,
    uint32_t page_end,
    uint64_t md_dst,
    uint32_t md_l1_addr,
    uint32_t md_size,
    uint64_t inc_addr,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header) {
    if (md_size > 0) {
        stream_pages<PageSize, MaxPayload, Depth, TridBase>(
            noc, conn, in_acc, out_acc, cb_l1_addr, page_start, page_end, header);
        fabric_write_bytes<MaxPayload>(conn, md_dst, md_l1_addr, md_size, header, inc_addr);
    } else if (page_end > page_start) {
        stream_pages<PageSize, MaxPayload, Depth, TridBase>(
            noc, conn, in_acc, out_acc, cb_l1_addr, page_start, page_end, header, inc_addr);
    } else {
        fabric_atomic_inc(conn, inc_addr, true, header);
    }
}

// Master side of the two-lane sync over the go_slot / go_count / done_count L1 words. Each go
// is a slot to copy, kGoOpenConns or kGoCloseConns; the sub acks every go with done_count, and
// post() waits for the previous ack first, so go_slot is never overwritten before it is read.
struct SubLane {
    uint32_t go_slot_addr;
    uint32_t go_count_addr;
    uint32_t done_count_addr;
    uint32_t termination_addr;
    uint32_t terminate_value;
    uint32_t go_local = 0;
    uint32_t last_done = 0;
    bool busy = false;

    // Wait for the ack of the last go. Gives up on termination: the sub may have exited.
    FORCE_INLINE void wait() {
        if (!busy) {
            return;
        }
        auto* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_count_addr);
        while (!advanced(done[0], last_done, 1)) {
            if (terminated(termination_addr, terminate_value)) {
                return;
            }
        }
        last_done += 1;
        busy = false;
    }

    FORCE_INLINE void post(uint32_t what) {
        wait();
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_slot_addr)[0] = what;
        go_local += 1;
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_count_addr)[0] = go_local;
        busy = true;
    }
};

// Sub side: run each go from the master and ack it. A go is checked before termination, so a
// posted close is never skipped; on termination the sub closes its connections if still open.
template <typename OpenFn, typename CloseFn, typename CopyFn>
FORCE_INLINE void run_sub_lane(
    uint32_t go_slot_addr,
    uint32_t go_count_addr,
    uint32_t done_count_addr,
    uint32_t termination_addr,
    uint32_t terminate_value,
    OpenFn open_conns,
    CloseFn close_conns,
    CopyFn copy) {
    auto* go_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_count_addr);
    auto* go_slot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_slot_addr);
    auto* done_count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_count_addr);
    uint32_t last_go = 0;
    bool conns_open = false;
    while (true) {
        if (!advanced(go_count[0], last_go, 1)) {
            if (terminated(termination_addr, terminate_value)) {
                break;
            }
            continue;
        }
        last_go += 1;
        const uint32_t what = go_slot[0];
        if (what == kGoOpenConns) {
            open_conns();
            conns_open = true;
        } else if (what == kGoCloseConns) {
            close_conns();
            conns_open = false;
        } else {
            copy(what);
        }
        done_count[0] += 1;
    }
    if (conns_open) {
        close_conns();
    }
}

}  // namespace d2d_relay
