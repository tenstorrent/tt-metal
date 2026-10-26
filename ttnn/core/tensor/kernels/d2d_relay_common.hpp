// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Helpers shared by the two D2D relay kernels (persistent_d2d_exit_relay.cpp and
// persistent_d2d_entry_relay.cpp). The page streaming and the fabric write are the same
// as in persistent_d2d_sender.cpp; here they take their sizes as template parameters,
// because the relays copy many tensors (one per source or target) instead of one.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"

namespace d2d_relay {

// Write `size` bytes from contiguous L1 to one remote NoC address over fabric, split into
// packets of at most MaxPayload bytes. The packet header must already hold the route.
template <uint32_t MaxPayload>
FORCE_INLINE void fabric_write_bytes(
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    uint64_t dst_addr,
    uint32_t l1_src_addr,
    uint32_t size,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header) {
    uint32_t remaining = size;
    while (remaining > 0) {
        const uint32_t packet_size = remaining > MaxPayload ? MaxPayload : remaining;
        header->to_noc_unicast_write(NocUnicastCommandHeader{dst_addr}, packet_size);
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

// Copy pages [page_start, page_end) of one tensor to another chip over fabric. DRAM reads
// run up to Depth pages ahead of the fabric writes, through a ring of Depth slots in the
// scratch buffer at cb_l1_addr. Each slot has its own NoC transaction id (TridBase + slot),
// so only the read for the page about to be sent is waited on. Same loop as
// stream_half in persistent_d2d_sender.cpp.
template <uint32_t PageSize, uint32_t MaxPayload, uint32_t Depth, uint32_t TridBase, typename InAcc, typename OutAcc>
FORCE_INLINE void stream_pages(
    const Noc& noc,
    tt::tt_fabric::WorkerToFabricEdmSender& conn,
    const InAcc& in_acc,
    const OutAcc& out_acc,
    uint32_t cb_l1_addr,
    uint32_t page_start,
    uint32_t page_end,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header) {
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
            conn, out_acc.get_noc_addr(page_start + k), cb_l1_addr + slot * PageSize, PageSize, header);
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

// Wrap-safe "counter advanced by at least n since last".
FORCE_INLINE bool advanced(uint32_t now, uint32_t last, uint32_t n) { return (now - last) >= n; }

FORCE_INLINE bool terminated(uint32_t termination_addr, uint32_t terminate_value) {
    invalidate_l1_cache();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_addr)[0] == terminate_value;
}

}  // namespace d2d_relay
