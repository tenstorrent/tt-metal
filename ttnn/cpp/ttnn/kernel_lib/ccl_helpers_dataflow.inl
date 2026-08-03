// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for ccl_helpers_dataflow.hpp
// Do not include directly - include ccl_helpers_dataflow.hpp instead

#pragma once

namespace dataflow_kernel_lib::ccl {

// Namespace shorthands (scoped to this namespace; do not leak to includers).
namespace linear_fabric = tt::tt_fabric::linear::experimental;
using tt::tt_fabric::common::experimental::UnicastAtomicIncUpdateMask;
using tt::tt_fabric::common::experimental::UnicastScatterWriteUpdateMask;
using tt::tt_fabric::common::experimental::UnicastWriteUpdateMask;

// ----------------------------------------------------------------------------
// FabricStream — armed unicast-write channel
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE UnicastWriteChannel<ConnT> FabricStream<ConnT>::arm_unicast_write(uint32_t page_size_bytes) {
    if (payload_hdr_ == nullptr) {
        payload_hdr_ = PacketHeaderPool::allocate_header();
    }
    // set_state programs the invariant on-wire payload size (+ the chip-unicast hop count); the
    // route util then writes the LowLatency 1-D routing fields (the proven-correct value, applied
    // last). Helper owns the PayloadSize mask. The route is the stream's, bound at open().
    linear_fabric::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        payload_hdr_,
        static_cast<uint8_t>(route_.distance_in_hops),
        nullptr,
        static_cast<uint16_t>(align(page_size_bytes, alignment_)));
    ccl_routing_utils::fabric_set_line_unicast_route(payload_hdr_, route_);
    return UnicastWriteChannel<ConnT>(conn_, payload_hdr_);
}

template <typename ConnT>
FORCE_INLINE void UnicastWriteChannel<ConnT>::write(uint64_t dst_noc_addr, uint32_t src_l1_addr) {
    // with_state issues the armed payload size, updating only the destination address.
    linear_fabric::fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
        conn_->sender(), hdr_, src_l1_addr, tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr});
}

template <typename ConnT>
template <class AddrGen>
FORCE_INLINE void UnicastWriteChannel<ConnT>::write_page(uint32_t src_l1_addr, uint32_t page_idx, const AddrGen& dst) {
    const uint64_t dst_noc_addr = tt::tt_fabric::linear::addrgen_detail::get_noc_address(dst, page_idx, 0);
    write(dst_noc_addr, src_l1_addr);
}

// ----------------------------------------------------------------------------
// FabricStream — armed scatter-write channel (<=4 chunks/packet)
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE ScatterWriteChannel<ConnT> FabricStream<ConnT>::arm_scatter_write(
    uint32_t chunk_size_bytes, uint32_t num_chunks) {
    if (scatter_hdr_ == nullptr) {
        scatter_hdr_ = PacketHeaderPool::allocate_header();
    }
    // set_state establishes the scatter send type + the stream's route + invariant chunk
    // sizes/payload; the dst addrs (and per-packet chunk count) are filled per-issue by write_scatter.
    uint64_t dummy_addrs[4] = {0, 0, 0, 0};
    uint16_t chunk_sizes[3] = {
        static_cast<uint16_t>(chunk_size_bytes),
        static_cast<uint16_t>(chunk_size_bytes),
        static_cast<uint16_t>(chunk_size_bytes)};
    linear_fabric::fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        scatter_hdr_,
        static_cast<uint8_t>(route_.distance_in_hops),
        tt::tt_fabric::NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, static_cast<uint8_t>(num_chunks)),
        static_cast<uint16_t>(chunk_size_bytes * num_chunks));
    ccl_routing_utils::fabric_set_line_unicast_route(scatter_hdr_, route_);
    return ScatterWriteChannel<ConnT>(conn_, scatter_hdr_, chunk_size_bytes);
}

template <typename ConnT>
FORCE_INLINE void ScatterWriteChannel<ConnT>::write_scatter(
    const uint64_t* dst_noc_addrs, uint32_t num_chunks, uint32_t src_l1_addr) {
    // with_state re-programs dst addrs + chunk count + payload size each call (the last packet of a
    // run can carry fewer chunks than the armed maximum).
    uint16_t chunk_sizes[3] = {
        static_cast<uint16_t>(chunk_size_bytes_),
        static_cast<uint16_t>(chunk_size_bytes_),
        static_cast<uint16_t>(chunk_size_bytes_)};
    linear_fabric::fabric_unicast_noc_scatter_write_with_state<
        UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
        UnicastScatterWriteUpdateMask::PayloadSize>(
        conn_->sender(),
        hdr_,
        src_l1_addr,
        tt::tt_fabric::NocUnicastScatterCommandHeader(dst_noc_addrs, chunk_sizes, static_cast<uint8_t>(num_chunks)),
        static_cast<uint16_t>(chunk_size_bytes_ * num_chunks));
}

// ----------------------------------------------------------------------------
// FabricStream — armed unicast atomic-inc channel
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE AtomicIncChannel<ConnT> FabricStream<ConnT>::arm_inc(uint32_t val) {
    if (sem_hdr_ == nullptr) {
        sem_hdr_ = PacketHeaderPool::allocate_header();
    }
    // set_state programs the invariant increment value + flush (the noc_address field is a
    // placeholder, filled per-issue by inc()). Helper owns the Val|Flush mask; route is the
    // stream's, bound at open().
    linear_fabric::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        sem_hdr_,
        static_cast<uint8_t>(route_.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, val});
    ccl_routing_utils::fabric_set_line_unicast_route(sem_hdr_, route_);
    return AtomicIncChannel<ConnT>(conn_, sem_hdr_);
}

template <typename ConnT>
FORCE_INLINE void AtomicIncChannel<ConnT>::inc(uint64_t remote_sem_noc_addr) {
    // with_state issues the armed value, updating only the destination semaphore address.
    linear_fabric::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        conn_->sender(), hdr_, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_sem_noc_addr, 0});
}

// ----------------------------------------------------------------------------
// FabricStream — armed multicast atomic-inc channel (the N-party barrier)
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE MulticastIncChannel<ConnT> FabricStream<ConnT>::arm_multicast_inc(
    const ccl_routing_utils::line_multicast_route_info_t& route, uint32_t val) {
    if (mcast_hdr_ == nullptr) {
        mcast_hdr_ = PacketHeaderPool::allocate_header();
    }
    // set_state programs the invariant inc value + flush on the dedicated multicast header for a
    // MULTICAST route; the dst sem addr is filled per-issue by multicast_inc. Independent of the
    // unicast sem_hdr_, so the barrier (multicast) and counting (unicast) channels may coexist.
    linear_fabric::fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        mcast_hdr_,
        static_cast<uint8_t>(route.start_distance_in_hops),
        static_cast<uint8_t>(route.range_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, val});
    ccl_routing_utils::fabric_set_line_multicast_route(mcast_hdr_, route);
    return MulticastIncChannel<ConnT>(conn_, mcast_hdr_);
}

template <typename ConnT>
FORCE_INLINE void MulticastIncChannel<ConnT>::multicast_inc(uint64_t remote_sem_noc_addr) {
    linear_fabric::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        conn_->sender(), hdr_, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_sem_noc_addr, 0});
}

// ----------------------------------------------------------------------------
// FabricStream — lifecycle
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE void FabricStream<ConnT>::drain() {
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}

template <typename ConnT>
FORCE_INLINE void FabricStream<ConnT>::close() {
    if (!closed_) {
        closed_ = true;
        // Flush outstanding fabric writes + atomic-incs BEFORE tearing the connection down, so a
        // trailing inc/write is never lost. Idempotent with an explicit drain() the caller may
        // have already issued.
        drain();
        conn_->close();
    }
}

// ----------------------------------------------------------------------------
// FabricStreamSender — one-shot signal()
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE void FabricStreamSender<ConnT>::signal(
    const ccl_routing_utils::line_unicast_route_info_t& route, uint64_t remote_sem_noc_addr, uint32_t val) {
    auto stream = open(route);
    auto ch = stream.arm_inc(val);
    ch.inc(remote_sem_noc_addr);
    stream.close();  // drains the inc, then closes
}

}  // namespace dataflow_kernel_lib::ccl
