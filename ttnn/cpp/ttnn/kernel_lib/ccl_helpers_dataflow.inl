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
using tt::tt_fabric::common::experimental::UnicastFusedAtomicIncUpdateMask;
using tt::tt_fabric::common::experimental::UnicastScatterWriteUpdateMask;
using tt::tt_fabric::common::experimental::UnicastWriteUpdateMask;

// The invariant fields a fused write+inc arms once (the inc value, the flush flag and a nominal
// payload size) and the fields each issue re-programs (both destinations + this packet's size).
// Named here so the masks appear exactly once and no op ever spells one out.
inline constexpr auto kFusedArmMask = UnicastFusedAtomicIncUpdateMask::Val | UnicastFusedAtomicIncUpdateMask::Flush |
                                      UnicastFusedAtomicIncUpdateMask::PayloadSize;
inline constexpr auto kFusedIssueMask = UnicastFusedAtomicIncUpdateMask::WriteDstAddr |
                                        UnicastFusedAtomicIncUpdateMask::SemaphoreAddr |
                                        UnicastFusedAtomicIncUpdateMask::PayloadSize;

/// Mirror one payload to the SAME logical destination on the local chip. The fabric carries a
/// noc0-encoded address; a local write needs it re-encoded for this chip's NoC, which is the
/// decompose-then-safe_get_noc_addr dance every reduction writer previously open-coded.
FORCE_INLINE void write_local_mirror(uint64_t noc0_dst_noc_addr, uint32_t src_l1_addr, uint32_t payload_size_bytes) {
    const auto [dst_noc_xy, dst_addr] = get_noc_address_components(noc0_dst_noc_addr);
    noc_async_write(src_l1_addr, safe_get_noc_addr(dst_noc_xy.x, dst_noc_xy.y, dst_addr), payload_size_bytes);
}

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

// ----------------------------------------------------------------------------
// FabricStream — armed fused write + atomic-inc channel
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE FusedWriteIncChannel<ConnT> FabricStream<ConnT>::arm_fused_write_inc(
    uint32_t page_size_bytes, uint32_t val, bool flush) {
    if (fused_hdr_ == nullptr) {
        fused_hdr_ = PacketHeaderPool::allocate_header();
    }
    // set_state programs the invariant inc value + flush + nominal payload size; the two addresses
    // are placeholders filled per-issue. Helper owns the mask. Route is the stream's, bound at open().
    linear_fabric::fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<kFusedArmMask>(
        fused_hdr_,
        static_cast<uint8_t>(route_.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0, 0, val, flush},
        static_cast<uint16_t>(align(page_size_bytes, alignment_)));
    ccl_routing_utils::fabric_set_line_unicast_route(fused_hdr_, route_);
    return FusedWriteIncChannel<ConnT>(conn_, fused_hdr_);
}

template <typename ConnT>
FORCE_INLINE void FusedWriteIncChannel<ConnT>::write_fused(
    uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr) {
    linear_fabric::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<
        UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
        conn_->sender(),
        hdr_,
        src_l1_addr,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, remote_sem_noc_addr, 0, false});
}

// ----------------------------------------------------------------------------
// FabricStream — armed chip-multicast payload-write channel
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE MulticastWriteChannel<ConnT> FabricStream<ConnT>::arm_multicast_write(
    const ccl_routing_utils::line_multicast_route_info_t& route, uint32_t page_size_bytes) {
    if (mcast_write_hdr_ == nullptr) {
        mcast_write_hdr_ = PacketHeaderPool::allocate_header();
    }
    linear_fabric::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        mcast_write_hdr_,
        static_cast<uint8_t>(route.start_distance_in_hops),
        static_cast<uint8_t>(route.range_hops),
        nullptr,
        static_cast<uint16_t>(align(page_size_bytes, alignment_)));
    ccl_routing_utils::fabric_set_line_multicast_route(mcast_write_hdr_, route);
    return MulticastWriteChannel<ConnT>(conn_, mcast_write_hdr_);
}

template <typename ConnT>
FORCE_INLINE void MulticastWriteChannel<ConnT>::write(uint64_t dst_noc_addr, uint32_t src_l1_addr) {
    linear_fabric::fabric_multicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
        conn_->sender(), hdr_, src_l1_addr, tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr});
}

// ----------------------------------------------------------------------------
// FabricStream — armed chip-multicast fused write + atomic-inc channel
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE MulticastFusedWriteIncChannel<ConnT> FabricStream<ConnT>::arm_multicast_fused_write_inc(
    const ccl_routing_utils::line_multicast_route_info_t& route, uint32_t page_size_bytes, uint32_t val, bool flush) {
    if (mcast_fused_hdr_ == nullptr) {
        mcast_fused_hdr_ = PacketHeaderPool::allocate_header();
    }
    linear_fabric::fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<kFusedArmMask>(
        mcast_fused_hdr_,
        static_cast<uint8_t>(route.start_distance_in_hops),
        static_cast<uint8_t>(route.range_hops),
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0, 0, val, flush},
        static_cast<uint16_t>(align(page_size_bytes, alignment_)));
    ccl_routing_utils::fabric_set_line_multicast_route(mcast_fused_hdr_, route);
    return MulticastFusedWriteIncChannel<ConnT>(conn_, mcast_fused_hdr_);
}

template <typename ConnT>
FORCE_INLINE void MulticastFusedWriteIncChannel<ConnT>::write_fused(
    uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr) {
    linear_fabric::fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
        UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>(
        conn_->sender(),
        hdr_,
        src_l1_addr,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, remote_sem_noc_addr, 0, false});
}

// ============================================================================
// The DUPLEX tier
// ============================================================================

// ----------------------------------------------------------------------------
// FabricDuplexStream — arming (per connected direction, on that direction's route)
// ----------------------------------------------------------------------------

template <Cast C, typename ConnT>
FORCE_INLINE DuplexWriteChannel<C, ConnT> FabricDuplexStream<C, ConnT>::arm_write(uint32_t page_size_bytes) {
    const uint16_t on_wire = static_cast<uint16_t>(align(page_size_bytes, alignment_));
    for (uint32_t d = 0; d < DuplexConn::kNumDirections; ++d) {
        // Only CONNECTED directions get a header: an end-of-line worker has one side unwired, and
        // arming it would burn a pooled header on a send that can never be issued.
        if (!conn_->has(d)) {
            continue;
        }
        if (write_hdr_[d] == nullptr) {
            write_hdr_[d] = PacketHeaderPool::allocate_header();
        }
        if constexpr (C == Cast::Multicast) {
            linear_fabric::fabric_multicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                write_hdr_[d],
                static_cast<uint8_t>(mcast_route_[d].start_distance_in_hops),
                static_cast<uint8_t>(mcast_route_[d].range_hops),
                nullptr,
                on_wire);
            ccl_routing_utils::fabric_set_line_multicast_route(write_hdr_[d], mcast_route_[d]);
        } else {
            linear_fabric::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
                write_hdr_[d], static_cast<uint8_t>(uni_route_[d].distance_in_hops), nullptr, on_wire);
            ccl_routing_utils::fabric_set_line_unicast_route(write_hdr_[d], uni_route_[d]);
        }
    }
    return DuplexWriteChannel<C, ConnT>(
        conn_, write_hdr_[DuplexConn::kForward], write_hdr_[DuplexConn::kBackward], on_wire);
}

template <Cast C, typename ConnT>
FORCE_INLINE DuplexFusedWriteIncChannel<C, ConnT> FabricDuplexStream<C, ConnT>::arm_fused_write_inc(
    uint32_t page_size_bytes, uint32_t val, bool flush) {
    const uint16_t on_wire = static_cast<uint16_t>(align(page_size_bytes, alignment_));
    const tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader arm_hdr{0, 0, val, flush};
    for (uint32_t d = 0; d < DuplexConn::kNumDirections; ++d) {
        if (!conn_->has(d)) {
            continue;
        }
        if (fused_hdr_[d] == nullptr) {
            fused_hdr_[d] = PacketHeaderPool::allocate_header();
        }
        if constexpr (C == Cast::Multicast) {
            linear_fabric::fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<kFusedArmMask>(
                fused_hdr_[d],
                static_cast<uint8_t>(mcast_route_[d].start_distance_in_hops),
                static_cast<uint8_t>(mcast_route_[d].range_hops),
                arm_hdr,
                on_wire);
            ccl_routing_utils::fabric_set_line_multicast_route(fused_hdr_[d], mcast_route_[d]);
        } else {
            linear_fabric::fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<kFusedArmMask>(
                fused_hdr_[d], static_cast<uint8_t>(uni_route_[d].distance_in_hops), arm_hdr, on_wire);
            ccl_routing_utils::fabric_set_line_unicast_route(fused_hdr_[d], uni_route_[d]);
        }
    }
    return DuplexFusedWriteIncChannel<C, ConnT>(
        conn_, fused_hdr_[DuplexConn::kForward], fused_hdr_[DuplexConn::kBackward], on_wire);
}

template <Cast C, typename ConnT>
FORCE_INLINE DuplexScatterWriteChannel<C, ConnT> FabricDuplexStream<C, ConnT>::arm_scatter_write(
    uint32_t chunk_size_bytes, uint32_t num_chunks) {
    uint64_t dummy_addrs[4] = {0, 0, 0, 0};
    uint16_t chunk_sizes[3] = {
        static_cast<uint16_t>(chunk_size_bytes),
        static_cast<uint16_t>(chunk_size_bytes),
        static_cast<uint16_t>(chunk_size_bytes)};
    const auto arm_hdr =
        tt::tt_fabric::NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, static_cast<uint8_t>(num_chunks));
    const uint16_t on_wire = static_cast<uint16_t>(chunk_size_bytes * num_chunks);
    for (uint32_t d = 0; d < DuplexConn::kNumDirections; ++d) {
        if (!conn_->has(d)) {
            continue;
        }
        if (scatter_hdr_[d] == nullptr) {
            scatter_hdr_[d] = PacketHeaderPool::allocate_header();
        }
        if constexpr (C == Cast::Multicast) {
            linear_fabric::fabric_multicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
                scatter_hdr_[d],
                static_cast<uint8_t>(mcast_route_[d].start_distance_in_hops),
                static_cast<uint8_t>(mcast_route_[d].range_hops),
                arm_hdr,
                on_wire);
            ccl_routing_utils::fabric_set_line_multicast_route(scatter_hdr_[d], mcast_route_[d]);
        } else {
            linear_fabric::fabric_unicast_noc_scatter_write_set_state<
                UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
                scatter_hdr_[d], static_cast<uint8_t>(uni_route_[d].distance_in_hops), arm_hdr, on_wire);
            ccl_routing_utils::fabric_set_line_unicast_route(scatter_hdr_[d], uni_route_[d]);
        }
    }
    return DuplexScatterWriteChannel<C, ConnT>(
        conn_, scatter_hdr_[DuplexConn::kForward], scatter_hdr_[DuplexConn::kBackward], chunk_size_bytes);
}

template <Cast C, typename ConnT>
FORCE_INLINE DuplexIncChannel<C, ConnT> FabricDuplexStream<C, ConnT>::arm_inc(uint32_t val, bool flush) {
    const tt::tt_fabric::NocUnicastAtomicIncCommandHeader arm_hdr{0, val, flush};
    for (uint32_t d = 0; d < DuplexConn::kNumDirections; ++d) {
        if (!conn_->has(d)) {
            continue;
        }
        if (inc_hdr_[d] == nullptr) {
            inc_hdr_[d] = PacketHeaderPool::allocate_header();
        }
        if constexpr (C == Cast::Multicast) {
            linear_fabric::fabric_multicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                inc_hdr_[d],
                static_cast<uint8_t>(mcast_route_[d].start_distance_in_hops),
                static_cast<uint8_t>(mcast_route_[d].range_hops),
                arm_hdr);
            ccl_routing_utils::fabric_set_line_multicast_route(inc_hdr_[d], mcast_route_[d]);
        } else {
            linear_fabric::fabric_unicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                inc_hdr_[d], static_cast<uint8_t>(uni_route_[d].distance_in_hops), arm_hdr);
            ccl_routing_utils::fabric_set_line_unicast_route(inc_hdr_[d], uni_route_[d]);
        }
    }
    return DuplexIncChannel<C, ConnT>(conn_, inc_hdr_[DuplexConn::kForward], inc_hdr_[DuplexConn::kBackward]);
}

// ----------------------------------------------------------------------------
// Duplex channels — issues fan out over every connected direction
// ----------------------------------------------------------------------------

template <Cast C, typename ConnT>
FORCE_INLINE void DuplexIncChannel<C, ConnT>::inc(uint64_t remote_sem_noc_addr) {
    const tt::tt_fabric::NocUnicastAtomicIncCommandHeader issue_hdr{remote_sem_noc_addr, 0};
    for (uint32_t d = 0; d < DuplexConn::kNumDirections; ++d) {
        if (!conn_->has(d)) {
            continue;
        }
        if constexpr (C == Cast::Multicast) {
            linear_fabric::fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                conn_->sender(d), hdr_[d], issue_hdr);
        } else {
            linear_fabric::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                conn_->sender(d), hdr_[d], issue_hdr);
        }
    }
}

template <Cast C, typename ConnT>
FORCE_INLINE void DuplexWriteChannel<C, ConnT>::write(
    uint64_t dst_noc_addr, uint32_t src_l1_addr, uint32_t payload_size_bytes) {
    for (uint32_t d = 0; d < DuplexConn::kNumDirections; ++d) {
        if (!conn_->has(d)) {
            continue;
        }
        if constexpr (C == Cast::Multicast) {
            linear_fabric::fabric_multicast_noc_unicast_write_with_state<
                UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                conn_->sender(d),
                hdr_[d],
                src_l1_addr,
                tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr},
                static_cast<uint16_t>(payload_size_bytes));
        } else {
            linear_fabric::fabric_unicast_noc_unicast_write_with_state<
                UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                conn_->sender(d),
                hdr_[d],
                src_l1_addr,
                tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr},
                static_cast<uint16_t>(payload_size_bytes));
        }
    }
}

template <Cast C, typename ConnT>
FORCE_INLINE void DuplexWriteChannel<C, ConnT>::write(uint64_t dst_noc_addr, uint32_t src_l1_addr) {
    write(dst_noc_addr, src_l1_addr, payload_size_);
}

template <Cast C, typename ConnT>
FORCE_INLINE void DuplexWriteChannel<C, ConnT>::write_with_local_copy(
    uint64_t dst_noc_addr, uint32_t src_l1_addr, uint32_t payload_size_bytes) {
    // Local copy first so it overlaps the fabric sends, then flush — the order and the trailing
    // flush of write_and_advance_local_read_address_for_fabric_write, preserved exactly.
    write_local_mirror(dst_noc_addr, src_l1_addr, payload_size_bytes);
    write(dst_noc_addr, src_l1_addr, payload_size_bytes);
    noc_async_writes_flushed();
}

template <Cast C, typename ConnT>
FORCE_INLINE void DuplexWriteChannel<C, ConnT>::write_with_local_copy(uint64_t dst_noc_addr, uint32_t src_l1_addr) {
    write_with_local_copy(dst_noc_addr, src_l1_addr, payload_size_);
}

template <Cast C, typename ConnT>
template <class AddrGen>
FORCE_INLINE void DuplexWriteChannel<C, ConnT>::write_page(
    uint32_t src_l1_addr, uint32_t page_idx, const AddrGen& dst) {
    write(tt::tt_fabric::linear::addrgen_detail::get_noc_address(dst, page_idx, 0), src_l1_addr, payload_size_);
}

template <Cast C, typename ConnT>
FORCE_INLINE void DuplexFusedWriteIncChannel<C, ConnT>::write_fused(
    uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr, uint32_t payload_size_bytes) {
    const tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader issue_hdr{dst_noc_addr, remote_sem_noc_addr, 0, false};
    for (uint32_t d = 0; d < DuplexConn::kNumDirections; ++d) {
        if (!conn_->has(d)) {
            continue;
        }
        if constexpr (C == Cast::Multicast) {
            linear_fabric::fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<kFusedIssueMask>(
                conn_->sender(d), hdr_[d], src_l1_addr, issue_hdr, static_cast<uint16_t>(payload_size_bytes));
        } else {
            linear_fabric::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<kFusedIssueMask>(
                conn_->sender(d), hdr_[d], src_l1_addr, issue_hdr, static_cast<uint16_t>(payload_size_bytes));
        }
    }
}

template <Cast C, typename ConnT>
FORCE_INLINE void DuplexFusedWriteIncChannel<C, ConnT>::write_fused(
    uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr) {
    write_fused(dst_noc_addr, src_l1_addr, remote_sem_noc_addr, payload_size_);
}

template <Cast C, typename ConnT>
FORCE_INLINE void DuplexFusedWriteIncChannel<C, ConnT>::write_fused_with_local_copy(
    uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr, uint32_t payload_size_bytes) {
    // Deliberately NO trailing flush — see the declaration's note: the fused free function leaves
    // flushing to the caller so it can be paired with the op's own semaphore protocol.
    write_local_mirror(dst_noc_addr, src_l1_addr, payload_size_bytes);
    write_fused(dst_noc_addr, src_l1_addr, remote_sem_noc_addr, payload_size_bytes);
}

template <Cast C, typename ConnT>
FORCE_INLINE void DuplexFusedWriteIncChannel<C, ConnT>::write_fused_with_local_copy(
    uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr) {
    write_fused_with_local_copy(dst_noc_addr, src_l1_addr, remote_sem_noc_addr, payload_size_);
}

template <Cast C, typename ConnT>
FORCE_INLINE void DuplexScatterWriteChannel<C, ConnT>::write_scatter(
    const uint64_t* dst_noc_addrs, uint32_t num_chunks, uint32_t src_l1_addr) {
    uint16_t chunk_sizes[3] = {
        static_cast<uint16_t>(chunk_size_bytes_),
        static_cast<uint16_t>(chunk_size_bytes_),
        static_cast<uint16_t>(chunk_size_bytes_)};
    const auto issue_hdr =
        tt::tt_fabric::NocUnicastScatterCommandHeader(dst_noc_addrs, chunk_sizes, static_cast<uint8_t>(num_chunks));
    const uint16_t on_wire = static_cast<uint16_t>(chunk_size_bytes_ * num_chunks);
    constexpr auto kScatterIssueMask = UnicastScatterWriteUpdateMask::DstAddrs |
                                       UnicastScatterWriteUpdateMask::ChunkSizes |
                                       UnicastScatterWriteUpdateMask::PayloadSize;
    for (uint32_t d = 0; d < DuplexConn::kNumDirections; ++d) {
        if (!conn_->has(d)) {
            continue;
        }
        if constexpr (C == Cast::Multicast) {
            linear_fabric::fabric_multicast_noc_scatter_write_with_state<kScatterIssueMask>(
                conn_->sender(d), hdr_[d], src_l1_addr, issue_hdr, on_wire);
        } else {
            linear_fabric::fabric_unicast_noc_scatter_write_with_state<kScatterIssueMask>(
                conn_->sender(d), hdr_[d], src_l1_addr, issue_hdr, on_wire);
        }
    }
}

// ----------------------------------------------------------------------------
// FabricDuplexStream — lifecycle
// ----------------------------------------------------------------------------

template <Cast C, typename ConnT>
FORCE_INLINE void FabricDuplexStream<C, ConnT>::drain() {
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}

template <Cast C, typename ConnT>
FORCE_INLINE void FabricDuplexStream<C, ConnT>::close() {
    if (!closed_) {
        closed_ = true;
        // Flush outstanding fabric writes + atomic-incs BEFORE teardown so a trailing packet on
        // either direction is never lost; then close both directions (self-gated per direction).
        drain();
        conn_->close();
    }
}

}  // namespace dataflow_kernel_lib::ccl
