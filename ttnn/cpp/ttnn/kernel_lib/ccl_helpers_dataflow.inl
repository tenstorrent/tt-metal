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

// ============================================================================================
// Out-of-line implementations of the methods declared inline in the .hpp interface. Moved here
// per the ".hpp = documentation + declarations, .inl = implementation" convention. The
// connection policies (DirectConn / MuxConn), the FabricStream move ctor, and the
// FabricStreamSender ctors + open() live here; the channel-arming / lifecycle and
// FabricStreamSender::signal implementations follow in their original sections below.
// ============================================================================================

// ----------------------------------------------------------------------------
// DirectConn — direct fabric-connection policy
// ----------------------------------------------------------------------------

FORCE_INLINE DirectConn::DirectConn(size_t& conn_arg_idx, bool is_forward) :
    conn_(FabricConnectionManager::build_from_args<
          FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx)),
    is_forward_(is_forward) {}

FORCE_INLINE void DirectConn::open() {
    conn_.open_finish();
    dir_ = is_forward_ ? &conn_.get_forward_connection() : &conn_.get_backward_connection();
}

FORCE_INLINE void DirectConn::close() { conn_.close(); }

FORCE_INLINE DirectConn::SenderT* DirectConn::sender() { return dir_; }

// ----------------------------------------------------------------------------
// MuxConn<NumBuffers> — worker-mux fabric-connection policy
// ----------------------------------------------------------------------------

template <uint8_t NumBuffers>
FORCE_INLINE MuxConn<NumBuffers>::MuxConn(
    size_t& arg_idx,
    size_t channel_buffer_size_bytes,
    size_t status_address,
    size_t termination_signal_address,
    uint32_t num_mux_clients) :
    termination_signal_address_(termination_signal_address), num_mux_clients_(num_mux_clients) {
    valid_ = get_arg_val<uint32_t>(arg_idx++) == 1;
    is_termination_master_ = get_arg_val<uint32_t>(arg_idx++);
    mux_x_ = get_arg_val<uint32_t>(arg_idx++);
    mux_y_ = get_arg_val<uint32_t>(arg_idx++);
    const size_t channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t channel_id = get_arg_val<uint32_t>(arg_idx++);
    termination_sync_address_ = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    termination_master_noc_x_ = get_arg_val<uint32_t>(arg_idx++);
    termination_master_noc_y_ = get_arg_val<uint32_t>(arg_idx++);
    if (valid_) {
        mux_ = tt::tt_fabric::build_connection_to_fabric_endpoint<NumBuffers>(
            mux_x_,
            mux_y_,
            channel_id,
            NumBuffers,
            channel_buffer_size_bytes,
            channel_base_address,
            connection_info_address,
            connection_handshake_address,
            flow_control_address,
            buffer_index_address,
            local_flow_control_address,
            local_teardown_address,
            local_buffer_index_address);
        // The mux endpoint is a separate kernel; block until it is ready to accept connections.
        tt::tt_fabric::wait_for_fabric_endpoint_ready(mux_x_, mux_y_, status_address, local_status_address);
    }
}

template <uint8_t NumBuffers>
FORCE_INLINE void MuxConn<NumBuffers>::open() {
    if (valid_) {
        tt::tt_fabric::fabric_client_connect(mux_);
    }
}

template <uint8_t NumBuffers>
FORCE_INLINE void MuxConn<NumBuffers>::close() {
    if (!valid_) {
        return;
    }
    tt::tt_fabric::fabric_client_disconnect(mux_);
    if (is_termination_master_) {
        auto* termination_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_sync_address_);
        noc_semaphore_wait(termination_sync_ptr, num_mux_clients_ - 1);
        tt::tt_fabric::fabric_endpoint_terminate(mux_x_, mux_y_, termination_signal_address_);
    } else {
        const uint64_t dest_addr =
            safe_get_noc_addr(termination_master_noc_x_, termination_master_noc_y_, termination_sync_address_, 0);
        noc_semaphore_inc(dest_addr, 1);
        noc_async_atomic_barrier();
    }
}

template <uint8_t NumBuffers>
FORCE_INLINE typename MuxConn<NumBuffers>::SenderT* MuxConn<NumBuffers>::sender() {
    return valid_ ? &mux_ : nullptr;
}

// ----------------------------------------------------------------------------
// FabricStream — move ctor
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE FabricStream<ConnT>::FabricStream(FabricStream&& o) :
    conn_(o.conn_), alignment_(o.alignment_), route_(o.route_), closed_(o.closed_) {
    o.closed_ = true;
}

// ----------------------------------------------------------------------------
// FabricStreamSender — ctors + open()
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE FabricStreamSender<ConnT>::FabricStreamSender(
    size_t& conn_arg_idx, bool is_forward, uint32_t alignment) :
    conn_(conn_arg_idx, is_forward), alignment_(alignment) {}

template <typename ConnT>
FORCE_INLINE FabricStreamSender<ConnT>::FabricStreamSender(ConnT conn, uint32_t alignment) :
    conn_(conn), alignment_(alignment) {}

template <typename ConnT>
FORCE_INLINE FabricStream<ConnT> FabricStreamSender<ConnT>::open(
    const ccl_routing_utils::line_unicast_route_info_t& route) {
    conn_.open();
    return FabricStream<ConnT>(&conn_, alignment_, route);
}

// ----------------------------------------------------------------------------
// FabricStream — armed unicast-write channel
// ----------------------------------------------------------------------------

template <typename ConnT>
FORCE_INLINE UnicastWriteChannel<ConnT> FabricStream<ConnT>::arm_unicast_write(uint32_t page_size_bytes) {
    // Each arm draws its OWN pooled header, which the returned channel then owns — arming the same
    // channel type twice yields two independent channels, never two aliases of one mutable slot.
    auto* hdr = PacketHeaderPool::allocate_header();
    // set_state programs the invariant on-wire payload size (+ the chip-unicast hop count); the
    // route util then writes the LowLatency 1-D routing fields (the proven-correct value, applied
    // last). Helper owns the PayloadSize mask. The route is the stream's, bound at open().
    linear_fabric::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        hdr,
        static_cast<uint8_t>(route_.distance_in_hops),
        nullptr,
        static_cast<uint16_t>(align(page_size_bytes, alignment_)));
    ccl_routing_utils::fabric_set_line_unicast_route(hdr, route_);
    return UnicastWriteChannel<ConnT>(conn_, hdr);
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
    auto* hdr = PacketHeaderPool::allocate_header();
    // set_state establishes the scatter send type + the stream's route + invariant chunk
    // sizes/payload; the dst addrs (and per-packet chunk count) are filled per-issue by write_scatter.
    uint64_t dummy_addrs[4] = {0, 0, 0, 0};
    uint16_t chunk_sizes[3] = {
        static_cast<uint16_t>(chunk_size_bytes),
        static_cast<uint16_t>(chunk_size_bytes),
        static_cast<uint16_t>(chunk_size_bytes)};
    linear_fabric::fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        hdr,
        static_cast<uint8_t>(route_.distance_in_hops),
        tt::tt_fabric::NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, static_cast<uint8_t>(num_chunks)),
        static_cast<uint16_t>(chunk_size_bytes * num_chunks));
    ccl_routing_utils::fabric_set_line_unicast_route(hdr, route_);
    return ScatterWriteChannel<ConnT>(conn_, hdr, chunk_size_bytes);
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
    auto* hdr = PacketHeaderPool::allocate_header();
    // set_state programs the invariant increment value + flush (the noc_address field is a
    // placeholder, filled per-issue by inc()). Helper owns the Val|Flush mask; route is the
    // stream's, bound at open().
    linear_fabric::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        hdr,
        static_cast<uint8_t>(route_.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, val});
    ccl_routing_utils::fabric_set_line_unicast_route(hdr, route_);
    return AtomicIncChannel<ConnT>(conn_, hdr);
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
    auto* hdr = PacketHeaderPool::allocate_header();
    // set_state programs the invariant inc value + flush on a dedicated multicast header for a
    // MULTICAST route; the dst sem addr is filled per-issue by multicast_inc. Independent of the
    // unicast atomic-inc channel, so the barrier (multicast) and counting (unicast) may coexist.
    linear_fabric::fabric_multicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        hdr,
        static_cast<uint8_t>(route.start_distance_in_hops),
        static_cast<uint8_t>(route.range_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, val});
    ccl_routing_utils::fabric_set_line_multicast_route(hdr, route);
    return MulticastIncChannel<ConnT>(conn_, hdr);
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

template <typename ConnT>
FORCE_INLINE void FabricStreamSender<ConnT>::signal_once(
    size_t& conn_arg_idx,
    bool is_forward,
    uint32_t alignment,
    const ccl_routing_utils::line_unicast_route_info_t& route,
    uint64_t remote_sem_noc_addr,
    uint32_t val) {
    // Fold the whole one-shot handshake — build the connection from args, open, arm the inc, issue
    // it, tear down — so a caller never has to name (or forget to tear down) an intermediate sender.
    FabricStreamSender<ConnT> sender(conn_arg_idx, is_forward, alignment);
    sender.signal(route, remote_sem_noc_addr, val);
}

}  // namespace dataflow_kernel_lib::ccl
