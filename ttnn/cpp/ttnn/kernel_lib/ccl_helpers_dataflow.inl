// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for ccl_helpers_dataflow.hpp
// Do not include directly - include ccl_helpers_dataflow.hpp instead

#pragma once

namespace dataflow_kernel_lib::ccl {

// ----------------------------------------------------------------------------
// FabricStreamSender — construction / lifecycle
// ----------------------------------------------------------------------------

FORCE_INLINE FabricStreamSender::FabricStreamSender(size_t& conn_arg_idx, bool is_forward, uint32_t alignment) :
    conn_(FabricConnectionManager::build_from_args<
          FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(conn_arg_idx)),
    alignment_(alignment),
    is_forward_(is_forward) {}

FORCE_INLINE void FabricStreamSender::open() {
    conn_.open_finish();
    dir_ = is_forward_ ? &conn_.get_forward_connection() : &conn_.get_backward_connection();
}

FORCE_INLINE void FabricStreamSender::close() { conn_.close(); }

// ----------------------------------------------------------------------------
// FabricStreamSender — route + lazy header
// ----------------------------------------------------------------------------

FORCE_INLINE void FabricStreamSender::set_route_unicast(uint32_t num_hops) {
    unicast_info_ = ccl_routing_utils::line_unicast_route_info_t{};
    // 1-D linear routing is intra-mesh and hop-distance based; dst_mesh_id is unused on
    // the LowLatencyPacketHeader path this resolves to.
    unicast_info_.dst_mesh_id = 0;
    unicast_info_.distance_in_hops = static_cast<uint16_t>(num_hops);
    route_set_ = true;
    if (payload_hdr_ != nullptr) {
        ccl_routing_utils::fabric_set_line_unicast_route(payload_hdr_, unicast_info_);
    }
}

FORCE_INLINE void FabricStreamSender::ensure_payload_header() {
    if (payload_hdr_ == nullptr) {
        payload_hdr_ = PacketHeaderPool::allocate_header();
        if (route_set_) {
            ccl_routing_utils::fabric_set_line_unicast_route(payload_hdr_, unicast_info_);
        }
    }
}

// ----------------------------------------------------------------------------
// FabricStreamSender — writes + atomic-inc
// ----------------------------------------------------------------------------

template <class AddrGen>
FORCE_INLINE void FabricStreamSender::write_page(
    uint32_t src_l1_addr, uint32_t size_bytes, uint32_t page_idx, const AddrGen& dst) {
    ensure_payload_header();
    // Header carries the alignment-rounded on-wire size; the payload send moves the
    // actual bytes (mirrors point_to_point's writer_send.cpp).
    tt::tt_fabric::linear::to_noc_unicast_write(align(size_bytes, alignment_), payload_hdr_, page_idx, dst);
    perform_payload_send(*dir_, src_l1_addr, size_bytes, payload_hdr_);
}

FORCE_INLINE void FabricStreamSender::inc_remote(uint64_t remote_sem_noc_addr, uint32_t val) {
    if (sem_hdr_ == nullptr) {
        sem_hdr_ = PacketHeaderPool::allocate_header();
    }
    if (route_set_) {
        ccl_routing_utils::fabric_set_line_unicast_route(sem_hdr_, unicast_info_);
    }
    sem_hdr_->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_sem_noc_addr, val});
    dir_->wait_for_empty_write_slot();
    dir_->send_payload_flush_blocking_from_address((uint32_t)sem_hdr_, sizeof(PACKET_HEADER_TYPE));
}

}  // namespace dataflow_kernel_lib::ccl
