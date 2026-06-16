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
using tt::tt_fabric::common::experimental::UnicastWriteUpdateMask;

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
// FabricStreamSender — route
// ----------------------------------------------------------------------------

FORCE_INLINE void FabricStreamSender::set_route_unicast(uint32_t num_hops) {
    unicast_info_ = ccl_routing_utils::line_unicast_route_info_t{};
    // 1-D linear routing is intra-mesh and hop-distance based; dst_mesh_id is unused on
    // the LowLatencyPacketHeader path this resolves to.
    unicast_info_.dst_mesh_id = 0;
    unicast_info_.distance_in_hops = static_cast<uint16_t>(num_hops);
}

// ----------------------------------------------------------------------------
// FabricStreamSender — armed unicast-write channel
// ----------------------------------------------------------------------------

FORCE_INLINE void FabricStreamSender::arm_unicast_write(uint32_t page_size_bytes) {
    if (payload_hdr_ == nullptr) {
        payload_hdr_ = PacketHeaderPool::allocate_header();
    }
    // set_state programs the invariant on-wire payload size (+ the chip-unicast hop
    // count); the route util then writes the LowLatency 1-D routing fields (the
    // proven-correct value, applied last). Helper owns the PayloadSize mask.
    linear_fabric::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        payload_hdr_,
        static_cast<uint8_t>(unicast_info_.distance_in_hops),
        nullptr,
        static_cast<uint16_t>(align(page_size_bytes, alignment_)));
    ccl_routing_utils::fabric_set_line_unicast_route(payload_hdr_, unicast_info_);
}

FORCE_INLINE void FabricStreamSender::write(uint64_t dst_noc_addr, uint32_t src_l1_addr) {
    // with_state issues the armed payload size, updating only the destination address.
    linear_fabric::fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
        dir_, payload_hdr_, src_l1_addr, tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr});
}

template <class AddrGen>
FORCE_INLINE void FabricStreamSender::write_page(uint32_t src_l1_addr, uint32_t page_idx, const AddrGen& dst) {
    const uint64_t dst_noc_addr = tt::tt_fabric::linear::addrgen_detail::get_noc_address(dst, page_idx, 0);
    write(dst_noc_addr, src_l1_addr);
}

// ----------------------------------------------------------------------------
// FabricStreamSender — armed unicast atomic-inc channel
// ----------------------------------------------------------------------------

FORCE_INLINE void FabricStreamSender::arm_inc(uint32_t val) {
    if (sem_hdr_ == nullptr) {
        sem_hdr_ = PacketHeaderPool::allocate_header();
    }
    // set_state programs the invariant increment value + flush (the noc_address field is
    // a placeholder, filled per-issue by inc()). Helper owns the Val|Flush mask.
    linear_fabric::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        sem_hdr_,
        static_cast<uint8_t>(unicast_info_.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, val});
    ccl_routing_utils::fabric_set_line_unicast_route(sem_hdr_, unicast_info_);
}

FORCE_INLINE void FabricStreamSender::inc(uint64_t remote_sem_noc_addr) {
    // with_state issues the armed value, updating only the destination semaphore address.
    linear_fabric::fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        dir_, sem_hdr_, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_sem_noc_addr, 0});
}

}  // namespace dataflow_kernel_lib::ccl
