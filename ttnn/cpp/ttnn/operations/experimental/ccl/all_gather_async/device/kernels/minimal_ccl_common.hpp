// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <cstdint>
#include <utility>

// Enum to hold high/low priority
enum class Priority {
    LOW = 0,
    HIGH = 1,
    INVALID = 2,
};

// Function to compare two priority tensors (self and other) and return which one has higher priority
template <uint32_t priority_tensor_cb_index>
FORCE_INLINE uint32_t get_priority(const uint32_t priority_tensor_a_addr, const uint32_t priority_tensor_b_addr) {
    // Set up the AddrGen
    constexpr uint32_t tile_hw = get_tile_hw(priority_tensor_cb_index);
    const uint32_t single_tile_size_bytes = get_tile_size(priority_tensor_cb_index);
    const DataFormat data_format = get_dataformat(priority_tensor_cb_index);
    const InterleavedAddrGenFast<true, tile_hw> addr_gen_a = {
        .bank_base_address = priority_tensor_a_addr, .page_size = single_tile_size_bytes, .data_format = data_format};
    const InterleavedAddrGenFast<true, tile_hw> addr_gen_b = {
        .bank_base_address = priority_tensor_b_addr, .page_size = single_tile_size_bytes, .data_format = data_format};

    // Read the priority tensors
    int32_t priority_a = 0;
    int32_t priority_b = 0;
    uint32_t l1_write_addr = 0;
    uint32_t tile_idx = 0;

    cb_reserve_back(priority_tensor_cb_index, 1);
    l1_write_addr = get_write_ptr(priority_tensor_cb_index);
    noc_async_read_tile(tile_idx, addr_gen_a, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(priority_tensor_cb_index, 1);
    priority_a = *(uint32_t*)l1_write_addr;

    cb_reserve_back(priority_tensor_cb_index, 1);
    l1_write_addr = get_write_ptr(priority_tensor_cb_index);
    noc_async_read_tile(tile_idx, addr_gen_b, l1_write_addr);
    noc_async_read_barrier();
    cb_push_back(priority_tensor_cb_index, 1);
    priority_b = *(uint32_t*)l1_write_addr;

    // Compare the priority tensors
    if (priority_a > priority_b) {
        return static_cast<uint32_t>(Priority::HIGH);
    } else if (priority_a < priority_b) {
        return static_cast<uint32_t>(Priority::LOW);
    } else {
        ASSERT(priority_a != priority_b);  // Only asserts when watcher is enabled
        return static_cast<uint32_t>(Priority::INVALID);
    }
}

FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes,
    bool skip_local_write = false) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;

    pkt_hdr_forward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
    pkt_hdr_backward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    if (!skip_local_write) {
        noc_async_write(
            payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
    }

    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    }

    if (fabric_connection.has_backward_connection()) {
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
    }

    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

// Function does not block or wait for writes to be sent out of L1. Caller must manage synchronization
FORCE_INLINE void fused_write_atomic_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes,
    uint64_t semaphore_noc_addr,
    const uint16_t val,
    const uint16_t wrap,
    const bool flush) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;

    noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
    if (fabric_connection.has_forward_connection()) {
        pkt_hdr_forward->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                noc0_dest_noc_addr, semaphore_noc_addr, val, wrap, flush},
            payload_size_bytes);
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    }

    if (fabric_connection.has_backward_connection()) {
        pkt_hdr_backward->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                noc0_dest_noc_addr, semaphore_noc_addr, val, wrap, flush},
            payload_size_bytes);
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
    }

    l1_read_addr += payload_size_bytes;
}
