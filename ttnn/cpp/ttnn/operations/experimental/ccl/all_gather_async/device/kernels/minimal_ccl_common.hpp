// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <cstdint>
#include <utility>

FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;

    pkt_hdr_forward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
    pkt_hdr_backward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    }

    if (fabric_connection.has_backward_connection()) {
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
            (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
    }

    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

static inline void local_barrier(
    FabricConnectionManager& fabric_connection,
    const size_t semaphore,
    uint32_t wait_num,
    const uint8_t sem_noc0_x,
    const uint8_t sem_noc0_y,
    uint32_t pkt_hdr_buffer_seminc,
    uint32_t num_targets_forward_direction,
    uint32_t num_targets_backward_direction,
    bool wait_semaphore,
    bool reset_semaphore) {
    // 2. mcast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt = safe_get_noc_addr(sem_noc0_x, sem_noc0_y, semaphore, 0);
    auto pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(pkt_hdr_buffer_seminc);
    pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the mcast packet (forward)
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            pkt_hdr_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
            pkt_hdr_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr = safe_get_noc_addr(sem_noc0_x, sem_noc0_y, semaphore);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);
    DPRINT << "inc done\n";

    // 3. wait for mcast output ready semaphore
    if (wait_semaphore) {
        while (*reinterpret_cast<volatile uint32_t*>(semaphore) < wait_num);
        DPRINT << "waitval done\n";
    }

    // 4. global semaphore reset
    if (reset_semaphore) {
        const uint64_t dest_noc_addr = get_noc_addr(my_x[0], my_y[0], semaphore);
        noc_inline_dw_write(dest_noc_addr, 0);
        DPRINT << "reset done\n";
    }
}

// Two phase barrier
template <bool byTwoPhase>
inline void ccl_barrier(
    FabricConnectionManager& fabric_connection,
    const size_t semaphore_wait,
    const size_t semaphore_release,
    uint32_t wait_num,
    const uint8_t sem_noc0_x,
    const uint8_t sem_noc0_y,
    uint32_t pkt_hdr_buffer_seminc,
    uint32_t num_targets_forward_direction,
    uint32_t num_targets_backward_direction,
    bool wait_semaphore,
    bool reset_semaphore) {
    // WAIT phase: ensure local chip got N signals from other N chips
    DPRINT << "ccl_barrier: WAIT phase\n";
    local_barrier(
        fabric_connection,
        semaphore_wait,
        wait_num,
        sem_noc0_x,
        sem_noc0_y,
        pkt_hdr_buffer_seminc,
        num_targets_forward_direction,
        num_targets_backward_direction,
        wait_semaphore,
        reset_semaphore);

    // This "if" is for backward compatibility
    if constexpr (byTwoPhase) {
        // RELEASE phase: ensure all chip got N signals from other N chips
        DPRINT << "ccl_barrier: RELEASE phase\n";
        local_barrier(
            fabric_connection,
            semaphore_release,
            wait_num,
            sem_noc0_x,
            sem_noc0_y,
            pkt_hdr_buffer_seminc,
            num_targets_forward_direction,
            num_targets_backward_direction,
            wait_semaphore,
            reset_semaphore);
    }
}
