// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <cstdint>
#include <utility>

FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write_forward(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);

    pkt_hdr_forward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    fabric_connection.get_forward_connection().wait_for_empty_write_slot();
    fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
        l1_read_addr, payload_size_bytes);
    fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
        (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write_backward(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);

    pkt_hdr_backward->to_noc_unicast_write(
        tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    fabric_connection.get_backward_connection().wait_for_empty_write_slot();
    fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
        l1_read_addr, payload_size_bytes);
    fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
        (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

FORCE_INLINE void fabric_write_unidir(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_direction_connection,
    size_t l1_read_addr,
    uint32_t payload_size_bytes) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);

    pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    fabric_direction_connection.wait_for_empty_write_slot();
    fabric_direction_connection.send_payload_without_header_non_blocking_from_address(l1_read_addr, payload_size_bytes);
    fabric_direction_connection.send_payload_flush_non_blocking_from_address(
        (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();
}

FORCE_INLINE void scatter_fabric_write_unidir(
    uint64_t noc0_dest_noc_addr,
    uint64_t noc0_dest_noc_addr_next_core,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_direction_connection,
    size_t l1_read_addr,
    uint16_t payload_size_bytes_first_core,
    uint32_t payload_size_bytes_second_core) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const auto [dest_noc_xy_next_core, dest_addr_next_core] = get_noc_address_components(noc0_dest_noc_addr_next_core);
    const size_t payload_l1_address = l1_read_addr;

    pkt_hdr->to_noc_unicast_scatter_write(
        tt::tt_fabric::NocUnicastScatterCommandHeader{
            noc0_dest_noc_addr, noc0_dest_noc_addr_next_core, payload_size_bytes_first_core},
        payload_size_bytes_first_core + payload_size_bytes_second_core);

    fabric_direction_connection.wait_for_empty_write_slot();
    fabric_direction_connection.send_payload_without_header_non_blocking_from_address(
        l1_read_addr, payload_size_bytes_first_core + payload_size_bytes_second_core);
    fabric_direction_connection.send_payload_flush_non_blocking_from_address(
        (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));

    noc_async_writes_flushed();
}
