// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include <cstdint>
#include <utility>

FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile tt::fabric::PacketHeader* pkt_hdr_forward,
    volatile tt::fabric::PacketHeader* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;

    size_t packet_send_size_bytes = payload_size_bytes + sizeof(tt::fabric::PacketHeader);
    pkt_hdr_forward->to_write()->to_noc_unicast(tt::fabric::NocUnicastCommandHeader{
        dest_addr, packet_send_size_bytes, static_cast<uint8_t>(dest_noc_xy.x), static_cast<uint8_t>(dest_noc_xy.y)});
    pkt_hdr_backward->to_write()->to_noc_unicast(tt::fabric::NocUnicastCommandHeader{
        dest_addr, packet_send_size_bytes, static_cast<uint8_t>(dest_noc_xy.x), static_cast<uint8_t>(dest_noc_xy.y)});

    noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_forward, sizeof(tt::fabric::PacketHeader));
    }

    if (fabric_connection.has_backward_connection()) {
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)pkt_hdr_backward, sizeof(tt::fabric::PacketHeader));
    }

    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}
