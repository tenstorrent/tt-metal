// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// CCL Kernel common includes
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/command_interpreter_base.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/ccl_command_base.hpp"

// Metal includes
#include "dataflow_api.h"

// System includes
#include <cstdint>
#include "debug/dprint.h"

template <typename CclCommandHeader>
FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    size_t packet_header_buffer_addr,
    const CclCommandHeader& current_cmd_header,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    const size_t payload_l1_address = l1_read_addr;

    auto pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
#ifdef DEBUG_PRINT_ENABLED
    pkt_hdr->reserved2 = my_chip_id;
#endif

    pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    switch (current_cmd_header.dest_type) {
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST: {
            const auto& unicast_args = current_cmd_header.get_unicast_dest_args();
            auto& fabric_conn = unicast_args.is_forward_direction ? fabric_connection.get_forward_connection()
                                                                  : fabric_connection.get_backward_connection();

            pkt_hdr->to_chip_unicast(unicast_args.distance_in_hops);
            fabric_conn.wait_for_empty_write_slot();
            fabric_conn.send_payload_without_header_non_blocking_from_address(l1_read_addr, payload_size_bytes);
            fabric_conn.send_payload_flush_blocking_from_address((uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
        } break;
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST: {
            noc_async_write(
                payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
            const auto& mcast_args = current_cmd_header.get_multicast_dest_args();
            if (fabric_connection.has_forward_connection()) {
                pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                    1, static_cast<uint8_t>(mcast_args.num_targets_forward_direction)});
                fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
                    l1_read_addr, payload_size_bytes);
                fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
            }

            if (fabric_connection.has_backward_connection()) {
                pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                    1, static_cast<uint8_t>(mcast_args.num_targets_backward_direction)});
                fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
                    l1_read_addr, payload_size_bytes);
                fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
            }
        } break;
        default: {
            ASSERT(false);
        } break;
    }

    l1_read_addr += payload_size_bytes;
}

template <typename CclCommandHeader>
FORCE_INLINE void write_payload_then_advance_read_address(
    uint64_t noc0_dest_noc_addr,
    size_t packet_header_buffer_addr,
    const CclCommandHeader& current_cmd_header,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    size_t payload_size_bytes) {
    static_assert(
        is_power_of_2(sizeof(PACKET_HEADER_TYPE)),
        "sizeof(tt::tt_fabric::PacketHeader) is not a power of two which violates the below assertion");

    switch (current_cmd_header.dest_type) {
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_UNICAST: [[fallthrough]];
        case ttnn::ccl::cmd::CclCommandDestType::CHIP_MULTICAST:
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                packet_header_buffer_addr,
                current_cmd_header,
                fabric_connection,
                l1_read_addr,
                payload_size_bytes);
            break;

        case ttnn::ccl::cmd::CclCommandDestType::CHIP_LOCAL_ONLY: {
            const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
            // Convert to our local noc_index based address
            noc_async_write(
                l1_read_addr, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
            l1_read_addr += payload_size_bytes;
        } break;
    }
}
