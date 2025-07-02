// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <cstdint>
#include <utility>

FORCE_INLINE void scatter_write_and_advance_local_read_address_for_fabric(
    uint64_t first_noc0_dest_noc_addr,
    uint64_t second_noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricEdmSender* fabric_direction_connection,
    size_t& l1_read_addr,
    uint32_t first_payload_size_bytes,
    uint32_t second_payload_size_bytes) {
    pkt_hdr->to_noc_unicast_scatter_write(
        tt::tt_fabric::NocUnicastScatterCommandHeader{
            {first_noc0_dest_noc_addr, second_noc0_dest_noc_addr}, (uint16_t)first_payload_size_bytes},
        first_payload_size_bytes + second_payload_size_bytes);

    fabric_direction_connection->wait_for_empty_write_slot();
    fabric_direction_connection->send_payload_without_header_non_blocking_from_address(
        l1_read_addr, first_payload_size_bytes + second_payload_size_bytes);
    fabric_direction_connection->send_payload_flush_non_blocking_from_address(
        (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();

    l1_read_addr += first_payload_size_bytes + second_payload_size_bytes;
}

FORCE_INLINE void write_and_advance_local_read_address_for_fabric(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricEdmSender* fabric_direction_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    fabric_direction_connection->wait_for_empty_write_slot();
    fabric_direction_connection->send_payload_without_header_non_blocking_from_address(
        l1_read_addr, payload_size_bytes);
    fabric_direction_connection->send_payload_flush_non_blocking_from_address(
        (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}
