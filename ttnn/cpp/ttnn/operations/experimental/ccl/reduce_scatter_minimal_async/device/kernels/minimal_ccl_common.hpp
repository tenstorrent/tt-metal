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

#ifdef ARCH_WORMHOLE
template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
void scatter_write_and_advance_local_read_address_for_fabric(
    uint64_t first_noc0_dest_noc_addr,
    uint64_t second_noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& fabric_mux_connection,
    size_t& l1_read_addr,
    uint32_t first_payload_size_bytes,
    uint32_t second_payload_size_bytes) {
    pkt_hdr->to_noc_unicast_scatter_write(
        tt::tt_fabric::NocUnicastScatterCommandHeader{
            {first_noc0_dest_noc_addr, second_noc0_dest_noc_addr}, (uint16_t)first_payload_size_bytes},
        first_payload_size_bytes + second_payload_size_bytes);

    tt::tt_fabric::fabric_async_write(
        fabric_mux_connection, pkt_hdr, l1_read_addr, first_payload_size_bytes + second_payload_size_bytes);
    noc_async_writes_flushed();

    l1_read_addr += first_payload_size_bytes + second_payload_size_bytes;
}
#endif

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
void write_and_advance_local_read_address_for_fabric(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& fabric_mux_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    tt::tt_fabric::fabric_async_write(fabric_mux_connection, pkt_hdr, l1_read_addr, payload_size_bytes);
    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}
