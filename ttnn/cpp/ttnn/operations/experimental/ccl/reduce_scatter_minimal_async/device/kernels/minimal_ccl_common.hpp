// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include <cstdint>
#include <utility>

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0, typename AddrGenType>
void scatter_write_and_advance_local_read_address_for_fabric(
    AddrGenType addrgen,
    uint32_t first_id,
    uint32_t second_id,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& fabric_mux_connection,
    size_t& l1_read_addr) {
    uint32_t payload_size_bytes = tt::tt_fabric::linear::addrgen_detail::get_page_size(addrgen) * 2;
    tt::tt_fabric::linear::to_noc_unicast_scatter_write(pkt_hdr, first_id, second_id, addrgen);

    tt::tt_fabric::fabric_async_write(fabric_mux_connection, pkt_hdr, l1_read_addr, payload_size_bytes);
    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0, typename AddrGenType>
void write_and_advance_local_read_address_for_fabric(
    AddrGenType addrgen,
    uint32_t id,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& fabric_mux_connection,
    size_t& l1_read_addr) {
    uint32_t payload_size_bytes = tt::tt_fabric::linear::addrgen_detail::get_page_size(addrgen);
    tt::tt_fabric::linear::to_noc_unicast_write(pkt_hdr, id, addrgen);

    tt::tt_fabric::fabric_async_write(fabric_mux_connection, pkt_hdr, l1_read_addr, payload_size_bytes);
    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}
