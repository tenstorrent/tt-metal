// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

template <bool blocking = false>
FORCE_INLINE void perform_payload_send(
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    size_t l1_read_addr,
    uint32_t payload_size_bytes,
    volatile PACKET_HEADER_TYPE* pkt_hdr) {
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_without_header_non_blocking_from_address(l1_read_addr, payload_size_bytes);
    if constexpr (blocking) {
        fabric_connection.send_payload_flush_blocking_from_address((uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
    } else {
        fabric_connection.send_payload_flush_non_blocking_from_address((uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
    }
}

template <typename AddrGenType>
FORCE_INLINE void perform_atomic_fabric_write(
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    uint32_t dest_id,
    AddrGenType addrgen,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes,
    uint64_t semaphore_noc_addr,
    const uint16_t val,
    const uint16_t wrap,
    const bool flush,
    uint32_t offset = 0) {
    tt::tt_fabric::linear::to_noc_fused_unicast_write_atomic_inc(
        payload_size_bytes,
        pkt_hdr,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_noc_addr, val, wrap, flush},
        dest_id,
        addrgen,
        offset);
    perform_payload_send(fabric_connection, l1_read_addr, payload_size_bytes, pkt_hdr);
}

FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;

    noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
    if (fabric_connection.has_forward_connection()) {
        pkt_hdr_forward->to_noc_unicast_write(
            tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
        perform_payload_send(
            fabric_connection.get_forward_connection(), l1_read_addr, payload_size_bytes, pkt_hdr_forward);
    }

    if (fabric_connection.has_backward_connection()) {
        pkt_hdr_backward->to_noc_unicast_write(
            tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
        perform_payload_send(
            fabric_connection.get_backward_connection(), l1_read_addr, payload_size_bytes, pkt_hdr_backward);
    }

    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

template <typename AddrGenType>

FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write(
    uint32_t dest_id,
    AddrGenType addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes,
    uint32_t offset = 0) {
    const size_t payload_l1_address = l1_read_addr;

    noc_async_write(payload_l1_address, addrgen.get_noc_addr(dest_id, offset), payload_size_bytes);

    if (fabric_connection.has_forward_connection()) {
        tt::tt_fabric::linear::to_noc_unicast_write(payload_size_bytes, pkt_hdr_forward, dest_id, addrgen, offset);
        perform_payload_send(
            fabric_connection.get_forward_connection(), l1_read_addr, payload_size_bytes, pkt_hdr_forward);
    }

    if (fabric_connection.has_backward_connection()) {
        tt::tt_fabric::linear::to_noc_unicast_write(payload_size_bytes, pkt_hdr_backward, dest_id, addrgen, offset);
        perform_payload_send(
            fabric_connection.get_backward_connection(), l1_read_addr, payload_size_bytes, pkt_hdr_backward);
    }

    noc_async_writes_flushed();

    l1_read_addr += payload_size_bytes;
}

template <bool advance = false, uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
void scatter_write_for_fabric_write(
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
    if constexpr (advance) {
        l1_read_addr += first_payload_size_bytes + second_payload_size_bytes;
    }
}

template <bool advance = false, uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0, typename AddrGenType>
void scatter_write_for_fabric_write(
    AddrGenType addrgen,
    uint32_t first_id,
    uint32_t second_id,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& fabric_mux_connection,
    size_t& l1_read_addr,
    uint32_t page_size_bytes,
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
    tt::tt_fabric::linear::to_noc_unicast_scatter_write(
        page_size_bytes, pkt_hdr, first_id, second_id, addrgen, offset0, offset1);
    tt::tt_fabric::fabric_async_write(fabric_mux_connection, pkt_hdr, l1_read_addr, page_size_bytes * 2);
    noc_async_writes_flushed();
    if constexpr (advance) {
        l1_read_addr += page_size_bytes * 2;
    }
}

template <bool advance = false, uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
void write_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& fabric_mux_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);

    tt::tt_fabric::fabric_async_write(
        payload_size_bytes, fabric_mux_connection, pkt_hdr, l1_read_addr, payload_size_bytes);
    noc_async_writes_flushed();
    if constexpr (advance) {
        l1_read_addr += payload_size_bytes;
    }
}

template <bool advance = false, uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0, typename AddrGenType>
void write_for_fabric_write(
    AddrGenType addrgen,
    uint32_t id,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& fabric_mux_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes,
    uint32_t offset = 0) {
    tt::tt_fabric::linear::to_noc_unicast_write(payload_size_bytes, pkt_hdr, id, addrgen, offset);

    tt::tt_fabric::fabric_async_write(fabric_mux_connection, pkt_hdr, l1_read_addr, payload_size_bytes);
    noc_async_writes_flushed();

    if constexpr (advance) {
        l1_read_addr += payload_size_bytes;
    }
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
        perform_payload_send(
            fabric_connection.get_forward_connection(), l1_read_addr, payload_size_bytes, pkt_hdr_forward);
    }

    if (fabric_connection.has_backward_connection()) {
        pkt_hdr_backward->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                noc0_dest_noc_addr, semaphore_noc_addr, val, wrap, flush},
            payload_size_bytes);
        perform_payload_send(
            fabric_connection.get_backward_connection(), l1_read_addr, payload_size_bytes, pkt_hdr_backward);
    }

    l1_read_addr += payload_size_bytes;
}

template <typename AddrGenType>
FORCE_INLINE void fused_write_atomic_and_advance_local_read_address_for_fabric_write(
    uint32_t dest_id,
    AddrGenType addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes,
    uint64_t semaphore_noc_addr,
    const uint16_t val,
    const uint16_t wrap,
    const bool flush,
    uint32_t offset = 0) {
    // This assumes payload size equals page size
    const size_t payload_l1_address = l1_read_addr;
    noc_async_write(payload_l1_address, addrgen.get_noc_addr(dest_id, offset), payload_size_bytes);
    if (fabric_connection.has_forward_connection()) {
        perform_atomic_fabric_write(
            pkt_hdr_forward,
            dest_id,
            addrgen,
            fabric_connection.get_forward_connection(),
            l1_read_addr,
            payload_size_bytes,
            semaphore_noc_addr,
            val,
            wrap,
            flush,
            offset);
    }

    if (fabric_connection.has_backward_connection()) {
        perform_atomic_fabric_write(
            pkt_hdr_backward,
            dest_id,
            addrgen,
            fabric_connection.get_backward_connection(),
            l1_read_addr,
            payload_size_bytes,
            semaphore_noc_addr,
            val,
            wrap,
            flush,
            offset);
    }

    l1_read_addr += payload_size_bytes;
}

FORCE_INLINE void fabric_write_unidir(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_direction_connection,
    size_t l1_read_addr,
    uint32_t payload_size_bytes) {
    pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
    perform_payload_send(fabric_direction_connection, l1_read_addr, payload_size_bytes, pkt_hdr);
    noc_async_writes_flushed();
}

template <typename AddrGenType>
FORCE_INLINE void fabric_write_unidir(
    uint32_t dest_id,
    AddrGenType addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_direction_connection,
    size_t l1_read_addr,
    uint32_t payload_size_bytes,
    uint32_t offset = 0) {
    tt::tt_fabric::linear::to_noc_unicast_write(payload_size_bytes, pkt_hdr, dest_id, addrgen, offset);
    perform_payload_send(fabric_direction_connection, l1_read_addr, payload_size_bytes, pkt_hdr);
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
    const size_t payload_l1_address = l1_read_addr;

    pkt_hdr->to_noc_unicast_scatter_write(
        tt::tt_fabric::NocUnicastScatterCommandHeader{
            noc0_dest_noc_addr, noc0_dest_noc_addr_next_core, payload_size_bytes_first_core},
        payload_size_bytes_first_core * 2);

    perform_payload_send(fabric_direction_connection, l1_read_addr, payload_size_bytes_first_core * 2, pkt_hdr);
    noc_async_writes_flushed();
}

template <typename AddrGenType>
FORCE_INLINE void scatter_fabric_write_unidir(
    uint32_t first_id,
    uint32_t second_id,
    AddrGenType addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_direction_connection,
    size_t l1_read_addr,
    uint16_t payload_size,
    uint32_t offset0 = 0,
    uint32_t offset1 = 0) {
    const size_t payload_l1_address = l1_read_addr;
    tt::tt_fabric::linear::to_noc_unicast_scatter_write(
        payload_size, pkt_hdr, first_id, second_id, addrgen, offset0, offset1);

    perform_payload_send(fabric_direction_connection, l1_read_addr, payload_size * 2, pkt_hdr);
    noc_async_writes_flushed();
}
