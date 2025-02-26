// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_types.hpp"
#include <cstdint>

// If the hop/distance counter equals to the below value, it indicates that it has
// arrived at (atleast one of) the intended destination(s)
static constexpr size_t DESTINATION_HOP_COUNT = 1;
// TODO: make 0 and the associated field to num mcast destinations
static constexpr size_t LAST_MCAST_DESTINATION = 1;

FORCE_INLINE void print_pkt_hdr_routing_fields(volatile tt::fabric::PacketHeader *const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    switch (packet_start->chip_send_type) {
        case tt::fabric::CHIP_UNICAST: {
            DPRINT << "C_UNI: dist:" << (uint32_t) (packet_start->routing_fields.value & tt::fabric::RoutingFields::HOP_DISTANCE_MASK) << "\n";
            break;
        }
        case tt::fabric::CHIP_MULTICAST: {
            DPRINT << "C_MCST: dist:" << (uint32_t) (packet_start->routing_fields.value & tt::fabric::RoutingFields::HOP_DISTANCE_MASK) <<
                ", rng:" << (uint32_t)((packet_start->routing_fields.value & tt::fabric::RoutingFields::RANGE_MASK) >> tt::fabric::RoutingFields::START_DISTANCE_FIELD_BIT_WIDTH)  << "\n";
            break;
        }
    };
#endif
}

FORCE_INLINE void print_pkt_hdr_routing_fields(volatile tt::fabric::LowLatencyPacketHeader *const packet_start) {
    #ifdef DEBUG_PRINT_ENABLED
        DPRINT << "ROUTE:" << packet_start->routing_fields.value << "\n";
    #endif
}

template <typename T>
FORCE_INLINE void print_pkt_header_noc_fields(volatile T *const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    switch (packet_start->noc_send_type) {
        case tt::fabric::NocSendType::NOC_UNICAST_WRITE: {
                DPRINT << "N_WR addr:"<<(uint64_t)packet_start->command_fields.unicast_write.noc_address << "\n";
        } break;
        case tt::fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            DPRINT << "N_WR addr:"<<(uint64_t)packet_start->command_fields.unicast_seminc.noc_address <<
                ", val:" << (uint32_t) packet_start->command_fields.unicast_seminc.val << "\n";

        } break;
        default:
        ASSERT(false); // unimplemented
        break;
    };
#endif
}

FORCE_INLINE void print_pkt_header(volatile tt::fabric::PacketHeader *const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    auto const& header = *packet_start;
    DPRINT << "PKT: nsnd_t:" << (uint32_t) packet_start->noc_send_type <<
        ", csnd_t:" << (uint32_t) packet_start->chip_send_type <<
        ", src_chip:" << (uint32_t) packet_start->src_ch_id <<
        ", payload_size_bytes:" << (uint32_t) packet_start->payload_size_bytes << "\n";
    print_pkt_hdr_routing_fields(packet_start);
    print_pkt_header_noc_fields(packet_start);
#endif
}

FORCE_INLINE void print_pkt_header(volatile tt::fabric::LowLatencyPacketHeader *const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    auto const& header = *packet_start;
    DPRINT << "PKT: nsnd_t:" << (uint32_t) packet_start->noc_send_type <<
        ", src_chip:" << (uint32_t) packet_start->src_ch_id <<
        ", payload_size_bytes:" << (uint32_t) packet_start->payload_size_bytes << "\n";
    print_pkt_hdr_routing_fields(packet_start);
    print_pkt_header_noc_fields(packet_start);
#endif
}


// Since we unicast to local, we must omit the packet header
FORCE_INLINE void execute_chip_unicast_to_local_chip(
    volatile PACKET_HEADER_TYPE *const packet_start, uint16_t payload_size_bytes, uint32_t transaction_id) {
    auto const& header = *packet_start;
    uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) + sizeof(PACKET_HEADER_TYPE);

    tt::fabric::NocSendType noc_send_type = packet_start->noc_send_type;
    switch (noc_send_type) {
        case tt::fabric::NocSendType::NOC_UNICAST_WRITE: {
            auto const dest_address = header.command_fields.unicast_write.noc_address;
            noc_async_write_one_packet_with_trid(payload_start_address, dest_address, payload_size_bytes, transaction_id);
        } break;

        case tt::fabric::NocSendType::NOC_MULTICAST_WRITE: {
            // TODO: confirm if we need to adjust dest core count if we span eth or dram cores
            auto const mcast_dest_address = get_noc_multicast_addr(
                header.command_fields.mcast_write.noc_x_start,
                header.command_fields.mcast_write.noc_y_start,
                header.command_fields.mcast_write.noc_x_start + header.command_fields.mcast_write.mcast_rect_size_x,
                header.command_fields.mcast_write.noc_y_start + header.command_fields.mcast_write.mcast_rect_size_y,
                header.command_fields.mcast_write.address);
            auto const num_dests = header.command_fields.mcast_write.mcast_rect_size_x * header.command_fields.mcast_write.mcast_rect_size_y;
            noc_async_write_one_packet_with_trid(payload_start_address, mcast_dest_address, payload_size_bytes, num_dests, transaction_id);
        } break;

        case tt::fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            uint64_t const dest_address = header.command_fields.unicast_seminc.noc_address;
            auto const increment = header.command_fields.unicast_seminc.val;
            noc_semaphore_inc(dest_address, increment);

        } break;

        case tt::fabric::NocSendType::NOC_UNICAST_INLINE_WRITE: {
            auto const dest_address = header.command_fields.unicast_inline_write.noc_address;
            auto const value = header.command_fields.unicast_inline_write.value;
            noc_inline_dw_write(dest_address, value);
        } break;

        case tt::fabric::NocSendType::NOC_MULTICAST_ATOMIC_INC:
        default: {
            ASSERT(false);
        } break;
    };
}

FORCE_INLINE void update_packet_header_for_next_hop(volatile tt::fabric::PacketHeader * packet_header, tt::fabric::RoutingFields cached_routing_fields) {
    // if the distance field is one, it means the range field decrements, else the start distance field decrements
    // TODO [optimization]: If we can make the terminal value 0, then we can save an instruction on the eq insn
    bool decrement_range = (cached_routing_fields.value & tt::fabric::RoutingFields::HOP_DISTANCE_MASK) == tt::fabric::RoutingFields::LAST_HOP_DISTANCE_VAL;
    uint8_t decrement_val = static_cast<uint8_t>(1) << (decrement_range * tt::fabric::RoutingFields::RANGE_HOPS_FIELD_BIT_WIDTH);
    packet_header->routing_fields.value = cached_routing_fields.value - decrement_val;
}

FORCE_INLINE void update_packet_header_for_next_hop(volatile tt::fabric::LowLatencyPacketHeader * packet_header, tt::fabric::LowLatencyRoutingFields cached_routing_fields) {
    packet_header->routing_fields.value >>= tt::fabric::LowLatencyRoutingFields::FIELD_WIDTH;
}

// This function forwards a packet to the downstream EDM channel for eventual sending
// to the next chip in the line/ring
//
// Modifies the packet header (decrements hop counts) so ...
//
// !!!WARNING!!!
// !!!WARNING!!! * do NOT call before determining if the packet should be consumed locally or forwarded
// !!!WARNING!!! * ENSURE DOWNSTREAM EDM HAS SPACE FOR PACKET BEFORE CALLING
// !!!WARNING!!!
template <uint8_t NUM_SENDER_BUFFERS>
FORCE_INLINE void forward_payload_to_downstream_edm(
    volatile PACKET_HEADER_TYPE *packet_header,
    uint16_t payload_size_bytes,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    tt::fabric::EdmToEdmSender<NUM_SENDER_BUFFERS> &downstream_edm_interface,
    uint8_t transaction_id
    ) {
    // TODO: PERF - this should already be getting checked by the caller so this should be redundant make it an ASSERT
    ASSERT(downstream_edm_interface.edm_has_space_for_packet()); // best effort check

    // This is a good place to print the packet header for debug if you are trying to inspect packets
    // because it is before we start manipulating the header for forwarding
    update_packet_header_for_next_hop(packet_header, cached_routing_fields);
    downstream_edm_interface.send_payload_non_blocking_from_address_with_trid(
        reinterpret_cast<size_t>(packet_header),
        payload_size_bytes + sizeof(PACKET_HEADER_TYPE),
        transaction_id);
}
