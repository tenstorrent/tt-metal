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

void print_pkt_hdr_routing_fields(volatile tt::fabric::PacketHeader *const packet_start) {
    switch (packet_start->chip_send_type) {
        case tt::fabric::CHIP_UNICAST: {
            DPRINT << "C_UNI: dist:" << (uint32_t) packet_start->routing_fields.chip_unicast.distance_in_hops << "\n";
            break;
        }
        case tt::fabric::CHIP_MULTICAST: {
            DPRINT << "C_MCST: dist:" << (uint32_t) packet_start->routing_fields.chip_mcast.start_distance_in_hops <<
                ", rng:" << (uint32_t) packet_start->routing_fields.chip_mcast.range_hops << "\n";
            break;
        }
    };
}

void print_pkt_header_noc_fields(volatile tt::fabric::PacketHeader *const packet_start) {
    switch (packet_start->noc_send_type) {
        case tt::fabric::NocSendType::NOC_UNICAST: {
            switch (packet_start->command_type) {
                case tt::fabric::CommandType::WRITE: {
                    DPRINT << "N_WR addr:"<<(uint32_t)packet_start->command_fields.unicast_write.address <<
                        ", size:" << (uint32_t) packet_start->command_fields.unicast_write.size <<
                        ", x:" << (uint32_t) packet_start->command_fields.unicast_write.noc_x <<
                        ", y:" << (uint32_t) packet_start->command_fields.unicast_write.noc_y << "\n";
                } break;
                case tt::fabric::CommandType::ATOMIC_INC: {
                    DPRINT << "N_WR addr:"<<(uint32_t)packet_start->command_fields.unicast_seminc.address <<
                        ", val:" << (uint32_t) packet_start->command_fields.unicast_seminc.val <<
                        ", x:" << (uint32_t) packet_start->command_fields.unicast_seminc.noc_x <<
                        ", y:" << (uint32_t) packet_start->command_fields.unicast_seminc.noc_y << "\n";

                } break;
            }
            break;
        }
        case tt::fabric::NocSendType::NOC_MULTICAST: {
            ASSERT(false); // unimplemented
            break;
        }
    }
}

void print_pkt_header(volatile tt::fabric::PacketHeader *const packet_start) {
    auto const& header = *packet_start;
    DPRINT << "PKT: cmd_t:" << (uint32_t) packet_start->command_type <<
        ", csnd_t:" << (uint32_t) packet_start->chip_send_type <<
        ", nsnd_t:" << (uint32_t) packet_start->noc_send_type <<
        ", src_chip:" << (uint32_t) packet_start->reserved2 << "\n";
    print_pkt_hdr_routing_fields(packet_start);
    print_pkt_header_noc_fields(packet_start);
}


// Since we unicast to local, we must omit the packet header
void execute_chip_unicast_to_local_chip(volatile tt::fabric::PacketHeader *const packet_start, uint32_t transaction_id) {
    auto const& header = *packet_start;
    uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) + sizeof(tt::fabric::PacketHeader);

    tt::fabric::CommandType command_type = packet_start->command_type;
    tt::fabric::NocSendType noc_send_type = packet_start->noc_send_type;
    switch (command_type) {
        case tt::fabric::CommandType::WRITE: {
            switch (noc_send_type) {
                case tt::fabric::NocSendType::NOC_UNICAST: {
                    DPRINT << "C_UNI to y|x" << (uint32_t)((header.command_fields.unicast_write.noc_y << 16) | header.command_fields.unicast_write.noc_x) <<
                        ", " << (uint32_t)header.command_fields.unicast_write.address << "\n";
                    auto const dest_address = get_noc_addr(
                        header.command_fields.unicast_write.noc_x,
                        header.command_fields.unicast_write.noc_y,
                        header.command_fields.unicast_write.address);
                    auto const size = header.command_fields.unicast_write.size - sizeof(tt::fabric::PacketHeader);
                    noc_async_write_one_packet_with_trid(payload_start_address, dest_address, size, transaction_id);

                }break;
                case tt::fabric::NocSendType::NOC_MULTICAST: {
                    // TODO: confirm if we need to adjust dest core count if we span eth or dram cores
                    auto const mcast_dest_address = get_noc_multicast_addr(
                        header.command_fields.mcast_write.noc_x_start,
                        header.command_fields.mcast_write.noc_y_start,
                        header.command_fields.mcast_write.noc_x_start + header.command_fields.mcast_write.mcast_rect_size_x,
                        header.command_fields.mcast_write.noc_y_start + header.command_fields.mcast_write.mcast_rect_size_y,
                        header.command_fields.mcast_write.address);
                    auto const num_dests = header.command_fields.mcast_write.mcast_rect_size_x * header.command_fields.mcast_write.mcast_rect_size_y;
                    auto const size = header.command_fields.mcast_write.size - sizeof(tt::fabric::PacketHeader);
                    noc_async_write_one_packet_with_trid(payload_start_address, mcast_dest_address, size, num_dests, transaction_id);

                }break;
                default: {
                    ASSERT(false);
                }
            }
            break;
        }
        case tt::fabric::CommandType::ATOMIC_INC: {
            DPRINT << "C_AT_INC\n";
            switch (noc_send_type) {
                case tt::fabric::NocSendType::NOC_UNICAST: {
                    auto const dest_address = get_noc_addr(
                        header.command_fields.unicast_seminc.noc_x,
                        header.command_fields.unicast_seminc.noc_y,
                        header.command_fields.unicast_seminc.address);
                    auto const increment = header.command_fields.unicast_seminc.val;
                    DPRINT << "\tx=" << (uint32_t)header.command_fields.unicast_seminc.noc_x <<
                        ", y=" << (uint32_t)header.command_fields.unicast_seminc.noc_y <<
                        ", addr=" << (uint32_t)header.command_fields.unicast_seminc.address <<
                        ", inc=" << (uint32_t)increment << "\n";
                    noc_semaphore_inc(dest_address, increment);

                }break;
                case tt::fabric::NocSendType::NOC_MULTICAST: {
                    ASSERT(false);
                    // noc_async_write(payload_start_address, header.dest_address, header.size_bytes);

                }break;
                default: {
                    ASSERT(false);
                }
            }
            break;

        };

        default: {
            ASSERT(false);
        }
    };
}



void update_packet_header_for_next_hop(volatile tt::fabric::PacketHeader * packet_header) {
    switch (packet_header->chip_send_type) {
        case tt::fabric::CHIP_UNICAST: {
            ASSERT(packet_header->routing_fields.chip_unicast.distance_in_hops > 0);
            packet_header->routing_fields.chip_unicast.distance_in_hops--;
        } break;
        case tt::fabric::CHIP_MULTICAST: {
            if (packet_header->routing_fields.chip_mcast.start_distance_in_hops == DESTINATION_HOP_COUNT) {
            ASSERT(packet_header->routing_fields.chip_mcast.range_hops > 0);
                packet_header->routing_fields.chip_mcast.range_hops--;
            } else {
                ASSERT(packet_header->routing_fields.chip_mcast.start_distance_in_hops > 0);
                packet_header->routing_fields.chip_mcast.start_distance_in_hops--;
            }
        } break;
    }
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
void forward_payload_to_downstream_edm(
    volatile tt::fabric::PacketHeader *packet_header,
    tt::fabric::WorkerToFabricEdmSender &downstream_edm_interface,
    uint8_t transaction_id
    ) {
    DPRINT << "Fwding pkt to downstream\n";
    // TODO: PERF - this should already be getting checked by the caller so this should be redundant make it an ASSERT
    ASSERT(downstream_edm_interface.edm_has_space_for_packet()); // best effort check

    // This is a good place to print the packet header for debug if you are trying to inspect packets
    // because it is before we start manipulating the header for forwarding
    update_packet_header_for_next_hop(packet_header);
    downstream_edm_interface.send_payload_non_blocking_from_address_with_trid(
        reinterpret_cast<size_t>(packet_header),
        packet_header->get_payload_size_including_header(),
        transaction_id);
}


bool packet_must_be_consumed_locally(volatile tt::fabric::PacketHeader const& packet_header) {
    switch (packet_header.chip_send_type) {
        case tt::fabric::ChipSendType::CHIP_UNICAST: {
            return packet_header.routing_fields.chip_unicast.distance_in_hops == DESTINATION_HOP_COUNT;
        }
        case tt::fabric::ChipSendType::CHIP_MULTICAST: {
            return packet_header.routing_fields.chip_mcast.start_distance_in_hops == DESTINATION_HOP_COUNT;
        }
        default: {
            ASSERT(false);
            return false;
        }
    }
}


bool packet_must_be_forwarded_to_next_chip(volatile tt::fabric::PacketHeader const& packet_header) {
    switch (packet_header.chip_send_type) {
        case tt::fabric::ChipSendType::CHIP_UNICAST:
            return packet_header.routing_fields.chip_unicast.distance_in_hops != DESTINATION_HOP_COUNT;

        case tt::fabric::ChipSendType::CHIP_MULTICAST:
            return packet_header.routing_fields.chip_mcast.range_hops != LAST_MCAST_DESTINATION;

        default:
            ASSERT(false);
            return false;
    }
}
