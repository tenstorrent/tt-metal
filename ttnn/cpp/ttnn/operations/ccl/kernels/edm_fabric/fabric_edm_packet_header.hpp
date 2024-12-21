// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace tt::fabric {

enum TerminationSignal : uint32_t {
    KEEP_RUNNING = 0,

    // Wait for messages to drain
    GRACEFULLY_TERMINATE = 1,

    // Immediately terminate - don't wait for any outstanding messages to arrive or drain out
    IMMEDIATELY_TERMINATE = 2
};

// 2 bits
enum CommandType : uint8_t {
    WRITE = 0,
    ATOMIC_INC = 1
};

// How to send the payload across the cluster
// 1 bit
enum ChipSendType : uint8_t {
    CHIP_UNICAST = 0,
    CHIP_MULTICAST = 1
};
enum NocSendType : uint8_t {
    NOC_UNICAST = 0,
    NOC_MULTICAST = 1
};


struct UnicastRoutingCommandHeader {
    uint8_t distance_in_hops;
};
static_assert(sizeof(UnicastRoutingCommandHeader) == 1, "UnicastRoutingCommandHeader size is not 1 byte");
struct MulticastRoutingCommandHeader {
    uint8_t start_distance_in_hops: 4;
    uint8_t range_hops: 4; // 0 implies unicast
};
static_assert(sizeof(MulticastRoutingCommandHeader) == 1, "MulticastRoutingCommandHeader size is not 1 byte");
union RoutingFields {
    UnicastRoutingCommandHeader chip_unicast;
    MulticastRoutingCommandHeader chip_mcast;
};
static_assert(sizeof(RoutingFields) == sizeof(UnicastRoutingCommandHeader), "RoutingFields size is not 1 bytes");

struct NocUnicastCommandHeader {
    // TODO: just encode the noc_addr as uint64_t directly
    uint32_t address;
    uint32_t size;
    uint8_t noc_x;
    uint8_t noc_y;
    uint16_t reserved;
    // ignores header size
    inline uint32_t get_payload_only_size() const {
        return size;
    }
};
struct NocUnicastAtomicIncCommandHeader {
    NocUnicastAtomicIncCommandHeader(uint32_t address, uint16_t val, uint16_t wrap, uint8_t noc_x, uint8_t noc_y)
        : address(address), val(val), wrap(wrap), noc_x(noc_x), noc_y(noc_y) {}

    uint32_t address;
    uint16_t val;
    uint16_t wrap;
    uint8_t noc_x;
    uint8_t noc_y;

};
struct NocMulticastCommandHeader {
    uint32_t address;
    uint32_t size;
    uint8_t noc_x_start;
    uint8_t noc_y_start;
    uint8_t mcast_rect_size_x;
    uint8_t mcast_rect_size_y;

    // ignores header size
    inline uint32_t get_payload_only_size() const {
        return size;
    }
};
struct NocMulticastAtomicIncCommandHeader {
    uint32_t address;
    uint16_t val;
    uint16_t wrap;
    uint8_t noc_x_start;
    uint8_t noc_y_start;
    uint8_t size_x;
    uint8_t size_y;
};
static_assert(sizeof(NocUnicastCommandHeader) == 12, "NocUnicastCommandHeader size is not 1 byte");
static_assert(sizeof(NocMulticastCommandHeader) == 12, "NocMulticastCommandHeader size is not 1 byte");
static_assert(sizeof(NocUnicastAtomicIncCommandHeader) == 12, "NocUnicastCommandHeader size is not 1 byte");
static_assert(sizeof(NocMulticastAtomicIncCommandHeader) == 12, "NocAtomicIncCommandHeader size is not 1 byte");
union CommandFields{
    NocUnicastCommandHeader unicast_write;
    NocMulticastCommandHeader mcast_write;
    NocUnicastAtomicIncCommandHeader unicast_seminc;
    NocMulticastAtomicIncCommandHeader mcast_seminc;
} ;
static_assert(sizeof(CommandFields) <= 15, "CommandFields size is not 15 bytes");

// TODO: wrap this in a debug version that holds type info so we can assert for field/command/
struct PacketHeader {
    // TODO: trim this down noc_send_type 2 bits (4 values):
    //   -> unicast_write, mcast_write, unicast_seminc, mcast_seminc
    // For now, kept it separate so I could do reads which would be handled differently
    // but for our purposes we shouldn't need read so we should be able to omit the support
    CommandType command_type : 2;
    ChipSendType chip_send_type : 1;
    NocSendType noc_send_type : 1;
    uint8_t reserved : 4;

    RoutingFields routing_fields;
    uint16_t reserved2;
    CommandFields command_fields;

    // Sort of hack to work-around DRAM read alignment issues that must be 32B aligned
    // To simplify worker kernel code, we for now decide to pad up the packet header
    // to 32B so the user can simplify shift into their CB chunk by sizeof(tt::fabric::PacketHeader)
    // and automatically work around the DRAM read alignment bug.
    //
    // Future changes will remove this padding and require the worker kernel to be aware of this bug
    // and pad their own CBs conditionally when reading from DRAM. It'll be up to the users to
    // manage this complexity.
    uint32_t padding0;
    uint32_t padding1;
    uint32_t padding2;
    uint32_t padding3;

    inline void set_command_type(CommandType &type) { this->command_type = type; }
    inline void set_chip_send_type(ChipSendType &type) { this->chip_send_type = type; }
    inline void set_noc_send_type(NocSendType &type) { this->noc_send_type = type; }
    inline void set_routing_fields(RoutingFields &fields) { this->routing_fields = fields; }
    inline void set_command_fields(CommandFields &fields) { this->command_fields = fields; }

    size_t get_payload_size_excluding_header() volatile const {
        switch(this->command_type) {
            case WRITE: {
                switch(this->noc_send_type) {
                    case NOC_UNICAST: {
                        return this->command_fields.unicast_write.size - sizeof(PacketHeader);
                    } break;
                    case NOC_MULTICAST: {
                        return this->command_fields.mcast_write.size - sizeof(PacketHeader);
                    } break;
                    default:
                        return 0;
                }
            } break;
            case ATOMIC_INC: {
                return 0;
            } break;
            default:
                return 0;
        }
    }
    inline size_t get_payload_size_including_header() volatile const {
        return get_payload_size_excluding_header() + sizeof(PacketHeader);
    }

    inline PacketHeader& to_write() { this->command_type = WRITE; return *this; }
    inline PacketHeader& to_atomic_inc() { this->command_type = ATOMIC_INC; return *this; }

    inline PacketHeader &to_chip_unicast(UnicastRoutingCommandHeader const &chip_unicast_command_header) {
        this->chip_send_type = CHIP_UNICAST;
        this->routing_fields.chip_unicast = chip_unicast_command_header;
        return *this;
    }
    inline PacketHeader &to_chip_multicast(MulticastRoutingCommandHeader const &chip_multicast_command_header) {
        this->chip_send_type = CHIP_MULTICAST;
        this->routing_fields.chip_mcast = chip_multicast_command_header;
        return *this;
    }
    inline PacketHeader &to_noc_unicast(NocUnicastCommandHeader const &noc_unicast_command_header) {
        this->noc_send_type = NOC_UNICAST;
        this->command_fields.unicast_write = noc_unicast_command_header;
        return *this;
    }
    inline PacketHeader &to_noc_multicast(NocMulticastCommandHeader const &noc_multicast_command_header) {
        this->noc_send_type = NOC_MULTICAST;
        this->command_fields.mcast_write = noc_multicast_command_header;
        return *this;
    }
    inline PacketHeader &to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader const &noc_unicast_atomic_inc_command_header) {
        this->noc_send_type = NOC_UNICAST;
        this->command_fields.unicast_seminc = noc_unicast_atomic_inc_command_header;
        return *this;
    }
    inline PacketHeader &to_noc_multicast_atomic_inc(
        NocMulticastAtomicIncCommandHeader const &noc_multicast_atomic_inc_command_header) {
        this->noc_send_type = NOC_MULTICAST;
        this->command_fields.mcast_seminc = noc_multicast_atomic_inc_command_header;
        return *this;
    }
};


// TODO: When we remove the 32B padding requirement, reduce to 16B size check
static_assert(sizeof(PacketHeader) == 32, "sizeof(PacketHeader) is not equal to 32B");

static constexpr size_t header_size_bytes = sizeof(PacketHeader);


} // namespace tt::fabric
