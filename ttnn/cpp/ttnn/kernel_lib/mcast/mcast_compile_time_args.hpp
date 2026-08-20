// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_common.hpp"

namespace dataflow_kernel_lib::mcast_wire {

// Internal v3 CT format: control, optional semaphore IDs, then only the numeric
// metadata used by this placement. Counts and IDs remain full-width uint32_t.
// ProgramSpec uses the same format without IDs: its native bindings own resources.
namespace ct_control {
constexpr uint32_t VERSION_MASK = 0xFu;
constexpr uint32_t FLAGS_SHIFT = 4, FLAGS_MASK = 0x1Fu;
constexpr uint32_t HAS_RECEIVERS = 1u << 9;
constexpr uint32_t MODE_SHIFT = 10, MODE_MASK = 7u;
constexpr uint32_t CAPACITY_SHIFT = 13, CAPACITY_MASK = 3u;
constexpr uint32_t ROLES_SHIFT = 15, ROLES_MASK = 3u;
constexpr uint32_t DYNAMIC = 1u << 17;
constexpr uint32_t CAPABILITIES_SHIFT = 18, CAPABILITIES_MASK = 3u;
constexpr uint32_t COORD_ENCODING_SHIFT = 20, COORD_ENCODING_MASK = 3u;
constexpr uint32_t REMOTE_KNOWN = 1u << 22;
constexpr uint32_t ROTATING = 1u << 23;
constexpr uint32_t ACK_SHIFT = 24, ACK_MASK = 3u;
enum class Ack : uint32_t { None, Constant, RemoteCount, Runtime };
constexpr Ack ack(uint32_t control) { return Ack((control >> ACK_SHIFT) & ACK_MASK); }
}  // namespace ct_control

constexpr bool valid_compile_time_control(uint32_t control) {
    return control == ABSENT || (control & ct_control::VERSION_MASK) == FAMILY;
}

// Canonicalize metadata that cannot affect this kernel's RT layout or pipe.
// In particular, do not remove dynamic ACK words from RT when compressing CT.
constexpr ArgumentMetadata compact_compile_time_metadata(ArgumentMetadata metadata) {
    auto& family = metadata.family;
    const bool chain = transfer_mode(family.flags) == TransferMode::ChainUnicast;
    const bool sends = (metadata.kernel.capabilities & CAN_SEND) != 0;
    const bool receives = (metadata.kernel.capabilities & CAN_RECEIVE) != 0;
    if (chain || !sends || !(family.flags & PRE_HANDSHAKE)) {
        family.ack_count = 0;
    }
    if (chain || !sends || family.rectangle_capacity != 1 || !family.remote_count_known) {
        family.uniform_remote_count = 0;
        family.remote_count_known = false;
    }
    if (chain || !receives) {
        metadata.coordinates = {};
    }
    if (metadata.kernel.capabilities == 0) {
        family.rotating_span = 0;
    }
    return metadata;
}

constexpr uint32_t compile_time_control(ArgumentMetadata input) {
    using namespace ct_control;
    const auto metadata = compact_compile_time_metadata(input);
    const auto& family = metadata.family;
    const bool sends = (metadata.kernel.capabilities & CAN_SEND) != 0;
    const bool needs_ack =
        sends && (family.flags & PRE_HANDSHAKE) && transfer_mode(family.flags) == TransferMode::Multicast;
    const Ack ack_encoding = !needs_ack                              ? Ack::None
                             : family.ack_count == ACK_EQUALS_FANOUT ? Ack::Runtime
                             : family.remote_count_known && family.ack_count == family.uniform_remote_count
                                 ? Ack::RemoteCount
                                 : Ack::Constant;
    return FAMILY | (family.flags << FLAGS_SHIFT) | (family.has_remote_receivers ? HAS_RECEIVERS : 0u) |
           (uint32_t(family.sender_mcast_mode) << MODE_SHIFT) | (family.rectangle_capacity << CAPACITY_SHIFT) |
           (metadata.kernel.roles == DYNAMIC_ROLES ? DYNAMIC : metadata.kernel.roles << ROLES_SHIFT) |
           (metadata.kernel.capabilities << CAPABILITIES_SHIFT) |
           (uint32_t(metadata.coordinates.encoding) << COORD_ENCODING_SHIFT) |
           (family.remote_count_known ? REMOTE_KNOWN : 0u) | (family.rotating_span ? ROTATING : 0u) |
           (uint32_t(ack_encoding) << ACK_SHIFT);
}

constexpr ArgumentMetadata compile_time_control_metadata(uint32_t control) {
    using namespace ct_control;
    ArgumentMetadata metadata;
    metadata.family.flags = (control >> FLAGS_SHIFT) & FLAGS_MASK;
    metadata.family.has_remote_receivers = (control & HAS_RECEIVERS) != 0;
    metadata.family.sender_mcast_mode = SenderMcastMode((control >> MODE_SHIFT) & MODE_MASK);
    metadata.family.rectangle_capacity = (control >> CAPACITY_SHIFT) & CAPACITY_MASK;
    metadata.family.remote_count_known = (control & REMOTE_KNOWN) != 0;
    metadata.kernel.roles = (control & DYNAMIC) ? DYNAMIC_ROLES : (control >> ROLES_SHIFT) & ROLES_MASK;
    metadata.kernel.capabilities = (control >> CAPABILITIES_SHIFT) & CAPABILITIES_MASK;
    metadata.coordinates.encoding = SenderCoordinateEncoding((control >> COORD_ENCODING_SHIFT) & COORD_ENCODING_MASK);
    return metadata;
}

struct CompileTimeLayout {
    uint32_t data_ready = OMITTED, consumer_ready = OMITTED, signal_source = OMITTED;
    uint32_t remote_count = OMITTED, ack_count = OMITTED, rotating_span = OMITTED;
    uint32_t coordinates = OMITTED;
    uint32_t words = 1;

    constexpr explicit CompileTimeLayout(uint32_t control, bool semaphore_ids = true) {
        if (control == ABSENT || !valid_compile_time_control(control)) {
            return;
        }
        const auto metadata = compile_time_control_metadata(control);
        if (semaphore_ids) {
            data_ready = words++;
            if (metadata.family.flags & PRE_HANDSHAKE) {
                consumer_ready = words++;
            }
            if (transfer_mode(metadata.family.flags) == TransferMode::ChainUnicast) {
                signal_source = words++;
            }
        }
        if (metadata.family.remote_count_known) {
            remote_count = words++;
        }
        if (ct_control::ack(control) == ct_control::Ack::Constant) {
            ack_count = words++;
        }
        if (control & ct_control::ROTATING) {
            rotating_span = words++;
        }
        if (metadata.coordinates.encoding != SenderCoordinateEncoding::ExplicitPairs) {
            coordinates = words;
            words += 4;  // Columns, rows, X ranges, Y ranges; no dimension/count truncation.
        }
    }

    constexpr uint32_t semaphore(SemaphoreRole role) const {
        return role == DATA_READY ? data_ready : role == CONSUMER_READY ? consumer_ready : signal_source;
    }
};

template <typename Words>
constexpr void encode_compile_time_metadata(Words& words, ArgumentMetadata input, bool semaphore_ids = true) {
    const auto metadata = compact_compile_time_metadata(input);
    const uint32_t control = compile_time_control(metadata);
    const CompileTimeLayout layout(control, semaphore_ids);
    words[0] = control;
    if (layout.remote_count != OMITTED) {
        words[layout.remote_count] = metadata.family.uniform_remote_count;
    }
    if (layout.ack_count != OMITTED) {
        words[layout.ack_count] = metadata.family.ack_count;
    }
    if (layout.rotating_span != OMITTED) {
        words[layout.rotating_span] = metadata.family.rotating_span;
    }
    if (layout.coordinates != OMITTED) {
        words[layout.coordinates] = metadata.coordinates.columns;
        words[layout.coordinates + 1] = metadata.coordinates.rows;
        words[layout.coordinates + 2] = metadata.coordinates.x_ranges;
        words[layout.coordinates + 3] = metadata.coordinates.y_ranges;
    }
}

template <typename Words>
constexpr ArgumentMetadata decode_compile_time_metadata(const Words& words, bool semaphore_ids = true) {
    const uint32_t control = words[0];
    if (control == ABSENT || !valid_compile_time_control(control)) {
        return {};
    }
    auto metadata = compile_time_control_metadata(control);
    const CompileTimeLayout layout(control, semaphore_ids);
    if (layout.remote_count != OMITTED) {
        metadata.family.uniform_remote_count = words[layout.remote_count];
    }
    const auto ack = ct_control::ack(control);
    metadata.family.ack_count = ack == ct_control::Ack::Constant      ? words[layout.ack_count]
                                : ack == ct_control::Ack::RemoteCount ? metadata.family.uniform_remote_count
                                : ack == ct_control::Ack::Runtime     ? ACK_EQUALS_FANOUT
                                                                      : 0u;
    if (layout.rotating_span != OMITTED) {
        metadata.family.rotating_span = words[layout.rotating_span];
    }
    if (layout.coordinates != OMITTED) {
        metadata.coordinates.columns = words[layout.coordinates];
        metadata.coordinates.rows = words[layout.coordinates + 1];
        metadata.coordinates.x_ranges = words[layout.coordinates + 2];
        metadata.coordinates.y_ranges = words[layout.coordinates + 3];
    }
    return metadata;
}

}  // namespace dataflow_kernel_lib::mcast_wire
