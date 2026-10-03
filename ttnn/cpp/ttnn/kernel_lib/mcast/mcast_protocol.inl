// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation for mcast_protocol.hpp. Do not include directly.

namespace dataflow_kernel_lib::mcast_wire {

constexpr TransferMode transfer_mode(uint32_t flags) {
    return static_cast<TransferMode>((flags & TRANSFER_MODE_MASK) >> TRANSFER_MODE_SHIFT);
}

constexpr SenderMcastMode classify(uint32_t remote_count, bool includes_sender) {
    if (remote_count == 0) {
        return SenderMcastMode::LocalCopy;
    }
    return includes_sender ? SenderMcastMode::MulticastIncludeSource : SenderMcastMode::MulticastExcludeSource;
}

constexpr bool concrete(SenderMcastMode sender_mcast_mode) {
    return sender_mcast_mode == SenderMcastMode::LocalCopy ||
           sender_mcast_mode == SenderMcastMode::MulticastExcludeSource ||
           sender_mcast_mode == SenderMcastMode::MulticastIncludeSource;
}

constexpr RuntimeLayout::RuntimeLayout(ArgumentMetadata metadata) {
    const auto& mcast = metadata.mcast;
    const auto& kernel = metadata.kernel;
    const auto& coordinates = metadata.coordinates;
    const bool sends = (kernel.capabilities & CAN_SEND) != 0;
    const bool receives = (kernel.capabilities & CAN_RECEIVE) != 0;
    roles = reserve(words, kernel.roles == DYNAMIC_ROLES);
    if (transfer_mode(mcast.flags) == TransferMode::ChainUnicast) {
        if (receives) {
            coordinate_words = SENDER_COORD_WORDS;
            sender_coordinates = reserve(words, true, coordinate_words);
        }
        chain_neighbors = reserve(words, true, CHAIN_WORDS);
        return;
    }
    sender_phase = reserve(words, sends && mcast.rotating_span != 0);
    rectangle_count = reserve(words, sends && mcast.rectangle_capacity > 1);
    ack = reserve(words, sends && (mcast.flags & PRE_HANDSHAKE) && mcast.ack_count == ACK_EQUALS_FANOUT);
    if (receives) {
        coordinate_words = coordinates.encoding == SenderCoordinateEncoding::ExplicitPairs
                               ? SENDER_COORD_WORDS * (mcast.rotating_span ? mcast.rotating_span : 1u)
                               : RANGE_WORDS * (coordinates.x_ranges + coordinates.y_ranges);
        sender_coordinates = reserve(words, true, coordinate_words);
    }
    if (sends) {
        rectangle_bounds = reserve(rectangle_stride, mcast.sender_mcast_mode != SenderMcastMode::LocalCopy, 4);
        rectangle_remote = reserve(rectangle_stride, !(mcast.rectangle_capacity == 1 && mcast.remote_count_known));
        rectangle_mode = reserve(rectangle_stride, mcast.sender_mcast_mode == SenderMcastMode::Unknown);
        rectangles = reserve(words, true, mcast.rectangle_capacity * rectangle_stride);
    }
}

constexpr uint32_t RuntimeLayout::reserve(uint32_t& end, bool present, uint32_t count) {
    if (!present) {
        return OMITTED;
    }
    const uint32_t offset = end;
    end += count;
    return offset;
}

template <typename Ranges>
constexpr uint32_t range_coordinate(const Ranges& ranges, uint32_t first, uint32_t count, uint32_t index) {
    for (uint32_t range = 0; range < count; ++range) {
        const uint32_t base = first + RANGE_WORDS * range;
        const uint32_t start = ranges[base + RANGE_START];
        const uint32_t length = ranges[base + RANGE_END] - start + 1;
        if (index < length) {
            return start + index;
        }
        index -= length;
    }
    return 0;  // Only reached by malformed input; host reconstruction validates every phase.
}

template <typename Coordinates>
constexpr uint32_t sender_coordinate(
    const Coordinates& payload, SenderCoordinateMetadata metadata, uint32_t phase, uint32_t axis) {
    if (metadata.encoding == SenderCoordinateEncoding::ExplicitPairs) {
        return payload[SENDER_COORD_WORDS * phase + axis];
    }
    const bool row_major = metadata.encoding == SenderCoordinateEncoding::RowMajorRanges;
    const uint32_t x = row_major ? phase % metadata.columns : phase / metadata.rows;
    const uint32_t y = row_major ? phase / metadata.columns : phase % metadata.rows;
    return axis == SENDER_X ? range_coordinate(payload, 0, metadata.x_ranges, x)
                            : range_coordinate(payload, RANGE_WORDS * metadata.x_ranges, metadata.y_ranges, y);
}

// Internal compact CT format: control, optional semaphore IDs, then only the numeric
// metadata used by this placement. Counts and IDs remain full-width uint32_t.
// ProgramSpec uses the same format without IDs: its native bindings own resources.
namespace ct_control {
constexpr uint32_t FLAGS_SHIFT = 0, FLAGS_MASK = 0x1Fu;
constexpr uint32_t HAS_RECEIVERS = 1u << 5;
constexpr uint32_t MODE_SHIFT = 6, MODE_MASK = 7u;
constexpr uint32_t CAPACITY_SHIFT = 9, CAPACITY_MASK = 3u;
constexpr uint32_t ROLES_SHIFT = 11, ROLES_MASK = 3u;
constexpr uint32_t DYNAMIC = 1u << 13;
constexpr uint32_t CAPABILITIES_SHIFT = 14, CAPABILITIES_MASK = 3u;
constexpr uint32_t COORD_ENCODING_SHIFT = 16, COORD_ENCODING_MASK = 3u;
constexpr uint32_t REMOTE_KNOWN = 1u << 18;
constexpr uint32_t ROTATING = 1u << 19;
constexpr uint32_t ACK_SHIFT = 20, ACK_MASK = 3u;
enum class Ack : uint32_t { None, Constant, RemoteCount, Runtime };
constexpr Ack ack(uint32_t control) { return Ack((control >> ACK_SHIFT) & ACK_MASK); }
}  // namespace ct_control

// Canonicalize metadata that cannot affect this kernel's RT layout or pipe.
// In particular, do not remove dynamic ACK words from RT when compressing CT.
constexpr ArgumentMetadata compact_compile_time_metadata(ArgumentMetadata metadata) {
    auto& mcast = metadata.mcast;
    const bool chain = transfer_mode(mcast.flags) == TransferMode::ChainUnicast;
    const bool sends = (metadata.kernel.capabilities & CAN_SEND) != 0;
    const bool receives = (metadata.kernel.capabilities & CAN_RECEIVE) != 0;
    if (chain || !sends || !(mcast.flags & PRE_HANDSHAKE)) {
        mcast.ack_count = 0;
    }
    if (chain || !sends || mcast.rectangle_capacity != 1 || !mcast.remote_count_known) {
        mcast.uniform_remote_count = 0;
        mcast.remote_count_known = false;
    }
    if (chain || !receives) {
        metadata.coordinates = {};
    }
    if (metadata.kernel.capabilities == 0) {
        mcast.rotating_span = 0;
    }
    return metadata;
}

constexpr uint32_t compile_time_control(ArgumentMetadata input) {
    using namespace ct_control;
    const auto metadata = compact_compile_time_metadata(input);
    const auto& mcast = metadata.mcast;
    const bool sends = (metadata.kernel.capabilities & CAN_SEND) != 0;
    const bool needs_ack =
        sends && (mcast.flags & PRE_HANDSHAKE) && transfer_mode(mcast.flags) == TransferMode::Multicast;
    Ack ack_encoding = Ack::None;
    if (needs_ack) {
        if (mcast.ack_count == ACK_EQUALS_FANOUT) {
            ack_encoding = Ack::Runtime;
        } else if (mcast.remote_count_known && mcast.ack_count == mcast.uniform_remote_count) {
            ack_encoding = Ack::RemoteCount;
        } else {
            ack_encoding = Ack::Constant;
        }
    }
    return (mcast.flags << FLAGS_SHIFT) | (mcast.has_remote_receivers ? HAS_RECEIVERS : 0u) |
           (uint32_t(mcast.sender_mcast_mode) << MODE_SHIFT) | (mcast.rectangle_capacity << CAPACITY_SHIFT) |
           (metadata.kernel.roles == DYNAMIC_ROLES ? DYNAMIC : metadata.kernel.roles << ROLES_SHIFT) |
           (metadata.kernel.capabilities << CAPABILITIES_SHIFT) |
           (uint32_t(metadata.coordinates.encoding) << COORD_ENCODING_SHIFT) |
           (mcast.remote_count_known ? REMOTE_KNOWN : 0u) | (mcast.rotating_span ? ROTATING : 0u) |
           (uint32_t(ack_encoding) << ACK_SHIFT);
}

constexpr ArgumentMetadata compile_time_control_metadata(uint32_t control) {
    using namespace ct_control;
    ArgumentMetadata metadata;
    metadata.mcast.flags = (control >> FLAGS_SHIFT) & FLAGS_MASK;
    metadata.mcast.has_remote_receivers = (control & HAS_RECEIVERS) != 0;
    metadata.mcast.sender_mcast_mode = SenderMcastMode((control >> MODE_SHIFT) & MODE_MASK);
    metadata.mcast.rectangle_capacity = (control >> CAPACITY_SHIFT) & CAPACITY_MASK;
    metadata.mcast.remote_count_known = (control & REMOTE_KNOWN) != 0;
    metadata.kernel.roles = (control & DYNAMIC) ? DYNAMIC_ROLES : (control >> ROLES_SHIFT) & ROLES_MASK;
    metadata.kernel.capabilities = (control >> CAPABILITIES_SHIFT) & CAPABILITIES_MASK;
    metadata.coordinates.encoding = SenderCoordinateEncoding((control >> COORD_ENCODING_SHIFT) & COORD_ENCODING_MASK);
    return metadata;
}

constexpr CompileTimeLayout::CompileTimeLayout(uint32_t control, bool semaphore_ids) {
    if (control == ABSENT) {
        return;
    }
    const auto metadata = compile_time_control_metadata(control);
    if (semaphore_ids) {
        data_ready = words++;
        if (metadata.mcast.flags & PRE_HANDSHAKE) {
            consumer_ready = words++;
        }
        if (transfer_mode(metadata.mcast.flags) == TransferMode::ChainUnicast) {
            signal_source = words++;
        }
    }
    if (metadata.mcast.remote_count_known) {
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

constexpr uint32_t CompileTimeLayout::semaphore(SemaphoreRole role) const {
    switch (role) {
        case DATA_READY: return data_ready;
        case CONSUMER_READY: return consumer_ready;
        case SIGNAL_SOURCE: return signal_source;
    }
    return OMITTED;
}

template <typename Words>
constexpr void encode_compile_time_metadata(Words& words, ArgumentMetadata input, bool semaphore_ids) {
    const auto metadata = compact_compile_time_metadata(input);
    const uint32_t control = compile_time_control(metadata);
    const CompileTimeLayout layout(control, semaphore_ids);
    words[0] = control;
    if (layout.remote_count != OMITTED) {
        words[layout.remote_count] = metadata.mcast.uniform_remote_count;
    }
    if (layout.ack_count != OMITTED) {
        words[layout.ack_count] = metadata.mcast.ack_count;
    }
    if (layout.rotating_span != OMITTED) {
        words[layout.rotating_span] = metadata.mcast.rotating_span;
    }
    if (layout.coordinates != OMITTED) {
        words[layout.coordinates] = metadata.coordinates.columns;
        words[layout.coordinates + 1] = metadata.coordinates.rows;
        words[layout.coordinates + 2] = metadata.coordinates.x_ranges;
        words[layout.coordinates + 3] = metadata.coordinates.y_ranges;
    }
}

template <typename Words>
constexpr ArgumentMetadata decode_compile_time_metadata(const Words& words, bool semaphore_ids) {
    const uint32_t control = words[0];
    if (control == ABSENT) {
        return {};
    }
    auto metadata = compile_time_control_metadata(control);
    const CompileTimeLayout layout(control, semaphore_ids);
    if (layout.remote_count != OMITTED) {
        metadata.mcast.uniform_remote_count = words[layout.remote_count];
    }
    const auto ack = ct_control::ack(control);
    switch (ack) {
        case ct_control::Ack::Constant: metadata.mcast.ack_count = words[layout.ack_count]; break;
        case ct_control::Ack::RemoteCount: metadata.mcast.ack_count = metadata.mcast.uniform_remote_count; break;
        case ct_control::Ack::Runtime: metadata.mcast.ack_count = ACK_EQUALS_FANOUT; break;
        case ct_control::Ack::None: metadata.mcast.ack_count = 0u; break;
    }
    if (layout.rotating_span != OMITTED) {
        metadata.mcast.rotating_span = words[layout.rotating_span];
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
