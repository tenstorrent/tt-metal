// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <array>

namespace dataflow_kernel_lib {

inline constexpr uint32_t MAX_MCAST_RECTANGLES = 3;

// How a family delivers payloads after the host resolves its receiver-set policy.
enum class TransferMode : uint32_t { Multicast = 0, ChainUnicast = 1 };

// Flag is cleared between events; Counter is monotonic and uses absolute work rounds.
enum class DataReadySignal : uint32_t { Flag = 0, Counter = 1 };

// Guard protects source L1 before returning. CallerManaged lets the caller provide that
// protection; modes may still need completion barriers for data-before-ready ordering.
enum class SourceL1Guard { Guard, CallerManaged };

inline constexpr uint32_t UNUSED_SEM_ID = 0xFFFFFFFFu;
inline constexpr uint32_t ACK_EQUALS_FANOUT = 0xFFFFFFFFu;

inline constexpr uint32_t NO_CHAIN_NEIGHBOR = 0xFFFFFFFFu;
struct ChainRuntimeArguments {
    uint32_t predecessor_x = NO_CHAIN_NEIGHBOR;
    uint32_t predecessor_y = NO_CHAIN_NEIGHBOR;
    uint32_t successor_x = NO_CHAIN_NEIGHBOR;
    uint32_t successor_y = NO_CHAIN_NEIGHBOR;
    bool includes_sender = false;
};

// Internal host/device encodings selecting the sender's per-rectangle multicast path.
enum class SenderMcastMode : uint32_t {
    Invalid = 0,
    LocalCopy = 1,
    MulticastExcludeSource = 2,
    // The sender is a receiver too; payload loopback is skipped when src == dst.
    MulticastIncludeSource = 3,
    // Use when the mode is unknown at compile time: the shared binary reads each
    // rectangle's concrete mode from its prepared record. Never valid in those records.
    Unknown = 4,
};

struct NocBounds {
    uint32_t sx, sy, ex, ey;
};

// One host-prepared rectangle. Readiness is shared by the whole group.
struct RectangleRuntimeArguments {
    NocBounds bounds;
    uint32_t remote_count;
    uint32_t loopback_count;
    SenderMcastMode sender_mcast_mode;
};

// Owning, capacity-sized values, independent of the serialized runtime-argument layout.
template <uint32_t Capacity>
struct SenderRuntimeArgumentsFor {
    static_assert(Capacity >= 1 && Capacity <= MAX_MCAST_RECTANGLES, "Multicast supports one to three rectangles");
    std::array<RectangleRuntimeArguments, Capacity> rectangles{};
    uint32_t num_rectangles = 0;  // Unconfigured: populate with 1..Capacity before constructing a sender.
    uint32_t ack_count = 0;

    constexpr SenderRuntimeArgumentsFor() = default;

    // Preserve existing direct single-rectangle initialization.
    constexpr SenderRuntimeArgumentsFor(
        NocBounds bounds, uint32_t remote, uint32_t loopback, uint32_t ack, SenderMcastMode sender_mcast_mode) :
        rectangles{{{bounds, remote, loopback, sender_mcast_mode}}}, num_rectangles(1), ack_count(ack) {
        static_assert(Capacity == 1, "Single-rectangle initialization requires capacity one");
    }
};

using SenderRuntimeArguments = SenderRuntimeArgumentsFor<1>;

namespace mcast_wire {
// Resource-neutral topology and protocol metadata shared by both argument frontends.
struct FamilyMetadata {
    uint32_t rotating_span = 0;
    uint32_t rectangle_capacity = 0;
    uint32_t ack_count = 0;
    uint32_t uniform_remote_count = 0;
    bool remote_count_known = false;
    dataflow_kernel_lib::SenderMcastMode sender_mcast_mode = dataflow_kernel_lib::SenderMcastMode::Unknown;
    bool has_remote_receivers = false;
    uint32_t flags = 0;
};

constexpr uint32_t ABSENT = 0;
// Low four bits of the compact CT control word. Reject both previous header formats.
constexpr uint32_t FAMILY = 3;
constexpr uint32_t NO_SENDER_ROUND = 0xFFFFFFFFu;
constexpr uint32_t CAN_SEND = 1u << 0;
constexpr uint32_t CAN_RECEIVE = 1u << 1;
constexpr uint32_t DYNAMIC_ROLES = 0xFFFFFFFFu;

struct KernelMetadata {
    uint32_t roles = DYNAMIC_ROLES;
    uint32_t capabilities = CAN_SEND | CAN_RECEIVE;
};

enum class SenderCoordinateEncoding : uint32_t { ExplicitPairs, RowMajorRanges, ColumnMajorRanges };

struct SenderCoordinateMetadata {
    SenderCoordinateEncoding encoding = SenderCoordinateEncoding::ExplicitPairs;
    uint32_t columns = 0;
    uint32_t rows = 0;
    uint32_t x_ranges = 0;
    uint32_t y_ranges = 0;
};

struct ArgumentMetadata {
    FamilyMetadata family;
    KernelMetadata kernel;
    SenderCoordinateMetadata coordinates;
};
constexpr uint32_t PRE_HANDSHAKE = 1u << 0;
constexpr uint32_t COUNTER_SIGNAL = 1u << 1;
constexpr uint32_t NOC1 = 1u << 2;
constexpr uint32_t TRANSFER_MODE_SHIFT = 3;
constexpr uint32_t TRANSFER_MODE_MASK = 3u << TRANSFER_MODE_SHIFT;
constexpr TransferMode transfer_mode(uint32_t flags) {
    return static_cast<TransferMode>((flags & TRANSFER_MODE_MASK) >> TRANSFER_MODE_SHIFT);
}

// Resource roles, not serialized CT positions (which depend on the control word).
enum SemaphoreRole : uint32_t { DATA_READY, CONSUMER_READY, SIGNAL_SOURCE };
// Complete prepared rectangle records, not the compact multicast wire stride.
enum RectOffset : uint32_t { SX = 0, SY, EX, EY, REMOTE, LOOPBACK, RECT_SENDER_MCAST_MODE, RECT_WORDS };
enum RuntimeOffset : uint32_t { NUM_RECTANGLES = 0, ACK, HEADER_WORDS };
enum SenderCoordOffset : uint32_t { SENDER_X = 0, SENDER_Y, SENDER_COORD_WORDS };
enum RoleOffset : uint32_t { ROLES = 0, SENDER_ROUND, ROLE_WORDS };
constexpr uint32_t ABSENT_CT_WORDS = 1;
enum ChainOffset : uint32_t {
    PREDECESSOR_X = 0,
    PREDECESSOR_Y,
    SUCCESSOR_X,
    SUCCESSOR_Y,
    INCLUDES_SENDER,
    CHAIN_WORDS
};
constexpr uint32_t CHAIN_COORDINATES = HEADER_WORDS;
constexpr uint32_t CHAIN_NEIGHBORS = CHAIN_COORDINATES + SENDER_COORD_WORDS;
constexpr uint32_t CHAIN_ROLES = CHAIN_NEIGHBORS + CHAIN_WORDS;
constexpr uint32_t CHAIN_RUNTIME_WORDS = CHAIN_ROLES + ROLE_WORDS;
constexpr SenderMcastMode classify(uint32_t remote_count, bool includes_sender) {
    return remote_count == 0 ? SenderMcastMode::LocalCopy
           : includes_sender ? SenderMcastMode::MulticastIncludeSource
                             : SenderMcastMode::MulticastExcludeSource;
}
constexpr bool concrete(SenderMcastMode sender_mcast_mode) {
    return sender_mcast_mode == SenderMcastMode::LocalCopy ||
           sender_mcast_mode == SenderMcastMode::MulticastExcludeSource ||
           sender_mcast_mode == SenderMcastMode::MulticastIncludeSource;
}
static_assert(RECT_WORDS == 7 && HEADER_WORDS == 2);
static_assert(CHAIN_WORDS == 5 && CHAIN_RUNTIME_WORDS == 11);
static_assert(sizeof(RectangleRuntimeArguments) == RECT_WORDS * sizeof(uint32_t));

constexpr uint32_t OMITTED = 0xFFFFFFFFu;
enum RangeOffset : uint32_t { RANGE_START = 0, RANGE_END, RANGE_WORDS };

// One fixed-width layout per kernel/channel, including its inactive placed cores.
// All offsets are relative to the helper block except the rectangle field offsets,
// which are relative to each capacity-sized rectangle record.
struct RuntimeLayout {
    uint32_t roles = OMITTED;
    uint32_t sender_phase = OMITTED;
    uint32_t rectangle_count = OMITTED;
    uint32_t ack = OMITTED;
    uint32_t sender_coordinates = OMITTED;
    uint32_t coordinate_words = 0;
    uint32_t rectangles = OMITTED;
    uint32_t rectangle_bounds = OMITTED;
    uint32_t rectangle_remote = OMITTED;
    uint32_t rectangle_mode = OMITTED;
    uint32_t rectangle_stride = 0;
    uint32_t chain_neighbors = OMITTED;
    uint32_t words = 0;

    constexpr explicit RuntimeLayout(ArgumentMetadata metadata) {
        const auto& family = metadata.family;
        const auto& kernel = metadata.kernel;
        const auto& coordinates = metadata.coordinates;
        if (transfer_mode(family.flags) == TransferMode::ChainUnicast) {
            // Preserve the original chain runtime bytes, including unused header fields.
            roles = CHAIN_ROLES + ROLES;
            sender_phase = roles + SENDER_ROUND;
            rectangle_count = NUM_RECTANGLES;
            ack = ACK;
            sender_coordinates = CHAIN_COORDINATES;
            coordinate_words = SENDER_COORD_WORDS;
            chain_neighbors = CHAIN_NEIGHBORS;
            words = CHAIN_RUNTIME_WORDS;
            return;
        }
        const bool sends = (kernel.capabilities & CAN_SEND) != 0;
        const bool receives = (kernel.capabilities & CAN_RECEIVE) != 0;
        roles = reserve(words, kernel.roles == DYNAMIC_ROLES);
        sender_phase = reserve(words, sends && family.rotating_span != 0);
        rectangle_count = reserve(words, sends && family.rectangle_capacity > 1);
        ack = reserve(words, sends && (family.flags & PRE_HANDSHAKE) && family.ack_count == ACK_EQUALS_FANOUT);
        if (receives) {
            coordinate_words = coordinates.encoding == SenderCoordinateEncoding::ExplicitPairs
                                   ? SENDER_COORD_WORDS * (family.rotating_span ? family.rotating_span : 1u)
                                   : RANGE_WORDS * (coordinates.x_ranges + coordinates.y_ranges);
            sender_coordinates = reserve(words, true, coordinate_words);
        }
        if (sends) {
            rectangle_bounds = reserve(rectangle_stride, family.sender_mcast_mode != SenderMcastMode::LocalCopy, 4);
            rectangle_remote =
                reserve(rectangle_stride, !(family.rectangle_capacity == 1 && family.remote_count_known));
            rectangle_mode = reserve(rectangle_stride, family.sender_mcast_mode == SenderMcastMode::Unknown);
            rectangles = reserve(words, true, family.rectangle_capacity * rectangle_stride);
        }
    }

private:
    static constexpr uint32_t reserve(uint32_t& end, bool present, uint32_t count = 1) {
        if (!present) {
            return OMITTED;
        }
        const uint32_t offset = end;
        end += count;
        return offset;
    }
};

// Common host/device range decoding; views can be pointers or native vararg views.
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
}  // namespace mcast_wire
}  // namespace dataflow_kernel_lib
