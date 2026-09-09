// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <array>

namespace dataflow_kernel_lib {

inline constexpr uint32_t MAX_MCAST_RECTANGLES = 3;

// Internal host/device encodings selecting the sender's transfer path.
enum class SenderTransferMode : uint32_t {
    Invalid = 0,
    LocalCopy = 1,
    MulticastExcludeSource = 2,
    // The sender is a receiver too; payload loopback is skipped when src == dst.
    MulticastIncludeSource = 3,
    // Use when the mode is unknown at compile time: the shared binary reads each
    // rectangle's concrete mode from its prepared record. Never valid in those records.
    TransferModeUnknown = 4,
};

struct NocBounds {
    uint32_t sx, sy, ex, ey;
};

// One host-prepared rectangle. Readiness is shared by the whole group.
struct RectangleRuntimeArguments {
    NocBounds bounds;
    uint32_t remote_count;
    uint32_t loopback_count;
    SenderTransferMode transfer_mode;
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
        NocBounds bounds, uint32_t remote, uint32_t loopback, uint32_t ack, SenderTransferMode mode) :
        rectangles{{{bounds, remote, loopback, mode}}}, num_rectangles(1), ack_count(ack) {
        static_assert(Capacity == 1, "Single-rectangle initialization requires capacity one");
    }
};

using SenderRuntimeArguments = SenderRuntimeArgumentsFor<1>;

namespace mcast_wire {
constexpr uint32_t ABSENT = 0;
constexpr uint32_t FAMILY = 1;
constexpr uint32_t NO_SENDER_ROUND = 0xFFFFFFFFu;
constexpr uint32_t CAN_SEND = 1u << 0;
constexpr uint32_t CAN_RECEIVE = 1u << 1;
constexpr uint32_t PRE_HANDSHAKE = 1u << 0;
constexpr uint32_t COUNTER_SIGNAL = 1u << 1;
constexpr uint32_t NOC1 = 1u << 2;

enum CT : uint32_t {
    TAG = 0,
    HAS_RECEIVERS,
    DATA_READY,
    CONSUMER_READY,
    ACK_COUNT,
    FLAGS,
    ROTATING_SPAN,
    TRANSFER_MODE,
    REMOTE_COUNT,
    LOOPBACK_COUNT,
    RECTANGLE_CAPACITY,
    CT_WORDS
};
// Every sender descriptor has the same layout, independently of mode/count uniformity.
enum RectOffset : uint32_t { SX = 0, SY, EX, EY, REMOTE, LOOPBACK, MODE, RECT_WORDS };
enum RuntimeOffset : uint32_t { NUM_RECTANGLES = 0, ACK, HEADER_WORDS };
constexpr uint32_t FIXED_SENDER_X = HEADER_WORDS;
constexpr uint32_t FIXED_SENDER_Y = HEADER_WORDS + 1;
constexpr uint32_t ROLE_WORDS = 2;
constexpr uint32_t ROLE_FROM_END = 2;
constexpr uint32_t PHASE_FROM_END = 1;
constexpr uint32_t sender_coords_offset(uint32_t) { return HEADER_WORDS; }
constexpr uint32_t rectangles_offset(uint32_t rotating_span) {
    return HEADER_WORDS + 2u * (rotating_span ? rotating_span : 1u);
}
constexpr uint32_t roles_offset(uint32_t rotating_span, uint32_t rectangle_capacity) {
    return rectangles_offset(rotating_span) + RECT_WORDS * rectangle_capacity;
}
constexpr uint32_t runtime_words(uint32_t rotating_span, uint32_t rectangle_capacity) {
    return roles_offset(rotating_span, rectangle_capacity) + ROLE_WORDS;
}
constexpr SenderTransferMode classify(uint32_t remote_count, bool includes_sender) {
    return remote_count == 0 ? SenderTransferMode::LocalCopy
           : includes_sender ? SenderTransferMode::MulticastIncludeSource
                             : SenderTransferMode::MulticastExcludeSource;
}
constexpr bool concrete(SenderTransferMode transfer_mode) {
    return transfer_mode == SenderTransferMode::LocalCopy ||
           transfer_mode == SenderTransferMode::MulticastExcludeSource ||
           transfer_mode == SenderTransferMode::MulticastIncludeSource;
}
static_assert(CT_WORDS == 11 && RECT_WORDS == 7 && HEADER_WORDS == 2);
static_assert(runtime_words(0, 1) == 13);
static_assert(runtime_words(0, 3) == 27);
static_assert(runtime_words(3, 2) == 24);
static_assert(sizeof(RectangleRuntimeArguments) == RECT_WORDS * sizeof(uint32_t));
}  // namespace mcast_wire
}  // namespace dataflow_kernel_lib
