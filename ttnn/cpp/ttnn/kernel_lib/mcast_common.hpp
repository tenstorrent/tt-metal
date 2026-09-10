// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <array>

namespace dataflow_kernel_lib {

inline constexpr uint32_t MAX_MCAST_RECTANGLES = 3;

// How a family delivers payloads after the host resolves its receiver-set policy.
enum class McastMode : uint32_t { Multicast = 0, ChainUnicast = 1 };

// Flag is cleared between events; Counter is monotonic and uses absolute work rounds.
enum class DataReadySignal { Flag, Counter };

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
constexpr uint32_t MCAST_MODE_SHIFT = 3;
constexpr uint32_t MCAST_MODE_MASK = 3u << MCAST_MODE_SHIFT;
constexpr McastMode mcast_mode(uint32_t flags) {
    return static_cast<McastMode>((flags & MCAST_MODE_MASK) >> MCAST_MODE_SHIFT);
}

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
enum ChainOffset : uint32_t {
    PREDECESSOR_X = 0,
    PREDECESSOR_Y,
    SUCCESSOR_X,
    SUCCESSOR_Y,
    INCLUDES_SENDER,
    CHAIN_WORDS
};
constexpr uint32_t chain_offset(uint32_t rotating_span, uint32_t rectangle_capacity) {
    return rectangles_offset(rotating_span) + RECT_WORDS * rectangle_capacity;
}
// Every layout query takes the multicast mode explicitly: multicast blocks end after the
// rectangles, while chain-unicast blocks carry CHAIN_WORDS of neighbor metadata.
constexpr uint32_t roles_offset(uint32_t rotating_span, uint32_t rectangle_capacity, McastMode mode) {
    return chain_offset(rotating_span, rectangle_capacity) + (mode == McastMode::Multicast ? 0u : CHAIN_WORDS);
}
constexpr uint32_t runtime_words(uint32_t rotating_span, uint32_t rectangle_capacity, McastMode mode) {
    return roles_offset(rotating_span, rectangle_capacity, mode) + ROLE_WORDS;
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
static_assert(runtime_words(0, 1, McastMode::Multicast) == 13);
static_assert(runtime_words(0, 3, McastMode::Multicast) == 27);
static_assert(runtime_words(3, 2, McastMode::Multicast) == 24);
static_assert(CHAIN_WORDS == 5 && runtime_words(0, 0, McastMode::ChainUnicast) == 11);
static_assert(sizeof(RectangleRuntimeArguments) == RECT_WORDS * sizeof(uint32_t));
}  // namespace mcast_wire
}  // namespace dataflow_kernel_lib
