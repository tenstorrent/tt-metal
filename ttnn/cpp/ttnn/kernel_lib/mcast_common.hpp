// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace dataflow_kernel_lib {

// Internal host/device encodings selecting the sender's transfer path.
enum class SenderTransferMode : uint32_t {
    Invalid = 0,
    LocalCopy = 1,
    MulticastExcludeSource = 2,
    // The sender is a receiver too; payload loopback is skipped when src == dst.
    MulticastIncludeSource = 3,
    // Use when the mode is unknown at compile time: the shared binary reads each
    // sender's concrete mode from SenderRuntimeArguments. Never valid in those arguments.
    TransferModeUnknown = 4,
};

struct NocBounds {
    uint32_t sx, sy, ex, ey;
};

struct SenderRuntimeArguments {
    NocBounds bounds;
    uint32_t remote_count;
    uint32_t loopback_count;
    uint32_t ack_count;
    SenderTransferMode transfer_mode;
};

namespace mcast_wire {
constexpr uint32_t ABSENT = 0;
constexpr uint32_t LEGACY_RECTANGLE = 1;  // Obsolete; deliberately not decoded.
constexpr uint32_t PREPARED_RECTANGLE = 2;
constexpr uint32_t RESERVED_FAMILY = 3;
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
    CT_WORDS
};
enum BoundsOffset : uint32_t { SX = 0, SY, EX, EY, BOUNDS_WORDS };
enum PreparedOffset : uint32_t { REMOTE = 0, LOOPBACK, ACK, MODE, PREPARED_WORDS };
constexpr uint32_t FIXED_SENDER_X = 0;
constexpr uint32_t FIXED_SENDER_Y = 1;
constexpr uint32_t ROLE_WORDS = 2;
constexpr uint32_t ROLE_FROM_END = 2;
constexpr uint32_t PHASE_FROM_END = 1;
constexpr uint32_t sender_coords_offset(uint32_t rotating_span) { return rotating_span ? BOUNDS_WORDS : 0u; }
constexpr uint32_t prepared_offset(uint32_t rotating_span) { return BOUNDS_WORDS + 2u * rotating_span; }
constexpr uint32_t roles_offset(uint32_t rotating_span, SenderTransferMode transfer_mode) {
    return prepared_offset(rotating_span) +
           (transfer_mode == SenderTransferMode::TransferModeUnknown ? PREPARED_WORDS : 0u);
}
constexpr uint32_t runtime_words(uint32_t rotating_span, SenderTransferMode transfer_mode) {
    return roles_offset(rotating_span, transfer_mode) + ROLE_WORDS;
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
static_assert(CT_WORDS == 10 && BOUNDS_WORDS == 4 && PREPARED_WORDS == 4);
static_assert(runtime_words(0, SenderTransferMode::MulticastExcludeSource) == 6);
static_assert(runtime_words(0, SenderTransferMode::TransferModeUnknown) == 10);
static_assert(runtime_words(3, SenderTransferMode::TransferModeUnknown) == 16);
static_assert(sizeof(SenderRuntimeArguments) == 8 * sizeof(uint32_t));
}  // namespace mcast_wire
}  // namespace dataflow_kernel_lib
