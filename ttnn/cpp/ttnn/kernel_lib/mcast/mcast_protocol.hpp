// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared host/device multicast protocol. Definitions are kept in
// mcast_protocol.inl so the serialized contract remains easy to scan here.
namespace dataflow_kernel_lib {

inline constexpr uint32_t MAX_MCAST_RECTANGLES = 3;

// How multicast delivers payloads after the host resolves the receiver-set policy.
enum class TransferMode : uint32_t { Multicast = 0, ChainUnicast = 1 };

// Flag is cleared between events; Counter is monotonic and uses absolute work rounds.
enum class DataReadySignal : uint32_t { Flag = 0, Counter = 1 };

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

namespace mcast_wire {

// Resource-neutral topology and protocol metadata shared by both argument frontends.
struct McastMetadata {
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
    McastMetadata mcast;
    KernelMetadata kernel;
    SenderCoordinateMetadata coordinates;
};

constexpr uint32_t PRE_HANDSHAKE = 1u << 0;
constexpr uint32_t COUNTER_SIGNAL = 1u << 1;
constexpr uint32_t NOC1 = 1u << 2;
constexpr uint32_t TRANSFER_MODE_SHIFT = 3;
constexpr uint32_t TRANSFER_MODE_MASK = 3u << TRANSFER_MODE_SHIFT;
constexpr TransferMode transfer_mode(uint32_t flags);

// Resource roles, not serialized CT positions (which depend on the control word).
enum SemaphoreRole : uint32_t { DATA_READY, CONSUMER_READY, SIGNAL_SOURCE };
// Complete prepared rectangle records, not the compact multicast wire stride.
enum RectOffset : uint32_t { SX = 0, SY, EX, EY, REMOTE, LOOPBACK, RECT_SENDER_MCAST_MODE, RECT_WORDS };
enum SenderCoordOffset : uint32_t { SENDER_X = 0, SENDER_Y, SENDER_COORD_WORDS };
constexpr uint32_t ABSENT_CT_WORDS = 1;
enum ChainOffset : uint32_t {
    PREDECESSOR_X = 0,
    PREDECESSOR_Y,
    SUCCESSOR_X,
    SUCCESSOR_Y,
    INCLUDES_SENDER,
    CHAIN_WORDS
};
constexpr SenderMcastMode classify(uint32_t remote_count, bool includes_sender);
constexpr bool concrete(SenderMcastMode sender_mcast_mode);

static_assert(RECT_WORDS == 7);
static_assert(CHAIN_WORDS == 5);

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

    constexpr explicit RuntimeLayout(ArgumentMetadata metadata);

private:
    static constexpr uint32_t reserve(uint32_t& end, bool present, uint32_t count = 1);
};

// Common host/device range decoding; views can be pointers or native vararg views.
template <typename Ranges>
constexpr uint32_t range_coordinate(const Ranges& ranges, uint32_t first, uint32_t count, uint32_t index);

template <typename Coordinates>
constexpr uint32_t sender_coordinate(
    const Coordinates& payload, SenderCoordinateMetadata metadata, uint32_t phase, uint32_t axis);

constexpr ArgumentMetadata compact_compile_time_metadata(ArgumentMetadata metadata);
constexpr uint32_t compile_time_control(ArgumentMetadata input);
constexpr ArgumentMetadata compile_time_control_metadata(uint32_t control);

struct CompileTimeLayout {
    uint32_t data_ready = OMITTED, consumer_ready = OMITTED, signal_source = OMITTED;
    uint32_t remote_count = OMITTED, ack_count = OMITTED, rotating_span = OMITTED;
    uint32_t coordinates = OMITTED;
    uint32_t words = 1;

    constexpr explicit CompileTimeLayout(uint32_t control, bool semaphore_ids = true);
    constexpr uint32_t semaphore(SemaphoreRole role) const;
};

template <typename Words>
constexpr void encode_compile_time_metadata(Words& words, ArgumentMetadata input, bool semaphore_ids = true);

template <typename Words>
constexpr ArgumentMetadata decode_compile_time_metadata(const Words& words, bool semaphore_ids = true);

}  // namespace mcast_wire
}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/mcast/mcast_protocol.inl"
