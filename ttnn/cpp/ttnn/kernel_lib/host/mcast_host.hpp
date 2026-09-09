// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "ttnn/cpp/ttnn/kernel_lib/mcast_common.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::kernel_lib::host {

// Data-ready signaling mode used by the kernel pipe.
enum class DataReadyMode : uint32_t { Flag = 0, Counter = 1 };

// Indicates that no consumer-ready semaphore is configured.
static constexpr uint32_t UNUSED_SEM_ID = 0xFFFFFFFFu;

// The host resolves this default from each sender's multicast fanout.
static constexpr uint32_t ACK_EQUALS_FANOUT = 0xFFFFFFFFu;

struct McastConfig {
    // NoC used by the kernel pipe.
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;
    // Wait for receiver readiness before sending.
    bool handshake = true;
    // Select the data-ready signaling mode.
    DataReadyMode data_ready = DataReadyMode::Flag;
    // First semaphore id allocated by the helper.
    uint32_t base_sem_id = 0;
    // Adopt caller-owned ids instead; consumer_ready is required when handshake is enabled.
    std::optional<std::vector<uint32_t>> sem_ids = std::nullopt;
    // Override the derived receiver acknowledgment count. Unlike the legacy Mcast2D
    // constructor's ack_count argument, zero is an explicit zero override.
    std::optional<uint32_t> ack_count_override = std::nullopt;
};

// Compile-time representation of an absent multicast channel. It emits only
// the false presence tag and therefore has no runtime payload or semaphores.
std::vector<uint32_t> absent_mcast_compile_time_args();

namespace detail {

template <typename Args>
void append_args_to(Args& destination, const std::vector<uint32_t>& args) {
    if constexpr (requires { destination.append(args); }) {
        destination.append(args);
    } else {
        destination.insert(destination.end(), args.begin(), args.end());
    }
}

}  // namespace detail

template <typename Args>
void append_absent_mcast_compile_time_args_to(Args& destination) {
    detail::append_args_to(destination, absent_mcast_compile_time_args());
}

// One exact receiver set and a nonempty ordered sender list.
// One sender is fixed; multiple senders rotate in the supplied order.
class McastGroup {
public:
    McastGroup(
        tt::tt_metal::CoreRangeSet receivers,
        std::vector<tt::tt_metal::CoreCoord> senders,
        std::optional<uint32_t> ack_count_override = std::nullopt);

    const tt::tt_metal::CoreRangeSet& receiver_cores() const { return receivers_; }
    const std::vector<tt::tt_metal::CoreCoord>& senders() const { return senders_; }
    bool rotating() const { return senders_.size() > 1; }
    std::optional<uint32_t> ack_count_override() const { return ack_count_override_; }

    const tt::tt_metal::CoreRangeSet& participating_cores() const { return participating_; }
    tt::tt_metal::CoreRangeSet sender_only_cores() const;
    bool is_sender(const tt::tt_metal::CoreCoord& core) const;
    uint32_t num_senders() const;
    // Remote fanout for a sender; zero for a non-sender.
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const;
    bool has_remote_receivers() const;

    // These methods require a prepared group, obtained through family.group(index).
    uint32_t ack_count(const tt::tt_metal::CoreCoord& core) const;
    uint32_t num_rectangles() const;
    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const;
    // Rejects cores outside this group's participating set, including cores in another group.
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const;

    template <typename Args>
    void append_compile_time_args_to(Args& destination, std::optional<bool> pre_handshake = std::nullopt) const {
        detail::append_args_to(destination, compile_time_args(pre_handshake));
    }
    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        detail::append_args_to(destination, runtime_args(core));
    }

private:
    friend class McastFamily;
    // The family derives one common layout after preparing all groups.
    struct ArgumentLayout {
        uint32_t rotating_span = 0;
        uint32_t rectangle_capacity = 1;
        uint32_t ack_count = 0;
        uint32_t uniform_remote_count = 0;
        uint32_t uniform_loopback_count = 0;
        dataflow_kernel_lib::SenderTransferMode transfer_mode =
            dataflow_kernel_lib::SenderTransferMode::TransferModeUnknown;
        bool has_remote_receivers = false;
        uint32_t data_ready_id = 0;
        uint32_t consumer_ready_id = UNUSED_SEM_ID;
        uint32_t flags = 0;
    };
    struct PreparedState {
        std::vector<tt::tt_metal::CoreRange> rectangles;
        std::vector<uint32_t> sender_coords;
        std::vector<std::vector<uint32_t>> sender_records;
        std::vector<uint32_t> acks;
        dataflow_kernel_lib::SenderTransferMode transfer_mode =
            dataflow_kernel_lib::SenderTransferMode::TransferModeUnknown;
    };
    void prepare_(tt::tt_metal::IDevice* device, const McastConfig& cfg);
    void set_argument_layout_(const ArgumentLayout& layout);
    const PreparedState& prepared_state_() const;
    const ArgumentLayout& argument_layout_() const;
    uint32_t sender_phase_(const tt::tt_metal::CoreCoord& core) const;
    tt::tt_metal::CoreRangeSet receivers_;
    std::vector<tt::tt_metal::CoreCoord> senders_;
    std::optional<uint32_t> ack_count_override_;
    tt::tt_metal::CoreRangeSet participating_;
    std::vector<uint32_t> fanouts_;
    std::optional<PreparedState> prepared_;
    std::optional<ArgumentLayout> layout_;
};

// Independent, disjoint groups sharing one protocol and semaphore allocation.
// All groups use the same sender mode and number of sender rounds.
class McastFamily {
public:
    McastFamily(tt::tt_metal::IDevice* device, std::vector<McastGroup> groups, const McastConfig& cfg = {});

    // Prepared member in constructor order; rejects an out-of-range index.
    const McastGroup& group(uint32_t index) const;
    std::vector<tt::tt_metal::SemaphoreDescriptor> owned_semaphores() const;
    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const;
    // Outside-family cores receive a correctly sized inactive argument block.
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const;

    template <typename Args>
    void append_compile_time_args_to(Args& destination, std::optional<bool> pre_handshake = std::nullopt) const {
        detail::append_args_to(destination, compile_time_args(pre_handshake));
    }
    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        detail::append_args_to(destination, runtime_args(core));
    }

    bool is_sender(const tt::tt_metal::CoreCoord& core) const;
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const;
    // Resolved count for this sender; zero for a non-sender. Zero overrides are explicit.
    uint32_t ack_count(const tt::tt_metal::CoreCoord& core) const;
    uint32_t num_senders() const;
    bool has_remote_receivers() const;
    const tt::tt_metal::CoreRangeSet& receiver_cores() const;
    const tt::tt_metal::CoreRangeSet& participating_cores() const;
    tt::tt_metal::CoreRangeSet sender_only_cores() const;
    uint32_t num_semaphores() const;
    uint32_t next_base_sem_id() const;
    uint32_t rectangle_capacity() const;
    uint32_t num_rectangles(const tt::tt_metal::CoreCoord& core) const;

private:
    friend class Mcast1D;
    friend class Mcast2D;
    const McastGroup* group_for_core_(const tt::tt_metal::CoreCoord& core) const;
    std::vector<McastGroup> groups_;
    McastConfig cfg_;
    tt::tt_metal::CoreRangeSet receivers_;
    tt::tt_metal::CoreRangeSet participating_;
    McastGroup::ArgumentLayout layout_;
};

// Mcast1D-specific types.

// Groups the receiver grid into independent row or column multicasts.
enum class Mcast1DShape {
    PerRow,
    PerColumn,
};

// Placement of the fixed sender on each row or column.
enum class Mcast1DSenderPlacement {
    Uniform,   // Use the same sender index on every line.
    Diagonal,  // Advance the sender index with each line.
};

struct Mcast1DFixedSenderConfig {
    uint32_t starting_sender_index = 0;
    Mcast1DSenderPlacement sender_placement = Mcast1DSenderPlacement::Uniform;
};

struct Mcast1DRotatingSenderConfig {
    // Rotate over receiver_grid when omitted.
    std::optional<tt::tt_metal::CoreRangeSet> sender_grid = std::nullopt;
};

using Mcast1DSenderConfig = std::variant<Mcast1DFixedSenderConfig, Mcast1DRotatingSenderConfig>;

// Configures independent row or column multicasts over a rectangular receiver grid.
// Fixed mode selects one sender per line. Rotating mode uses every core in sender_grid, or
// receiver_grid when sender_grid is omitted.
class Mcast1D {
public:
    // Unified API: the sender configuration selects fixed or rotating behavior.
    Mcast1D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& receiver_grid,
        Mcast1DShape shape,
        const Mcast1DSenderConfig& sender_config,
        const McastConfig& cfg = {});

    // Add these semaphore descriptors to the program. Empty when sem_ids are supplied.
    std::vector<tt::tt_metal::SemaphoreDescriptor> owned_semaphores() const;

    // Arguments consumed by McastArgs. pre_handshake overrides this kernel face only.
    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const;

    template <typename Args>
    void append_compile_time_args_to(Args& destination, std::optional<bool> pre_handshake = std::nullopt) const {
        detail::append_args_to(destination, compile_time_args(pre_handshake));
    }

    // Per-core runtime arguments consumed by McastArgs. Cores outside the participating
    // topology receive a correctly sized argument block with neither role enabled.
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const;

    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        detail::append_args_to(destination, runtime_args(core));
    }

    bool is_sender(const tt::tt_metal::CoreCoord& core) const;

    // Number of receivers reached by core, or zero when core is not an active sender.
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const;

    // Number of receiver acknowledgments the sender waits for.
    uint32_t ack_count() const;

    // Number of sender rounds; one in fixed mode.
    uint32_t num_senders() const;

    // True when at least one sender reaches a remote receiver. A degenerate
    // local-copy channel returns false here.
    bool has_remote_receivers() const;

    const tt::tt_metal::CoreRangeSet& receiver_cores() const;
    const tt::tt_metal::CoreRangeSet& participating_cores() const;
    tt::tt_metal::CoreRangeSet sender_only_cores() const;

    // Number of descriptors returned by owned_semaphores().
    uint32_t num_semaphores() const;

    // Use as base_sem_id for the next helper. Requires helper-owned semaphores.
    uint32_t next_base_sem_id() const;

private:
    static std::vector<std::vector<tt::tt_metal::CoreCoord>> sender_lines_from_grid_(
        const tt::tt_metal::CoreRangeSet& receiver_grid,
        const tt::tt_metal::CoreRangeSet& sender_grid,
        Mcast1DShape shape);

    std::optional<McastFamily> family_;
};

// Mcast2D-specific types.

// Traversal order used to assign rotating sender rounds over a 2D sender grid.
enum class Mcast2DSenderOrder {
    RowMajor,
    ColumnMajor,
};

struct Mcast2DFixedSenderConfig {
    tt::tt_metal::CoreCoord sender;
};

struct Mcast2DRotatingSenderConfig {
    // Rotate over mcast_rect when omitted.
    std::optional<tt::tt_metal::CoreRangeSet> sender_grid = std::nullopt;
    Mcast2DSenderOrder sender_order = Mcast2DSenderOrder::RowMajor;
};

using Mcast2DSenderConfig = std::variant<Mcast2DFixedSenderConfig, Mcast2DRotatingSenderConfig>;

// Configures one multicast over a rectangular receiver grid.
// The fixed sender may be inside or outside the receiver grid. Rotating mode uses sender_grid, or
// mcast_rect when sender_grid is omitted.
class Mcast2D {
public:
    // Unified API: the sender configuration selects fixed or rotating behavior.
    Mcast2D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& mcast_rect,
        const Mcast2DSenderConfig& sender_config,
        const McastConfig& cfg = {});

    // Add these semaphore descriptors to the program. Empty when sem_ids are supplied.
    std::vector<tt::tt_metal::SemaphoreDescriptor> owned_semaphores() const;

    // Arguments consumed by McastArgs. pre_handshake overrides this kernel face only.
    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const;

    template <typename Args>
    void append_compile_time_args_to(Args& destination, std::optional<bool> pre_handshake = std::nullopt) const {
        detail::append_args_to(destination, compile_time_args(pre_handshake));
    }

    // Per-core runtime arguments consumed by McastArgs.
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const;

    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        detail::append_args_to(destination, runtime_args(core));
    }

    bool is_sender(const tt::tt_metal::CoreCoord& core) const;

    // Number of receivers reached by core, or zero when core is not a sender with receivers.
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const;

    // Number of receiver acknowledgments the sender waits for.
    uint32_t ack_count() const;

    // Number of sender rounds; one in fixed mode.
    uint32_t num_senders() const;

    // True when at least one sender reaches a remote receiver. A degenerate
    // local-copy channel returns false here.
    bool has_remote_receivers() const;

    // Whether the fixed sender, or first rotating sender, is inside the receiver grid.
    bool sender_in_rect() const;

    // Number of descriptors returned by owned_semaphores().
    uint32_t num_semaphores() const;

    // Use as base_sem_id for the next helper. Requires helper-owned semaphores.
    uint32_t next_base_sem_id() const;

private:
    static std::vector<tt::tt_metal::CoreCoord> senders_from_grid_(
        const tt::tt_metal::CoreRangeSet& sender_grid, Mcast2DSenderOrder sender_order);

    std::optional<McastFamily> family_;
    bool sender_in_rect_ = false;
};

}  // namespace ttnn::kernel_lib::host
