// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::kernel_lib::host {

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

// Data-ready signaling mode used by the kernel pipe.
enum class DataReadyMode : uint32_t { Flag = 0, Counter = 1 };

// Indicates that no consumer-ready semaphore is configured.
static constexpr uint32_t UNUSED_SEM_ID = 0xFFFFFFFFu;

struct McastConfig {
    // NoC used by the kernel pipe.
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;
    // Wait for receiver readiness before sending.
    bool handshake = true;
    // Select the data-ready signaling mode.
    DataReadyMode data_ready = DataReadyMode::Flag;
    // Rotate the sender role across the sender grid.
    bool rotating_sender = false;
    // First semaphore id allocated by the helper.
    uint32_t base_sem_id = 0;
    // Adopt caller-owned ids instead; consumer_ready is required when handshake is enabled.
    std::optional<std::vector<uint32_t>> sem_ids = std::nullopt;
};

// Configures independent row or column multicasts over a rectangular receiver grid.
// Fixed mode selects one sender per line. Rotating mode uses every core in sender_grid, or
// receiver_grid when sender_grid is omitted.
class Mcast1D {
public:
    Mcast1D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& receiver_grid,
        Mcast1DShape shape,
        uint32_t starting_sender_index,
        const McastConfig& cfg,
        Mcast1DSenderPlacement sender_placement = Mcast1DSenderPlacement::Uniform,
        std::optional<tt::tt_metal::CoreRangeSet> sender_grid = std::nullopt);

    // Add these semaphore descriptors to the program. Empty when sem_ids are supplied.
    std::vector<tt::tt_metal::SemaphoreDescriptor> owned_semaphores() const;

    // Append these arguments to the kernel compile-time arguments consumed by McastArgs.
    // `pre_handshake` overrides handshake behavior for this kernel only.
    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const;

    // Use these as the core's runtime arguments consumed by McastArgs.
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const;

    bool is_sender(const tt::tt_metal::CoreCoord& core) const;

    // Number of receivers reached by `core`, or zero when `core` is not an active sender.
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const;

    // Number of receiver acknowledgments the sender waits for.
    uint32_t num_active() const;

    // Number of sender rounds; one in fixed mode.
    uint32_t num_senders() const;

    bool active() const;

    // Number of descriptors returned by owned_semaphores().
    uint32_t num_semaphores() const;

    // Use as base_sem_id for the next helper. Requires helper-owned semaphores.
    uint32_t next_base_sem_id() const;

private:
    static std::vector<std::vector<tt::tt_metal::CoreCoord>> sender_lines_from_grid_(
        const tt::tt_metal::CoreRangeSet& receiver_grid,
        const tt::tt_metal::CoreRangeSet& sender_grid,
        Mcast1DShape shape);

    std::pair<uint32_t, uint32_t> virt_(const tt::tt_metal::CoreCoord& logical) const;
    tt::tt_metal::CoreCoord sender_of_(const tt::tt_metal::CoreCoord& core) const;
    uint32_t line_index_(const tt::tt_metal::CoreCoord& core) const;
    uint32_t sender_index_for_line_(uint32_t line) const;
    tt::tt_metal::CoreCoord line_coord_(const tt::tt_metal::CoreCoord& core, uint32_t i) const;
    std::vector<uint32_t> noc_ordered_bbox_(const std::vector<std::pair<uint32_t, uint32_t>>& vs) const;
    std::vector<uint32_t> sender_rect_(const tt::tt_metal::CoreCoord& core) const;
    std::vector<uint32_t> rotating_rt_(const tt::tt_metal::CoreCoord& core) const;

    tt::tt_metal::IDevice* device_;
    tt::tt_metal::CoreRangeSet grid_;
    tt::tt_metal::CoreRangeSet receiver_grid_;
    Mcast1DShape shape_;
    uint32_t starting_sender_index_;
    Mcast1DSenderPlacement sender_placement_;
    McastConfig cfg_;
    uint32_t origin_x_ = 0;
    uint32_t origin_y_ = 0;
    uint32_t GR_ = 1;
    uint32_t GC_ = 1;
    uint32_t span_ = 1;
    uint32_t receiver_span_ = 1;
    std::vector<std::vector<tt::tt_metal::CoreCoord>> sender_lines_;
    bool active_ = false;
    uint32_t ack_count_ = 0;
    bool owns_sems_ = true;
    uint32_t data_ready_id_ = 0;
    uint32_t consumer_ready_id_ = UNUSED_SEM_ID;
};

// Configures one multicast over a rectangular receiver grid.
// The sender may be inside or outside the receiver grid. Rotating mode uses sender_grid, or mcast_rect
// when sender_grid is omitted. Set num_active only when fewer than all receivers acknowledge readiness.
class Mcast2D {
public:
    Mcast2D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& mcast_rect,
        tt::tt_metal::CoreCoord sender,
        const McastConfig& cfg,
        uint32_t num_active = 0,
        std::optional<tt::tt_metal::CoreRangeSet> sender_grid = std::nullopt);

    // Add these semaphore descriptors to the program. Empty when sem_ids are supplied.
    std::vector<tt::tt_metal::SemaphoreDescriptor> owned_semaphores() const;

    // Append these arguments to the kernel compile-time arguments consumed by McastArgs.
    // `pre_handshake` overrides handshake behavior for this kernel only.
    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const;

    // Use these as the core's runtime arguments consumed by McastArgs.
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const;

    bool is_sender(const tt::tt_metal::CoreCoord& core) const;

    // Number of receivers reached by `core`, or zero when `core` is not an active sender.
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const;

    // Number of receiver acknowledgments the sender waits for.
    uint32_t num_active() const;

    // Number of sender rounds; one in fixed mode.
    uint32_t num_senders() const;

    bool active() const;

    // Whether the fixed sender, or first rotating sender, is inside the receiver grid.
    bool sender_in_rect() const;

    // Number of descriptors returned by owned_semaphores().
    uint32_t num_semaphores() const;

    // Use as base_sem_id for the next helper. Requires helper-owned semaphores.
    uint32_t next_base_sem_id() const;

private:
    static std::vector<tt::tt_metal::CoreCoord> senders_from_grid_(const tt::tt_metal::CoreRangeSet& sender_grid);

    bool in_rect_(const tt::tt_metal::CoreCoord& core) const;
    std::vector<std::pair<uint32_t, uint32_t>> rect_virt_coords_() const;
    std::vector<uint32_t> rect_corners_() const;
    std::vector<uint32_t> rotating_rt_() const;

    tt::tt_metal::IDevice* device_;
    tt::tt_metal::CoreRangeSet participating_;
    tt::tt_metal::CoreCoord sender_;
    McastConfig cfg_;
    uint32_t rx0_ = 0;
    uint32_t ry0_ = 0;
    uint32_t rx1_ = 0;
    uint32_t ry1_ = 0;
    uint32_t area_ = 1;
    std::vector<tt::tt_metal::CoreCoord> senders_;
    bool sender_in_rect_ = true;
    bool active_ = false;
    bool owns_sems_ = true;
    uint32_t ack_count_ = 0;
    uint32_t data_ready_id_ = 0;
    uint32_t consumer_ready_id_ = UNUSED_SEM_ID;
};

}  // namespace ttnn::kernel_lib::host
