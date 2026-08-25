// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mcast_host.hpp"

#include <algorithm>
#include <cstddef>

#include <tt_stl/assert.hpp>

namespace ttnn::kernel_lib::host {
namespace detail {

std::pair<uint32_t, uint32_t> virt_coord(tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord& logical) {
    const auto worker = device->worker_core_from_logical_core(logical);
    return {static_cast<uint32_t>(worker.x), static_cast<uint32_t>(worker.y)};
}

uint32_t mcast_flags(const McastConfig& cfg, std::optional<bool> pre_handshake_override = std::nullopt) {
    uint32_t flags = 0;
    if (pre_handshake_override.value_or(cfg.handshake)) {
        flags |= 0x1u;
    }
    if (cfg.data_ready == DataReadyMode::Counter) {
        flags |= 0x2u;
    }
    return flags;
}

std::vector<uint32_t> noc_ordered_bbox(
    tt::tt_metal::NOC noc, const std::vector<std::pair<uint32_t, uint32_t>>& coordinates) {
    uint32_t xlo = coordinates[0].first;
    uint32_t xhi = coordinates[0].first;
    uint32_t ylo = coordinates[0].second;
    uint32_t yhi = coordinates[0].second;
    for (const auto& coordinate : coordinates) {
        xlo = std::min(xlo, coordinate.first);
        xhi = std::max(xhi, coordinate.first);
        ylo = std::min(ylo, coordinate.second);
        yhi = std::max(yhi, coordinate.second);
    }
    if (noc == tt::tt_metal::NOC::NOC_1) {
        return {xhi, yhi, xlo, ylo};
    }
    return {xlo, ylo, xhi, yhi};
}

}  // namespace detail

Mcast1D::Mcast1D(
    tt::tt_metal::IDevice* device,
    const tt::tt_metal::CoreRangeSet& receiver_grid,
    Mcast1DShape shape,
    uint32_t starting_sender_index,
    const McastConfig& cfg,
    Mcast1DSenderPlacement sender_placement,
    std::optional<tt::tt_metal::CoreRangeSet> sender_grid) :
    device_(device),
    shape_(shape),
    starting_sender_index_(starting_sender_index),
    sender_placement_(sender_placement),
    cfg_(cfg) {
    TT_FATAL(device_ != nullptr, "Mcast1D: device must not be null");

    const auto receiver_box = receiver_grid.bounding_box();
    TT_FATAL(
        receiver_grid.num_cores() == receiver_box.size(),
        "Mcast1D: receiver grid must be one dense rectangle (bounding box has {} cores, set has {})",
        receiver_box.size(),
        receiver_grid.num_cores());
    origin_x_ = static_cast<uint32_t>(receiver_box.start_coord.x);
    origin_y_ = static_cast<uint32_t>(receiver_box.start_coord.y);
    GC_ = static_cast<uint32_t>(receiver_box.end_coord.x - receiver_box.start_coord.x) + 1;
    GR_ = static_cast<uint32_t>(receiver_box.end_coord.y - receiver_box.start_coord.y) + 1;

    receiver_span_ = (shape_ == Mcast1DShape::PerRow) ? GC_ : GR_;
    receiver_grid_ = receiver_grid;

    const auto& effective_sender_grid = sender_grid.has_value() ? *sender_grid : receiver_grid;
    sender_lines_ = sender_lines_from_grid_(receiver_grid, effective_sender_grid, shape_);
    span_ = sender_lines_.front().size();
    for (uint32_t line = 0; line < sender_lines_.size(); ++line) {
        TT_FATAL(
            sender_lines_[line].size() == span_,
            "Mcast1D: sender line {} has {} cores; expected {}",
            line,
            sender_lines_[line].size(),
            span_);
    }

    TT_FATAL(
        !cfg_.rotating_sender || sender_placement_ == Mcast1DSenderPlacement::Uniform,
        "Mcast1D: Diagonal sender placement cannot be combined with rotating_sender");

    if (!cfg_.rotating_sender) {
        TT_FATAL(
            starting_sender_index_ < span_,
            "Mcast1D: starting_sender_index {} must be less than the sender-line span {}",
            starting_sender_index_,
            span_);
    }

    std::vector<tt::tt_metal::CoreRange> participating_ranges = receiver_grid.ranges();
    bool have_fanout = false;
    bool uniform_fanout = true;
    uint32_t first_fanout = 0;
    const auto add_sender = [&](const tt::tt_metal::CoreCoord& sender) {
        const bool sender_in_receiver_grid = receiver_box.contains(sender);
        if (!sender_in_receiver_grid) {
            participating_ranges.emplace_back(sender, sender);
        }
        const uint32_t fanout = receiver_span_ - (sender_in_receiver_grid ? 1u : 0u);
        active_ = active_ || fanout > 0;
        if (!have_fanout) {
            first_fanout = fanout;
            have_fanout = true;
        } else {
            uniform_fanout = uniform_fanout && fanout == first_fanout;
        }
    };
    for (uint32_t line = 0; line < sender_lines_.size(); ++line) {
        if (cfg_.rotating_sender) {
            for (const auto& sender : sender_lines_[line]) {
                add_sender(sender);
            }
        } else {
            add_sender(sender_lines_[line][sender_index_for_line_(line)]);
        }
    }
    ack_count_ = uniform_fanout ? first_fanout : 0xFFFFFFFFu;
    grid_ = tt::tt_metal::CoreRangeSet(std::move(participating_ranges));

    if (cfg_.sem_ids.has_value()) {
        const auto& ids = *cfg_.sem_ids;
        TT_FATAL(!ids.empty(), "Mcast1D: adopted sem_ids must contain at least the data_ready id");
        data_ready_id_ = ids[0];
        consumer_ready_id_ = cfg_.handshake ? (ids.size() > 1 ? ids[1] : UNUSED_SEM_ID) : UNUSED_SEM_ID;
        owns_sems_ = false;
    } else {
        data_ready_id_ = cfg_.base_sem_id;
        consumer_ready_id_ = cfg_.handshake ? (cfg_.base_sem_id + 1) : UNUSED_SEM_ID;
        owns_sems_ = true;
    }
}

std::vector<tt::tt_metal::SemaphoreDescriptor> Mcast1D::owned_semaphores() const {
    std::vector<tt::tt_metal::SemaphoreDescriptor> semaphores;
    if (!owns_sems_) {
        return semaphores;
    }
    semaphores.push_back(
        tt::tt_metal::SemaphoreDescriptor{.id = data_ready_id_, .core_ranges = grid_, .initial_value = 0});
    if (cfg_.handshake) {
        semaphores.push_back(
            tt::tt_metal::SemaphoreDescriptor{.id = consumer_ready_id_, .core_ranges = grid_, .initial_value = 0});
    }
    return semaphores;
}

std::vector<uint32_t> Mcast1D::compile_time_args(std::optional<bool> pre_handshake) const {
    // TODO: Share this CT argument layout and count with kernel McastArgs.
    return {
        active_ ? 1u : 0u,
        data_ready_id_,
        consumer_ready_id_,
        num_active(),
        detail::mcast_flags(cfg_, pre_handshake),
        cfg_.rotating_sender ? span_ : 0u};
}

uint32_t Mcast1D::num_active() const { return ack_count_; }

std::vector<uint32_t> Mcast1D::runtime_args(const tt::tt_metal::CoreCoord& core) const {
    // TODO: Share this RT argument layout and count with McastArgs.
    if (cfg_.rotating_sender) {
        return rotating_rt_(core);
    }
    if (is_sender(core)) {
        return line_rect_(core);
    }
    const auto sender = sender_of_(core);
    const auto virtual_sender = virt_(sender);
    return {virtual_sender.first, virtual_sender.second, 0, 0};
}

bool Mcast1D::is_sender(const tt::tt_metal::CoreCoord& core) const {
    for (uint32_t line = 0; line < sender_lines_.size(); ++line) {
        if (cfg_.rotating_sender) {
            if (std::find(sender_lines_[line].begin(), sender_lines_[line].end(), core) != sender_lines_[line].end()) {
                return true;
            }
        } else if (sender_lines_[line][sender_index_for_line_(line)] == core) {
            return true;
        }
    }
    return false;
}

uint32_t Mcast1D::num_receivers(const tt::tt_metal::CoreCoord& core) const {
    if (!active_ || !is_sender(core)) {
        return 0;
    }
    return receiver_span_ - (receiver_grid_.bounding_box().contains(core) ? 1u : 0u);
}

uint32_t Mcast1D::num_senders() const { return cfg_.rotating_sender ? span_ : 1u; }

bool Mcast1D::active() const { return active_; }

uint32_t Mcast1D::num_semaphores() const { return owns_sems_ ? (cfg_.handshake ? 2u : 1u) : 0u; }

uint32_t Mcast1D::next_base_sem_id() const {
    TT_FATAL(
        owns_sems_,
        "Mcast1D::next_base_sem_id() is only valid when the helper created its own semaphores; this "
        "instance adopted explicit sem_ids, so the caller owns semaphore-id allocation.");
    return cfg_.base_sem_id + num_semaphores();
}

std::vector<std::vector<tt::tt_metal::CoreCoord>> Mcast1D::sender_lines_from_grid_(
    const tt::tt_metal::CoreRangeSet& receiver_grid,
    const tt::tt_metal::CoreRangeSet& sender_grid,
    Mcast1DShape shape) {
    TT_FATAL(sender_grid.num_cores() > 0, "Mcast1D: sender grid must not be empty");

    const auto receiver_box = receiver_grid.bounding_box();
    const uint32_t num_lines = shape == Mcast1DShape::PerRow
                                   ? static_cast<uint32_t>(receiver_box.end_coord.y - receiver_box.start_coord.y) + 1
                                   : static_cast<uint32_t>(receiver_box.end_coord.x - receiver_box.start_coord.x) + 1;
    std::vector<std::vector<tt::tt_metal::CoreCoord>> sender_lines(num_lines);

    for (const auto& range : sender_grid.ranges()) {
        for (std::size_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (std::size_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                const tt::tt_metal::CoreCoord sender{x, y};
                const bool aligned = shape == Mcast1DShape::PerRow
                                         ? y >= receiver_box.start_coord.y && y <= receiver_box.end_coord.y
                                         : x >= receiver_box.start_coord.x && x <= receiver_box.end_coord.x;
                TT_FATAL(
                    aligned,
                    "Mcast1D: sender ({},{}) does not align with any receiver {}",
                    x,
                    y,
                    shape == Mcast1DShape::PerRow ? "row" : "column");
                const uint32_t line = shape == Mcast1DShape::PerRow
                                          ? static_cast<uint32_t>(y - receiver_box.start_coord.y)
                                          : static_cast<uint32_t>(x - receiver_box.start_coord.x);
                sender_lines[line].push_back(sender);
            }
        }
    }

    for (uint32_t line = 0; line < num_lines; ++line) {
        auto& senders = sender_lines[line];
        TT_FATAL(!senders.empty(), "Mcast1D: receiver line {} has no sender cores", line);
        std::sort(senders.begin(), senders.end(), [shape](const auto& lhs, const auto& rhs) {
            return shape == Mcast1DShape::PerRow ? lhs.x < rhs.x : lhs.y < rhs.y;
        });
        TT_FATAL(
            std::adjacent_find(senders.begin(), senders.end()) == senders.end(),
            "Mcast1D: sender grid contains a duplicate core on line {}",
            line);
    }
    return sender_lines;
}

std::pair<uint32_t, uint32_t> Mcast1D::virt_(const tt::tt_metal::CoreCoord& logical) const {
    return detail::virt_coord(device_, logical);
}

tt::tt_metal::CoreCoord Mcast1D::sender_of_(const tt::tt_metal::CoreCoord& core) const {
    const uint32_t line = line_index_(core);
    return sender_lines_[line][sender_index_for_line_(line)];
}

uint32_t Mcast1D::line_index_(const tt::tt_metal::CoreCoord& core) const {
    return (shape_ == Mcast1DShape::PerRow) ? static_cast<uint32_t>(core.y) - origin_y_
                                            : static_cast<uint32_t>(core.x) - origin_x_;
}

uint32_t Mcast1D::sender_index_for_line_(uint32_t line) const {
    if (sender_placement_ == Mcast1DSenderPlacement::Diagonal) {
        return (starting_sender_index_ + line) % span_;
    }
    return starting_sender_index_;
}

tt::tt_metal::CoreCoord Mcast1D::line_coord_(const tt::tt_metal::CoreCoord& core, uint32_t i) const {
    return (shape_ == Mcast1DShape::PerRow) ? tt::tt_metal::CoreCoord{origin_x_ + i, core.y}
                                            : tt::tt_metal::CoreCoord{core.x, origin_y_ + i};
}

std::vector<uint32_t> Mcast1D::noc_ordered_bbox_(const std::vector<std::pair<uint32_t, uint32_t>>& coordinates) const {
    return detail::noc_ordered_bbox(cfg_.noc, coordinates);
}

std::vector<uint32_t> Mcast1D::line_rect_(const tt::tt_metal::CoreCoord& core) const {
    std::vector<std::pair<uint32_t, uint32_t>> coordinates;
    coordinates.reserve(receiver_span_);
    for (uint32_t i = 0; i < receiver_span_; ++i) {
        coordinates.push_back(virt_(line_coord_(core, i)));
    }
    return noc_ordered_bbox_(coordinates);
}

std::vector<uint32_t> Mcast1D::rotating_rt_(const tt::tt_metal::CoreCoord& core) const {
    const uint32_t line = line_index_(core);
    std::vector<std::pair<uint32_t, uint32_t>> receiver_coordinates;
    receiver_coordinates.reserve(receiver_span_);
    for (uint32_t i = 0; i < receiver_span_; ++i) {
        receiver_coordinates.push_back(virt_(line_coord_(core, i)));
    }
    std::vector<uint32_t> runtime_args =
        active_ ? noc_ordered_bbox_(receiver_coordinates) : std::vector<uint32_t>{0, 0, 0, 0};
    for (const auto& sender : sender_lines_[line]) {
        const auto coordinate = virt_(sender);
        runtime_args.push_back(coordinate.first);
        runtime_args.push_back(coordinate.second);
    }
    return runtime_args;
}

Mcast2D::Mcast2D(
    tt::tt_metal::IDevice* device,
    const tt::tt_metal::CoreRangeSet& mcast_rect,
    tt::tt_metal::CoreCoord sender,
    const McastConfig& cfg,
    uint32_t num_active,
    std::optional<tt::tt_metal::CoreRangeSet> sender_grid) :
    device_(device), sender_(sender), cfg_(cfg) {
    TT_FATAL(device_ != nullptr, "Mcast2D: device must not be null");

    const auto receiver_box = mcast_rect.bounding_box();
    TT_FATAL(
        mcast_rect.num_cores() == receiver_box.size(),
        "Mcast2D: receiver set must be one dense rectangle (bounding box has {} cores, set has {})",
        receiver_box.size(),
        mcast_rect.num_cores());
    rx0_ = static_cast<uint32_t>(receiver_box.start_coord.x);
    ry0_ = static_cast<uint32_t>(receiver_box.start_coord.y);
    rx1_ = static_cast<uint32_t>(receiver_box.end_coord.x);
    ry1_ = static_cast<uint32_t>(receiver_box.end_coord.y);
    area_ = (rx1_ - rx0_ + 1) * (ry1_ - ry0_ + 1);

    std::vector<tt::tt_metal::CoreRange> participating_ranges = mcast_rect.ranges();
    if (cfg_.rotating_sender) {
        TT_FATAL(
            sender_grid.has_value() || receiver_box.contains(sender_),
            "Mcast2D: rotating_sender without sender_grid rotates within the receiver rect, so sender "
            "({},{}) must lie inside rect [{},{}]-[{},{}]",
            sender_.x,
            sender_.y,
            rx0_,
            ry0_,
            rx1_,
            ry1_);
        const auto& effective_sender_grid = sender_grid.has_value() ? *sender_grid : mcast_rect;
        senders_ = senders_from_grid_(effective_sender_grid);
        sender_ = senders_.front();
        sender_in_rect_ = in_rect_(sender_);

        bool uniform_fanout = true;
        const uint32_t first_fanout = area_ - (sender_in_rect_ ? 1u : 0u);
        uint32_t minimum_fanout = first_fanout;
        for (const auto& rotating_sender : senders_) {
            const bool sender_in_rect = in_rect_(rotating_sender);
            const uint32_t fanout = area_ - (sender_in_rect ? 1u : 0u);
            uniform_fanout = uniform_fanout && fanout == first_fanout;
            minimum_fanout = std::min(minimum_fanout, fanout);
            active_ = active_ || fanout > 0;
            if (!sender_in_rect) {
                participating_ranges.emplace_back(rotating_sender, rotating_sender);
            }
        }
        ack_count_ = num_active == 0 ? (uniform_fanout ? first_fanout : 0xFFFFFFFFu) : num_active;
        TT_FATAL(
            num_active == 0 || num_active <= minimum_fanout,
            "Mcast2D: num_active ({}) exceeds the minimum rotating sender fan-out ({})",
            num_active,
            minimum_fanout);
    } else {
        TT_FATAL(!sender_grid.has_value(), "Mcast2D: sender_grid is only valid when rotating_sender=true");
        sender_in_rect_ = receiver_box.contains(sender_);
        const uint32_t receivers = sender_in_rect_ ? (area_ - 1) : area_;
        active_ = receivers > 0;
        ack_count_ = (num_active == 0) ? receivers : num_active;
        TT_FATAL(
            ack_count_ <= receivers,
            "Mcast2D: num_active ({}) exceeds the receiver fan-out ({})",
            ack_count_,
            receivers);
        if (!sender_in_rect_) {
            participating_ranges.emplace_back(sender_, sender_);
        }
    }
    participating_ = tt::tt_metal::CoreRangeSet(std::move(participating_ranges));

    if (cfg_.sem_ids.has_value()) {
        const auto& ids = *cfg_.sem_ids;
        TT_FATAL(!ids.empty(), "Mcast2D: adopted sem_ids must contain at least the data_ready id");
        data_ready_id_ = ids[0];
        consumer_ready_id_ = cfg_.handshake ? (ids.size() > 1 ? ids[1] : UNUSED_SEM_ID) : UNUSED_SEM_ID;
        owns_sems_ = false;
    } else {
        data_ready_id_ = cfg_.base_sem_id;
        consumer_ready_id_ = cfg_.handshake ? (cfg_.base_sem_id + 1) : UNUSED_SEM_ID;
        owns_sems_ = true;
    }
}

std::vector<tt::tt_metal::SemaphoreDescriptor> Mcast2D::owned_semaphores() const {
    std::vector<tt::tt_metal::SemaphoreDescriptor> semaphores;
    if (!owns_sems_) {
        return semaphores;
    }
    semaphores.push_back(
        tt::tt_metal::SemaphoreDescriptor{.id = data_ready_id_, .core_ranges = participating_, .initial_value = 0});
    if (cfg_.handshake) {
        semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = consumer_ready_id_, .core_ranges = participating_, .initial_value = 0});
    }
    return semaphores;
}

std::vector<uint32_t> Mcast2D::compile_time_args(std::optional<bool> pre_handshake) const {
    // TODO: Share this CT argument layout and count with McastArgs.
    return {
        active_ ? 1u : 0u,
        data_ready_id_,
        consumer_ready_id_,
        ack_count_,
        detail::mcast_flags(cfg_, pre_handshake),
        cfg_.rotating_sender ? num_senders() : 0u};
}

std::vector<uint32_t> Mcast2D::runtime_args(const tt::tt_metal::CoreCoord& core) const {
    // TODO: Share this RT argument layout and count with McastArgs.
    if (cfg_.rotating_sender) {
        return rotating_rt_();
    }
    if (is_sender(core)) {
        return rect_corners_();
    }
    const auto virtual_sender = detail::virt_coord(device_, sender_);
    return {virtual_sender.first, virtual_sender.second, 0, 0};
}

bool Mcast2D::is_sender(const tt::tt_metal::CoreCoord& core) const {
    if (cfg_.rotating_sender) {
        return active_ && std::find(senders_.begin(), senders_.end(), core) != senders_.end();
    }
    return core == sender_;
}

uint32_t Mcast2D::num_receivers(const tt::tt_metal::CoreCoord& core) const {
    if (!active_) {
        return 0;
    }
    if (cfg_.rotating_sender) {
        return is_sender(core) ? area_ - (in_rect_(core) ? 1u : 0u) : 0u;
    }
    return is_sender(core) ? area_ - (sender_in_rect_ ? 1u : 0u) : 0u;
}

uint32_t Mcast2D::num_active() const { return ack_count_; }

uint32_t Mcast2D::num_senders() const { return cfg_.rotating_sender ? senders_.size() : 1u; }

bool Mcast2D::active() const { return active_; }

bool Mcast2D::sender_in_rect() const { return sender_in_rect_; }

uint32_t Mcast2D::num_semaphores() const { return owns_sems_ ? (cfg_.handshake ? 2u : 1u) : 0u; }

uint32_t Mcast2D::next_base_sem_id() const {
    TT_FATAL(
        owns_sems_,
        "Mcast2D::next_base_sem_id() is only valid when the helper created its own semaphores; this "
        "instance adopted explicit sem_ids, so the caller owns semaphore-id allocation.");
    return cfg_.base_sem_id + num_semaphores();
}

std::vector<tt::tt_metal::CoreCoord> Mcast2D::senders_from_grid_(const tt::tt_metal::CoreRangeSet& sender_grid) {
    TT_FATAL(sender_grid.num_cores() > 0, "Mcast2D: sender grid must not be empty");
    std::vector<tt::tt_metal::CoreCoord> senders;
    senders.reserve(sender_grid.num_cores());
    for (const auto& range : sender_grid.ranges()) {
        for (std::size_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (std::size_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                senders.emplace_back(x, y);
            }
        }
    }
    std::sort(senders.begin(), senders.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.y == rhs.y ? lhs.x < rhs.x : lhs.y < rhs.y;
    });
    TT_FATAL(
        std::adjacent_find(senders.begin(), senders.end()) == senders.end(),
        "Mcast2D: sender grid contains a duplicate core");
    return senders;
}

bool Mcast2D::in_rect_(const tt::tt_metal::CoreCoord& core) const {
    const auto x = static_cast<uint32_t>(core.x);
    const auto y = static_cast<uint32_t>(core.y);
    return x >= rx0_ && x <= rx1_ && y >= ry0_ && y <= ry1_;
}

std::vector<std::pair<uint32_t, uint32_t>> Mcast2D::rect_virt_coords_() const {
    std::vector<std::pair<uint32_t, uint32_t>> coordinates;
    coordinates.reserve(area_);
    for (uint32_t y = ry0_; y <= ry1_; ++y) {
        for (uint32_t x = rx0_; x <= rx1_; ++x) {
            coordinates.push_back(detail::virt_coord(device_, tt::tt_metal::CoreCoord{x, y}));
        }
    }
    return coordinates;
}

std::vector<uint32_t> Mcast2D::rect_corners_() const { return detail::noc_ordered_bbox(cfg_.noc, rect_virt_coords_()); }

std::vector<uint32_t> Mcast2D::rotating_rt_() const {
    const auto rectangle_coordinates = rect_virt_coords_();
    std::vector<uint32_t> runtime_args = detail::noc_ordered_bbox(cfg_.noc, rectangle_coordinates);
    for (const auto& sender : senders_) {
        const auto coordinate = detail::virt_coord(device_, sender);
        runtime_args.push_back(coordinate.first);
        runtime_args.push_back(coordinate.second);
    }
    return runtime_args;
}

}  // namespace ttnn::kernel_lib::host
