// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp"

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
        flags |= dataflow_kernel_lib::mcast_wire::PRE_HANDSHAKE;
    }
    if (cfg.data_ready == DataReadyMode::Counter) {
        flags |= dataflow_kernel_lib::mcast_wire::COUNTER_SIGNAL;
    }
    if (cfg.noc == tt::tt_metal::NOC::NOC_1) {
        flags |= dataflow_kernel_lib::mcast_wire::NOC1;
    }
    return flags;
}

void append_role_args(std::vector<uint32_t>& args, bool can_send, bool can_receive, uint32_t sender_round) {
    args.push_back(
        (can_send ? dataflow_kernel_lib::mcast_wire::CAN_SEND : 0u) |
        (can_receive ? dataflow_kernel_lib::mcast_wire::CAN_RECEIVE : 0u));
    args.push_back(sender_round);
}

void append_sender_coords(
    std::vector<uint32_t>& args, tt::tt_metal::IDevice* device, const std::vector<tt::tt_metal::CoreCoord>& senders) {
    args.reserve(args.size() + 2u * senders.size());
    for (const auto& sender : senders) {
        const auto coordinate = virt_coord(device, sender);
        args.push_back(coordinate.first);
        args.push_back(coordinate.second);
    }
}

// Coordinates are already mapped; never take a bounding box across omitted workers.
dataflow_kernel_lib::NocBounds noc_ordered_bounds(tt::tt_metal::NOC noc, const tt::tt_metal::CoreRange& rectangle) {
    const auto& lo = rectangle.start_coord;
    const auto& hi = rectangle.end_coord;
    if (noc == tt::tt_metal::NOC::NOC_1) {
        return {uint32_t(hi.x), uint32_t(hi.y), uint32_t(lo.x), uint32_t(lo.y)};
    }
    return {uint32_t(lo.x), uint32_t(lo.y), uint32_t(hi.x), uint32_t(hi.y)};
}

}  // namespace detail

using dataflow_kernel_lib::SenderTransferMode;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
namespace wire = dataflow_kernel_lib::mcast_wire;

std::vector<uint32_t> absent_mcast_compile_time_args() { return {wire::ABSENT}; }

McastGroup::McastGroup(
    CoreRangeSet receivers, std::vector<CoreCoord> senders, std::optional<uint32_t> ack_count_override) :
    receivers_(std::move(receivers)), senders_(std::move(senders)), ack_count_override_(ack_count_override) {
    TT_FATAL(!receivers_.empty(), "McastGroup: receiver set must not be empty; self-only groups are supported");
    TT_FATAL(!senders_.empty(), "McastGroup: sender schedule must not be empty");
    for (size_t i = 0; i < senders_.size(); ++i) {
        TT_FATAL(
            std::find(senders_.begin(), senders_.begin() + i, senders_[i]) == senders_.begin() + i,
            "McastGroup: duplicate sender ({},{})",
            senders_[i].x,
            senders_[i].y);
    }
    participating_ = receivers_;
    for (const auto& sender : senders_) {
        participating_ = participating_.merge(CoreRangeSet(CoreRange(sender, sender)));
        fanouts_.push_back(receivers_.num_cores() - receivers_.contains(sender));
    }
}

void McastGroup::prepare_(tt::tt_metal::IDevice* device, const McastConfig& cfg) {
    // A prepared group can be copied into a new family with a different device/configuration.
    layout_.reset();
    auto& state = prepared_.emplace(PreparedState{});
    std::vector<CoreRange> mapped;
    for (const auto& logical : tt::tt_metal::corerange_to_cores(receivers_, std::nullopt, true)) {
        const auto worker = device->worker_core_from_logical_core(logical);
        mapped.emplace_back(worker, worker);
    }
    state.rectangles = CoreRangeSet(std::move(mapped)).merge_ranges().ranges();
    TT_FATAL(
        state.rectangles.size() <= dataflow_kernel_lib::MAX_MCAST_RECTANGLES,
        "McastGroup: requires {} mapped NoC rectangles; at most {} are supported",
        state.rectangles.size(),
        dataflow_kernel_lib::MAX_MCAST_RECTANGLES);
    detail::append_sender_coords(state.sender_coords, device, senders_);
    const auto override = ack_count_override_.has_value() ? ack_count_override_ : cfg.ack_count_override;
    std::optional<SenderTransferMode> first_mode;
    bool uniform_mode = true;
    for (size_t phase = 0; phase < senders_.size(); ++phase) {
        const CoreCoord worker(state.sender_coords[2 * phase], state.sender_coords[2 * phase + 1]);
        const uint32_t fanout = fanouts_[phase];
        const uint32_t ack = override.value_or(fanout);
        TT_FATAL(ack <= fanout, "McastGroup: acknowledgment count ({}) exceeds sender fan-out ({})", ack, fanout);
        state.acks.push_back(ack);
        auto& records = state.sender_records.emplace_back();
        for (const auto& rectangle : state.rectangles) {
            const bool includes_sender = rectangle.contains(worker);
            const uint32_t remote = rectangle.size() - includes_sender;
            const auto mode = wire::classify(remote, includes_sender);
            if (!first_mode) {
                first_mode = mode;
            } else {
                uniform_mode = uniform_mode && *first_mode == mode;
            }
            const auto bounds = detail::noc_ordered_bounds(cfg.noc, rectangle);
            records.insert(
                records.end(), {bounds.sx, bounds.sy, bounds.ex, bounds.ey, remote, remote + 1u, uint32_t(mode)});
        }
    }
    state.transfer_mode = uniform_mode && first_mode ? *first_mode : SenderTransferMode::TransferModeUnknown;
}

void McastGroup::set_argument_layout_(const ArgumentLayout& layout) { layout_ = layout; }

const McastGroup::PreparedState& McastGroup::prepared_state_() const {
    TT_FATAL(prepared_.has_value(), "McastGroup: use family.group(index) to access a prepared group");
    return *prepared_;
}
const McastGroup::ArgumentLayout& McastGroup::argument_layout_() const {
    TT_FATAL(layout_.has_value(), "McastGroup: use family.group(index) to access a prepared group");
    return *layout_;
}
uint32_t McastGroup::sender_phase_(const CoreCoord& core) const {
    const auto it = std::find(senders_.begin(), senders_.end(), core);
    return it == senders_.end() ? wire::NO_SENDER_ROUND : uint32_t(it - senders_.begin());
}
CoreRangeSet McastGroup::sender_only_cores() const { return participating_.subtract(receivers_); }
bool McastGroup::is_sender(const CoreCoord& core) const { return sender_phase_(core) != wire::NO_SENDER_ROUND; }
uint32_t McastGroup::num_senders() const { return senders_.size(); }
uint32_t McastGroup::num_receivers(const CoreCoord& core) const {
    const auto phase = sender_phase_(core);
    return phase == wire::NO_SENDER_ROUND ? 0u : fanouts_[phase];
}
bool McastGroup::has_remote_receivers() const {
    return std::any_of(fanouts_.begin(), fanouts_.end(), [](uint32_t fanout) { return fanout > 0; });
}
uint32_t McastGroup::ack_count(const CoreCoord& core) const {
    const auto& state = prepared_state_();
    const auto phase = sender_phase_(core);
    return phase == wire::NO_SENDER_ROUND ? 0u : state.acks[phase];
}
uint32_t McastGroup::num_rectangles() const { return prepared_state_().rectangles.size(); }

std::vector<uint32_t> McastGroup::compile_time_args(std::optional<bool> pre_handshake) const {
    const auto& layout = argument_layout_();
    auto flags = layout.flags;
    if (pre_handshake.has_value()) {
        flags = (flags & ~wire::PRE_HANDSHAKE) | (*pre_handshake ? wire::PRE_HANDSHAKE : 0u);
    }
    TT_FATAL(
        !(flags & wire::PRE_HANDSHAKE) || layout.consumer_ready_id != UNUSED_SEM_ID,
        "McastGroup: pre_handshake requires a consumer_ready semaphore");
    return {
        wire::FAMILY,
        layout.has_remote_receivers ? 1u : 0u,
        layout.data_ready_id,
        layout.consumer_ready_id,
        layout.ack_count,
        flags,
        layout.rotating_span,
        uint32_t(layout.transfer_mode),
        layout.uniform_remote_count,
        layout.uniform_loopback_count,
        layout.rectangle_capacity};
}

std::vector<uint32_t> McastGroup::runtime_args(const CoreCoord& core) const {
    TT_FATAL(participating_.contains(core), "McastGroup: core ({},{}) is not in this group", core.x, core.y);
    const auto& layout = argument_layout_();
    const auto& state = prepared_state_();
    std::vector<uint32_t> args(wire::roles_offset(layout.rotating_span, layout.rectangle_capacity), 0u);
    const auto phase = sender_phase_(core);
    const bool sender = phase != wire::NO_SENDER_ROUND;
    std::copy(
        state.sender_coords.begin(),
        state.sender_coords.end(),
        args.begin() + wire::sender_coords_offset(layout.rotating_span));
    if (sender) {
        args[wire::NUM_RECTANGLES] = state.rectangles.size();
        args[wire::ACK] = state.acks[phase];
        const auto& records = state.sender_records[phase];
        std::copy(records.begin(), records.end(), args.begin() + wire::rectangles_offset(layout.rotating_span));
    }
    const bool receiver = receivers_.contains(core) && (!sender || rotating());
    detail::append_role_args(args, sender, receiver, phase);
    return args;
}

McastFamily::McastFamily(tt::tt_metal::IDevice* device, std::vector<McastGroup> groups, const McastConfig& cfg) :
    groups_(std::move(groups)), cfg_(cfg) {
    TT_FATAL(device != nullptr, "McastFamily: device must not be null");
    TT_FATAL(!groups_.empty(), "McastFamily: at least one group is required");
    const bool rotating = groups_.front().rotating();
    layout_.rotating_span = rotating ? groups_.front().num_senders() : 0;
    layout_.flags = detail::mcast_flags(cfg_);
    std::optional<uint32_t> first_ack, first_remote;
    std::optional<SenderTransferMode> first_mode;
    bool uniform_ack = true, uniform_remote = true, uniform_mode = true;
    for (auto& group : groups_) {
        TT_FATAL(group.rotating() == rotating, "McastFamily: groups must use the same sender mode");
        TT_FATAL(
            !rotating || group.num_senders() == layout_.rotating_span,
            "McastFamily: groups must use the same rotation length");
        TT_FATAL(!participating_.intersects(group.participating_cores()), "McastFamily: group footprints overlap");
        participating_ = participating_.merge(group.participating_cores());
        receivers_ = receivers_.merge(group.receiver_cores());
        group.prepare_(device, cfg_);
        layout_.rectangle_capacity = std::max(layout_.rectangle_capacity, group.num_rectangles());
        layout_.has_remote_receivers |= group.has_remote_receivers();
        const auto& state = group.prepared_state_();
        for (size_t phase = 0; phase < group.num_senders(); ++phase) {
            const auto ack = state.acks[phase];
            const auto fanout = group.fanouts_[phase];
            if (!first_ack) {
                first_ack = ack;
            } else {
                uniform_ack = uniform_ack && *first_ack == ack;
            }
            if (!first_remote) {
                first_remote = fanout;
            } else {
                uniform_remote = uniform_remote && *first_remote == fanout;
            }
        }
        const auto mode = state.transfer_mode;
        if (!first_mode) {
            first_mode = mode;
        } else {
            uniform_mode = uniform_mode && *first_mode == mode;
        }
    }
    layout_.ack_count = uniform_ack ? *first_ack : ACK_EQUALS_FANOUT;
    layout_.uniform_remote_count = uniform_remote ? *first_remote : 0u;
    layout_.uniform_loopback_count = uniform_remote ? *first_remote + 1u : 0u;
    // A concrete CT mode is a per-rectangle specialization, never a whole-group inference.
    layout_.transfer_mode = uniform_mode && first_mode ? *first_mode : SenderTransferMode::TransferModeUnknown;
    if (cfg_.sem_ids.has_value()) {
        const auto& ids = *cfg_.sem_ids;
        TT_FATAL(!ids.empty(), "McastFamily: adopted sem_ids must contain the data_ready id");
        TT_FATAL(
            !cfg_.handshake || (ids.size() > 1 && ids[1] != UNUSED_SEM_ID),
            "McastFamily: handshake requires an adopted consumer_ready id");
        layout_.data_ready_id = ids[0];
        layout_.consumer_ready_id = cfg_.handshake ? ids[1] : UNUSED_SEM_ID;
    } else {
        layout_.data_ready_id = cfg_.base_sem_id;
        layout_.consumer_ready_id = cfg_.handshake ? cfg_.base_sem_id + 1u : UNUSED_SEM_ID;
    }
    for (auto& group : groups_) {
        group.set_argument_layout_(layout_);
    }
}

const McastGroup& McastFamily::group(uint32_t index) const {
    TT_FATAL(index < groups_.size(), "McastFamily: group index {} is out of range", index);
    return groups_[index];
}
const McastGroup* McastFamily::group_for_core_(const CoreCoord& core) const {
    for (const auto& group : groups_) {
        if (group.participating_cores().contains(core)) {
            return &group;
        }
    }
    return nullptr;
}
std::vector<tt::tt_metal::SemaphoreDescriptor> McastFamily::owned_semaphores() const {
    if (cfg_.sem_ids) {
        return {};
    }
    std::vector<tt::tt_metal::SemaphoreDescriptor> semaphores;
    semaphores.push_back({.id = layout_.data_ready_id, .core_ranges = participating_, .initial_value = 0});
    if (cfg_.handshake) {
        semaphores.push_back({.id = layout_.consumer_ready_id, .core_ranges = participating_, .initial_value = 0});
    }
    return semaphores;
}
std::vector<uint32_t> McastFamily::compile_time_args(std::optional<bool> pre_handshake) const {
    return groups_.front().compile_time_args(pre_handshake);
}
std::vector<uint32_t> McastFamily::runtime_args(const CoreCoord& core) const {
    if (const auto* group = group_for_core_(core)) {
        return group->runtime_args(core);
    }
    std::vector<uint32_t> args(wire::roles_offset(layout_.rotating_span, layout_.rectangle_capacity), 0u);
    detail::append_role_args(args, false, false, wire::NO_SENDER_ROUND);
    return args;
}
bool McastFamily::is_sender(const CoreCoord& core) const {
    const auto* group = group_for_core_(core);
    return group && group->is_sender(core);
}
uint32_t McastFamily::num_receivers(const CoreCoord& core) const {
    const auto* group = group_for_core_(core);
    return group ? group->num_receivers(core) : 0u;
}
uint32_t McastFamily::ack_count(const CoreCoord& core) const {
    const auto* group = group_for_core_(core);
    return group ? group->ack_count(core) : 0u;
}
uint32_t McastFamily::num_senders() const { return groups_.front().num_senders(); }
bool McastFamily::has_remote_receivers() const { return layout_.has_remote_receivers; }
const CoreRangeSet& McastFamily::receiver_cores() const { return receivers_; }
const CoreRangeSet& McastFamily::participating_cores() const { return participating_; }
CoreRangeSet McastFamily::sender_only_cores() const { return participating_.subtract(receivers_); }
uint32_t McastFamily::num_semaphores() const { return cfg_.sem_ids ? 0u : cfg_.handshake ? 2u : 1u; }
uint32_t McastFamily::next_base_sem_id() const {
    TT_FATAL(!cfg_.sem_ids, "McastFamily::next_base_sem_id() requires helper-owned semaphores");
    return cfg_.base_sem_id + num_semaphores();
}
uint32_t McastFamily::rectangle_capacity() const { return layout_.rectangle_capacity; }
uint32_t McastFamily::num_rectangles(const CoreCoord& core) const {
    const auto* group = group_for_core_(core);
    return group ? group->num_rectangles() : 0u;
}

Mcast1D::Mcast1D(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& receiver_grid,
    Mcast1DShape shape,
    const Mcast1DSenderConfig& sender_config,
    const McastConfig& cfg) {
    TT_FATAL(device != nullptr, "Mcast1D: device must not be null");
    TT_FATAL(!receiver_grid.empty(), "Mcast1D: receiver grid must not be empty");
    const auto box = receiver_grid.bounding_box();
    TT_FATAL(receiver_grid.num_cores() == box.size(), "Mcast1D: receiver grid must be one dense rectangle");
    const auto* rotating = std::get_if<Mcast1DRotatingSenderConfig>(&sender_config);
    const auto& sender_grid = rotating && rotating->sender_grid ? *rotating->sender_grid : receiver_grid;
    const auto lines = sender_lines_from_grid_(receiver_grid, sender_grid, shape);
    const uint32_t span = lines.front().size();
    std::vector<McastGroup> groups;
    for (uint32_t line = 0; line < lines.size(); ++line) {
        TT_FATAL(lines[line].size() == span, "Mcast1D: sender lines must have equal length");
        auto lo = box.start_coord, hi = box.end_coord;
        if (shape == Mcast1DShape::PerRow) {
            lo.y += line;
            hi.y = lo.y;
        } else {
            lo.x += line;
            hi.x = lo.x;
        }
        CoreRangeSet receivers(CoreRange(lo, hi));
        if (rotating) {
            groups.emplace_back(receivers, lines[line]);
        } else {
            const auto& fixed = std::get<Mcast1DFixedSenderConfig>(sender_config);
            TT_FATAL(
                fixed.starting_sender_index < span,
                "Mcast1D: starting_sender_index must be less than sender-line span");
            const uint32_t phase = fixed.sender_placement == Mcast1DSenderPlacement::Diagonal
                                       ? (fixed.starting_sender_index + line) % span
                                       : fixed.starting_sender_index;
            groups.emplace_back(receivers, std::vector<CoreCoord>{lines[line][phase]});
        }
    }
    family_.emplace(device, std::move(groups), cfg);
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

Mcast2D::Mcast2D(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& receivers,
    const Mcast2DSenderConfig& sender_config,
    const McastConfig& cfg) {
    TT_FATAL(device != nullptr, "Mcast2D: device must not be null");
    TT_FATAL(!receivers.empty(), "Mcast2D: receiver set must not be empty");
    TT_FATAL(
        receivers.num_cores() == receivers.bounding_box().size(), "Mcast2D: receiver set must be one dense rectangle");
    std::vector<McastGroup> groups;
    if (const auto* rotating = std::get_if<Mcast2DRotatingSenderConfig>(&sender_config)) {
        auto senders = senders_from_grid_(rotating->sender_grid.value_or(receivers), rotating->sender_order);
        sender_in_rect_ = receivers.contains(senders.front());
        groups.emplace_back(receivers, std::move(senders));
    } else {
        const auto sender = std::get<Mcast2DFixedSenderConfig>(sender_config).sender;
        sender_in_rect_ = receivers.contains(sender);
        groups.emplace_back(receivers, std::vector<CoreCoord>{sender});
    }
    family_.emplace(device, std::move(groups), cfg);
}

std::vector<tt::tt_metal::CoreCoord> Mcast2D::senders_from_grid_(
    const tt::tt_metal::CoreRangeSet& sender_grid, Mcast2DSenderOrder sender_order) {
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
    std::sort(senders.begin(), senders.end(), [sender_order](const auto& lhs, const auto& rhs) {
        if (sender_order == Mcast2DSenderOrder::RowMajor) {
            return lhs.y == rhs.y ? lhs.x < rhs.x : lhs.y < rhs.y;
        }
        return lhs.x == rhs.x ? lhs.y < rhs.y : lhs.x < rhs.x;
    });
    TT_FATAL(
        std::adjacent_find(senders.begin(), senders.end()) == senders.end(),
        "Mcast2D: sender grid contains a duplicate core");
    return senders;
}

std::vector<tt::tt_metal::SemaphoreDescriptor> Mcast1D::owned_semaphores() const { return family_->owned_semaphores(); }
std::vector<uint32_t> Mcast1D::compile_time_args(std::optional<bool> pre_handshake) const {
    return family_->compile_time_args(pre_handshake);
}
std::vector<uint32_t> Mcast1D::runtime_args(const CoreCoord& core) const { return family_->runtime_args(core); }
bool Mcast1D::is_sender(const CoreCoord& core) const { return family_->is_sender(core); }
uint32_t Mcast1D::num_receivers(const CoreCoord& core) const { return family_->num_receivers(core); }
uint32_t Mcast1D::ack_count() const { return family_->layout_.ack_count; }
uint32_t Mcast1D::num_senders() const { return family_->num_senders(); }
bool Mcast1D::has_remote_receivers() const { return family_->has_remote_receivers(); }
uint32_t Mcast1D::num_semaphores() const { return family_->num_semaphores(); }
uint32_t Mcast1D::next_base_sem_id() const { return family_->next_base_sem_id(); }

std::vector<tt::tt_metal::SemaphoreDescriptor> Mcast2D::owned_semaphores() const { return family_->owned_semaphores(); }
std::vector<uint32_t> Mcast2D::compile_time_args(std::optional<bool> pre_handshake) const {
    return family_->compile_time_args(pre_handshake);
}
std::vector<uint32_t> Mcast2D::runtime_args(const CoreCoord& core) const { return family_->runtime_args(core); }
bool Mcast2D::is_sender(const CoreCoord& core) const { return family_->is_sender(core); }
uint32_t Mcast2D::num_receivers(const CoreCoord& core) const { return family_->num_receivers(core); }
uint32_t Mcast2D::ack_count() const { return family_->layout_.ack_count; }
uint32_t Mcast2D::num_senders() const { return family_->num_senders(); }
bool Mcast2D::has_remote_receivers() const { return family_->has_remote_receivers(); }
uint32_t Mcast2D::num_semaphores() const { return family_->num_semaphores(); }
uint32_t Mcast2D::next_base_sem_id() const { return family_->next_base_sem_id(); }

const CoreRangeSet& Mcast1D::receiver_cores() const { return family_->receiver_cores(); }
const CoreRangeSet& Mcast1D::participating_cores() const { return family_->participating_cores(); }
CoreRangeSet Mcast1D::sender_only_cores() const { return family_->sender_only_cores(); }
bool Mcast2D::sender_in_rect() const { return sender_in_rect_; }

}  // namespace ttnn::kernel_lib::host
