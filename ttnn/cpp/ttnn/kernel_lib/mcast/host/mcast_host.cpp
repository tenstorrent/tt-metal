// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"

#include <algorithm>
#include <cstddef>

#include <tt_stl/assert.hpp>

namespace ttnn::kernel_lib::host {
namespace detail {

// Families contain disjoint ranges, most often rows/columns of one dense grid.
// Avoid the general grid-based merge when their union is already a rectangle.
tt::tt_metal::CoreRangeSet merge_disjoint_ranges(std::vector<tt::tt_metal::CoreRange> ranges) {
    tt::tt_metal::CoreRangeSet result(std::move(ranges));
    if (result.ranges().size() <= 1) {
        return result;
    }
    const auto box = result.bounding_box();
    if (result.num_cores() == box.size()) {
        return tt::tt_metal::CoreRangeSet(box);
    }
    return result.merge_ranges();
}

std::pair<uint32_t, uint32_t> virt_coord(tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord& logical) {
    const auto worker = device->worker_core_from_logical_core(logical);
    return {static_cast<uint32_t>(worker.x), static_cast<uint32_t>(worker.y)};
}

uint32_t mcast_flags(const McastConfig& cfg) {
    uint32_t flags = 0;
    if (cfg.handshake) {
        flags |= dataflow_kernel_lib::mcast_wire::PRE_HANDSHAKE;
    }
    if (cfg.data_ready == dataflow_kernel_lib::DataReadySignal::Counter) {
        flags |= dataflow_kernel_lib::mcast_wire::COUNTER_SIGNAL;
    }
    if (cfg.noc == tt::tt_metal::NOC::NOC_1) {
        flags |= dataflow_kernel_lib::mcast_wire::NOC1;
    }
    return flags;
}

void write_role_args(
    std::vector<uint32_t>& args, uint32_t base, bool can_send, bool can_receive, uint32_t sender_round) {
    namespace wire = dataflow_kernel_lib::mcast_wire;
    args[base + wire::ROLES] = (can_send ? wire::CAN_SEND : 0u) | (can_receive ? wire::CAN_RECEIVE : 0u);
    args[base + wire::SENDER_ROUND] = sender_round;
}

void append_sender_coords(
    std::vector<uint32_t>& args, tt::tt_metal::IDevice* device, const std::vector<tt::tt_metal::CoreCoord>& senders) {
    namespace wire = dataflow_kernel_lib::mcast_wire;
    const auto start = args.size();
    args.resize(start + wire::SENDER_COORD_WORDS * senders.size());
    for (size_t sender_index = 0; sender_index < senders.size(); ++sender_index) {
        const auto [x, y] = virt_coord(device, senders[sender_index]);
        const auto base = start + wire::SENDER_COORD_WORDS * sender_index;
        args[base + wire::SENDER_X] = x;
        args[base + wire::SENDER_Y] = y;
    }
}

// Corners of one logical worker rectangle, already mapped to virtual NoC coordinates.
dataflow_kernel_lib::NocBounds noc_ordered_bounds(tt::tt_metal::NOC noc, const tt::tt_metal::CoreRange& rectangle) {
    const auto& lo = rectangle.start_coord;
    const auto& hi = rectangle.end_coord;
    if (noc == tt::tt_metal::NOC::NOC_1) {
        return {uint32_t(hi.x), uint32_t(hi.y), uint32_t(lo.x), uint32_t(lo.y)};
    }
    return {uint32_t(lo.x), uint32_t(lo.y), uint32_t(hi.x), uint32_t(hi.y)};
}

}  // namespace detail

using dataflow_kernel_lib::SenderMcastMode;
using dataflow_kernel_lib::TransferMode;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
namespace wire = dataflow_kernel_lib::mcast_wire;

std::vector<uint32_t> detail::absent_mcast_compile_time_args() {
    std::vector<uint32_t> args(wire::ABSENT_CT_WORDS);
    args[wire::TAG] = wire::ABSENT;
    return args;
}

McastFamily::Group::Group(
    CoreRangeSet receivers, std::vector<CoreCoord> senders, std::optional<uint32_t> ack_count_override) :
    receivers_(std::move(receivers)), senders_(std::move(senders)), ack_count_override_(ack_count_override) {
    TT_FATAL(
        !receivers_.empty(), "McastFamily::add_group: receiver set must not be empty; self-only groups are supported");
    TT_FATAL(!senders_.empty(), "McastFamily::add_group: sender schedule must not be empty");
    for (size_t i = 0; i < senders_.size(); ++i) {
        TT_FATAL(
            std::find(senders_.begin(), senders_.begin() + i, senders_[i]) == senders_.begin() + i,
            "McastFamily::add_group: duplicate sender ({},{})",
            senders_[i].x,
            senders_[i].y);
    }
    // Normalize once in logical worker space; preserve holes and reuse during mapping.
    receivers_ = receivers_.merge_ranges();
    std::vector<CoreRange> external_senders;
    const auto num_receivers = receivers_.num_cores();
    fanouts_.reserve(senders_.size());
    for (const auto& sender : senders_) {
        const bool sender_is_receiver = receivers_.contains(sender);
        if (!sender_is_receiver) {
            external_senders.emplace_back(sender, sender);
        }
        fanouts_.push_back(num_receivers - sender_is_receiver);
    }
    participating_ = external_senders.empty() ? receivers_ : receivers_.merge(external_senders);
}

void McastFamily::Group::prepare_(
    tt::tt_metal::IDevice* device, const McastConfig& cfg, dataflow_kernel_lib::TransferMode transfer_mode) {
    auto& state = prepared_.emplace(PreparedState{});
    // Map the normalized logical rectangles: non-worker NoC rows/columns are transparent to multicast.
    // Preserve holes in the logical receiver set, and count actual workers rather than NoC area.
    for (const auto& logical : receivers_.ranges()) {
        const auto start = device->worker_core_from_logical_core(logical.start_coord);
        const auto end = device->worker_core_from_logical_core(logical.end_coord);
        state.rectangles.push_back({logical, CoreRange(start, end)});
    }
    TT_FATAL(
        state.rectangles.size() <= dataflow_kernel_lib::MAX_MCAST_RECTANGLES,
        "McastFamily::prepare_arguments: requires {} logical worker rectangles; at most {} are supported",
        state.rectangles.size(),
        dataflow_kernel_lib::MAX_MCAST_RECTANGLES);
    detail::append_sender_coords(state.sender_coords, device, senders_);
    if (transfer_mode == TransferMode::ChainUnicast) {
        TT_FATAL(!rotating(), "McastFamily::prepare_arguments: chain forwarding requires one fixed sender");
        TT_FATAL(
            cfg.handshake, "McastFamily::prepare_arguments: chain forwarding requires receiver readiness handshakes");
        TT_FATAL(
            !ack_count_override_.has_value() && !cfg.ack_count_override.has_value(),
            "McastFamily::prepare_arguments: chain forwarding owns per-hop acknowledgments; overrides are not "
            "supported");
        state.transport = prepare_chain_(device, state);
    } else {
        state.transport = prepare_multicast_(cfg, state);
    }
}

McastFamily::Group::PreparedMulticast McastFamily::Group::prepare_multicast_(
    const McastConfig& cfg, PreparedState& state) const {
    PreparedMulticast multicast;
    const auto override = ack_count_override_.has_value() ? ack_count_override_ : cfg.ack_count_override;
    std::optional<SenderMcastMode> first_sender_mcast_mode;
    bool uniform_sender_mcast_mode = true;
    for (size_t phase = 0; phase < senders_.size(); ++phase) {
        const uint32_t fanout = fanouts_[phase];
        const uint32_t ack = override.value_or(fanout);
        TT_FATAL(
            ack <= fanout,
            "McastFamily::prepare_arguments: acknowledgment count ({}) exceeds sender fan-out ({})",
            ack,
            fanout);
        state.acks.push_back(ack);
        auto& rectangle_args =
            multicast.receiver_rectangle_args_per_sender.emplace_back(state.rectangles.size() * wire::RECT_WORDS, 0u);
        for (size_t rectangle_index = 0; rectangle_index < state.rectangles.size(); ++rectangle_index) {
            const auto& rectangle = state.rectangles[rectangle_index];
            const bool includes_sender = rectangle.logical.contains(senders_[phase]);
            const uint32_t remote = rectangle.logical.size() - includes_sender;
            const auto sender_mcast_mode = wire::classify(remote, includes_sender);
            if (!first_sender_mcast_mode) {
                first_sender_mcast_mode = sender_mcast_mode;
            } else {
                uniform_sender_mcast_mode = uniform_sender_mcast_mode && *first_sender_mcast_mode == sender_mcast_mode;
            }
            const auto bounds = detail::noc_ordered_bounds(cfg.noc, rectangle.noc);
            const auto base = rectangle_index * wire::RECT_WORDS;
            rectangle_args[base + wire::SX] = bounds.sx;
            rectangle_args[base + wire::SY] = bounds.sy;
            rectangle_args[base + wire::EX] = bounds.ex;
            rectangle_args[base + wire::EY] = bounds.ey;
            rectangle_args[base + wire::REMOTE] = remote;
            rectangle_args[base + wire::LOOPBACK] = remote + 1u;
            rectangle_args[base + wire::RECT_SENDER_MCAST_MODE] = uint32_t(sender_mcast_mode);
        }
    }
    multicast.sender_mcast_mode =
        uniform_sender_mcast_mode && first_sender_mcast_mode ? *first_sender_mcast_mode : SenderMcastMode::Unknown;
    return multicast;
}

McastFamily::Group::PreparedChain McastFamily::Group::prepare_chain_(
    tt::tt_metal::IDevice* device, PreparedState& state) const {
    PreparedChain chain;
    chain.order.push_back(senders_.front());
    // Logical row-major order across every receiver range, sender first.
    auto receivers = tt::tt_metal::corerange_to_cores(receivers_);
    std::sort(receivers.begin(), receivers.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y == b.y ? a.x < b.x : a.y < b.y;
    });
    for (const auto& core : receivers) {
        if (core != senders_.front()) {
            chain.order.push_back(core);
        }
    }
    const bool includes_sender = receivers_.contains(senders_.front());
    chain.nodes.resize(chain.order.size());
    for (size_t i = 0; i < chain.order.size(); ++i) {
        auto& node = chain.nodes[i];
        if (i > 0) {
            const auto [x, y] = detail::virt_coord(device, chain.order[i - 1]);
            node.predecessor_x = x;
            node.predecessor_y = y;
        }
        if (i + 1 < chain.order.size()) {
            const auto [x, y] = detail::virt_coord(device, chain.order[i + 1]);
            node.successor_x = x;
            node.successor_y = y;
        }
        node.includes_sender = includes_sender;
    }
    state.acks.push_back(chain.order.size() > 1 ? 1u : 0u);  // One successor, or a local-only chain.
    return chain;
}

const McastFamily::Group::PreparedState& McastFamily::Group::prepared_state_() const {
    TT_FATAL(prepared_.has_value(), "McastFamily: missing prepared group state");
    return *prepared_;
}
uint32_t McastFamily::Group::sender_phase_(const CoreCoord& core) const {
    const auto it = std::find(senders_.begin(), senders_.end(), core);
    return it == senders_.end() ? wire::NO_SENDER_ROUND : uint32_t(it - senders_.begin());
}
uint32_t McastFamily::Group::num_senders() const { return senders_.size(); }
bool McastFamily::Group::has_remote_receivers() const {
    return std::any_of(fanouts_.begin(), fanouts_.end(), [](uint32_t fanout) { return fanout > 0; });
}
uint32_t McastFamily::Group::num_rectangles() const { return prepared_state_().rectangles.size(); }

std::vector<uint32_t> McastFamily::Group::runtime_args(
    const CoreCoord& core, const dataflow_kernel_lib::mcast_wire::FamilyMetadata& layout) const {
    const auto& state = prepared_state_();
    std::vector<uint32_t> args(
        wire::runtime_words(layout.rotating_span, layout.rectangle_capacity, wire::transfer_mode(layout.flags)), 0u);
    const auto phase = sender_phase_(core);
    const bool sender = phase != wire::NO_SENDER_ROUND;
    std::copy(
        state.sender_coords.begin(),
        state.sender_coords.end(),
        args.begin() + wire::sender_coords_offset(layout.rotating_span));
    if (sender) {
        args[wire::ACK] = state.acks[phase];
        if (const auto* multicast = std::get_if<PreparedMulticast>(&state.transport)) {
            args[wire::NUM_RECTANGLES] = state.rectangles.size();
            const auto& rectangle_args = multicast->receiver_rectangle_args_per_sender[phase];
            std::copy(
                rectangle_args.begin(),
                rectangle_args.end(),
                args.begin() + wire::rectangles_offset(layout.rotating_span));
        }
    }
    const auto* chain = std::get_if<PreparedChain>(&state.transport);
    if (chain) {
        const auto it = std::find(chain->order.begin(), chain->order.end(), core);
        TT_FATAL(
            it != chain->order.end(), "McastFamily::runtime_args: participating core is missing from chain topology");
        const auto& node = chain->nodes[it - chain->order.begin()];
        const auto base = wire::chain_offset(layout.rotating_span, layout.rectangle_capacity);
        args[base + wire::PREDECESSOR_X] = node.predecessor_x;
        args[base + wire::PREDECESSOR_Y] = node.predecessor_y;
        args[base + wire::SUCCESSOR_X] = node.successor_x;
        args[base + wire::SUCCESSOR_Y] = node.successor_y;
        args[base + wire::INCLUDES_SENDER] = node.includes_sender;
    }
    const bool receiver = receivers_.contains(core) && (!sender || rotating());
    detail::write_role_args(
        args,
        wire::roles_offset(layout.rotating_span, layout.rectangle_capacity, wire::transfer_mode(layout.flags)),
        sender,
        receiver,
        phase);
    return args;
}

McastFamily::McastFamily(tt::tt_metal::IDevice* device, const McastConfig& cfg) : device_(device), cfg_(cfg) {
    TT_FATAL(device_ != nullptr, "McastFamily: device must not be null");
}

void McastFamily::add_group(
    CoreRangeSet receivers, std::vector<CoreCoord> senders, std::optional<uint32_t> ack_count_override) {
    TT_FATAL(!arguments_prepared_, "McastFamily::add_group: cannot add groups after prepare_arguments");
    Group candidate(std::move(receivers), std::move(senders), ack_count_override);
    if (!groups_.empty()) {
        const auto& first = groups_.front();
        TT_FATAL(
            candidate.rotating() == first.rotating(), "McastFamily::add_group: groups must use the same sender mode");
        TT_FATAL(
            candidate.num_senders() == first.num_senders(),
            "McastFamily::add_group: groups must use the same rotation length");
    }
    for (const auto& group : groups_) {
        for (const auto& range : group.participating_cores().ranges()) {
            TT_FATAL(
                !candidate.participating_cores().intersects(range), "McastFamily::add_group: group footprints overlap");
        }
    }
    groups_.push_back(std::move(candidate));
}

void McastFamily::prepare_arguments() {
    if (arguments_prepared_) {
        return;
    }
    TT_FATAL(!groups_.empty(), "McastFamily::prepare_arguments: at least one group is required");
    // Discard any partial preparation from a previous failed attempt. Input geometry survives.
    receivers_ = {};
    participating_ = {};
    layout_ = {};
    for (auto& group : groups_) {
        group.prepared_.reset();
    }
    const bool has_irregular_receiver_set = std::any_of(
        groups_.begin(), groups_.end(), [](const auto& group) { return group.receivers_.ranges().size() > 1; });
    // Resolve the irregular-set policy once because every group uses the same compile-time TransferMode.
    const auto transfer_mode =
        has_irregular_receiver_set && cfg_.irregular_receiver_set_mode == TransferMode::ChainUnicast
            ? TransferMode::ChainUnicast
            : TransferMode::Multicast;
    const bool rotating = groups_.front().rotating();
    layout_.rotating_span = rotating ? groups_.front().num_senders() : 0;
    layout_.flags = detail::mcast_flags(cfg_);
    std::optional<uint32_t> first_ack, first_remote;
    std::optional<SenderMcastMode> first_sender_mcast_mode;
    bool uniform_ack = true, uniform_remote = true, uniform_sender_mcast_mode = true;
    std::vector<CoreRange> participating_ranges, receiver_ranges;
    for (auto& group : groups_) {
        const auto& participants = group.participating_cores().ranges();
        participating_ranges.insert(participating_ranges.end(), participants.begin(), participants.end());
        const auto& receivers = group.receiver_cores().ranges();
        receiver_ranges.insert(receiver_ranges.end(), receivers.begin(), receivers.end());
        group.prepare_(device_, cfg_, transfer_mode);
        layout_.has_remote_receivers |= group.has_remote_receivers();
        const auto& state = group.prepared_state_();
        // Compare every sender turn across all groups; first_ack/first_remote persist between groups.
        // Uniform counts become shared compile-time constants; differing counts use runtime arguments.
        // The kernel uses the fanout constants only when the family has one rectangle per group.
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
        if (const auto* multicast = std::get_if<McastFamily::Group::PreparedMulticast>(&state.transport)) {
            layout_.rectangle_capacity = std::max(layout_.rectangle_capacity, group.num_rectangles());
            const auto sender_mcast_mode = multicast->sender_mcast_mode;
            if (!first_sender_mcast_mode) {
                first_sender_mcast_mode = sender_mcast_mode;
            } else {
                uniform_sender_mcast_mode = uniform_sender_mcast_mode && *first_sender_mcast_mode == sender_mcast_mode;
            }
        }
    }
    // Groups are disjoint. Coalesce their ranges once, rather than rebuilding the
    // growing union after each row/column in a multicast family.
    participating_ = detail::merge_disjoint_ranges(std::move(participating_ranges));
    receivers_ = detail::merge_disjoint_ranges(std::move(receiver_ranges));
    layout_.flags |= uint32_t(transfer_mode) << wire::TRANSFER_MODE_SHIFT;
    layout_.ack_count = uniform_ack ? *first_ack : ACK_EQUALS_FANOUT;
    layout_.uniform_remote_count = uniform_remote ? *first_remote : 0u;
    layout_.uniform_loopback_count = uniform_remote ? *first_remote + 1u : 0u;
    // A concrete CT mode is a per-rectangle specialization, never a whole-group inference.
    layout_.sender_mcast_mode =
        uniform_sender_mcast_mode && first_sender_mcast_mode ? *first_sender_mcast_mode : SenderMcastMode::Unknown;
    if (cfg_.sem_ids.has_value()) {
        const auto& ids = *cfg_.sem_ids;
        TT_FATAL(!ids.empty(), "McastFamily::prepare_arguments: adopted sem_ids must contain the data_ready id");
        TT_FATAL(
            !cfg_.handshake || (ids.size() > 1 && ids[1] != UNUSED_SEM_ID),
            "McastFamily::prepare_arguments: handshake requires an adopted consumer_ready id");
        if (transfer_mode == TransferMode::ChainUnicast) {
            TT_FATAL(
                ids.size() > 2 && ids[2] != UNUSED_SEM_ID,
                "McastFamily::prepare_arguments: chain forwarding requires an adopted signal_source id");
            TT_FATAL(
                ids[0] != UNUSED_SEM_ID && ids[0] != ids[1] && ids[2] != ids[0] && ids[2] != ids[1],
                "McastFamily::prepare_arguments: chain semaphore ids must be valid and distinct");
        }
    }
    prepared_arch_ = device_->arch();
    prepared_device_grid_ = device_->compute_with_storage_grid_size();
    arguments_prepared_ = true;
}

void McastFamily::require_arguments_prepared_() const {
    TT_FATAL(arguments_prepared_, "McastFamily: call prepare_arguments() before querying the family");
}
const McastFamily::Group* McastFamily::group_for_core_(const CoreCoord& core) const {
    for (const auto& group : groups_) {
        if (group.participating_cores().contains(core)) {
            return &group;
        }
    }
    return nullptr;
}
std::vector<uint32_t> McastFamily::compile_time_args_(const std::array<uint32_t, 3>& ids) const {
    const auto& layout = layout_;
    const auto flags = layout.flags;
    std::vector<uint32_t> args(wire::compile_time_words(wire::transfer_mode(flags)));
    args[wire::TAG] = wire::FAMILY;
    args[wire::HAS_RECEIVERS] = layout.has_remote_receivers;
    args[wire::DATA_READY] = ids[0];
    args[wire::CONSUMER_READY] = ids[1];
    args[wire::ACK_COUNT] = layout.ack_count;
    args[wire::FLAGS] = flags;
    args[wire::ROTATING_SPAN] = layout.rotating_span;
    args[wire::SENDER_MCAST_MODE] = uint32_t(layout.sender_mcast_mode);
    args[wire::REMOTE_COUNT] = layout.uniform_remote_count;
    args[wire::LOOPBACK_COUNT] = layout.uniform_loopback_count;
    args[wire::RECTANGLE_CAPACITY] = layout.rectangle_capacity;
    if (wire::transfer_mode(flags) == TransferMode::ChainUnicast) {
        args[wire::SIGNAL_SOURCE] = ids[2];
    }
    return args;
}

std::vector<uint32_t> McastFamily::runtime_args_(const CoreCoord& core) const {
    require_arguments_prepared_();
    if (const auto* group = group_for_core_(core)) {
        return group->runtime_args(core, layout_);
    }
    std::vector<uint32_t> args(
        wire::runtime_words(layout_.rotating_span, layout_.rectangle_capacity, wire::transfer_mode(layout_.flags)), 0u);
    detail::write_role_args(
        args,
        wire::roles_offset(layout_.rotating_span, layout_.rectangle_capacity, wire::transfer_mode(layout_.flags)),
        false,
        false,
        wire::NO_SENDER_ROUND);
    return args;
}
const CoreRangeSet& McastFamily::participating_cores() const {
    require_arguments_prepared_();
    return participating_;
}
CoreRangeSet McastFamily::sender_only_cores() const {
    require_arguments_prepared_();
    return participating_.subtract(receivers_);
}
uint32_t McastFamily::required_semaphores_() const {
    return wire::transfer_mode(layout_.flags) == TransferMode::ChainUnicast ? 3u : cfg_.handshake ? 2u : 1u;
}

Mcast1D::Mcast1D(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& receivers,
    Mcast1DShape shape,
    const Mcast1DSenderConfig& sender_config,
    const McastConfig& cfg) {
    TT_FATAL(device != nullptr, "Mcast1D: device must not be null");
    TT_FATAL(!receivers.empty(), "Mcast1D: receiver grid must not be empty");
    const auto box = receivers.bounding_box();
    TT_FATAL(receivers.num_cores() == box.size(), "Mcast1D: receiver grid must be one dense rectangle");
    const bool per_row = shape == Mcast1DShape::PerRow;
    const uint32_t num_lines =
        per_row ? box.end_coord.y - box.start_coord.y + 1 : box.end_coord.x - box.start_coord.x + 1;
    const uint32_t line_length =
        per_row ? box.end_coord.x - box.start_coord.x + 1 : box.end_coord.y - box.start_coord.y + 1;
    const auto* rotating = std::get_if<Mcast1DRotatingSenderConfig>(&sender_config);
    const auto* fixed = rotating ? nullptr : &std::get<Mcast1DFixedSenderConfig>(sender_config);
    std::vector<std::vector<CoreCoord>> senders_by_line;
    if (rotating) {
        const auto& sender_grid = rotating->sender_grid ? *rotating->sender_grid : receivers;
        senders_by_line = sender_lines_from_grid_(box, sender_grid, shape);
    } else {
        TT_FATAL(
            fixed->starting_sender_index < line_length,
            "Mcast1D: starting_sender_index must be less than receiver-line length");
    }

    family_.emplace(device, cfg);
    for (uint32_t line = 0; line < num_lines; ++line) {
        auto lo = box.start_coord, hi = box.end_coord;
        if (per_row) {
            lo.y += line;
            hi.y = lo.y;
        } else {
            lo.x += line;
            hi.x = lo.x;
        }
        CoreRangeSet line_receivers(CoreRange(lo, hi));
        if (rotating) {
            family_->add_group(std::move(line_receivers), std::move(senders_by_line[line]));
        } else {
            const uint32_t sender_index = fixed->sender_placement == Mcast1DSenderPlacement::Diagonal
                                              ? (fixed->starting_sender_index + line) % line_length
                                              : fixed->starting_sender_index;
            auto sender = lo;
            if (per_row) {
                sender.x += sender_index;
            } else {
                sender.y += sender_index;
            }
            family_->add_group(std::move(line_receivers), std::vector<CoreCoord>{sender});
        }
    }
    family_->prepare_arguments();
}

std::vector<std::vector<tt::tt_metal::CoreCoord>> Mcast1D::sender_lines_from_grid_(
    const tt::tt_metal::CoreRange& receiver_box, const tt::tt_metal::CoreRangeSet& sender_grid, Mcast1DShape shape) {
    TT_FATAL(sender_grid.num_cores() > 0, "Mcast1D: sender grid must not be empty");

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

    // Every group in the family must have the same number of sender rounds.
    const auto senders_per_line = sender_lines.front().size();
    for (uint32_t line = 0; line < num_lines; ++line) {
        auto& senders = sender_lines[line];
        TT_FATAL(!senders.empty(), "Mcast1D: receiver line {} has no sender cores", line);
        TT_FATAL(senders.size() == senders_per_line, "Mcast1D: sender lines must have equal length");
        std::sort(senders.begin(), senders.end(), [shape](const auto& lhs, const auto& rhs) {
            return shape == Mcast1DShape::PerRow ? lhs.x < rhs.x : lhs.y < rhs.y;
        });
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
    family_.emplace(device, cfg);
    if (const auto* rotating = std::get_if<Mcast2DRotatingSenderConfig>(&sender_config)) {
        auto senders = senders_from_grid_(rotating->sender_grid.value_or(receivers), rotating->sender_order);
        family_->add_group(receivers, std::move(senders));
    } else {
        const auto sender = std::get<Mcast2DFixedSenderConfig>(sender_config).sender;
        family_->add_group(receivers, std::vector<CoreCoord>{sender});
    }
    family_->prepare_arguments();
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
    return senders;
}

const CoreRangeSet& Mcast1D::participating_cores() const { return family_->participating_cores(); }
CoreRangeSet Mcast1D::sender_only_cores() const { return family_->sender_only_cores(); }

}  // namespace ttnn::kernel_lib::host
