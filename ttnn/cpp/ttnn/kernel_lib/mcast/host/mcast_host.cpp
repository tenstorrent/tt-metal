// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"

#include <algorithm>
#include <cstddef>
#include <set>

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

namespace {
// Preserve first-appearance axis order, including gaps. The reconstruction check
// below, not a bounding box, determines whether this is a Cartesian prefix.
std::vector<uint32_t> coordinate_axis(const std::vector<uint32_t>& coordinates, uint32_t axis) {
    std::vector<uint32_t> values;
    std::set<uint32_t> seen;
    for (size_t i = axis; i < coordinates.size(); i += wire::SENDER_COORD_WORDS) {
        if (seen.insert(coordinates[i]).second) {
            values.push_back(coordinates[i]);
        }
    }
    return values;
}

void append_coordinate_ranges(std::vector<uint32_t>& ranges, const std::vector<uint32_t>& axis) {
    for (const uint32_t coordinate : axis) {
        if (!ranges.empty() && ranges.back() != UINT32_MAX && coordinate == ranges.back() + 1) {
            ranges.back() = coordinate;
        } else {
            const size_t offset = ranges.size();
            ranges.resize(offset + wire::RANGE_WORDS);
            ranges[offset + wire::RANGE_START] = coordinate;
            ranges[offset + wire::RANGE_END] = coordinate;
        }
    }
}

std::pair<wire::SenderCoordinateMetadata, std::vector<uint32_t>> compress_coordinates(
    const std::vector<uint32_t>& coordinates) {
    const auto xs = coordinate_axis(coordinates, wire::SENDER_X);
    const auto ys = coordinate_axis(coordinates, wire::SENDER_Y);
    std::vector<uint32_t> x_ranges, y_ranges;
    append_coordinate_ranges(x_ranges, xs);
    append_coordinate_ranges(y_ranges, ys);
    if (x_ranges.size() + y_ranges.size() >= coordinates.size()) {
        return {};
    }
    wire::SenderCoordinateMetadata metadata;
    metadata.columns = xs.size();
    metadata.rows = ys.size();
    metadata.x_ranges = x_ranges.size() / wire::RANGE_WORDS;
    metadata.y_ranges = y_ranges.size() / wire::RANGE_WORDS;
    x_ranges.insert(x_ranges.end(), y_ranges.begin(), y_ranges.end());
    for (const auto encoding :
         {wire::SenderCoordinateEncoding::RowMajorRanges, wire::SenderCoordinateEncoding::ColumnMajorRanges}) {
        metadata.encoding = encoding;
        bool matches = true;
        for (uint32_t word = 0; word < coordinates.size(); ++word) {
            if (wire::sender_coordinate(
                    x_ranges, metadata, word / wire::SENDER_COORD_WORDS, word % wire::SENDER_COORD_WORDS) !=
                coordinates[word]) {
                matches = false;
                break;
            }
        }
        if (matches) {
            return {metadata, std::move(x_ranges)};
        }
    }
    return {};
}

bool same_encoding(wire::SenderCoordinateMetadata left, wire::SenderCoordinateMetadata right) {
    return left.encoding == right.encoding && left.columns == right.columns && left.rows == right.rows &&
           left.x_ranges == right.x_ranges && left.y_ranges == right.y_ranges;
}
}  // namespace

std::vector<uint32_t> detail::absent_mcast_compile_time_args() {
    std::vector<uint32_t> args(wire::ABSENT_CT_WORDS);
    args[0] = wire::ABSENT;
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
    tt::tt_metal::IDevice* device,
    const McastConfig& cfg,
    dataflow_kernel_lib::TransferMode transfer_mode,
    const CoreRangeSet* handshake_cores) const {
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
        state.transport = prepare_multicast_(cfg, state, handshake_cores);
        // Sender-coordinate encoding is independent of receiver membership.
        // Exact reconstruction and size checks select ranges or explicit pairs.
        if (rotating()) {
            auto compressed = compress_coordinates(state.sender_coords);
            state.coordinate_metadata = compressed.first;
            state.sender_ranges = std::move(compressed.second);
        }
    }
}

McastFamily::Group::PreparedMulticast McastFamily::Group::prepare_multicast_(
    const McastConfig& cfg, PreparedState& state, const CoreRangeSet* handshake_cores) const {
    PreparedMulticast multicast;
    const auto override = ack_count_override_.has_value() ? ack_count_override_ : cfg.ack_count_override;
    const uint32_t handshake_count = handshake_cores ? receivers_.intersection(*handshake_cores).num_cores() : 0;
    std::optional<SenderMcastMode> first_sender_mcast_mode;
    bool uniform_sender_mcast_mode = true;
    for (size_t phase = 0; phase < senders_.size(); ++phase) {
        const uint32_t fanout = fanouts_[phase];
        const uint32_t ack =
            handshake_cores ? handshake_count - handshake_cores->contains(senders_[phase]) : override.value_or(fanout);
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
    const CoreCoord& core, const wire::ArgumentMetadata& metadata) const {
    const auto& state = prepared_state_();
    const wire::RuntimeLayout runtime(metadata);
    std::vector<uint32_t> args(runtime.words, 0u);
    const auto phase = sender_phase_(core);
    const bool sender = phase != wire::NO_SENDER_ROUND;
    const bool receiver = receivers_.contains(core) && (!sender || rotating());
    if (runtime.sender_coordinates != wire::OMITTED) {
        const auto& coordinates = metadata.coordinates.encoding == wire::SenderCoordinateEncoding::ExplicitPairs
                                      ? state.sender_coords
                                      : state.sender_ranges;
        std::copy(coordinates.begin(), coordinates.end(), args.begin() + runtime.sender_coordinates);
    }
    if (sender) {
        if (runtime.ack != wire::OMITTED) {
            args[runtime.ack] = state.acks[phase];
        }
        if (const auto* multicast = std::get_if<PreparedMulticast>(&state.transport)) {
            if (runtime.rectangle_count != wire::OMITTED) {
                args[runtime.rectangle_count] = state.rectangles.size();
            }
            const auto& rectangle_args = multicast->receiver_rectangle_args_per_sender[phase];
            for (uint32_t rectangle = 0; rectangle < state.rectangles.size(); ++rectangle) {
                const uint32_t source = rectangle * wire::RECT_WORDS;
                const uint32_t target = runtime.rectangles + rectangle * runtime.rectangle_stride;
                if (runtime.rectangle_bounds != wire::OMITTED) {
                    for (const uint32_t bound : {wire::SX, wire::SY, wire::EX, wire::EY}) {
                        args[target + runtime.rectangle_bounds + bound] = rectangle_args[source + bound];
                    }
                }
                if (runtime.rectangle_remote != wire::OMITTED) {
                    args[target + runtime.rectangle_remote] = rectangle_args[source + wire::REMOTE];
                }
                if (runtime.rectangle_mode != wire::OMITTED) {
                    args[target + runtime.rectangle_mode] = rectangle_args[source + wire::RECT_SENDER_MCAST_MODE];
                }
            }
        }
    }
    const auto* chain = std::get_if<PreparedChain>(&state.transport);
    if (chain) {
        const auto it = std::find(chain->order.begin(), chain->order.end(), core);
        TT_FATAL(
            it != chain->order.end(), "McastFamily::runtime_args: participating core is missing from chain topology");
        const auto& node = chain->nodes[it - chain->order.begin()];
        const auto base = runtime.chain_neighbors;
        args[base + wire::PREDECESSOR_X] = node.predecessor_x;
        args[base + wire::PREDECESSOR_Y] = node.predecessor_y;
        args[base + wire::SUCCESSOR_X] = node.successor_x;
        args[base + wire::SUCCESSOR_Y] = node.successor_y;
        args[base + wire::INCLUDES_SENDER] = node.includes_sender;
    }
    if (runtime.roles != wire::OMITTED) {
        args[runtime.roles] = (sender ? wire::CAN_SEND : 0u) | (receiver ? wire::CAN_RECEIVE : 0u);
    }
    if (runtime.sender_phase != wire::OMITTED) {
        args[runtime.sender_phase] = phase;
    }
    return args;
}

McastFamily::McastFamily(tt::tt_metal::IDevice* device, const McastConfig& cfg) : device_(device), cfg_(cfg) {
    TT_FATAL(device_ != nullptr, "McastFamily: device must not be null");
}

McastFamily::McastFamily(
    tt::tt_metal::IDevice* device, const McastConfig& cfg, std::optional<CoreRangeSet> handshake_cores) :
    McastFamily(device, cfg) {
    TT_FATAL(cfg.handshake || !handshake_cores, "Mcast: handshake_cores must be null when handshaking is disabled");
    handshake_cores_ = std::move(handshake_cores);
}

void McastFamily::add_group(
    CoreRangeSet receivers, std::vector<CoreCoord> senders, std::optional<uint32_t> ack_count_override) {
    TT_FATAL(!arguments_prepared_, "McastFamily::add_group: cannot add groups after successful preparation");
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
    topology_current_ = false;
}

void McastFamily::prepare_topology_() const {
    if (topology_current_) {
        return;
    }
    std::vector<CoreRange> participating_ranges, receiver_ranges;
    for (const auto& group : groups_) {
        const auto& participants = group.participating_cores().ranges();
        participating_ranges.insert(participating_ranges.end(), participants.begin(), participants.end());
        const auto& receivers = group.receiver_cores().ranges();
        receiver_ranges.insert(receiver_ranges.end(), receivers.begin(), receivers.end());
    }
    // Cache logical topology independently of device mapping. Rebuild only after an addition.
    auto participants = detail::merge_disjoint_ranges(std::move(participating_ranges));
    auto receivers = detail::merge_disjoint_ranges(std::move(receiver_ranges));
    participating_ = std::move(participants);
    receivers_ = std::move(receivers);
    topology_current_ = true;
}

void McastFamily::prepare_arguments_() const {
    if (arguments_prepared_) {
        return;
    }
    TT_FATAL(!groups_.empty(), "McastFamily::prepare_arguments: at least one group is required");
    // Discard any partial preparation from a previous failed attempt. Input geometry survives.
    prepare_topology_();
    layout_ = {};
    for (auto& group : groups_) {
        group.prepared_.reset();
    }
    const bool has_irregular_receiver_set = std::any_of(
        groups_.begin(), groups_.end(), [](const Group& group) { return group.receivers_.ranges().size() > 1; });
    // Resolve the irregular-set policy once because every group uses the same compile-time TransferMode.
    const auto transfer_mode =
        has_irregular_receiver_set && cfg_.irregular_receiver_set_mode == TransferMode::ChainUnicast
            ? TransferMode::ChainUnicast
            : TransferMode::Multicast;
    if (handshake_cores_) {
        TT_FATAL(
            handshake_cores_->subtract(receivers_).empty(), "Mcast: handshake_cores must be a subset of receivers");
        TT_FATAL(
            transfer_mode != TransferMode::ChainUnicast || receivers_.subtract(*handshake_cores_).empty(),
            "Mcast: chain forwarding requires all receivers in handshake_cores");
    }
    const bool rotating = groups_.front().rotating();
    layout_.rotating_span = rotating ? groups_.front().num_senders() : 0;
    layout_.flags = detail::mcast_flags(cfg_);
    std::optional<uint32_t> first_ack, first_remote;
    std::optional<SenderMcastMode> first_sender_mcast_mode;
    bool uniform_ack = true, uniform_remote = true, uniform_sender_mcast_mode = true;
    for (auto& group : groups_) {
        group.prepare_(device_, cfg_, transfer_mode, handshake_cores_ ? &*handshake_cores_ : nullptr);
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
    layout_.flags |= uint32_t(transfer_mode) << wire::TRANSFER_MODE_SHIFT;
    layout_.ack_count = uniform_ack ? *first_ack : ACK_EQUALS_FANOUT;
    layout_.uniform_remote_count = uniform_remote ? *first_remote : 0u;
    layout_.remote_count_known = uniform_remote;
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
    generic_metadata_ = {.family = layout_};
    if (transfer_mode == TransferMode::Multicast) {
        const auto candidate = groups_.front().prepared_state_().coordinate_metadata;
        if (std::all_of(groups_.begin(), groups_.end(), [&](const Group& group) {
                return same_encoding(candidate, group.prepared_state_().coordinate_metadata);
            })) {
            generic_metadata_.coordinates = candidate;
        }
    }
    arguments_prepared_ = true;
}

void McastFamily::require_arguments_prepared_() const {
    TT_FATAL(arguments_prepared_, "McastFamily: attach or call append_semaphores() before requesting arguments");
}
const McastFamily::Group* McastFamily::group_for_core_(const CoreCoord& core) const {
    for (const auto& group : groups_) {
        if (group.participating_cores().contains(core)) {
            return &group;
        }
    }
    return nullptr;
}
wire::ArgumentMetadata McastFamily::argument_metadata_(const CoreRangeSet* placement) const {
    require_arguments_prepared_();
    // Empty descriptor placements still compile their source, including direct
    // pipe construction. No roles can be inferred; retain the generic metadata.
    if (!placement || placement->empty() || wire::transfer_mode(layout_.flags) == TransferMode::ChainUnicast) {
        return generic_metadata_;
    }
    wire::ArgumentMetadata metadata{.family = layout_};
    metadata.kernel.capabilities = 0;
    std::optional<uint32_t> first_roles;
    bool uniform_roles = true;
    std::optional<wire::SenderCoordinateMetadata> encoding;
    bool compatible_coordinates = true;
    for (const auto& core : tt::tt_metal::corerange_to_cores(*placement)) {
        uint32_t roles = 0;
        if (const auto* group = group_for_core_(core)) {
            const bool sends = group->sender_phase_(core) != wire::NO_SENDER_ROUND;
            const bool receives = group->receivers_.contains(core) && (!sends || group->rotating());
            roles = (sends ? wire::CAN_SEND : 0u) | (receives ? wire::CAN_RECEIVE : 0u);
            // Coordinate accessors are valid on every participating core when
            // this kernel carries coordinates, including its sender-only cores.
            if (compatible_coordinates) {
                const auto candidate = group->prepared_state_().coordinate_metadata;
                compatible_coordinates = candidate.encoding != wire::SenderCoordinateEncoding::ExplicitPairs &&
                                         (!encoding || same_encoding(*encoding, candidate));
                encoding = candidate;
            }
        }
        if (!first_roles) {
            first_roles = roles;
        } else {
            uniform_roles &= *first_roles == roles;
        }
        metadata.kernel.capabilities |= roles;
    }
    metadata.kernel.roles = uniform_roles ? *first_roles : wire::DYNAMIC_ROLES;
    if (compatible_coordinates && encoding) {
        metadata.coordinates = *encoding;
    }
    return metadata;
}

std::vector<uint32_t> McastFamily::compile_time_args_(
    const std::array<uint32_t, 3>& ids, const wire::ArgumentMetadata& metadata) const {
    const wire::CompileTimeLayout layout(wire::compile_time_control(metadata));
    std::vector<uint32_t> args(layout.words);
    wire::encode_compile_time_metadata(args, metadata);
    for (const auto role : {wire::DATA_READY, wire::CONSUMER_READY, wire::SIGNAL_SOURCE}) {
        if (const auto offset = layout.semaphore(role); offset != wire::OMITTED) {
            args[offset] = ids[role];
        }
    }
    return args;
}

std::vector<uint32_t> McastFamily::runtime_args_(const CoreCoord& core, const wire::ArgumentMetadata& metadata) const {
    require_arguments_prepared_();
    if (const auto* group = group_for_core_(core)) {
        return group->runtime_args(core, metadata);
    }
    const wire::RuntimeLayout runtime(metadata);
    std::vector<uint32_t> args(runtime.words, 0u);
    if (runtime.sender_phase != wire::OMITTED) {
        args[runtime.sender_phase] = wire::NO_SENDER_ROUND;
    }
    return args;
}
const CoreRangeSet& McastFamily::participating_cores() const {
    prepare_topology_();
    return participating_;
}
CoreRangeSet McastFamily::sender_only_cores() const {
    prepare_topology_();
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
    family_->prepare_arguments_();
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
        std::sort(senders.begin(), senders.end(), [shape](const CoreCoord& lhs, const CoreCoord& rhs) {
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
    family_->prepare_arguments_();
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
    std::sort(senders.begin(), senders.end(), [sender_order](const CoreCoord& lhs, const CoreCoord& rhs) {
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
