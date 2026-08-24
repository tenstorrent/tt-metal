// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// mcast_host — the HOST counterpart of kernel_lib/mcast_pipe.hpp.
// =============================================================================
//
// A program factory that drives a NoC-multicast + semaphore-handshake channel has to emit,
// per core, the wire the device-side `SenderPipe`/`ReceiverPipe` decode: the mcast config
// (semaphore ids + whether the family is active) as compile-time args, and the per-core
// sender-rectangle / receiver-sender-coords as runtime args. Hand-rolling that on the host is
// where the bug-prone parts live — logical->virtual conversion, rect corner ordering (the
// per-NoC start/end swap), Blackhole virtualization non-monotonicity, and the degenerate
// single-line case.
//
// `McastFamily` owns all of it for one semantic stream across independent exact `McastGroup`s.
// `Mcast1D` builds one group per row or column; `Mcast2D` is the one-group convenience form.
//
// It serves TWO sender modes over the same 1D line:
//   * FIXED sender (default): one core on the line broadcasts to the rest. The fixed sender placement
//     may be uniform (the same axis index on every line) or diagonal (the index advances by one per
//     line). An interior sender targets the full line and the kernel pipe excludes the source.
//   * ROTATING sender (`config.rotating_sender`): the sender role follows an ordered sequence over
//     `span` rounds. The default sequence is every core on the receiver line. An overload accepts an
//     independent ordered sequence whose senders may extend beyond that line while the receiver
//     rectangle stays fixed.
//
// This header is HOST-ONLY (no dataflow_api.h). It shares the *wire* with mcast_pipe.hpp — the CT + RT
// layout the one McastArgs<CT_BASE, RT_BASE> decoder self-parses — so the two evolve in lockstep.
// See helper_design/mcast_pipe/plan.md.
//
//   CT, COMPACT PRESENT (tag 1, 7 words):
//                                [ 1, has_receivers, data_ready_sem_id, consumer_ready_sem_id, ack_count,
//                                  flags, rotating_span ]
//   CT, EXTENDED FAMILY (tag 2, 7 words):
//                                [ 2, has_receivers, data_ready_sem_id, consumer_ready_sem_id, flags,
//                                  rotating_span, max_rectangles ]
//   CT, ABSENT (per family, contiguous, 1 word): [ present = 0 ]
//                                flags bit0 = pre_handshake, bit1 = data-ready signal (0 Flag / 1 Counter)
//                                rotating_span = 0 fixed; sender count when rotating
//   RT, FIXED (per family, 6 words):
//                                sender   -> [ rect_x0, rect_y0, rect_x1, rect_y1 ]  (virtual, NOC-ordered)
//                                receiver -> [ sender_x, sender_y, 0, 0 ]
//                                degenerate -> the true self rectangle
//                                followed by [ role_flags, sender_phase ]
//   RT, ROTATING (per family, 6 + 2*span words):
//                                every core -> [ rect_x0, rect_y0, rect_x1, rect_y1,     (full-line rect)
//                                                s0_x, s0_y, ... s{span-1}_x, s{span-1}_y,
//                                                role_flags, sender_phase ]
//   RT, EXTENDED FAMILY:
//                                [4*max_rectangles rectangle words, 2*num_senders sender-coordinate words,
//                                 4 transport-topology words, role flags, sender phase,
//                                 rectangle count, ack count, transport]
// =============================================================================

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>         // tt::tt_metal::NOC
#include <tt-metalium/program_descriptors.hpp>  // tt::tt_metal::SemaphoreDescriptor
#include <tt_stl/assert.hpp>

namespace ttnn::kernel_lib::host {

// Which 1D topology the mcast rides. The kernel decoder (McastArgs) is shape-agnostic; the shape is a
// pure host concern (which cores send, what rect they target). A 2D single-sender->whole-grid mcast is
// out of scope for a 1D helper — express it as two families (one PerRow, one PerColumn).
enum class Mcast1DShape {
    PerRow,     // one mcast per ROW: one sender broadcasts ACROSS its row
    PerColumn,  // one mcast per COLUMN: one sender broadcasts DOWN its column
};

// Placement of the one fixed sender on each independent line. Uniform preserves the original
// Mcast1D behavior. Diagonal advances the sender's broadcast-axis index by the line index and wraps
// at the broadcast span:
//   PerRow    -> sender_x(y) = (starting_sender_index + y) % columns
//   PerColumn -> sender_y(x) = (starting_sender_index + x) % rows
enum class Mcast1DSenderPlacement {
    Uniform,
    Diagonal,
};

// Mirrors the kernel's DataReadySignal. Flag = level flag (default, fastest); Counter = reset-free.
enum class DataReadyMode : uint32_t { Flag = 0, Counter = 1 };

// "no consumer-ready semaphore" — matches mcast_pipe.hpp UNUSED_SEM_ID. Emitted as the consumer_ready
// CT word when there is no handshake, so the wire stays a fixed-size block the kernel always reads.
static constexpr uint32_t UNUSED_SEM_ID = 0xFFFFFFFFu;

struct McastConfig {
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;  // drives virtualization + rect corner order.
    // Whether the sender gates each broadcast on a receiver-ready ack (allocates the consumer_ready
    // semaphore). true = handshaked (default); false = fire-and-forget broadcast, no consumer_ready.
    bool handshake = true;
    DataReadyMode data_ready = DataReadyMode::Flag;
    // Rotating sender: the sender role follows an ordered sequence over `span` rounds. By default the
    // sequence is the receiver line/rectangle; independent-sender overloads provide it explicitly.
    // When set, the fixed sender placement is ignored and runtime_args() emits the rotating layout.
    bool rotating_sender = false;
    // Semaphore ids the helper assigns, starting here (data_ready = base, consumer_ready = base+1).
    // Two independent families on one grid pass base 0 and base 2. Ignored when `sem_ids` adopts.
    uint32_t base_sem_id = 0;
    // Escape hatch: adopt the factory's own ids [data_ready, consumer_ready] instead of creating.
    // When set, semaphores() returns {} (the factory owns creation).
    std::optional<std::vector<uint32_t>> sem_ids = std::nullopt;
};

// Materialize helper-owned descriptors in a legacy Program. Program semaphore IDs are allocated
// sequentially, so sort by the declared ID and fail immediately if surrounding factory allocations
// do not line up with the helper's base_sem_id. Descriptor factories can append the same vector
// directly to ProgramDescriptor::semaphores.
inline void create_owned_semaphores(
    tt::tt_metal::Program& program, std::vector<tt::tt_metal::SemaphoreDescriptor> descriptors) {
    std::sort(descriptors.begin(), descriptors.end(), [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; });
    for (const auto& descriptor : descriptors) {
        TT_FATAL(
            descriptor.core_type == tt::CoreType::WORKER,
            "mcast helper legacy Program bridge only supports WORKER semaphores");
        const uint32_t created_id =
            tt::tt_metal::CreateSemaphore(program, descriptor.core_ranges, descriptor.initial_value);
        TT_FATAL(
            created_id == descriptor.id,
            "mcast helper declared semaphore id {} but legacy Program allocated id {}",
            descriptor.id,
            created_id);
    }
}

// Shared coord math, used by both Mcast1D and Mcast2D.
namespace detail {

static constexpr uint32_t CAN_SEND = 1u << 0;
static constexpr uint32_t CAN_RECEIVE = 1u << 1;
static constexpr uint32_t NO_SENDER_ROUND = 0xFFFFFFFFu;

// logical -> virtual (worker) coord.
inline std::pair<uint32_t, uint32_t> virt_coord(tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord& logical) {
    const auto w = device->worker_core_from_logical_core(logical);
    return {static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
}

// The `flags` CT word (6th word of every present mcast CT block) the kernel's McastArgs decodes:
//   bit0 = pre_handshake — this face gates on the receiver->sender readiness ack.
//   bit1 = data-ready signal — 0 = Flag, 1 = Counter (== cfg.data_ready).
// Baking these onto the wire is what lets the kernel's sender()/receiver() take no behaviour knobs.
// `pre_handshake_override` lets ONE mcast object emit different pre_handshake per kernel (one semantic
// mcast whose faces pick their own handshake: a sender and a non-acking receiver ride the same family
// with opposite handshake); when unset, pre_handshake tracks cfg.handshake (the common case, all faces
// alike).
inline uint32_t mcast_flags(const McastConfig& cfg, std::optional<bool> pre_handshake_override = std::nullopt) {
    uint32_t f = 0;
    if (pre_handshake_override.value_or(cfg.handshake)) {
        f |= 0x1u;
    }
    if (cfg.data_ready == DataReadyMode::Counter) {
        f |= 0x2u;
    }
    return f;
}

template <typename Args>
inline void append_args_to(Args& destination, const std::vector<uint32_t>& args) {
    if constexpr (requires { destination.append(args); }) {
        destination.append(args);
    } else {
        destination.insert(destination.end(), args.begin(), args.end());
    }
}

inline void append_role_args(
    std::vector<uint32_t>& args, bool can_send, bool can_receive, uint32_t sender_round = NO_SENDER_ROUND) {
    args.push_back((can_send ? CAN_SEND : 0u) | (can_receive ? CAN_RECEIVE : 0u));
    args.push_back(sender_round);
}

// Bounding box over a set of virtual coords, NOC-ordered. NoC0 walks +x/+y (start = low corner);
// NoC1 walks -x/-y (start = high corner). Taking min/max over the ACTUAL virtual coords is robust
// to non-monotonic virtualization on Blackhole (where virtual-x does not track logical-x).
inline std::vector<uint32_t> noc_ordered_bbox(
    tt::tt_metal::NOC noc, const std::vector<std::pair<uint32_t, uint32_t>>& vs) {
    uint32_t xlo = vs[0].first, xhi = vs[0].first, ylo = vs[0].second, yhi = vs[0].second;
    for (const auto& v : vs) {
        xlo = std::min(xlo, v.first);
        xhi = std::max(xhi, v.first);
        ylo = std::min(ylo, v.second);
        yhi = std::max(yhi, v.second);
    }
    if (noc == tt::tt_metal::NOC::NOC_1) {
        return {xhi, yhi, xlo, ylo};
    }
    return {xlo, ylo, xhi, yhi};
}

}  // namespace detail

inline std::vector<uint32_t> absent_mcast_compile_time_args() { return {0u}; }

template <typename Args>
void append_absent_mcast_compile_time_args_to(Args& destination) {
    detail::append_args_to(destination, absent_mcast_compile_time_args());
}

// =============================================================================
// McastGroup / McastFamily — exact receiver groups for one semantic stream.
// =============================================================================

class McastGroup {
public:
    McastGroup(
        tt::tt_metal::CoreRangeSet receiver_set,
        tt::tt_metal::CoreCoord sender,
        bool use_chain_forwarding = false,
        std::optional<uint32_t> ack_count = std::nullopt) :
        receiver_set_(std::move(receiver_set)),
        senders_{sender},
        rotating_sender_(false),
        use_chain_forwarding_(use_chain_forwarding),
        ack_count_(ack_count) {}

    McastGroup(
        tt::tt_metal::CoreRangeSet receiver_set,
        std::vector<tt::tt_metal::CoreCoord> rotating_senders,
        bool use_chain_forwarding = false,
        std::optional<uint32_t> ack_count = std::nullopt) :
        receiver_set_(std::move(receiver_set)),
        senders_(std::move(rotating_senders)),
        rotating_sender_(true),
        use_chain_forwarding_(use_chain_forwarding),
        ack_count_(ack_count) {
        TT_FATAL(!senders_.empty(), "McastGroup: rotating sender schedule must not be empty");
    }

    const tt::tt_metal::CoreRangeSet& receiver_set() const { return receiver_set_; }
    const std::vector<tt::tt_metal::CoreCoord>& senders() const { return senders_; }
    bool rotating_sender() const { return rotating_sender_; }
    bool use_chain_forwarding() const { return use_chain_forwarding_; }
    std::optional<uint32_t> ack_count() const { return ack_count_; }

private:
    tt::tt_metal::CoreRangeSet receiver_set_;
    std::vector<tt::tt_metal::CoreCoord> senders_;
    bool rotating_sender_ = false;
    bool use_chain_forwarding_ = false;
    std::optional<uint32_t> ack_count_;
};

class McastFamily {
public:
    McastFamily(tt::tt_metal::IDevice* device, std::vector<McastGroup> groups, const McastConfig& cfg) :
        device_(device), groups_(std::move(groups)), cfg_(cfg) {
        TT_FATAL(device_ != nullptr, "McastFamily: device must not be null");
        TT_FATAL(!groups_.empty(), "McastFamily: at least one group is required");

        rotating_span_ = cfg_.rotating_sender ? static_cast<uint32_t>(groups_.front().senders().size()) : 0u;
        std::optional<uint32_t> common_ack_count;
        bool all_acks_derived = true;
        bool common_ack = true;

        for (uint32_t group_index = 0; group_index < groups_.size(); ++group_index) {
            const auto& group = groups_[group_index];
            TT_FATAL(
                group.rotating_sender() == cfg_.rotating_sender,
                "McastFamily: group {} sender mode does not match McastConfig::rotating_sender",
                group_index);
            TT_FATAL(
                !cfg_.rotating_sender || group.senders().size() == rotating_span_,
                "McastFamily: group {} has rotation span {}; expected {}",
                group_index,
                group.senders().size(),
                rotating_span_);

            for (uint32_t sender_index = 0; sender_index < group.senders().size(); ++sender_index) {
                TT_FATAL(
                    std::find(
                        group.senders().begin(),
                        group.senders().begin() + sender_index,
                        group.senders()[sender_index]) == group.senders().begin() + sender_index,
                    "McastFamily: group {} contains duplicate sender ({},{})",
                    group_index,
                    group.senders()[sender_index].x,
                    group.senders()[sender_index].y);
            }

            GroupState state{.group_index = group_index};
            state.rectangles = exact_rectangles_(group.receiver_set());
            state.footprint = group.receiver_set();
            for (const auto& sender : group.senders()) {
                state.footprint =
                    state.footprint.merge(tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(sender, sender)));
            }
            for (const auto& previous : states_) {
                TT_FATAL(
                    !state.footprint.intersects(previous.footprint),
                    "McastFamily: group footprints {} and {} overlap",
                    previous.group_index,
                    group_index);
            }

            state.fanouts.reserve(group.senders().size());
            for (const auto& sender : group.senders()) {
                const uint32_t fanout = static_cast<uint32_t>(group.receiver_set().num_cores()) -
                                        (group.receiver_set().contains(sender) ? 1u : 0u);
                TT_FATAL(
                    !group.ack_count().has_value() || *group.ack_count() <= fanout,
                    "McastFamily: group {} ack_count {} exceeds sender ({},{}) fan-out {}",
                    group_index,
                    group.ack_count().value_or(0),
                    sender.x,
                    sender.y,
                    fanout);
                const uint32_t ack = group.ack_count().value_or(fanout);
                state.fanouts.push_back(fanout);
                has_receivers_ = has_receivers_ || fanout > 0;
                if (!common_ack_count.has_value()) {
                    common_ack_count = ack;
                } else {
                    common_ack = common_ack && *common_ack_count == ack;
                }
            }
            all_acks_derived = all_acks_derived && !group.ack_count().has_value();
            max_rectangles_ = std::max(max_rectangles_, static_cast<uint32_t>(state.rectangles.size()));
            receiver_cores_ = receiver_cores_.merge(group.receiver_set());
            participating_cores_ = participating_cores_.merge(state.footprint);
            states_.push_back(std::move(state));
        }

        max_rectangles_ = std::max(max_rectangles_, 1u);
        compact_wire_ = max_rectangles_ == 1u &&
                        std::all_of(
                            groups_.begin(),
                            groups_.end(),
                            [](const McastGroup& group) { return !group.receiver_set().empty(); }) &&
                        (common_ack || all_acks_derived);
        compact_ack_count_ = common_ack ? common_ack_count.value_or(0u) : 0xFFFFFFFFu;

        // Stage 1 materializes exact hardware multicast. Irregular chain groups are accepted only
        // once Stage 3 supplies their device relay transport. Dense groups ignore the chain flag.
        for (uint32_t group_index = 0; group_index < groups_.size(); ++group_index) {
            TT_FATAL(
                !groups_[group_index].use_chain_forwarding() || states_[group_index].rectangles.size() <= 1,
                "McastFamily: irregular chain forwarding is not available until the chain transport is enabled");
        }

        if (cfg_.sem_ids.has_value()) {
            const auto& ids = *cfg_.sem_ids;
            TT_FATAL(!ids.empty(), "McastFamily: adopted sem_ids must contain at least the data_ready id");
            data_ready_id_ = ids[0];
            consumer_ready_id_ = cfg_.handshake ? (ids.size() > 1 ? ids[1] : UNUSED_SEM_ID) : UNUSED_SEM_ID;
            owns_sems_ = false;
        } else {
            data_ready_id_ = cfg_.base_sem_id;
            consumer_ready_id_ = cfg_.handshake ? cfg_.base_sem_id + 1u : UNUSED_SEM_ID;
            owns_sems_ = true;
        }
    }

    std::vector<tt::tt_metal::SemaphoreDescriptor> owned_semaphores() const {
        if (!owns_sems_) {
            return {};
        }
        std::vector<tt::tt_metal::SemaphoreDescriptor> out;
        out.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = data_ready_id_, .core_ranges = participating_cores_, .initial_value = 0});
        if (cfg_.handshake) {
            out.push_back(tt::tt_metal::SemaphoreDescriptor{
                .id = consumer_ready_id_, .core_ranges = participating_cores_, .initial_value = 0});
        }
        return out;
    }

    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const {
        if (compact_wire_) {
            return {
                1u,
                has_receivers_ ? 1u : 0u,
                data_ready_id_,
                consumer_ready_id_,
                compact_ack_count_,
                detail::mcast_flags(cfg_, pre_handshake),
                rotating_span_};
        }
        return {
            2u,
            has_receivers_ ? 1u : 0u,
            data_ready_id_,
            consumer_ready_id_,
            detail::mcast_flags(cfg_, pre_handshake),
            rotating_span_,
            max_rectangles_};
    }

    template <typename Args>
    void append_compile_time_args_to(Args& destination, std::optional<bool> pre_handshake = std::nullopt) const {
        detail::append_args_to(destination, compile_time_args(pre_handshake));
    }

    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const {
        const GroupState* state = state_for_(core);
        if (compact_wire_) {
            return compact_runtime_args_(core, state);
        }
        return extended_runtime_args_(core, state);
    }

    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        detail::append_args_to(destination, runtime_args(core));
    }

    bool is_sender(const tt::tt_metal::CoreCoord& core) const {
        const GroupState* state = state_for_(core);
        return state != nullptr && sender_phase_(*state, core) != detail::NO_SENDER_ROUND;
    }

    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const {
        const GroupState* state = state_for_(core);
        if (state == nullptr) {
            return 0;
        }
        const uint32_t phase = sender_phase_(*state, core);
        return phase == detail::NO_SENDER_ROUND ? 0u : state->fanouts[phase];
    }

    bool has_receivers() const { return has_receivers_; }
    uint32_t max_rectangles() const { return max_rectangles_; }
    bool uses_compact_wire() const { return compact_wire_; }
    const tt::tt_metal::CoreRangeSet& receiver_cores() const { return receiver_cores_; }
    const tt::tt_metal::CoreRangeSet& participating_cores() const { return participating_cores_; }
    tt::tt_metal::CoreRangeSet sender_only_cores() const { return participating_cores_.subtract(receiver_cores_); }

    uint32_t next_base_sem_id() const {
        TT_FATAL(
            owns_sems_, "McastFamily::next_base_sem_id() is only valid when the helper created its own semaphores");
        return cfg_.base_sem_id + (cfg_.handshake ? 2u : 1u);
    }

private:
    struct GroupState {
        uint32_t group_index = 0;
        std::vector<tt::tt_metal::CoreRange> rectangles;
        tt::tt_metal::CoreRangeSet footprint;
        std::vector<uint32_t> fanouts;
    };

    static std::vector<tt::tt_metal::CoreRange> exact_rectangles_(const tt::tt_metal::CoreRangeSet& receiver_set) {
        auto rectangles = receiver_set.merge_ranges().ranges();
        std::sort(rectangles.begin(), rectangles.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.start_coord.y != rhs.start_coord.y) {
                return lhs.start_coord.y < rhs.start_coord.y;
            }
            if (lhs.start_coord.x != rhs.start_coord.x) {
                return lhs.start_coord.x < rhs.start_coord.x;
            }
            if (lhs.end_coord.y != rhs.end_coord.y) {
                return lhs.end_coord.y < rhs.end_coord.y;
            }
            return lhs.end_coord.x < rhs.end_coord.x;
        });
        return rectangles;
    }

    const GroupState* state_for_(const tt::tt_metal::CoreCoord& core) const {
        for (const auto& state : states_) {
            if (state.footprint.contains(core)) {
                return &state;
            }
        }
        return nullptr;
    }

    uint32_t sender_phase_(const GroupState& state, const tt::tt_metal::CoreCoord& core) const {
        const auto& senders = groups_[state.group_index].senders();
        const auto it = std::find(senders.begin(), senders.end(), core);
        return it == senders.end() ? detail::NO_SENDER_ROUND
                                   : static_cast<uint32_t>(std::distance(senders.begin(), it));
    }

    std::vector<uint32_t> rect_args_(const GroupState& state) const {
        std::vector<uint32_t> args;
        args.reserve(4u * max_rectangles_);
        for (const auto& rectangle : state.rectangles) {
            std::vector<std::pair<uint32_t, uint32_t>> virtual_cores;
            virtual_cores.reserve(rectangle.size());
            for (std::size_t y = rectangle.start_coord.y; y <= rectangle.end_coord.y; ++y) {
                for (std::size_t x = rectangle.start_coord.x; x <= rectangle.end_coord.x; ++x) {
                    virtual_cores.push_back(detail::virt_coord(device_, tt::tt_metal::CoreCoord{x, y}));
                }
            }
            const auto bounds = detail::noc_ordered_bbox(cfg_.noc, virtual_cores);
            args.insert(args.end(), bounds.begin(), bounds.end());
        }
        args.resize(4u * max_rectangles_, 0u);
        return args;
    }

    void append_sender_coords_(std::vector<uint32_t>& args, const GroupState& state) const {
        for (const auto& sender : groups_[state.group_index].senders()) {
            const auto coord = detail::virt_coord(device_, sender);
            args.push_back(coord.first);
            args.push_back(coord.second);
        }
    }

    bool can_receive_(const GroupState& state, const tt::tt_metal::CoreCoord& core, uint32_t sender_phase) const {
        if (!groups_[state.group_index].receiver_set().contains(core)) {
            return false;
        }
        return sender_phase == detail::NO_SENDER_ROUND || (cfg_.rotating_sender && rotating_span_ > 1u);
    }

    uint32_t sender_ack_count_(const GroupState& state, uint32_t sender_phase) const {
        if (sender_phase == detail::NO_SENDER_ROUND) {
            return 0u;
        }
        return groups_[state.group_index].ack_count().value_or(state.fanouts[sender_phase]);
    }

    std::vector<uint32_t> compact_runtime_args_(const tt::tt_metal::CoreCoord& core, const GroupState* state) const {
        const uint32_t topology_words = cfg_.rotating_sender ? 4u + 2u * rotating_span_ : 4u;
        if (state == nullptr) {
            std::vector<uint32_t> args(topology_words, 0u);
            detail::append_role_args(args, false, false);
            return args;
        }

        const uint32_t phase = sender_phase_(*state, core);
        const bool can_send = phase != detail::NO_SENDER_ROUND;
        std::vector<uint32_t> args;
        if (cfg_.rotating_sender || can_send) {
            const bool group_has_receivers =
                std::any_of(state->fanouts.begin(), state->fanouts.end(), [](uint32_t fanout) { return fanout > 0u; });
            args = cfg_.rotating_sender && !group_has_receivers ? std::vector<uint32_t>(4u, 0u) : rect_args_(*state);
        } else {
            const auto sender = detail::virt_coord(device_, groups_[state->group_index].senders().front());
            args = {sender.first, sender.second, 0u, 0u};
        }
        if (cfg_.rotating_sender) {
            append_sender_coords_(args, *state);
        }
        detail::append_role_args(args, can_send, can_receive_(*state, core, phase), phase);
        return args;
    }

    std::vector<uint32_t> extended_runtime_args_(const tt::tt_metal::CoreCoord& core, const GroupState* state) const {
        const uint32_t topology_words = 4u * max_rectangles_ + 2u * (cfg_.rotating_sender ? rotating_span_ : 1u) + 4u;
        if (state == nullptr) {
            std::vector<uint32_t> args(topology_words, 0u);
            args.insert(args.end(), {0u, detail::NO_SENDER_ROUND, 0u, 0u, 0u});
            return args;
        }

        const uint32_t phase = sender_phase_(*state, core);
        const bool can_send = phase != detail::NO_SENDER_ROUND;
        std::vector<uint32_t> args = rect_args_(*state);
        append_sender_coords_(args, *state);
        // Reserved predecessor/successor coordinates. Stage 3 fills these for chain groups.
        args.insert(args.end(), {0u, 0u, 0u, 0u});
        args.push_back(
            (can_send ? detail::CAN_SEND : 0u) | (can_receive_(*state, core, phase) ? detail::CAN_RECEIVE : 0u));
        args.push_back(phase);
        args.push_back(static_cast<uint32_t>(state->rectangles.size()));
        args.push_back(sender_ack_count_(*state, phase));
        args.push_back(0u);  // hardware multicast transport
        return args;
    }

    tt::tt_metal::IDevice* device_;
    std::vector<McastGroup> groups_;
    McastConfig cfg_;
    std::vector<GroupState> states_;
    tt::tt_metal::CoreRangeSet receiver_cores_;
    tt::tt_metal::CoreRangeSet participating_cores_;
    uint32_t rotating_span_ = 0;
    uint32_t max_rectangles_ = 0;
    uint32_t compact_ack_count_ = 0;
    bool compact_wire_ = false;
    bool has_receivers_ = false;
    bool owns_sems_ = true;
    uint32_t data_ready_id_ = 0;
    uint32_t consumer_ready_id_ = UNUSED_SEM_ID;
};

// =============================================================================
// Mcast1D — one row- or column-family of mcasts over a rectangular grid.
// =============================================================================
class Mcast1D {
public:
    Mcast1D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& grid,
        Mcast1DShape shape,
        uint32_t starting_sender_index,
        const McastConfig& cfg,
        Mcast1DSenderPlacement sender_placement = Mcast1DSenderPlacement::Uniform) :
        device_(device),
        shape_(shape),
        starting_sender_index_(starting_sender_index),
        sender_placement_(sender_placement),
        cfg_(cfg) {
        TT_FATAL(device_ != nullptr, "Mcast1D: device must not be null");

        // Grid extent. Preserve the logical origin so every line calculation remains relative to
        // the supplied rectangle rather than assuming the device's (0,0).
        const auto bb = grid.bounding_box();
        TT_FATAL(
            grid.num_cores() == bb.size(),
            "Mcast1D: grid must be one dense rectangle (bounding box has {} cores, set has {})",
            bb.size(),
            grid.num_cores());
        origin_x_ = static_cast<uint32_t>(bb.start_coord.x);
        origin_y_ = static_cast<uint32_t>(bb.start_coord.y);
        GC_ = static_cast<uint32_t>(bb.end_coord.x - bb.start_coord.x) + 1;  // columns
        GR_ = static_cast<uint32_t>(bb.end_coord.y - bb.start_coord.y) + 1;  // rows

        // The broadcast extent along the mcast axis; >1 => the family actually multicasts.
        span_ = (shape_ == Mcast1DShape::PerRow) ? GC_ : GR_;
        receiver_span_ = span_;
        has_receivers_ = span_ > 1;

        // Diagonal is a FIXED-sender placement; combining it with rotating sender is contradictory.
        TT_FATAL(
            !cfg_.rotating_sender || sender_placement_ == Mcast1DSenderPlacement::Uniform,
            "Mcast1D: Diagonal sender placement cannot be combined with rotating_sender");

        // FIXED sender only: every selected sender must lie on its line. Interior senders are valid:
        // their sender RT carries the full-line rect and SenderPipe's EXCLUDE-source mode reaches the
        // other span-1 cores without loopback.
        if (!cfg_.rotating_sender) {
            TT_FATAL(
                starting_sender_index_ < span_,
                "Mcast1D: starting_sender_index {} must be less than the broadcast span {}",
                starting_sender_index_,
                span_);
        }

        // Semaphore ids: adopt the factory's, or assign from base (data_ready, consumer_ready).
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
        grid_ = grid;
        receiver_grid_ = grid;
        family_.emplace(device_, make_groups_(), cfg_);
    }

    // Rotating line families whose ordered sender set is independent of the receiver lines. The
    // outer vector is one entry per receiver line (row for PerRow, column for PerColumn); every
    // inner vector is the sender order for that line and must have the same non-zero size.
    Mcast1D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& receiver_grid,
        Mcast1DShape shape,
        const std::vector<std::vector<tt::tt_metal::CoreCoord>>& rotating_senders,
        const McastConfig& cfg) :
        device_(device), shape_(shape), starting_sender_index_(0), cfg_(cfg), independent_rotating_senders_(true) {
        TT_FATAL(device_ != nullptr, "Mcast1D: device must not be null");
        TT_FATAL(cfg_.rotating_sender, "Mcast1D: an independent sender set requires rotating_sender=true");

        const auto bb = receiver_grid.bounding_box();
        TT_FATAL(
            receiver_grid.num_cores() == bb.size(),
            "Mcast1D: receiver grid must be one dense rectangle (bounding box has {} cores, set has {})",
            bb.size(),
            receiver_grid.num_cores());
        origin_x_ = static_cast<uint32_t>(bb.start_coord.x);
        origin_y_ = static_cast<uint32_t>(bb.start_coord.y);
        GC_ = static_cast<uint32_t>(bb.end_coord.x - bb.start_coord.x) + 1;
        GR_ = static_cast<uint32_t>(bb.end_coord.y - bb.start_coord.y) + 1;
        receiver_span_ = (shape_ == Mcast1DShape::PerRow) ? GC_ : GR_;

        const uint32_t num_lines = (shape_ == Mcast1DShape::PerRow) ? GR_ : GC_;
        TT_FATAL(
            rotating_senders.size() == num_lines,
            "Mcast1D: expected {} rotating sender lines, got {}",
            num_lines,
            rotating_senders.size());
        TT_FATAL(
            !rotating_senders.empty() && !rotating_senders.front().empty(), "Mcast1D: sender lines must not be empty");

        span_ = rotating_senders.front().size();
        sender_lines_ = rotating_senders;
        std::vector<tt::tt_metal::CoreRange> participating_ranges = receiver_grid.ranges();
        for (uint32_t line = 0; line < num_lines; ++line) {
            TT_FATAL(
                sender_lines_[line].size() == span_,
                "Mcast1D: sender line {} has {} rounds; expected {}",
                line,
                sender_lines_[line].size(),
                span_);
            std::vector<tt::tt_metal::CoreCoord> seen;
            seen.reserve(span_);
            for (const auto& sender : sender_lines_[line]) {
                const bool on_line = (shape_ == Mcast1DShape::PerRow)
                                         ? static_cast<uint32_t>(sender.y) == origin_y_ + line
                                         : static_cast<uint32_t>(sender.x) == origin_x_ + line;
                TT_FATAL(on_line, "Mcast1D: sender ({},{}) is not on receiver line {}", sender.x, sender.y, line);
                TT_FATAL(
                    std::find(seen.begin(), seen.end(), sender) == seen.end(),
                    "Mcast1D: duplicate sender ({},{}) on line {}",
                    sender.x,
                    sender.y,
                    line);
                seen.push_back(sender);
                const bool sender_in_receiver_line = bb.contains(sender);
                if (!sender_in_receiver_line) {
                    participating_ranges.emplace_back(sender, sender);
                }
                has_receivers_ = has_receivers_ || receiver_span_ > (sender_in_receiver_line ? 1u : 0u);
            }
        }

        receiver_grid_ = receiver_grid;
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
        family_.emplace(device_, make_groups_(), cfg_);
    }

    // Grid-native form of the independent rotating-sender topology above. The receiver grid defines
    // the fixed multicast lines, while every core in sender_grid takes one sender turn on its aligned
    // receiver line. Sender order is increasing logical coordinate along the broadcast axis. This is
    // the common shape for a sharded input feeding a differently-sized output-work grid; callers that
    // need a non-geometric sender order can use the explicit sender-lines overload.
    Mcast1D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& receiver_grid,
        const tt::tt_metal::CoreRangeSet& sender_grid,
        Mcast1DShape shape,
        const McastConfig& cfg) :
        Mcast1D(device, receiver_grid, shape, sender_lines_from_grid_(receiver_grid, sender_grid, shape), cfg) {}

    // ---- args (the wire) -----------------------------------------------------

    // The semaphores THIS helper created, for the factory to add to the program. Empty when sem_ids
    // were adopted — the factory already owns those, so returning them would make it add them twice.
    // "owned" names exactly that: add what the helper created, never what the caller already has.
    std::vector<tt::tt_metal::SemaphoreDescriptor> owned_semaphores() const { return family_->owned_semaphores(); }

    // Uniform (grid-wide) config, spliced into the reader CT list. Fixed 7-word block the kernel's
    // McastArgs<CT, RT> self-parses:
    // [present, has_receivers, data_ready, consumer_ready, ack_count, flags, rotating_span].
    // `consumer_ready` is UNUSED_SEM_ID with no handshake; `ack_count` is the sender's ack wait-count
    // (or ACK_EQUALS_FANOUT for an independent sequence containing both inside and outside senders);
    // `flags` carries pre_handshake + the data-ready signal (see detail::mcast_flags). rotating_span
    // is zero for fixed mode and the sender count for rotating mode, making the CT wire the only source
    // of truth for the RT layout and receiver type.
    //
    // `pre_handshake` overrides the flags word's pre_handshake bit for THIS emission only (the sems and
    // geometry are unchanged) — one semantic mcast whose faces pick their own handshake per kernel: a
    // sender kernel and a non-acking receiver kernel splice the SAME family with opposite pre_handshake,
    // off ONE object. Omit it for the common case (all faces = cfg.handshake).
    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const {
        return family_->compile_time_args(pre_handshake);
    }

    template <typename Args>
    void append_compile_time_args_to(Args& destination, std::optional<bool> pre_handshake = std::nullopt) const {
        detail::append_args_to(destination, compile_time_args(pre_handshake));
    }

    // Sender's handshake ACK policy on the wire. The default line family uses its dense EXCLUDE
    // fan-out. Independent sequences use ACK_EQUALS_FANOUT because inside and outside senders have
    // different dense fan-outs for the same receiver line.
    uint32_t ack_count() const {
        return independent_rotating_senders_ ? 0xFFFFFFFFu : (has_receivers_ ? (span_ - 1u) : 0u);
    }

    // Per-core runtime args. The topology block is followed by [role flags, sender phase].
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const {
        return family_->runtime_args(core);
    }

    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        detail::append_args_to(destination, runtime_args(core));
    }

    // ---- queryables (not args) ----------------------------------------------

    bool is_sender(const tt::tt_metal::CoreCoord& core) const {
        // Rotating: every core on the axis takes a sender turn, so every active core "is a sender".
        if (cfg_.rotating_sender) {
            if (independent_rotating_senders_) {
                for (const auto& line : sender_lines_) {
                    if (std::find(line.begin(), line.end(), core) != line.end()) {
                        return true;
                    }
                }
                return false;
            }
            return receiver_grid_.bounding_box().contains(core);
        }
        const uint32_t sender_index = sender_index_for_(core);
        return (shape_ == Mcast1DShape::PerRow) ? (static_cast<uint32_t>(core.x) == origin_x_ + sender_index)
                                                : (static_cast<uint32_t>(core.y) == origin_y_ + sender_index);
    }

    // Number of receiver cores a broadcast lands on (0 for a non-sender or a degenerate sender).
    // Rotating: each sender round reaches the other span-1 cores, so every active core sees span-1.
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const {
        if (!has_receivers_) {
            return 0;
        }
        if (cfg_.rotating_sender) {
            if (independent_rotating_senders_) {
                if (!is_sender(core)) {
                    return 0;
                }
                return receiver_span_ - (receiver_grid_.bounding_box().contains(core) ? 1u : 0u);
            }
            return span_ - 1;
        }
        return is_sender(core) ? (span_ - 1) : 0;
    }

    bool has_receivers() const { return has_receivers_; }

    // Logical topology partitions. These let a program factory place operation work on the receiver
    // grid while still launching sender-only participants and allocating shared resources on the
    // complete topology, without reconstructing the helper's sender/receiver union itself.
    const tt::tt_metal::CoreRangeSet& receiver_cores() const { return family_->receiver_cores(); }
    const tt::tt_metal::CoreRangeSet& participating_cores() const { return family_->participating_cores(); }
    tt::tt_metal::CoreRangeSet sender_only_cores() const { return family_->sender_only_cores(); }

    // The base_sem_id the NEXT family on the same grid should use so their ids don't overlap. Mirrors
    // the CT-chaining idiom (McastArgs::next_compile_time_args_offset()). Only valid when this family
    // created its own semaphores — under adopted sem_ids there is no base to chain from and the caller
    // owns id allocation, so calling this is a usage error rather than a silently-wrong value.
    uint32_t next_base_sem_id() const {
        TT_FATAL(
            owns_sems_,
            "Mcast1D::next_base_sem_id() is only valid when the helper created its own semaphores; this "
            "instance adopted explicit sem_ids, so the caller owns semaphore-id allocation.");
        return cfg_.base_sem_id + (cfg_.handshake ? 2u : 1u);
    }

private:
    std::vector<McastGroup> make_groups_() const {
        const uint32_t num_lines = shape_ == Mcast1DShape::PerRow ? GR_ : GC_;
        std::vector<McastGroup> groups;
        groups.reserve(num_lines);
        for (uint32_t line = 0; line < num_lines; ++line) {
            const tt::tt_metal::CoreCoord first = shape_ == Mcast1DShape::PerRow
                                                      ? tt::tt_metal::CoreCoord{origin_x_, origin_y_ + line}
                                                      : tt::tt_metal::CoreCoord{origin_x_ + line, origin_y_};
            const tt::tt_metal::CoreCoord last = shape_ == Mcast1DShape::PerRow
                                                     ? tt::tt_metal::CoreCoord{origin_x_ + GC_ - 1u, origin_y_ + line}
                                                     : tt::tt_metal::CoreCoord{origin_x_ + line, origin_y_ + GR_ - 1u};
            tt::tt_metal::CoreRangeSet receiver_set(tt::tt_metal::CoreRange(first, last));
            if (cfg_.rotating_sender) {
                std::vector<tt::tt_metal::CoreCoord> senders;
                if (independent_rotating_senders_) {
                    senders = sender_lines_[line];
                } else {
                    senders.reserve(span_);
                    for (uint32_t sender_index = 0; sender_index < span_; ++sender_index) {
                        senders.push_back(line_coord_(first, sender_index));
                    }
                }
                groups.emplace_back(std::move(receiver_set), std::move(senders));
            } else {
                groups.emplace_back(std::move(receiver_set), sender_of_(first));
            }
        }
        return groups;
    }

    static std::vector<std::vector<tt::tt_metal::CoreCoord>> sender_lines_from_grid_(
        const tt::tt_metal::CoreRangeSet& receiver_grid,
        const tt::tt_metal::CoreRangeSet& sender_grid,
        Mcast1DShape shape) {
        TT_FATAL(sender_grid.num_cores() > 0, "Mcast1D: sender grid must not be empty");

        const auto receiver_bb = receiver_grid.bounding_box();
        const uint32_t num_lines = shape == Mcast1DShape::PerRow
                                       ? static_cast<uint32_t>(receiver_bb.end_coord.y - receiver_bb.start_coord.y) + 1
                                       : static_cast<uint32_t>(receiver_bb.end_coord.x - receiver_bb.start_coord.x) + 1;
        std::vector<std::vector<tt::tt_metal::CoreCoord>> sender_lines(num_lines);

        for (const auto& range : sender_grid.ranges()) {
            for (std::size_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (std::size_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    const tt::tt_metal::CoreCoord sender{x, y};
                    const bool aligned = shape == Mcast1DShape::PerRow
                                             ? y >= receiver_bb.start_coord.y && y <= receiver_bb.end_coord.y
                                             : x >= receiver_bb.start_coord.x && x <= receiver_bb.end_coord.x;
                    TT_FATAL(
                        aligned,
                        "Mcast1D: sender ({},{}) does not align with any receiver {}",
                        x,
                        y,
                        shape == Mcast1DShape::PerRow ? "row" : "column");
                    const uint32_t line = shape == Mcast1DShape::PerRow
                                              ? static_cast<uint32_t>(y - receiver_bb.start_coord.y)
                                              : static_cast<uint32_t>(x - receiver_bb.start_coord.x);
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

    // logical -> virtual (worker) coord.
    std::pair<uint32_t, uint32_t> virt_(const tt::tt_metal::CoreCoord& logical) const {
        return detail::virt_coord(device_, logical);
    }

    // The sender core a given receiver listens to (FIXED mode).
    tt::tt_metal::CoreCoord sender_of_(const tt::tt_metal::CoreCoord& core) const {
        const uint32_t sender_index = sender_index_for_(core);
        return (shape_ == Mcast1DShape::PerRow) ? tt::tt_metal::CoreCoord{origin_x_ + sender_index, core.y}
                                                : tt::tt_metal::CoreCoord{core.x, origin_y_ + sender_index};
    }

    // The independent line index: row for PerRow, column for PerColumn.
    uint32_t line_index_(const tt::tt_metal::CoreCoord& core) const {
        return (shape_ == Mcast1DShape::PerRow) ? static_cast<uint32_t>(core.y) - origin_y_
                                                : static_cast<uint32_t>(core.x) - origin_x_;
    }

    // The fixed sender's broadcast-axis index on the line containing `core`.
    uint32_t sender_index_for_(const tt::tt_metal::CoreCoord& core) const {
        if (sender_placement_ == Mcast1DSenderPlacement::Diagonal) {
            return (starting_sender_index_ + line_index_(core)) % span_;
        }
        return starting_sender_index_;
    }

    // The logical core at axis position `i` on the line `core` belongs to.
    tt::tt_metal::CoreCoord line_coord_(const tt::tt_metal::CoreCoord& core, uint32_t i) const {
        return (shape_ == Mcast1DShape::PerRow) ? tt::tt_metal::CoreCoord{origin_x_ + i, core.y}
                                                : tt::tt_metal::CoreCoord{core.x, origin_y_ + i};
    }

    // Bounding box over a set of virtual coords, NOC-ordered (see detail::noc_ordered_bbox).
    std::vector<uint32_t> noc_ordered_bbox_(const std::vector<std::pair<uint32_t, uint32_t>>& vs) const {
        return detail::noc_ordered_bbox(cfg_.noc, vs);
    }

    // Full receiver line, virtualized and NOC-ordered. SenderPipe excludes an in-line source.
    std::vector<uint32_t> line_rect_(const tt::tt_metal::CoreCoord& core) const {
        std::vector<std::pair<uint32_t, uint32_t>> coords;
        coords.reserve(receiver_span_);
        for (uint32_t i = 0; i < receiver_span_; ++i) {
            coords.push_back(virt_(line_coord_(core, i)));
        }
        return noc_ordered_bbox_(coords);
    }

    // ROTATING runtime block: the full-line dest rect (all span cores) followed by the ordered sender
    // coords, one per round. Line-uniform (identical for every core on the same line); every core
    // reads its own rect for its sender round and indexes the coord list by round when receiving.
    std::vector<uint32_t> rotating_rt_(const tt::tt_metal::CoreCoord& core) const {
        if (independent_rotating_senders_) {
            const uint32_t line = line_index_(core);
            std::vector<std::pair<uint32_t, uint32_t>> receiver_coords;
            receiver_coords.reserve(receiver_span_);
            for (uint32_t i = 0; i < receiver_span_; ++i) {
                receiver_coords.push_back(virt_(line_coord_(core, i)));
            }
            std::vector<uint32_t> out = noc_ordered_bbox_(receiver_coords);
            for (const auto& sender : sender_lines_[line]) {
                const auto coord = virt_(sender);
                out.push_back(coord.first);
                out.push_back(coord.second);
            }
            return out;
        }
        std::vector<std::pair<uint32_t, uint32_t>> coords;
        coords.reserve(span_);
        for (uint32_t i = 0; i < span_; ++i) {
            coords.push_back(virt_(line_coord_(core, i)));
        }
        std::vector<uint32_t> out = has_receivers_ ? line_rect_(core) : std::vector<uint32_t>{0, 0, 0, 0};
        for (const auto& c : coords) {
            out.push_back(c.first);
            out.push_back(c.second);
        }
        return out;
    }

    bool is_receiver_(const tt::tt_metal::CoreCoord& core) const {
        if (!receiver_grid_.bounding_box().contains(core)) {
            return false;
        }
        return !is_sender(core) || (cfg_.rotating_sender && span_ > 1u);
    }

    uint32_t sender_round_(const tt::tt_metal::CoreCoord& core) const {
        if (!cfg_.rotating_sender) {
            return is_sender(core) ? 0u : detail::NO_SENDER_ROUND;
        }
        if (independent_rotating_senders_) {
            const auto& senders = sender_lines_[line_index_(core)];
            const auto it = std::find(senders.begin(), senders.end(), core);
            return it == senders.end() ? detail::NO_SENDER_ROUND
                                       : static_cast<uint32_t>(std::distance(senders.begin(), it));
        }
        if (!is_sender(core)) {
            return detail::NO_SENDER_ROUND;
        }
        return shape_ == Mcast1DShape::PerRow ? static_cast<uint32_t>(core.x) - origin_x_
                                              : static_cast<uint32_t>(core.y) - origin_y_;
    }

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
    uint32_t span_ = 1;  // cores on the broadcast axis
    uint32_t receiver_span_ = 1;
    bool independent_rotating_senders_ = false;
    std::vector<std::vector<tt::tt_metal::CoreCoord>> sender_lines_;
    bool has_receivers_ = false;
    bool owns_sems_ = true;
    uint32_t data_ready_id_ = 0;
    uint32_t consumer_ready_id_ = UNUSED_SEM_ID;
    std::optional<McastFamily> family_;
};

// =============================================================================
// Mcast2D — convenience wrapper for one exact McastGroup.
// =============================================================================
// Dense receiver sets retain the compact one-rectangle wire. An irregular CoreRangeSet uses the same
// exact multi-rectangle family wire as an explicitly constructed McastFamily. Sender membership in
// the receiver set determines whether the active sender is excluded from that round's fan-out. The
// optional ack_count remains a divergent-handshake override; zero derives the exact active fan-out.
// =============================================================================
class Mcast2D {
public:
    Mcast2D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& mcast_rect,
        tt::tt_metal::CoreCoord sender,
        const McastConfig& cfg,
        uint32_t ack_count = 0) :
        device_(device), receiver_set_(mcast_rect), sender_(sender), cfg_(cfg) {
        TT_FATAL(device_ != nullptr, "Mcast2D: device must not be null");

        // Preserve the exact set. Ordinary Mcast2D remains a compact one-rectangle family, while an
        // irregular set naturally selects the family's extended multi-rectangle wire.
        const auto bb = mcast_rect.bounding_box();
        rx0_ = static_cast<uint32_t>(bb.start_coord.x);
        ry0_ = static_cast<uint32_t>(bb.start_coord.y);
        rx1_ = static_cast<uint32_t>(bb.end_coord.x);
        ry1_ = static_cast<uint32_t>(bb.end_coord.y);
        area_ = static_cast<uint32_t>(mcast_rect.num_cores());

        sender_in_rect_ = mcast_rect.contains(sender_);

        // Rotating rotates the sender role over the rect, so the sender must be part of that rect;
        // a separate + rotating sender is contradictory.
        TT_FATAL(
            !cfg_.rotating_sender || sender_in_rect_,
            "Mcast2D: rotating_sender rotates the sender role within the rect, so the sender must lie "
            "inside the mcast rect; got sender ({},{}) outside rect [{},{}]-[{},{}]",
            sender_.x,
            sender_.y,
            rx0_,
            ry0_,
            rx1_,
            ry1_);

        // Receiver fan-out: the sender is excluded from the receivers only when it sits in the rect.
        const uint32_t receivers = sender_in_rect_ ? (area_ - 1) : area_;
        has_receivers_ = receivers > 0;

        // ack_count = handshake ack wait-count; 0 => dense (every receiver acks = the fan-out).
        ack_count_ = (ack_count == 0) ? receivers : ack_count;
        TT_FATAL(
            ack_count_ <= receivers,
            "Mcast2D: ack_count ({}) exceeds the receiver fan-out ({})",
            ack_count_,
            receivers);

        // Participating set (sems + reader RT): the rect, or rect ∪ {sender} when the sender is separate.
        if (sender_in_rect_) {
            participating_ = mcast_rect;
        } else {
            std::vector<tt::tt_metal::CoreRange> ranges = mcast_rect.ranges();
            ranges.push_back(tt::tt_metal::CoreRange(sender_, sender_));
            participating_ = tt::tt_metal::CoreRangeSet(std::move(ranges));
        }

        // Semaphore ids: adopt the factory's, or assign from base (data_ready, consumer_ready).
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
        if (cfg_.rotating_sender) {
            std::vector<tt::tt_metal::CoreCoord> rotating_senders;
            rotating_senders.reserve(mcast_rect.num_cores());
            for (uint32_t y = ry0_; y <= ry1_; ++y) {
                for (uint32_t x = rx0_; x <= rx1_; ++x) {
                    const tt::tt_metal::CoreCoord core{x, y};
                    if (mcast_rect.contains(core)) {
                        rotating_senders.push_back(core);
                    }
                }
            }
            family_.emplace(
                device_, std::vector<McastGroup>{McastGroup(mcast_rect, std::move(rotating_senders))}, cfg_);
        } else {
            family_.emplace(
                device_,
                std::vector<McastGroup>{McastGroup(
                    mcast_rect, sender_, false, ack_count == 0u ? std::nullopt : std::optional<uint32_t>{ack_count})},
                cfg_);
        }
    }

    // Rotating senders independent of the fixed receiver rectangle. The vector order is the round
    // order carried on the existing rotating RT wire. Dense handshakes use the device-side fan-out
    // sentinel because inside senders wait for area-1 ACKs while outside senders wait for area ACKs.
    Mcast2D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& mcast_rect,
        const std::vector<tt::tt_metal::CoreCoord>& rotating_senders,
        const McastConfig& cfg) :
        device_(device),
        receiver_set_(mcast_rect),
        cfg_(cfg),
        independent_rotating_senders_(true),
        senders_(rotating_senders) {
        TT_FATAL(device_ != nullptr, "Mcast2D: device must not be null");
        TT_FATAL(cfg_.rotating_sender, "Mcast2D: an independent sender set requires rotating_sender=true");
        TT_FATAL(!senders_.empty(), "Mcast2D: rotating sender set must not be empty");

        const auto bb = mcast_rect.bounding_box();
        rx0_ = static_cast<uint32_t>(bb.start_coord.x);
        ry0_ = static_cast<uint32_t>(bb.start_coord.y);
        rx1_ = static_cast<uint32_t>(bb.end_coord.x);
        ry1_ = static_cast<uint32_t>(bb.end_coord.y);
        area_ = static_cast<uint32_t>(mcast_rect.num_cores());

        std::vector<tt::tt_metal::CoreRange> participating_ranges = mcast_rect.ranges();
        for (std::size_t i = 0; i < senders_.size(); ++i) {
            TT_FATAL(
                std::find(senders_.begin(), senders_.begin() + i, senders_[i]) == senders_.begin() + i,
                "Mcast2D: duplicate rotating sender ({},{})",
                senders_[i].x,
                senders_[i].y);
            const bool sender_in_rect = in_rect_(senders_[i]);
            if (!sender_in_rect) {
                participating_ranges.emplace_back(senders_[i], senders_[i]);
            }
            has_receivers_ = has_receivers_ || area_ > (sender_in_rect ? 1u : 0u);
        }
        sender_ = senders_.front();
        sender_in_rect_ = in_rect_(sender_);
        participating_ = tt::tt_metal::CoreRangeSet(std::move(participating_ranges));
        ack_count_ = 0xFFFFFFFFu;

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
        family_.emplace(device_, std::vector<McastGroup>{McastGroup(mcast_rect, rotating_senders)}, cfg_);
    }

    // ---- args (the wire) -----------------------------------------------------

    // The semaphores THIS helper created, placed on the participating set (rect, or rect ∪ {sender}).
    // Empty when sem_ids were adopted (the factory already owns those).
    std::vector<tt::tt_metal::SemaphoreDescriptor> owned_semaphores() const { return family_->owned_semaphores(); }

    // Uniform 7-word CT block owned by the one-group family. Dense sets use the compact tag-1 form;
    // irregular sets use tag 2 and move rectangle count plus ack count into the per-core RT block.
    // `pre_handshake` overrides the flags bit for this emission only.
    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const {
        return family_->compile_time_args(pre_handshake);
    }

    template <typename Args>
    void append_compile_time_args_to(Args& destination, std::optional<bool> pre_handshake = std::nullopt) const {
        detail::append_args_to(destination, compile_time_args(pre_handshake));
    }

    // Per-core topology followed by [role flags, sender phase].
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const {
        return family_->runtime_args(core);
    }

    template <typename Args>
    void append_runtime_args_to(Args& destination, const tt::tt_metal::CoreCoord& core) const {
        detail::append_args_to(destination, runtime_args(core));
    }

    // ---- queryables (not args) ----------------------------------------------

    bool is_sender(const tt::tt_metal::CoreCoord& core) const {
        // Rotating: every core in the rect takes a sender turn, so every active rect core "is a sender".
        if (cfg_.rotating_sender) {
            if (independent_rotating_senders_) {
                return std::find(senders_.begin(), senders_.end(), core) != senders_.end();
            }
            return in_rect_(core);
        }
        return core == sender_;
    }

    // Number of receiver cores a broadcast lands on (the geometric fan-out: area-1 when the sender is
    // in the rect, else area). Distinct from ack_count (the ack subset — noop cores still receive).
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const {
        if (!has_receivers_) {
            return 0;
        }
        const uint32_t receivers = sender_in_rect_ ? (area_ - 1) : area_;
        if (cfg_.rotating_sender) {
            if (independent_rotating_senders_) {
                return is_sender(core) ? area_ - (in_rect_(core) ? 1u : 0u) : 0u;
            }
            return receivers;
        }
        return is_sender(core) ? receivers : 0;
    }

    // The handshake ACK wait-count on the wire (== fan-out in the dense case; smaller when divergent).
    uint32_t ack_count() const { return ack_count_; }

    bool has_receivers() const { return has_receivers_; }

    // Whether the sender sits inside the rect (fully-inside mode) vs is a separate core.
    bool sender_in_rect() const { return sender_in_rect_; }

    // The base_sem_id the NEXT family on the same grid should use (mirrors Mcast1D). Only valid when
    // this instance created its own semaphores.
    uint32_t next_base_sem_id() const {
        TT_FATAL(
            owns_sems_,
            "Mcast2D::next_base_sem_id() is only valid when the helper created its own semaphores; this "
            "instance adopted explicit sem_ids, so the caller owns semaphore-id allocation.");
        return cfg_.base_sem_id + (cfg_.handshake ? 2u : 1u);
    }

private:
    bool in_rect_(const tt::tt_metal::CoreCoord& core) const { return receiver_set_.contains(core); }

    bool is_receiver_(const tt::tt_metal::CoreCoord& core) const {
        if (!in_rect_(core)) {
            return false;
        }
        const uint32_t sender_count = independent_rotating_senders_ ? senders_.size() : area_;
        return !is_sender(core) || (cfg_.rotating_sender && sender_count > 1u);
    }

    uint32_t sender_round_(const tt::tt_metal::CoreCoord& core) const {
        if (!cfg_.rotating_sender) {
            return is_sender(core) ? 0u : detail::NO_SENDER_ROUND;
        }
        if (independent_rotating_senders_) {
            const auto it = std::find(senders_.begin(), senders_.end(), core);
            return it == senders_.end() ? detail::NO_SENDER_ROUND
                                        : static_cast<uint32_t>(std::distance(senders_.begin(), it));
        }
        if (!in_rect_(core)) {
            return detail::NO_SENDER_ROUND;
        }
        return (static_cast<uint32_t>(core.y) - ry0_) * (rx1_ - rx0_ + 1u) + (static_cast<uint32_t>(core.x) - rx0_);
    }

    // Virtual coords of every core in the rect, row-major (y outer, x inner).
    std::vector<std::pair<uint32_t, uint32_t>> rect_virt_coords_() const {
        std::vector<std::pair<uint32_t, uint32_t>> vs;
        vs.reserve(area_);
        for (uint32_t y = ry0_; y <= ry1_; ++y) {
            for (uint32_t x = rx0_; x <= rx1_; ++x) {
                const tt::tt_metal::CoreCoord core{x, y};
                if (receiver_set_.contains(core)) {
                    vs.push_back(detail::virt_coord(device_, core));
                }
            }
        }
        return vs;
    }

    // The whole rect's dest corners, virtualized + NOC-ordered. Min/max over ALL rect cores (not just
    // two diagonal corners) so it stays correct under non-monotonic Blackhole virtualization.
    std::vector<uint32_t> rect_corners_() const { return detail::noc_ordered_bbox(cfg_.noc, rect_virt_coords_()); }

    // ROTATING runtime block: the full-rect destination followed by the ordered per-round sender
    // coordinates. The default order is row-major over the rect; the independent overload preserves
    // its explicit sequence. Every participant receives the same block and indexes it by round.
    std::vector<uint32_t> rotating_rt_() const {
        const auto rect_coords = rect_virt_coords_();
        // True rect corners (the 1x1 self-rect too, if area==1) — same reasoning as the fixed path.
        std::vector<uint32_t> out = detail::noc_ordered_bbox(cfg_.noc, rect_coords);
        if (independent_rotating_senders_) {
            for (const auto& sender : senders_) {
                const auto c = detail::virt_coord(device_, sender);
                out.push_back(c.first);
                out.push_back(c.second);
            }
            return out;
        }
        for (const auto& c : rect_coords) {
            out.push_back(c.first);
            out.push_back(c.second);
        }
        return out;
    }

    tt::tt_metal::IDevice* device_;
    tt::tt_metal::CoreRangeSet receiver_set_;
    tt::tt_metal::CoreRangeSet participating_;
    tt::tt_metal::CoreCoord sender_;
    McastConfig cfg_;
    uint32_t rx0_ = 0, ry0_ = 0, rx1_ = 0, ry1_ = 0;
    uint32_t area_ = 1;
    bool independent_rotating_senders_ = false;
    std::vector<tt::tt_metal::CoreCoord> senders_;
    bool sender_in_rect_ = true;
    bool has_receivers_ = false;
    bool owns_sems_ = true;
    uint32_t ack_count_ = 0;
    uint32_t data_ready_id_ = 0;
    uint32_t consumer_ready_id_ = UNUSED_SEM_ID;
    std::optional<McastFamily> family_;
};

}  // namespace ttnn::kernel_lib::host
