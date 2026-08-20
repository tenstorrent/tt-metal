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
// `Mcast1D` owns all of it. The developer picks a SHAPE (a mcast per row, or a mcast per
// column), the helper owns the semaphores + coord math + per-core packing. Two of these — one
// PerRow, one PerColumn — express a 2D dual-multicast.
//
// It serves TWO sender modes over the same 1D line:
//   * FIXED sender (default): one core on the line broadcasts to the rest. The fixed sender placement
//     may be uniform (the same axis index on every line) or diagonal (the index advances by one per
//     line). An interior sender targets the full line and the kernel pipe excludes the source.
//   * ROTATING sender (`config.rotating_sender`): the sender role follows the sender grid over `span`
//     rounds. By default the sender grid is the receiver grid. An optional independent sender grid may
//     extend beyond the receiver line while the receiver rectangle stays fixed.
//
// This header is HOST-ONLY (no dataflow_api.h). It shares the *wire* with mcast_pipe.hpp — the CT + RT
// layout the one McastArgs<CT_BASE, RT_BASE> decoder self-parses — so the two version in
// lockstep. See helper_design/NEW_HOST_HELPER/{API_SKETCH,IMPL_PLAN}.md.
//
//   CT (per family, contiguous, 6 words):
//                                [ active, data_ready_sem_id, consumer_ready_sem_id, num_active, flags,
//                                  rotating_span ]
//                                flags bit0 = pre_handshake, bit1 = data-ready signal (0 Flag / 1 Counter)
//                                rotating_span = 0 fixed; sender count when rotating
//   RT, FIXED (per family, 4 words):
//                                sender   -> [ rect_x0, rect_y0, rect_x1, rect_y1 ]  (virtual, NOC-ordered)
//                                receiver -> [ sender_x, sender_y, 0, 0 ]
//                                degenerate (no receivers) -> [ 0, 0, 0, 0 ]
//   RT, ROTATING (per family, 4 + 2*span words):
//                                every core -> [ rect_x0, rect_y0, rect_x1, rect_y1,     (full-line rect)
//                                                s0_x, s0_y, ... s{span-1}_x, s{span-1}_y ]  (sender per round)
// =============================================================================

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
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
// at the sender-line span:
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
    // Rotating sender: the sender role walks every core on the sender line over `span` rounds. By
    // default the sender grid is the receiver grid; Mcast1D accepts an independent sender grid.
    // When set, the fixed sender placement is ignored and runtime_args() emits the rotating layout.
    bool rotating_sender = false;
    // Semaphore ids the helper assigns, starting here (data_ready = base, consumer_ready = base+1).
    // Two independent families on one grid pass base 0 and base 2. Ignored when `sem_ids` adopts.
    uint32_t base_sem_id = 0;
    // Escape hatch: adopt the factory's own ids [data_ready, consumer_ready] instead of creating.
    // When set, semaphores() returns {} (the factory owns creation).
    std::optional<std::vector<uint32_t>> sem_ids = std::nullopt;
};

// Shared coord math, used by both Mcast1D and Mcast2D.
namespace detail {

// logical -> virtual (worker) coord.
inline std::pair<uint32_t, uint32_t> virt_coord(tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord& logical) {
    const auto w = device->worker_core_from_logical_core(logical);
    return {static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
}

// The `flags` CT word (5th word of every mcast CT block) the kernel's McastArgs decodes:
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

// =============================================================================
// Mcast1D — one row- or column-family of mcasts over a rectangular grid.
// =============================================================================
class Mcast1D {
public:
    Mcast1D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& receiver_grid,
        Mcast1DShape shape,
        uint32_t starting_sender_index,
        const McastConfig& cfg,
        Mcast1DSenderPlacement sender_placement = Mcast1DSenderPlacement::Uniform,
        std::optional<tt::tt_metal::CoreRangeSet> sender_grid = std::nullopt) :
        device_(device),
        shape_(shape),
        starting_sender_index_(starting_sender_index),
        sender_placement_(sender_placement),
        cfg_(cfg) {
        TT_FATAL(device_ != nullptr, "Mcast1D: device must not be null");

        // Receiver extent. Preserve the logical origin so every line calculation remains relative to
        // the supplied rectangle rather than assuming the device's (0,0).
        const auto bb = receiver_grid.bounding_box();
        TT_FATAL(
            receiver_grid.num_cores() == bb.size(),
            "Mcast1D: receiver grid must be one dense rectangle (bounding box has {} cores, set has {})",
            bb.size(),
            receiver_grid.num_cores());
        origin_x_ = static_cast<uint32_t>(bb.start_coord.x);
        origin_y_ = static_cast<uint32_t>(bb.start_coord.y);
        GC_ = static_cast<uint32_t>(bb.end_coord.x - bb.start_coord.x) + 1;  // columns
        GR_ = static_cast<uint32_t>(bb.end_coord.y - bb.start_coord.y) + 1;  // rows

        receiver_span_ = (shape_ == Mcast1DShape::PerRow) ? GC_ : GR_;
        receiver_grid_ = receiver_grid;

        // Normalize both the legacy topology and an independent sender grid to one ordered sender
        // line per receiver line. CoreRangeSet may be sparse; geometric order along the mcast axis is
        // the round order. The default is exactly the old sender_grid == receiver_grid behavior.
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

        // Diagonal is a FIXED-sender placement; combining it with rotating sender is contradictory.
        TT_FATAL(
            !cfg_.rotating_sender || sender_placement_ == Mcast1DSenderPlacement::Uniform,
            "Mcast1D: Diagonal sender placement cannot be combined with rotating_sender");

        // FIXED sender only: starting_sender_index selects an ordered sender-grid position on each
        // line. Senders inside the receiver line are excluded from their own multicast; outside
        // senders target the complete receiver line.
        if (!cfg_.rotating_sender) {
            TT_FATAL(
                starting_sender_index_ < span_,
                "Mcast1D: starting_sender_index {} must be less than the sender-line span {}",
                starting_sender_index_,
                span_);
        }

        // The participating topology is the receiver grid plus the senders that actually run: one
        // selected core per line in fixed mode, or every sender-grid core in rotating mode. Derive the
        // dense handshake count when it is uniform; mixed inside/outside senders use the device-side
        // ACK_EQUALS_FANOUT sentinel because their fan-outs differ by one.
        std::vector<tt::tt_metal::CoreRange> participating_ranges = receiver_grid.ranges();
        bool have_fanout = false;
        bool uniform_fanout = true;
        uint32_t first_fanout = 0;
        const auto add_sender = [&](const tt::tt_metal::CoreCoord& sender) {
            const bool sender_in_receiver_grid = bb.contains(sender);
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
    }

    // ---- args (the wire) -----------------------------------------------------

    // The semaphores THIS helper created, for the factory to add to the program. Empty when sem_ids
    // were adopted — the factory already owns those, so returning them would make it add them twice.
    // "owned" names exactly that: add what the helper created, never what the caller already has.
    std::vector<tt::tt_metal::SemaphoreDescriptor> owned_semaphores() const {
        std::vector<tt::tt_metal::SemaphoreDescriptor> out;
        if (!owns_sems_) {
            return out;
        }
        // data_ready: always needed (the sender->receiver signal). initial 0. core_type defaults to
        // WORKER (SemaphoreDescriptor's default member initializer).
        out.push_back(
            tt::tt_metal::SemaphoreDescriptor{.id = data_ready_id_, .core_ranges = grid_, .initial_value = 0});
        // consumer_ready: only when a handshake is used. MUST init to 0 (a remote receiver may ack
        // before the sender core even runs — see mcast_pipe.hpp).
        if (cfg_.handshake) {
            out.push_back(
                tt::tt_metal::SemaphoreDescriptor{.id = consumer_ready_id_, .core_ranges = grid_, .initial_value = 0});
        }
        return out;
    }

    // Uniform (grid-wide) config, spliced into the reader CT list. Fixed 6-word block the kernel's
    // McastArgs<CT, RT> self-parses:
    // [active, data_ready, consumer_ready, num_active, flags, rotating_span].
    // `consumer_ready` is UNUSED_SEM_ID with no handshake; `num_active` is the sender's ack wait-count
    // (or ACK_EQUALS_FANOUT when the selected sender grid contains both inside and outside senders);
    // `flags` carries pre_handshake + the data-ready signal (see detail::mcast_flags). rotating_span
    // is zero for fixed mode and the sender count for rotating mode, making the CT wire the only source
    // of truth for the RT layout and receiver type.
    //
    // `pre_handshake` overrides the flags word's pre_handshake bit for THIS emission only (the sems and
    // geometry are unchanged) — one semantic mcast whose faces pick their own handshake per kernel: a
    // sender kernel and a non-acking receiver kernel splice the SAME family with opposite pre_handshake,
    // off ONE object. Omit it for the common case (all faces = cfg.handshake).
    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const {
        return {
            active_ ? 1u : 0u,
            data_ready_id_,
            consumer_ready_id_,
            num_active(),
            detail::mcast_flags(cfg_, pre_handshake),
            cfg_.rotating_sender ? span_ : 0u};
    }

    // Sender's handshake ACK policy on the wire. A literal count is used when every selected sender
    // has the same dense fan-out; ACK_EQUALS_FANOUT lets each sender derive its count when an
    // independent grid mixes senders inside and outside the receiver line.
    uint32_t num_active() const { return ack_count_; }

    // Per-core runtime args. FIXED: 4 words (sender rect | receiver sender-coords). ROTATING:
    // 4 + 2*span words (full-line rect, then one sender coord pair per round). See file header.
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const {
        if (cfg_.rotating_sender) {
            return rotating_rt_(core);
        }
        if (is_sender(core)) {
            return sender_rect_(core);
        }
        // Receiver: the sender it listens to, in virtual coords.
        const auto s = sender_of_(core);
        const auto v = virt_(s);
        return {v.first, v.second, 0, 0};
    }

    // ---- queryables (not args) ----------------------------------------------

    bool is_sender(const tt::tt_metal::CoreCoord& core) const {
        for (uint32_t line = 0; line < sender_lines_.size(); ++line) {
            if (cfg_.rotating_sender) {
                if (std::find(sender_lines_[line].begin(), sender_lines_[line].end(), core) !=
                    sender_lines_[line].end()) {
                    return true;
                }
            } else if (sender_lines_[line][sender_index_for_line_(line)] == core) {
                return true;
            }
        }
        return false;
    }

    // Number of receiver cores a broadcast lands on (0 for a non-sender or a degenerate sender).
    // A sender inside the receiver line reaches receiver_span-1; an outside sender reaches all of it.
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const {
        if (!active_) {
            return 0;
        }
        if (!is_sender(core)) {
            return 0;
        }
        return receiver_span_ - (receiver_grid_.bounding_box().contains(core) ? 1u : 0u);
    }

    // Rounds the sender role rotates through = cores on the axis (1 when the line is degenerate).
    // FIXED mode has a single sender. This is the count of sender-coord pairs in the rotating RT block.
    uint32_t num_senders() const { return cfg_.rotating_sender ? span_ : 1u; }

    bool active() const { return active_; }

    // Semaphores this helper created from base_sem_id: 0 (sem_ids adopted) | 1 (no handshake) | 2.
    // Answers "how many did this family consume".
    uint32_t num_semaphores() const { return owns_sems_ ? (cfg_.handshake ? 2u : 1u) : 0u; }
    // The base_sem_id the NEXT family on the same grid should use so their ids don't overlap. Mirrors
    // the CT-chaining idiom (McastArgs::next_compile_time_args_offset()). Only valid when this family
    // created its own semaphores — under adopted sem_ids there is no base to chain from and the caller
    // owns id allocation, so calling this is a usage error rather than a silently-wrong value.
    uint32_t next_base_sem_id() const {
        TT_FATAL(
            owns_sems_,
            "Mcast1D::next_base_sem_id() is only valid when the helper created its own semaphores; this "
            "instance adopted explicit sem_ids, so the caller owns semaphore-id allocation.");
        return cfg_.base_sem_id + num_semaphores();
    }

private:
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

    // The selected sender core for the line containing `core` (FIXED mode).
    tt::tt_metal::CoreCoord sender_of_(const tt::tt_metal::CoreCoord& core) const {
        const uint32_t line = line_index_(core);
        return sender_lines_[line][sender_index_for_line_(line)];
    }

    // The independent line index: row for PerRow, column for PerColumn.
    uint32_t line_index_(const tt::tt_metal::CoreCoord& core) const {
        return (shape_ == Mcast1DShape::PerRow) ? static_cast<uint32_t>(core.y) - origin_y_
                                                : static_cast<uint32_t>(core.x) - origin_x_;
    }

    // The fixed sender's ordered position on sender line `line`.
    uint32_t sender_index_for_line_(uint32_t line) const {
        if (sender_placement_ == Mcast1DSenderPlacement::Diagonal) {
            return (starting_sender_index_ + line) % span_;
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

    // FIXED sender's destination rectangle, virtualized + NOC-ordered. Preserve the compact
    // receiver-only rectangle for an edge sender inside the receiver line. An interior sender uses
    // the full receiver line and SenderPipe excludes it; an outside sender also uses the full line and
    // therefore reaches every receiver.
    std::vector<uint32_t> sender_rect_(const tt::tt_metal::CoreCoord& core) const {
        const auto sender = sender_of_(core);
        const auto receiver_bb = receiver_grid_.bounding_box();
        if (!receiver_bb.contains(sender)) {
            std::vector<std::pair<uint32_t, uint32_t>> coords;
            coords.reserve(receiver_span_);
            for (uint32_t i = 0; i < receiver_span_; ++i) {
                coords.push_back(virt_(line_coord_(core, i)));
            }
            return noc_ordered_bbox_(coords);
        }

        // A one-core receiver line whose sender is that core has no receivers. Still emit its true
        // self rectangle: SenderPipe recognizes area==1 && in_rect as degenerate. A synthetic zero
        // rectangle is a real NoC rectangle and can become a stray multicast when another line keeps
        // the family-wide `active` bit set.
        if (receiver_span_ == 1) {
            return noc_ordered_bbox_({virt_(sender)});
        }

        const uint32_t sender_index = shape_ == Mcast1DShape::PerRow ? static_cast<uint32_t>(sender.x) - origin_x_
                                                                     : static_cast<uint32_t>(sender.y) - origin_y_;
        if (sender_index == 0) {
            return noc_ordered_bbox_({virt_(line_coord_(core, 1u)), virt_(line_coord_(core, receiver_span_ - 1u))});
        }
        if (sender_index == receiver_span_ - 1u) {
            return noc_ordered_bbox_({virt_(line_coord_(core, 0u)), virt_(line_coord_(core, receiver_span_ - 2u))});
        }

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
        const uint32_t line = line_index_(core);
        std::vector<std::pair<uint32_t, uint32_t>> receiver_coords;
        receiver_coords.reserve(receiver_span_);
        for (uint32_t i = 0; i < receiver_span_; ++i) {
            receiver_coords.push_back(virt_(line_coord_(core, i)));
        }
        // No receivers anywhere in a degenerate family => zeroed rect, preserving the existing wire.
        std::vector<uint32_t> out = active_ ? noc_ordered_bbox_(receiver_coords) : std::vector<uint32_t>{0, 0, 0, 0};
        for (const auto& sender : sender_lines_[line]) {
            const auto coord = virt_(sender);
            out.push_back(coord.first);
            out.push_back(coord.second);
        }
        return out;
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
    uint32_t span_ = 1;  // ordered sender cores per line (rotating rounds)
    uint32_t receiver_span_ = 1;
    std::vector<std::vector<tt::tt_metal::CoreCoord>> sender_lines_;
    bool active_ = false;
    uint32_t ack_count_ = 0;
    bool owns_sems_ = true;
    uint32_t data_ready_id_ = 0;
    uint32_t consumer_ready_id_ = UNUSED_SEM_ID;
};

// =============================================================================
// Mcast2D — ONE mcast over a single rectangle (a single sender -> the whole rectangle).
// =============================================================================
// Where Mcast1D builds MANY per-line mcasts and DERIVES each line's receiver rect from
// shape + sender_index, Mcast2D is one mcast over one rectangle handed straight to the ctor. The
// sender is a specific core; whether it sits INSIDE the rect is read off the geometry and picks the
// whole mode — no extra flag:
//   * sender IN rect ("fully inside"): the rect includes the sender, so the wire carries the rect
//     verbatim and the kernel's SenderPipe auto-excludes the sender (in_rect_ => EXCLUDE_SRC).
//     Fan-out = area - 1. ROTATING is allowed here (every core in the rect takes a sender turn).
//   * sender SEPARATE (outside rect): every core in the rect is a receiver, fan-out = area. Fixed mode
//     accepts one outside sender. Rotating mode accepts an optional independent sender grid containing
//     any mix of inside and outside senders without widening the receiver rectangle.
//
// num_active is the sender's handshake ACK wait-count (how many receivers actually ack): the dense
// default (ctor arg 0) is the whole fan-out, a divergent caller (mcast box holds noop cores that
// receive but never ack) passes a smaller count. It rides CT as the 4th word; the receiver ignores
// it. The participating set that needs the semaphores (and reader runtime args) is the rect, or
// rect ∪ {sender} when the sender is separate; the helper owns that union in owned_semaphores().
//
//   CT (6 words): [ active, data_ready_sem_id, consumer_ready_sem_id, num_active, flags, rotating_span ]
//                 flags bit0 = pre_handshake, bit1 = data-ready signal (0 Flag / 1 Counter)
//                 rotating_span = 0 fixed; ordered sender count when rotating
//   RT, FIXED (4 words):    sender   -> [ rect_x0, rect_y0, rect_x1, rect_y1 ]  (virtual, NOC-ordered)
//                           receiver -> [ sender_x, sender_y, 0, 0 ]
//                           degenerate (single-core rect, no receivers) -> [ 0, 0, 0, 0 ]
//   RT, ROTATING (4 + 2*sender_count words):
//                           every core -> [ rect_x0, rect_y0, rect_x1, rect_y1,     (full-rect rect)
//                                           s0_x, s0_y, ... ]  (sender coords, row-major over sender grid)
//
// Kernel side: one McastArgs<CT_BASE, RT_BASE> — the same decoder as Mcast1D. The rotating rectangle's
// area rides the shared CT block; the sender/receiver take no knobs.
// =============================================================================
class Mcast2D {
public:
    Mcast2D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& mcast_rect,
        tt::tt_metal::CoreCoord sender,
        const McastConfig& cfg,
        uint32_t num_active = 0,
        std::optional<tt::tt_metal::CoreRangeSet> sender_grid = std::nullopt) :
        device_(device), sender_(sender), cfg_(cfg) {
        TT_FATAL(device_ != nullptr, "Mcast2D: device must not be null");

        // Mcast2D is one dense receiver rectangle.
        const auto bb = mcast_rect.bounding_box();
        TT_FATAL(
            mcast_rect.num_cores() == bb.size(),
            "Mcast2D: receiver set must be one dense rectangle (bounding box has {} cores, set has {})",
            bb.size(),
            mcast_rect.num_cores());
        rx0_ = static_cast<uint32_t>(bb.start_coord.x);
        ry0_ = static_cast<uint32_t>(bb.start_coord.y);
        rx1_ = static_cast<uint32_t>(bb.end_coord.x);
        ry1_ = static_cast<uint32_t>(bb.end_coord.y);
        area_ = (rx1_ - rx0_ + 1) * (ry1_ - ry0_ + 1);

        std::vector<tt::tt_metal::CoreRange> participating_ranges = mcast_rect.ranges();
        if (cfg_.rotating_sender) {
            // Preserve the original API contract: without an explicit sender grid, the scalar sender
            // must identify a core in the receiver rect even though the rotation covers the whole rect.
            TT_FATAL(
                sender_grid.has_value() || bb.contains(sender_),
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
            sender_in_rect_ = bb.contains(sender_);
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
    }

    // ---- args (the wire) -----------------------------------------------------

    // The semaphores THIS helper created, placed on the participating set (rect, or rect ∪ {sender}).
    // Empty when sem_ids were adopted (the factory already owns those).
    std::vector<tt::tt_metal::SemaphoreDescriptor> owned_semaphores() const {
        std::vector<tt::tt_metal::SemaphoreDescriptor> out;
        if (!owns_sems_) {
            return out;
        }
        out.push_back(
            tt::tt_metal::SemaphoreDescriptor{.id = data_ready_id_, .core_ranges = participating_, .initial_value = 0});
        if (cfg_.handshake) {
            out.push_back(tt::tt_metal::SemaphoreDescriptor{
                .id = consumer_ready_id_, .core_ranges = participating_, .initial_value = 0});
        }
        return out;
    }

    // Uniform (grid-wide) config, spliced into the reader CT list. 6-word block the kernel's
    // McastArgs<CT, RT> self-parses:
    // [active, data_ready, consumer_ready, num_active, flags, rotating_span].
    // num_active is the sender's ack wait-count (receiver ignores it); flags carries pre_handshake +
    // the data-ready signal (see detail::mcast_flags). `pre_handshake` overrides the flags bit for THIS
    // emission (one semantic mcast whose faces pick their own handshake per kernel — e.g. a divergent
    // ack-count where some receivers ack and some don't, off ONE object).
    std::vector<uint32_t> compile_time_args(std::optional<bool> pre_handshake = std::nullopt) const {
        return {
            active_ ? 1u : 0u,
            data_ready_id_,
            consumer_ready_id_,
            ack_count_,
            detail::mcast_flags(cfg_, pre_handshake),
            cfg_.rotating_sender ? num_senders() : 0u};
    }

    // Per-core runtime args. FIXED: 4 words (sender rect | receiver sender-coords).
    // ROTATING: 4 + 2*sender_count words (full rect, then one sender coord pair per round). See header.
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const {
        if (cfg_.rotating_sender) {
            return rotating_rt_();
        }
        if (is_sender(core)) {
            // Always the TRUE rect corners — including the fully-inside area==1 self-rect. The kernel's
            // SenderPipe reads degenerate as (area==1 && in_rect_), so it needs the box on the sender's
            // OWN core to collapse to a local copy; a synthetic {0,0,0,0} would place it off-core and
            // turn a local copy into a stray unicast.
            return rect_corners_();
        }
        // Receiver: the sender it listens to, in virtual coords.
        const auto v = detail::virt_coord(device_, sender_);
        return {v.first, v.second, 0, 0};
    }

    // ---- queryables (not args) ----------------------------------------------

    bool is_sender(const tt::tt_metal::CoreCoord& core) const {
        if (cfg_.rotating_sender) {
            return active_ && std::find(senders_.begin(), senders_.end(), core) != senders_.end();
        }
        return core == sender_;
    }

    // Number of receiver cores a broadcast lands on (the geometric fan-out: area-1 when the sender is
    // in the rect, else area). Distinct from num_active (the ack subset — noop cores still receive).
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const {
        if (!active_) {
            return 0;
        }
        if (cfg_.rotating_sender) {
            return is_sender(core) ? area_ - (in_rect_(core) ? 1u : 0u) : 0u;
        }
        return is_sender(core) ? area_ - (sender_in_rect_ ? 1u : 0u) : 0u;
    }

    // The handshake ACK wait-count on the wire (== fan-out in the dense case; smaller when divergent).
    uint32_t num_active() const { return ack_count_; }

    // Rounds the sender role rotates through = cores in the effective sender grid (1 in fixed mode).
    uint32_t num_senders() const { return cfg_.rotating_sender ? senders_.size() : 1u; }

    bool active() const { return active_; }

    // Whether the sender sits inside the rect (fully-inside mode) vs is a separate core.
    bool sender_in_rect() const { return sender_in_rect_; }

    // Semaphores this helper created: 0 (sem_ids adopted) | 1 (no handshake) | 2.
    uint32_t num_semaphores() const { return owns_sems_ ? (cfg_.handshake ? 2u : 1u) : 0u; }
    // The base_sem_id the NEXT family on the same grid should use (mirrors Mcast1D). Only valid when
    // this instance created its own semaphores.
    uint32_t next_base_sem_id() const {
        TT_FATAL(
            owns_sems_,
            "Mcast2D::next_base_sem_id() is only valid when the helper created its own semaphores; this "
            "instance adopted explicit sem_ids, so the caller owns semaphore-id allocation.");
        return cfg_.base_sem_id + num_semaphores();
    }

private:
    static std::vector<tt::tt_metal::CoreCoord> senders_from_grid_(const tt::tt_metal::CoreRangeSet& sender_grid) {
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

    bool in_rect_(const tt::tt_metal::CoreCoord& core) const {
        const auto x = static_cast<uint32_t>(core.x);
        const auto y = static_cast<uint32_t>(core.y);
        return x >= rx0_ && x <= rx1_ && y >= ry0_ && y <= ry1_;
    }

    // Virtual coords of every core in the rect, row-major (y outer, x inner).
    std::vector<std::pair<uint32_t, uint32_t>> rect_virt_coords_() const {
        std::vector<std::pair<uint32_t, uint32_t>> vs;
        vs.reserve(area_);
        for (uint32_t y = ry0_; y <= ry1_; ++y) {
            for (uint32_t x = rx0_; x <= rx1_; ++x) {
                vs.push_back(detail::virt_coord(device_, tt::tt_metal::CoreCoord{x, y}));
            }
        }
        return vs;
    }

    // The whole rect's dest corners, virtualized + NOC-ordered. Min/max over ALL rect cores (not just
    // two diagonal corners) so it stays correct under non-monotonic Blackhole virtualization.
    std::vector<uint32_t> rect_corners_() const { return detail::noc_ordered_bbox(cfg_.noc, rect_virt_coords_()); }

    // ROTATING runtime block: the full receiver rectangle followed by the effective sender grid in
    // row-major order. Every participant receives the same block and indexes it by round.
    std::vector<uint32_t> rotating_rt_() const {
        const auto rect_coords = rect_virt_coords_();
        // True rect corners (the 1x1 self-rect too, if area==1) — same reasoning as the fixed path.
        std::vector<uint32_t> out = detail::noc_ordered_bbox(cfg_.noc, rect_coords);
        for (const auto& sender : senders_) {
            const auto c = detail::virt_coord(device_, sender);
            out.push_back(c.first);
            out.push_back(c.second);
        }
        return out;
    }

    tt::tt_metal::IDevice* device_;
    tt::tt_metal::CoreRangeSet participating_;
    tt::tt_metal::CoreCoord sender_;
    McastConfig cfg_;
    uint32_t rx0_ = 0, ry0_ = 0, rx1_ = 0, ry1_ = 0;
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
