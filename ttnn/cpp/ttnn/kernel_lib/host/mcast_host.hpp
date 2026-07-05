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
//   * FIXED sender (default): one core on the line broadcasts to the rest. The sender sits on an
//     edge of the axis so the receivers form ONE contiguous rect.
//   * ROTATING sender (`config.rotating_sender`): the sender role walks the whole line — over `span`
//     rounds every core takes a turn broadcasting to the other `span-1`. Each core therefore acts
//     as BOTH faces of the channel, so its runtime args carry its own dest rect AND the ordered
//     coords of every sender (one per round), which the receiver indexes by round.
//
// This header is HOST-ONLY (no dataflow_api.h). It shares the *wire* with mcast_pipe.hpp — the CT + RT
// layout the one McastArgs<CT_BASE, RT_BASE[, SPAN]> decoder self-parses — so the two version in
// lockstep. See helper_design/NEW_HOST_HELPER/{API_SKETCH,IMPL_PLAN}.md.
//
//   CT (per family, contiguous, 5 words):
//                                [ active, data_ready_sem_id, consumer_ready_sem_id, num_active, flags ]
//                                flags bit0 = pre_handshake, bit1 = data-ready signal (0 Flag / 1 Counter)
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
    PerRow,     // one mcast per ROW: sender in a fixed COLUMN, broadcasts ACROSS its row
    PerColumn,  // one mcast per COLUMN: sender in a fixed ROW, broadcasts DOWN its column
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
    // Rotating sender: the sender role walks the line over `span` rounds; every core is a sender once
    // and a receiver span-1 times. When set, `sender_index` is ignored (there is no single sender) and
    // runtime_args() emits the rotating layout.
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
        const tt::tt_metal::CoreRangeSet& grid,
        Mcast1DShape shape,
        uint32_t sender_index,
        const McastConfig& cfg) :
        device_(device), shape_(shape), sender_index_(sender_index), cfg_(cfg) {
        TT_FATAL(device_ != nullptr, "Mcast1D: device must not be null");

        // Grid extent. The grid must be a single 0-anchored rectangle.
        const auto bb = grid.bounding_box();
        TT_FATAL(
            bb.start_coord.x == 0 && bb.start_coord.y == 0,
            "Mcast1D: grid must be anchored at (0,0) (got start ({},{}))",
            bb.start_coord.x,
            bb.start_coord.y);
        GC_ = static_cast<uint32_t>(bb.end_coord.x) + 1;  // columns
        GR_ = static_cast<uint32_t>(bb.end_coord.y) + 1;  // rows

        // The broadcast extent along the mcast axis; >1 => the family actually multicasts.
        span_ = (shape_ == Mcast1DShape::PerRow) ? GC_ : GR_;
        active_ = span_ > 1;

        // FIXED sender only: the sender must sit on an EDGE of the broadcast axis, so the receivers
        // form ONE contiguous rect. A middle sender splits the row/column into two rects (multi-rect,
        // deferred). ROTATING has no single sender — every index sends on its own round — so the
        // constraint does not apply and sender_index is ignored.
        TT_FATAL(
            cfg_.rotating_sender || !active_ || sender_index_ == 0 || sender_index_ == span_ - 1,
            "Mcast1D: sender_index {} must be on an edge of the {}-wide broadcast axis (0 or {}); "
            "a middle sender needs multi-rect, which is deferred",
            sender_index_,
            span_,
            span_ - 1);

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

    // Uniform (grid-wide) config, spliced into the reader CT list. Fixed 5-word block the kernel's
    // McastArgs<CT, RT[, SPAN]> self-parses: [active, data_ready, consumer_ready, num_active, flags].
    // `consumer_ready` is UNUSED_SEM_ID with no handshake; `num_active` is the sender's ack wait-count
    // (the dense EXCLUDE fan-out span-1); `flags` carries pre_handshake + the data-ready signal (see
    // detail::mcast_flags). Shared by both sender modes — rotating carries no extra CT (the round count
    // is num_senders(), a queryable the caller splices where its kernel expects it).
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
            detail::mcast_flags(cfg_, pre_handshake)};
    }

    // Sender's handshake ACK wait-count on the wire. Mcast1D is always dense, so this is the EXCLUDE
    // fan-out (the other span-1 cores on the line); 0 for a degenerate/inactive line.
    uint32_t num_active() const { return active_ ? (span_ - 1u) : 0u; }

    // Per-core runtime args. FIXED: 4 words (sender rect | receiver sender-coords). ROTATING:
    // 4 + 2*span words (full-line rect, then one sender coord pair per round). See file header.
    std::vector<uint32_t> runtime_args(const tt::tt_metal::CoreCoord& core) const {
        if (cfg_.rotating_sender) {
            return rotating_rt_(core);
        }
        if (is_sender(core)) {
            if (!active_) {
                return {0, 0, 0, 0};  // degenerate: the only core on the broadcast axis, no receivers.
            }
            return sender_rect_(core);
        }
        // Receiver: the sender it listens to, in virtual coords.
        const auto s = sender_of_(core);
        const auto v = virt_(s);
        return {v.first, v.second, 0, 0};
    }

    // ---- queryables (not args) ----------------------------------------------

    bool is_sender(const tt::tt_metal::CoreCoord& core) const {
        // Rotating: every core on the axis takes a sender turn, so every active core "is a sender".
        if (cfg_.rotating_sender) {
            return active_;
        }
        return (shape_ == Mcast1DShape::PerRow) ? (static_cast<uint32_t>(core.x) == sender_index_)
                                                : (static_cast<uint32_t>(core.y) == sender_index_);
    }

    // Number of receiver cores a broadcast lands on (0 for a non-sender or a degenerate sender).
    // Rotating: each sender round reaches the other span-1 cores, so every active core sees span-1.
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const {
        if (!active_) {
            return 0;
        }
        if (cfg_.rotating_sender) {
            return span_ - 1;
        }
        return is_sender(core) ? (span_ - 1) : 0;
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
    // logical -> virtual (worker) coord.
    std::pair<uint32_t, uint32_t> virt_(const tt::tt_metal::CoreCoord& logical) const {
        return detail::virt_coord(device_, logical);
    }

    // The sender core a given receiver listens to (FIXED mode).
    tt::tt_metal::CoreCoord sender_of_(const tt::tt_metal::CoreCoord& core) const {
        return (shape_ == Mcast1DShape::PerRow) ? tt::tt_metal::CoreCoord{sender_index_, core.y}
                                                : tt::tt_metal::CoreCoord{core.x, sender_index_};
    }

    // The logical core at axis position `i` on the line `core` belongs to.
    tt::tt_metal::CoreCoord line_coord_(const tt::tt_metal::CoreCoord& core, uint32_t i) const {
        return (shape_ == Mcast1DShape::PerRow) ? tt::tt_metal::CoreCoord{i, core.y}
                                                : tt::tt_metal::CoreCoord{core.x, i};
    }

    // Bounding box over a set of virtual coords, NOC-ordered (see detail::noc_ordered_bbox).
    std::vector<uint32_t> noc_ordered_bbox_(const std::vector<std::pair<uint32_t, uint32_t>>& vs) const {
        return detail::noc_ordered_bbox(cfg_.noc, vs);
    }

    // FIXED sender's dest rectangle (receivers only), virtualized + NOC-ordered.
    std::vector<uint32_t> sender_rect_(const tt::tt_metal::CoreCoord& core) const {
        // Receiver range along the broadcast axis = [lo, hi], excluding the edge sender.
        const uint32_t lo = (sender_index_ == 0) ? 1u : 0u;
        const uint32_t hi = (sender_index_ == 0) ? (span_ - 1) : (span_ - 2);
        return noc_ordered_bbox_({virt_(line_coord_(core, lo)), virt_(line_coord_(core, hi))});
    }

    // ROTATING runtime block: the full-line dest rect (all span cores) followed by the ordered sender
    // coords, one per round. Line-uniform (identical for every core on the same line); every core
    // reads its own rect for its sender round and indexes the coord list by round when receiving.
    std::vector<uint32_t> rotating_rt_(const tt::tt_metal::CoreCoord& core) const {
        std::vector<std::pair<uint32_t, uint32_t>> coords;
        coords.reserve(span_);
        for (uint32_t i = 0; i < span_; ++i) {
            coords.push_back(virt_(line_coord_(core, i)));
        }
        // rect: the full line (all span cores). No receivers on a degenerate line => zeroed rect.
        std::vector<uint32_t> out = active_ ? noc_ordered_bbox_(coords) : std::vector<uint32_t>{0, 0, 0, 0};
        for (const auto& c : coords) {
            out.push_back(c.first);
            out.push_back(c.second);
        }
        return out;
    }

    tt::tt_metal::IDevice* device_;
    tt::tt_metal::CoreRangeSet grid_;
    Mcast1DShape shape_;
    uint32_t sender_index_;
    McastConfig cfg_;
    uint32_t GR_ = 1;
    uint32_t GC_ = 1;
    uint32_t span_ = 1;  // cores on the broadcast axis
    bool active_ = false;
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
//   * sender SEPARATE (outside rect): every core in the rect is a receiver, fan-out = area.
//     FIXED only — rotation would need the sender to be a member of the set it broadcasts to.
//
// num_active is the sender's handshake ACK wait-count (how many receivers actually ack): the dense
// default (ctor arg 0) is the whole fan-out, a divergent caller (mcast box holds noop cores that
// receive but never ack) passes a smaller count. It rides CT as the 4th word; the receiver ignores
// it. The participating set that needs the semaphores (and reader runtime args) is the rect, or
// rect ∪ {sender} when the sender is separate; the helper owns that union in owned_semaphores().
//
//   CT (5 words): [ active, data_ready_sem_id, consumer_ready_sem_id, num_active, flags ]
//                 flags bit0 = pre_handshake, bit1 = data-ready signal (0 Flag / 1 Counter)
//   RT, FIXED (4 words):    sender   -> [ rect_x0, rect_y0, rect_x1, rect_y1 ]  (virtual, NOC-ordered)
//                           receiver -> [ sender_x, sender_y, 0, 0 ]
//                           degenerate (single-core rect, no receivers) -> [ 0, 0, 0, 0 ]
//   RT, ROTATING (4 + 2*area words):
//                           every core -> [ rect_x0, rect_y0, rect_x1, rect_y1,     (full-rect rect)
//                                           s0_x, s0_y, ... ]  (sender coords, row-major over the rect)
//
// Kernel side: one McastArgs<CT_BASE, RT_BASE[, SPAN]> — the same decoder as Mcast1D (SPAN = area for
// the rotating rect). num_active + flags ride the shared CT block; the sender/receiver take no knobs.
// =============================================================================
class Mcast2D {
public:
    Mcast2D(
        tt::tt_metal::IDevice* device,
        const tt::tt_metal::CoreRangeSet& mcast_rect,
        tt::tt_metal::CoreCoord sender,
        const McastConfig& cfg,
        uint32_t num_active = 0) :
        device_(device), sender_(sender), cfg_(cfg) {
        TT_FATAL(device_ != nullptr, "Mcast2D: device must not be null");

        // Mcast2D is one rectangle: take the bounding box of the passed set as THE rect.
        const auto bb = mcast_rect.bounding_box();
        rx0_ = static_cast<uint32_t>(bb.start_coord.x);
        ry0_ = static_cast<uint32_t>(bb.start_coord.y);
        rx1_ = static_cast<uint32_t>(bb.end_coord.x);
        ry1_ = static_cast<uint32_t>(bb.end_coord.y);
        area_ = (rx1_ - rx0_ + 1) * (ry1_ - ry0_ + 1);

        // One containment test picks the whole mode. bb.contains is exact for a single rectangle.
        sender_in_rect_ = bb.contains(sender_);

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
        active_ = receivers > 0;

        // num_active = handshake ack wait-count; 0 => dense (every receiver acks = the fan-out).
        ack_count_ = (num_active == 0) ? receivers : num_active;
        TT_FATAL(
            ack_count_ <= receivers,
            "Mcast2D: num_active ({}) exceeds the receiver fan-out ({})",
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

    // Uniform (grid-wide) config, spliced into the reader CT list. 5-word block the kernel's
    // McastArgs<CT, RT[, SPAN]> self-parses: [active, data_ready, consumer_ready, num_active, flags].
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
            detail::mcast_flags(cfg_, pre_handshake)};
    }

    // Per-core runtime args. FIXED: 4 words (sender rect | receiver sender-coords).
    // ROTATING: 4 + 2*area words (full-rect rect, then one sender coord pair per round). See header.
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
        // Rotating: every core in the rect takes a sender turn, so every active rect core "is a sender".
        if (cfg_.rotating_sender) {
            return active_ && in_rect_(core);
        }
        return core == sender_;
    }

    // Number of receiver cores a broadcast lands on (the geometric fan-out: area-1 when the sender is
    // in the rect, else area). Distinct from num_active (the ack subset — noop cores still receive).
    uint32_t num_receivers(const tt::tt_metal::CoreCoord& core) const {
        if (!active_) {
            return 0;
        }
        const uint32_t receivers = sender_in_rect_ ? (area_ - 1) : area_;
        if (cfg_.rotating_sender) {
            return receivers;
        }
        return is_sender(core) ? receivers : 0;
    }

    // The handshake ACK wait-count on the wire (== fan-out in the dense case; smaller when divergent).
    uint32_t num_active() const { return ack_count_; }

    // Rounds the sender role rotates through = cores in the rect (1 in fixed mode).
    uint32_t num_senders() const { return cfg_.rotating_sender ? area_ : 1u; }

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

    // ROTATING runtime block: the full-rect dest rect followed by the ordered per-round sender coords
    // (row-major over the rect). Uniform for every core in the rect; each reads its own rect on its
    // sender round and indexes the coord list by round when receiving.
    std::vector<uint32_t> rotating_rt_() const {
        const auto coords = rect_virt_coords_();
        // True rect corners (the 1x1 self-rect too, if area==1) — same reasoning as the fixed path.
        std::vector<uint32_t> out = detail::noc_ordered_bbox(cfg_.noc, coords);
        for (const auto& c : coords) {
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
    bool sender_in_rect_ = true;
    bool active_ = false;
    bool owns_sems_ = true;
    uint32_t ack_count_ = 0;
    uint32_t data_ready_id_ = 0;
    uint32_t consumer_ready_id_ = UNUSED_SEM_ID;
};

}  // namespace ttnn::kernel_lib::host
