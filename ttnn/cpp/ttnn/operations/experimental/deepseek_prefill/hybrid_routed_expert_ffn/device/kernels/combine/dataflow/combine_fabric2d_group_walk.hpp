// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Which tile-rows an untilizer group produces for one local expert, and in what order. Kernel-only: both the
// untilizer that produces them and the readers that consume them build this from the same replicated control
// tensors, so a batch index names the same tile-row on every core with nothing exchanged between them.
//
// A group is one ring direction, feeding the num_links senders travelling that way. Those senders take their
// destinations furthest-first — descending dispatch-group index clockwise, ascending counter-clockwise — and
// walk the pages inside a run the same way. Runs sit in dispatch-group order inside an expert's region, so
// each destination's run continues where the previous one stopped, and the same-chip run the local phase
// copies continues where the last of them stops. The walk is therefore ONE contiguous page interval, or two
// where the dispatch-group index wraps the region boundary — at most once, since the destinations plus the
// same-chip run are m + 1 consecutive indices out of the ring's 2m.
//
// The group produces every one of those runs WHOLE. A sender takes only its own share, so it skips what the
// other sender of its group takes and must release those batches on the way past; and the two runs both
// directions need — the diametrically opposite chip's, and the same-chip one — are simply produced by both
// groups, which is what keeps this free of any notion of a seam.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "combine_fabric2d_kernel_interface.hpp"

namespace hyb_cmbf2d {

// Where origin chip `dg_index`'s tokens for expert `e` start and end. One run ends where the next begins; the
// last one ends at the expert's total count past its region base, which is the only thing expert_offsets
// cannot say.
struct ControlTables {
    volatile tt_l1_ptr uint32_t* offsets;
    volatile tt_l1_ptr uint32_t* counts;
    volatile tt_l1_ptr uint32_t* region;
    uint32_t num_routed_experts;
    uint32_t dispatch_group_size;
    // This ring's rows of global_expert_idx_table, one per chip, `expert_row_words` apart.
    volatile tt_l1_ptr uint32_t* expert_ids;
    uint32_t expert_row_words;

    // The global id of the expert in slot `local` of ring chip `dg_index`.
    uint32_t expert_id(uint32_t dg_index, uint32_t local) const {
        return expert_ids[dg_index * expert_row_words + local];
    }

    uint32_t run_begin(uint32_t dg_index, uint32_t e) const { return offsets[dg_index * num_routed_experts + e]; }
    uint32_t run_end(uint32_t dg_index, uint32_t e) const {
        return dg_index + 1 < dispatch_group_size ? offsets[(dg_index + 1) * num_routed_experts + e]
                                                  : region[e] + counts[e];
    }
};

// The global expert a chip handles at walk step `step`, in the order the routed expert finishes them. The
// routed expert runs two passes, every expert with count <= threshold then the rest, each in ascending
// index, so combine walks that order too and each expert is picked up as soon as it is ready. With no
// threshold there is only the second pass and the order is the identity.
//
// The test is on the RAW count, exactly the routed expert's count_in_band: a different reading here would
// put an expert in the other pass and combine would read that expert's rows before they are written.
// Built from the replicated counts, so any core can compute it for any chip: a relay sizes a chunk with its
// ORIGIN chip's order, not its own.
// Whether the routed expert runs the expert in slot `local` of ring chip `dg_index` in its first, fused pass.
inline bool in_fused_pass(const ControlTables& ctl, uint32_t dg_index, uint32_t threshold, uint32_t local) {
    return threshold != 0 && ctl.counts[ctl.expert_id(dg_index, local)] <= threshold;
}

// The slot, not the global id, of the expert walked at `step`.
inline uint32_t local_at_step(
    const ControlTables& ctl, uint32_t dg_index, uint32_t experts_per_chip, uint32_t threshold, uint32_t step) {
    uint32_t seen = 0;
    for (uint32_t pass = 0; pass < 2; pass++) {
        for (uint32_t local = 0; local < experts_per_chip; local++) {
            if (in_fused_pass(ctl, dg_index, threshold, local) == (pass == 0)) {
                if (seen == step) {
                    return local;
                }
                seen++;
            }
        }
    }
    ASSERT(false);
    return step;
}

inline uint32_t expert_at_step(
    const ControlTables& ctl, uint32_t dg_index, uint32_t experts_per_chip, uint32_t threshold, uint32_t step) {
    return ctl.expert_id(dg_index, local_at_step(ctl, dg_index, experts_per_chip, threshold, step));
}

// The `ready` value that says the routed expert has written the expert in slot `local` of THIS chip. Every
// routed-expert writer reports once per slot it walks, in every pass, and the collector publishes how many
// walked slots all of them have passed. So a fused expert is ready once pass A has walked past its slot, and
// an unfused one once pass A has walked all of them and pass B has walked past its slot.
inline uint32_t ready_target(
    const ControlTables& ctl, uint32_t dg_index, uint32_t experts_per_chip, uint32_t threshold, uint32_t local) {
    if (in_fused_pass(ctl, dg_index, threshold, local)) {
        return local + 1;
    }
    const uint32_t fused_pass_slots = threshold != 0 ? experts_per_chip : 0;
    return fused_pass_slots + local + 1;
}

// Blocks until this core's `ready` count reaches `target`.
inline void wait_for_ready(uint32_t ready_sem, uint32_t target) {
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem));
    invalidate_l1_cache();
    while (*ready < target) {
        invalidate_l1_cache();
    }
}

class GroupWalk {
public:
    explicit GroupWalk(bool walks_down) : walks_down_(walks_down) {}

    // Runs in the order the group walks them. An empty one is skipped and does not separate its neighbours,
    // because a run always begins exactly where the previous one ended.
    void add_run(uint32_t lo, uint32_t hi) {
        if (hi == lo) {
            return;
        }
        if (num_segments_ > 0) {
            Segment& last = segments_[num_segments_ - 1];
            if (walks_down_ && last.lo == hi) {
                last.lo = lo;
                return;
            }
            if (!walks_down_ && last.hi == lo) {
                last.hi = hi;
                return;
            }
        }
        segments_[num_segments_++] = Segment{lo, hi, 0};
    }

    // No more runs: number the tile-rows.
    void seal() {
        for (uint32_t s = 0; s < num_segments_; s++) {
            segments_[s].first_batch = num_batches_;
            num_batches_ += segments_[s].batches();
        }
    }

    uint32_t num_batches() const { return num_batches_; }

    uint32_t batch_of(uint32_t page) const {
        const Segment& s = segments_[segment_of_page(page)];
        return s.first_batch + (walks_down_ ? tile_row(s.hi - 1) - tile_row(page) : tile_row(page) - tile_row(s.lo));
    }

    uint32_t tile_row_of(uint32_t batch) const {
        const Segment& s = segments_[segment_of_batch(batch)];
        return walks_down_ ? tile_row(s.hi - 1) - (batch - s.first_batch) : tile_row(s.lo) + (batch - s.first_batch);
    }

private:
    // Two, because the walk breaks only where the dispatch-group index wraps, and it wraps at most once.
    static constexpr uint32_t MAX_SEGMENTS = 2;

    struct Segment {
        uint32_t lo;
        uint32_t hi;
        uint32_t first_batch;

        uint32_t batches() const { return (hi - 1) / UNT_BATCH_ROWS - lo / UNT_BATCH_ROWS + 1; }
    };

    static uint32_t tile_row(uint32_t page) { return page / UNT_BATCH_ROWS; }

    // The segments are disjoint and there are at most two, so falling through to the last one is the answer
    // rather than a guess.
    uint32_t segment_of_page(uint32_t page) const {
        return (num_segments_ > 1 && page >= segments_[0].lo && page < segments_[0].hi) ? 0 : num_segments_ - 1;
    }
    uint32_t segment_of_batch(uint32_t batch) const {
        return (num_segments_ > 1 && batch < segments_[1].first_batch) ? 0 : num_segments_ - 1;
    }

    Segment segments_[MAX_SEGMENTS];
    uint32_t num_segments_ = 0;
    uint32_t num_batches_ = 0;
    bool walks_down_;
};

// The walk for one local expert. `destination(k)` gives the k-th own destination's dispatch-group index in
// the order the group emits them; the same-chip run the local phase copies closes the walk.
template <typename Destination>
GroupWalk group_walk(
    const ControlTables& ctl,
    bool walks_down,
    uint32_t expert,
    uint32_t my_dg_index,
    uint32_t num_destinations,
    Destination destination) {
    GroupWalk walk(walks_down);
    for (uint32_t k = 0; k < num_destinations; k++) {
        walk.add_run(ctl.run_begin(destination(k), expert), ctl.run_end(destination(k), expert));
    }
    walk.add_run(ctl.run_begin(my_dg_index, expert), ctl.run_end(my_dg_index, expert));
    walk.seal();
    return walk;
}

}  // namespace hyb_cmbf2d
