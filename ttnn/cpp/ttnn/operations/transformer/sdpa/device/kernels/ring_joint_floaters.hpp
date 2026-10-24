// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Floaters for the expanded-K K split (ring_joint_sdpa_program_factory.cpp). The K-split units leave cores
// idle; they become floater groups that take over pieces of the home cores' work. Every home core donates a piece of
// its band in two ring iterations; a floater runs each piece as a fresh accumulation in ring order and saves the
// partial (max, sum, out) to a DRAM slot. The home's writer lands it in L1 as soon as it is saved and hands the slot
// back to the floater. A floater core cycles through kFloaterRingSlots slots, so it saves visit v only once the home of
// visit v - kFloaterRingSlots has read that slot. Compute merges the first piece at a fixed ring step, after the
// floater has saved it (the schedule's merge_after), which frees the L1 for the second; the second merges before the
// K-split role. Fixed merge points keep the result deterministic. A piece is a pure function of the band and the
// donated fraction, so homes and floaters cut the same chunks.

#pragma once

#include <cstdint>

#include "ring_joint_ksplit.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Donated fractions are in units of 1 / kDonationFracOne of a band.
constexpr uint32_t kDonationFracOne = 3072;
// Ring iterations a home core donates in, and so the floater partials it merges.
constexpr uint32_t kDonatedPieces = 2;
// Visits a floater runs in one ring iteration.
constexpr uint32_t kFloaterMaxVisitsPerIter = 8;
// A scratch slot holds one partial's max, sum and out tiles, one float32-tile page each.
constexpr uint32_t kFloaterSlotPageBytes = 4096;
// Floater groups of four cores (a 2x2 block's worth), and the scratch slots each floater core cycles through: core c
// owns slots [c kFloaterRingSlots, (c + 1) kFloaterRingSlots). Two let a floater save a visit while its home still
// lands the one before.
constexpr uint32_t kFloaterGroups = 5;
constexpr uint32_t kFloaterCores = 4 * kFloaterGroups;
constexpr uint32_t kFloaterRingSlots = 2;
// Runtime args per floater visit: reader / compute [iteration, unit, band]; writer [iteration, slot, home x, home y].
// A visit's index v in the floater's list is its own semaphore ordinal v + 1, which the home sets once it has read the
// slot; the home's ordinal for it follows the K-split senders' (piece_ordinal).
constexpr uint32_t kFloaterVisitArgs = 3;
constexpr uint32_t kFloaterWriterVisitArgs = 4;
// Runtime args per donated piece of a home's writer: [slot, floater x, floater y, floater semaphore ordinal].
constexpr uint32_t kFloaterHomePieceArgs = 4;

// The program's donation schedule, the same on every home (named compile-time args donate_iter0 / 1 and
// donate_frac0 / 1): a home donates fracs[p] of its band in ring iteration iters[p], iters ascending.
struct DonationSchedule {
    uint32_t iters[kDonatedPieces];
    uint32_t fracs[kDonatedPieces];

    // The fraction donated in `ring_iter`, 0 outside the schedule.
    constexpr uint32_t frac(uint32_t ring_iter) const {
        return ring_iter == iters[0] ? fracs[0] : ring_iter == iters[1] ? fracs[1] : 0;
    }
    // The piece donated in a scheduled `ring_iter`.
    constexpr uint32_t piece(uint32_t ring_iter) const { return ring_iter == iters[0] ? 0 : 1; }
};
static_assert(kDonatedPieces == 2, "DonationSchedule spells out two pieces");

// A home's semaphore ordinal for the floater partial of `piece`: its bits follow the K-split senders'.
constexpr uint32_t piece_ordinal(uint32_t ksplit_count, uint32_t piece) { return ksplit_count + 1 + piece; }

// Home blocks of the rotation layout: 2x2 blocks of (2 heads x 2 Q chunks) per K-split band on a 10x10 sub-grid.
constexpr uint32_t kFloaterHomeBlocks = 24;
// First-piece visits a floater group runs, at most (the home blocks go round-robin over the groups).
constexpr uint32_t kFloaterFirstPieceVisits = (kFloaterHomeBlocks + kFloaterGroups - 1) / kFloaterGroups;

// A shape's schedule: the ring iterations a home donates in, the fraction of its band donated in each, which end of
// the band it donates, and the ring iteration after which a home merges its first piece, by that piece's position in
// its floater's queue (a floater runs them back to back). Tuned on the Blackhole 8-chip ring (Kimi final chunk, 50k
// cached + 5k new, q320 k256).
struct FloaterSchedule {
    uint32_t iters[kDonatedPieces];
    uint32_t fracs[kDonatedPieces];
    bool donate_tail;
    uint32_t merge_after[kFloaterFirstPieceVisits];
};

// The schedule of a validated shape: heads per device x K-split bands (heads / 2 x bands = kFloaterHomeBlocks).
inline bool floater_schedule(uint32_t heads, uint32_t bands, FloaterSchedule& schedule) {
    if (heads == 24 && bands == 2) {  // Kimi K3: 0.35 of iteration 2, 0.9 of iteration 4, from the band's start
        schedule = {{2, 4}, {1075, 2765}, false, {3, 3, 3, 3, 3}};
        return true;
    }
    if (heads == 16 && bands == 3) {  // Kimi K2.7: 0.6 of iteration 1, 0.5 of iteration 3, from the band's end
        schedule = {{1, 3}, {1843, 1536}, true, {3, 4, 5, 6, 6}};
        return true;
    }
    return false;
}

// Chunks of `band` that `frac` donates.
inline uint32_t donated_cut(KSplitRange band, uint32_t frac) {
    const uint32_t len = band.end > band.begin ? band.end - band.begin : 0;
    return len * frac / kDonationFracOne;
}

// The floater's piece of a home band: at the band's end when donate_tail, else at its start.
inline KSplitRange donated_piece(KSplitRange band, uint32_t frac, bool donate_tail) {
    const uint32_t cut = donated_cut(band, frac);
    return donate_tail ? KSplitRange{band.end - cut, band.end} : KSplitRange{band.begin, band.begin + cut};
}

// A floater's visits in one ring step: the donated piece of each band it visits, and that band's unit. Reader and
// compute collect them from the same args, so they cut the same chunks.
template <uint32_t max_visits>
struct FloaterStepVisits {
    KSplitRange ranges[max_visits];
    uint32_t units[max_visits];
    uint32_t count = 0;

    // The visits of `ring_iter` among a floater's `visit_count` visit args ([iteration, unit, band] each, from
    // `args`), whose bands are this step's split of `num_valid` chunks; `frac` is the iteration's donated fraction.
    template <bool donate_tail, typename ReadArg>
    void collect(
        ReadArg read_arg,
        uint32_t args,
        uint32_t visit_count,
        uint32_t ring_iter,
        uint32_t frac,
        uint32_t num_valid,
        uint32_t bands,
        uint32_t ring_id) {
        for (uint32_t v = 0; v < visit_count && count < max_visits; ++v) {
            const uint32_t a = args + v * kFloaterVisitArgs;
            if (read_arg(a) == ring_iter) {
                ranges[count] =
                    donated_piece(ksplit_range(num_valid, read_arg(a + 2), bands, ring_id), frac, donate_tail);
                units[count++] = read_arg(a + 1);
            }
        }
    }
};

// What the home keeps of its band.
inline KSplitRange kept_after_donation(KSplitRange band, uint32_t frac, bool donate_tail) {
    const uint32_t cut = donated_cut(band, frac);
    return donate_tail ? KSplitRange{band.begin, band.end - cut} : KSplitRange{band.begin + cut, band.end};
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
