// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Floaters for the expanded-K K split (ring_joint_sdpa_program_factory.cpp). The K-split units leave cores
// idle; they become floater groups that take over pieces of the home cores' work. Every home core donates a piece of
// its band in two ring iterations; a floater runs each piece as a fresh accumulation in ring order and saves the
// partial (max, sum, out) to the home core's DRAM slot, which the home merges before its K-split role. A piece is a
// pure function of the band and the donated fraction, so homes and floaters cut the same chunks.

#pragma once

#include <cstdint>

#include "ring_joint_ksplit.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Donated fractions are in units of 1 / kDonationFracOne of a band.
constexpr uint32_t kDonationFracOne = 3072;
// Ring iterations a home core donates in, and so the floater partials it merges (one scratch slot each).
constexpr uint32_t kDonatedPieces = 2;
// Ring iterations covered by the per-iteration fraction args.
constexpr uint32_t kDonationIters = 8;
// Visits a floater runs in one ring iteration.
constexpr uint32_t kFloaterMaxVisitsPerIter = 8;
// A scratch slot holds one partial's max, sum and out tiles, one float32-tile page each.
constexpr uint32_t kFloaterSlotPageBytes = 4096;
// Runtime args per floater visit: reader / compute [iteration, unit, band, frac]; writer [iteration, slot, home x,
// home y, home semaphore ordinal].
constexpr uint32_t kFloaterVisitArgs = 4;
constexpr uint32_t kFloaterWriterVisitArgs = 5;

// Home blocks of the rotation layout: 2x2 blocks of (2 heads x 2 Q chunks) per K-split band on a 10x10 sub-grid.
constexpr uint32_t kFloaterHomeBlocks = 24;

// A shape's schedule: the ring iterations a home donates in, the fraction of its band donated in each, and which end of
// the band it donates. Tuned on the Blackhole 8-chip ring (Kimi final chunk, 50k cached + 5k new, q320 k256).
struct FloaterSchedule {
    uint32_t iters[kDonatedPieces];
    uint32_t fracs[kDonatedPieces];
    bool donate_tail;
};

// The schedule of a validated shape: heads per device x K-split bands (heads / 2 x bands = kFloaterHomeBlocks).
inline bool floater_schedule(uint32_t heads, uint32_t bands, FloaterSchedule& schedule) {
    if (heads == 24 && bands == 2) {  // Kimi K3: 0.35 of iteration 2, 0.9 of iteration 4, from the band's start
        schedule = {{2, 4}, {1075, 2765}, false};
        return true;
    }
    if (heads == 16 && bands == 3) {  // Kimi K2.7: 0.6 of iteration 1, 0.5 of iteration 3, from the band's end
        schedule = {{1, 3}, {1843, 1536}, true};
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

// What the home keeps of its band.
inline KSplitRange kept_after_donation(KSplitRange band, uint32_t frac, bool donate_tail) {
    const uint32_t cut = donated_cut(band, frac);
    return donate_tail ? KSplitRange{band.begin, band.end - cut} : KSplitRange{band.begin + cut, band.end};
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
