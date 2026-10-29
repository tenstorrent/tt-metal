// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Floaters for the expanded-K K split (ring_joint_sdpa_program_factory.cpp). The K-split units leave cores
// idle; they become floater groups that take over pieces of the home cores' work. Every home core donates a piece of
// its band in two ring iterations; a floater runs each piece as a fresh accumulation in ring order and stages the
// partial (max, sum, out) in its accumulator half A. The home's writer reads it from there into its restore CBs and
// tells the floater, which only then reuses half A for its next visit. Compute merges the first piece at a fixed ring
// step (the schedule's merge_after), which frees the restore CBs for the second; the second merges before the K-split
// role. Fixed merge points keep the result deterministic; a late first merge holds back the floater whose second
// piece waits for it. A piece is a pure function of the band and the
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
// Runtime args per floater visit: reader / compute [iteration, unit, band]; writer [iteration, home x, home y].
// A visit's index v in the floater's list is its own semaphore ordinal v + 1, which the home sets once it has read the
// partial; the home's ordinal for it follows the K-split senders' (piece_ordinal).
constexpr uint32_t kFloaterVisitArgs = 3;
constexpr uint32_t kFloaterWriterVisitArgs = 3;
// Runtime args per donated piece of a home's writer: [floater x, floater y, floater semaphore ordinal].
constexpr uint32_t kFloaterHomePieceArgs = 3;

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

// Chunks of `band` that `frac` donates.
// Out of line: homes and floaters both cut (kept_after_donation, donated_piece), in one binary near the kernel config
// buffer's size. Only floater programs call it.
__attribute__((noinline)) inline uint32_t donated_cut(KSplitRange band, uint32_t frac) {
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
