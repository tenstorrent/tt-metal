// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host side of the expanded-K floaters (kernels/ring_joint_floaters.hpp holds the device ABI): the tuned schedules,
// and the layout of home blocks, floater groups, their visit queues and scratch slots. Pure integer math, so it is
// checked without hardware (tests/ttnn/unit_tests/gtests/sdpa/test_ring_joint_floater_layout.cpp).

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_floaters.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Floater groups of four cores (a 2x2 block's worth): group core k mirrors home core k of the block it visits.
constexpr uint32_t kFloaterGroupCores = 4;
constexpr uint32_t kFloaterGroups = 5;
// Floater cores; core c owns scratch slots [c kFloaterRingSlots, (c + 1) kFloaterRingSlots).
constexpr uint32_t kFloaterCores = kFloaterGroupCores * kFloaterGroups;
// Home blocks of the layout: 2x2 blocks of (2 heads x 2 Q chunks) per K-split band on a 10x10 sub-grid.
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

constexpr uint32_t kNoFloaterGroup = std::numeric_limits<uint32_t>::max();

// A floater group's visit: a home block's donated piece, run in ring iteration `iter`.
struct FloaterVisit {
    uint32_t iter, block, piece;
};

// A home core's donated piece: the scratch slot it lands, and the floater core whose semaphore ordinal it sets.
struct HomePiece {
    uint32_t slot, floater, ordinal;
};

// Core indices are the program's: the SDPA grid row-major [0, num_cores), then the gather column's cores below its
// workers [num_cores, num_cores_all) (floater programs only). Defaults describe a core with no floater role: band 0,
// no unit (total_q_chunks), no group.
struct FloaterLayout {
    uint32_t grid_x = 0;
    uint32_t num_cores = 0;
    uint32_t gather_workers = 0;
    uint32_t num_cores_all = 0;
    // Per core.
    std::vector<uint32_t> home_band, home_unit, home_merge_iter, group_of, k_of;
    std::vector<std::array<HomePiece, kDonatedPieces>> home_pieces;
    // Per (band, unit): the home core.
    std::vector<uint32_t> unit_home;
    std::vector<std::array<uint32_t, kFloaterGroupCores>> groups;
    // Per group, in run order: first pieces, then second pieces.
    std::vector<std::vector<FloaterVisit>> visits;

    // Home block g is the 2x2 block at (2 (g % cols), 2 (g / cols)) of the sub-grid left of the last grid column.
    uint32_t block_core(uint32_t g, uint32_t k) const {
        const uint32_t cols = (grid_x - 1) / 2;
        return (2 * (g / cols) + k / 2) * grid_x + 2 * (g % cols) + k % 2;
    }
    // Group core k of group f is floater core 4 f + k, and saves its visit v in its ring slot v % kFloaterRingSlots.
    static uint32_t slot(uint32_t f, uint32_t k, uint32_t v) {
        return (f * kFloaterGroupCores + k) * kFloaterRingSlots + v % kFloaterRingSlots;
    }
    bool is_floater(uint32_t core) const { return group_of[core] != kNoFloaterGroup; }
    // Logical (x, y) of a core: past the SDPA grid, the gather column below its workers.
    std::pair<uint32_t, uint32_t> logical_xy(uint32_t core) const {
        return core < num_cores ? std::pair{core % grid_x, core / grid_x}
                                : std::pair{grid_x, gather_workers + (core - num_cores)};
    }
};

// A program without floaters: every core keeps the defaults.
inline FloaterLayout no_floater_layout(uint32_t grid_x, uint32_t num_cores, uint32_t total_q_chunks) {
    FloaterLayout layout;
    layout.grid_x = grid_x;
    layout.num_cores = num_cores;
    layout.num_cores_all = num_cores;
    layout.home_band.assign(num_cores, 0);
    layout.home_unit.assign(num_cores, total_q_chunks);
    layout.home_merge_iter.assign(num_cores, 0);
    layout.group_of.assign(num_cores, kNoFloaterGroup);
    layout.k_of.assign(num_cores, 0);
    layout.home_pieces.resize(num_cores);
    return layout;
}

// The floater layout of an eligible shape (ring_joint_plan::floater_eligibility): grid_x x grid_y SDPA cores with the
// gather's gather_workers in the column right of them, `heads` heads of two Q chunks in `bands` K-split bands.
//
// Home block g holds head pair g % (heads / 2) of band g / (heads / 2); its core k = (r, c) owns unit
// (2 pair + r) * 2 + c (head 2 pair + r, Q chunk c), so a head's two Q chunks (an expansion pair, one V chain) share a
// row. Floater groups: the 25th block, the last column's rows 0-3 and 4-7, the gather column's rows 4-7, and the 2x2
// of both columns' rows 8-9; their expansion pairs are (0, 1) and (2, 3). The visits go round-robin over the groups
// in ring order, each home block's two pieces.
inline FloaterLayout build_floater_layout(
    const FloaterSchedule& schedule,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t heads,
    uint32_t bands,
    uint32_t gather_workers,
    uint32_t total_q_chunks,
    uint32_t ring_size) {
    // Homes merge the partials in the order of their pieces' iterations; every piece donates something, and a home
    // keeps part of every non-empty band (so a partial with state always has the home's own to merge into).
    for (uint32_t piece = 0; piece < kDonatedPieces; ++piece) {
        TT_FATAL(
            schedule.fracs[piece] > 0 && schedule.fracs[piece] < kDonationFracOne &&
                schedule.iters[piece] < ring_size && (piece == 0 || schedule.iters[piece - 1] < schedule.iters[piece]),
            "ring_joint floaters: invalid schedule for {} heads x {} bands",
            heads,
            bands);
    }
    TT_FATAL(
        (heads / 2) * bands == kFloaterHomeBlocks && gather_workers == 4 && grid_y == 10 && grid_x == 11,
        "ring_joint floaters: the layout covers {} home blocks on an 11x10 grid beside 4 gather workers (got {} heads "
        "x "
        "{} bands, {}x{}, {} workers)",
        kFloaterHomeBlocks,
        heads,
        bands,
        grid_x,
        grid_y,
        gather_workers);
    const uint32_t num_cores = grid_x * grid_y;
    FloaterLayout layout = no_floater_layout(grid_x, num_cores, total_q_chunks);
    layout.gather_workers = gather_workers;
    layout.num_cores_all = num_cores + grid_y - gather_workers;
    for (auto* per_core : {&layout.home_band, &layout.home_merge_iter, &layout.k_of}) {
        per_core->resize(layout.num_cores_all, 0);
    }
    layout.home_unit.resize(layout.num_cores_all, total_q_chunks);
    layout.group_of.resize(layout.num_cores_all, kNoFloaterGroup);
    layout.home_pieces.resize(layout.num_cores_all);
    layout.unit_home.assign(bands * total_q_chunks, 0);

    const uint32_t pairs = heads / 2;
    for (uint32_t g = 0; g < kFloaterHomeBlocks; ++g) {
        for (uint32_t k = 0; k < kFloaterGroupCores; ++k) {
            const uint32_t core = layout.block_core(g, k);
            layout.home_band[core] = g / pairs;
            layout.home_unit[core] = (2 * (g % pairs) + k / 2) * 2 + k % 2;
            layout.unit_home[layout.home_band[core] * total_q_chunks + layout.home_unit[core]] = core;
        }
    }

    const uint32_t x = grid_x - 1;
    auto at = [&](uint32_t cy) { return cy * grid_x + x; };
    const uint32_t spare = kFloaterHomeBlocks;  // the block after the homes'
    layout.groups = {
        {layout.block_core(spare, 0),
         layout.block_core(spare, 1),
         layout.block_core(spare, 2),
         layout.block_core(spare, 3)},
        {at(0), at(1), at(2), at(3)},
        {at(4), at(5), at(6), at(7)},
        {num_cores, num_cores + 1, num_cores + 2, num_cores + 3},
        {at(8), num_cores + 4, at(9), num_cores + 5}};
    TT_FATAL(
        layout.groups.size() == kFloaterGroups,
        "ring_joint floaters: {} groups, the scratch and schedules assume {}",
        layout.groups.size(),
        kFloaterGroups);
    layout.visits.resize(layout.groups.size());
    uint32_t next = 0;
    for (uint32_t piece = 0; piece < kDonatedPieces; ++piece) {
        for (uint32_t g = 0; g < kFloaterHomeBlocks; ++g) {
            layout.visits[next++ % layout.groups.size()].push_back({schedule.iters[piece], g, piece});
        }
    }
    for (uint32_t f = 0; f < layout.groups.size(); ++f) {
        const auto& visits = layout.visits[f];
        for (uint32_t piece = 0; piece < kDonatedPieces; ++piece) {
            const auto visits_in_iter = std::count_if(visits.begin(), visits.end(), [&](const FloaterVisit& visit) {
                return visit.iter == schedule.iters[piece];
            });
            TT_FATAL(
                visits_in_iter <= kFloaterMaxVisitsPerIter,
                "ring_joint floaters: group {} has {} visits in ring iteration {}",
                f,
                visits_in_iter,
                schedule.iters[piece]);
        }
        // A visit's ordinal (v + 1) is a bit of the floater's semaphore.
        TT_FATAL(visits.size() < 32, "ring_joint floaters: group {} has {} visits", f, visits.size());
        for (uint32_t k = 0; k < kFloaterGroupCores; ++k) {
            layout.group_of[layout.groups[f][k]] = f;
            layout.k_of[layout.groups[f][k]] = k;
            for (uint32_t v = 0; v < visits.size(); ++v) {
                const auto& visit = visits[v];
                const uint32_t home = layout.block_core(visit.block, k);
                layout.home_pieces[home][visit.piece] = {FloaterLayout::slot(f, k, v), layout.groups[f][k], v + 1};
                TT_FATAL(
                    v == 0 || visits[v - 1].piece <= visit.piece,
                    "ring_joint floaters: group {} runs its visits out of piece order",
                    f);
                if (visit.piece == 0) {
                    // A group runs its first-piece visits first, so v is the visit's place in that queue.
                    TT_FATAL(
                        v < kFloaterFirstPieceVisits && schedule.merge_after[v] >= schedule.iters[0] &&
                            schedule.merge_after[v] < ring_size,
                        "ring_joint floaters: no merge iteration for group {}'s first-piece visit {}",
                        f,
                        v);
                    layout.home_merge_iter[home] = schedule.merge_after[v];
                }
            }
        }
    }
    return layout;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
