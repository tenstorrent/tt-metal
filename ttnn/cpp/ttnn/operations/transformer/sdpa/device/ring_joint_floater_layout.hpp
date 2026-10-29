// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host side of the expanded-K floaters (kernels/ring_joint_floaters.hpp holds the device ABI): the tuned schedules,
// and the layout of home blocks, floater groups and their visit queues. Pure integer math, so it is checked without
// hardware (tests/ttnn/unit_tests/gtests/sdpa/test_ring_joint_floater_layout.cpp).

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_floaters.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Floater groups of four cores (a 2x2 block's worth): group core k mirrors home core k of the block it visits.
constexpr uint32_t kFloaterGroupCores = 4;
constexpr uint32_t kFloaterGroups = 5;
// Floater cores.
constexpr uint32_t kFloaterCores = kFloaterGroupCores * kFloaterGroups;
// Home blocks of the layout: 2x2 blocks of (2 heads x 2 Q chunks) per K-split band on a 10x10 sub-grid.
constexpr uint32_t kFloaterHomeBlocks = 24;
// First-piece visits a floater group runs, at most (the home blocks go round-robin over the groups).
constexpr uint32_t kFloaterFirstPieceVisits = (kFloaterHomeBlocks + kFloaterGroups - 1) / kFloaterGroups;

// A shape's schedule: the ring iterations a home donates in, the fraction of its band donated in each, which end of
// the band it donates, and the ring iteration after which a home merges its first piece, by that piece's position in
// its floater's queue (a floater runs them back to back).
struct FloaterSchedule {
    uint32_t iters[kDonatedPieces];
    uint32_t fracs[kDonatedPieces];
    bool donate_tail;
    uint32_t merge_after[kFloaterFirstPieceVisits];
};

// The layouts the floaters run on, by heads per device x K-split bands, each with its schedule: the two the gtest
// checks exactly (Kimi K3 and K2.7). Other shapes that fill the 24 home blocks (12 heads x 4 bands) are not covered by
// those checks and are unsupported. The schedules are tuned on the Blackhole Galaxy 8x4's 8-chip SP ring at 5k chunks
// (models/demos/deepseek_v3_d_p/tests/perf/test_tune_ring_joint_floaters.py); whether floaters run at all is
// floater_compute_to_transfer's call.
inline constexpr std::array<std::pair<std::pair<uint32_t, uint32_t>, FloaterSchedule>, 2> kFloaterLayoutSchedules = {{
    // Kimi K3: 0.35 of iteration 1, 0.9 of iteration 4, from the band's start.
    {{24, 2}, {{1, 4}, {1075, 2765}, false, {2, 2, 2, 2, 2}}},
    // Kimi K2.7: 0.6 of iteration 1, 0.54 of iteration 3, from the band's end.
    {{16, 3}, {{1, 3}, {1843, 1664}, true, {2, 3, 4, 5, 5}}},
}};

// The schedule of the layout for `heads` x `bands`, or nullopt for an unsupported layout.
inline std::optional<FloaterSchedule> floater_layout_schedule(uint32_t heads, uint32_t bands) {
    for (const auto& [heads_bands, schedule] : kFloaterLayoutSchedules) {
        if (heads_bands == std::pair{heads, bands}) {
            return schedule;
        }
    }
    return std::nullopt;
}

inline bool floater_layout_supported(uint32_t heads, uint32_t bands) {
    return floater_layout_schedule(heads, bands).has_value();
}

// Matmul multiply-adds per Tensix core per cycle at LoFi, as sdpa_perf_model.cpp counts them.
inline constexpr double kTensixMacsPerCycle = 4096.0;
// One Blackhole Ethernet link's bandwidth per direction (25 GB/s) at the 1.35 GHz AI clock, in bytes per cycle.
inline constexpr double kEthLinkBytesPerCycle = 25e9 / 1.35e9;

// One device's ring-joint SDPA work per K/V token that arrives over the ring, for floater_compute_to_transfer.
struct FloaterWorkload {
    uint32_t heads;                // per device
    uint32_t q_rows;               // per device
    uint32_t qk_dim, v_dim;        // QK^T width (K expanded: no-PE + PE) and PV width
    uint32_t expand_in;            // K expansion per head: latent width in ...
    uint32_t expand_out;           // ... no-PE width out
    uint32_t fidelity_multiplier;  // of the QK^T / PV matmuls: LoFi 1 to HiFi4 4
    double bytes_per_token;        // one token's K row and V heads on the ring
    uint32_t cores;                // SDPA compute cores
    uint32_t ring_links;           // links into the busiest device: both directions on a wrapped axis, one on a line
};

// Floaters add compute (20 idle cores beside the 96 working ones) at a fixed cost per piece (K expanded again, the
// partial handed over and merged). Compute helps only while a ring step's work outlasts the arrival of its K/V slice,
// so the floaters pay off where the SDPA's compute per arriving token is large against that token's transfer: this
// ratio of the two, both in ideal cycles (the matmuls at their fidelity, the links at full bandwidth). It rises with
// fidelity, Q rows and heads.
inline double floater_compute_to_transfer(const FloaterWorkload& w) {
    const double macs =
        static_cast<double>(w.heads) * (static_cast<double>(w.q_rows) * (w.qk_dim + w.v_dim) +
                                        static_cast<double>(w.expand_in) * static_cast<double>(w.expand_out));
    const double compute_cycles = macs * w.fidelity_multiplier / (static_cast<double>(w.cores) * kTensixMacsPerCycle);
    const double transfer_cycles = w.bytes_per_token / (static_cast<double>(w.ring_links) * kEthLinkBytesPerCycle);
    return compute_cycles / transfer_cycles;
}

// The floaters run from this floater_compute_to_transfer up. Measured on the Galaxy 8x4 (Kimi K3 and K2.7, 5k and 2k
// chunks, LoFi to HiFi4; torus and unwrapped line): every shape at 0.45 or below runs slower with floaters or no
// faster, every shape at 0.51 or above 12-16% faster at 512k cached tokens. A shape near the boundary gains or loses
// little either way.
inline constexpr double kFloaterMinComputeToTransfer = 0.48;

// Values of a schedule in SDPAProgramConfig::floater_schedule's order: iters, fracs, donate_tail, merge_after.
constexpr size_t kFloaterScheduleValues = 2 * kDonatedPieces + 1 + kFloaterFirstPieceVisits;

// Whether floater_schedule_from_values accepts `values` without a TT_FATAL: its two checks, kept in step with it (the
// gtest checks the two agree).
inline bool floater_schedule_values_well_formed(const std::vector<uint32_t>& values) {
    return values.empty() || (values.size() == kFloaterScheduleValues && values[2 * kDonatedPieces] <= 1);
}

// A schedule given as SDPAProgramConfig::floater_schedule: nullopt for an empty list (no floaters).
// build_floater_layout validates the values against the layout.
inline std::optional<FloaterSchedule> floater_schedule_from_values(const std::vector<uint32_t>& values) {
    if (values.empty()) {
        return std::nullopt;
    }
    TT_FATAL(
        values.size() == kFloaterScheduleValues,
        "floater_schedule: expected {} values (iter0, iter1, frac0, frac1, donate_tail, merge0..merge{}) or none, got "
        "{}",
        kFloaterScheduleValues,
        kFloaterFirstPieceVisits - 1,
        values.size());
    FloaterSchedule schedule{};
    for (uint32_t piece = 0; piece < kDonatedPieces; ++piece) {
        schedule.iters[piece] = values[piece];
        schedule.fracs[piece] = values[kDonatedPieces + piece];
    }
    TT_FATAL(
        values[2 * kDonatedPieces] <= 1, "floater_schedule: donate_tail is {}, not 0 or 1", values[2 * kDonatedPieces]);
    schedule.donate_tail = values[2 * kDonatedPieces] == 1;
    for (uint32_t v = 0; v < kFloaterFirstPieceVisits; ++v) {
        schedule.merge_after[v] = values[2 * kDonatedPieces + 1 + v];
    }
    return schedule;
}

constexpr uint32_t kNoFloaterGroup = std::numeric_limits<uint32_t>::max();

// A floater group's visit: a home block's donated piece, run in ring iteration `iter`.
struct FloaterVisit {
    uint32_t iter, block, piece;
};

// A home core's donated piece: the floater core that runs it, and that floater's semaphore ordinal the home sets.
struct HomePiece {
    uint32_t floater, ordinal;
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
    // The homes' K multicast groups: per block row and band, that band's blocks in the row. Every home of a band reads
    // the same K chunks, so one injector (the first core; the rest ascending) reads each chunk once for the strip. The
    // injectors sit in distinct columns, on a strip's top or bottom row by strip index, as the dense layout's row-wide
    // multicast spreads its own: each injector's DRAM reads arrive through its column, and injectors sharing one
    // slow every home's K and the gather's prefetch (the floater groups' injectors, their first cores, are not part of
    // this spread). Each strip fills a 2-row rectangle (the multicast writes all of
    // it), and its homes merge their first floater piece at the same iteration, so the strip stalls at one merge
    // iteration rather than at each home's.
    std::vector<std::vector<uint32_t>> home_strips;

    // Home block g is the 2x2 block at (2 (g % cols), 2 (g / cols)) of the sub-grid left of the last grid column.
    uint32_t block_core(uint32_t g, uint32_t k) const {
        const uint32_t cols = (grid_x - 1) / 2;
        return (2 * (g / cols) + k / 2) * grid_x + 2 * (g % cols) + k % 2;
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
    // Home blocks hold head pairs, so every head is covered only with an even count that fills them exactly.
    TT_FATAL(
        heads % 2 == 0 && (heads / 2) * bands == kFloaterHomeBlocks && gather_workers == 4 && grid_y == 10 &&
            grid_x == 11,
        "ring_joint floaters: the layout covers {} home blocks of head pairs on an 11x10 grid beside 4 gather workers "
        "(got {} heads x {} bands, {}x{}, {} workers)",
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
        "ring_joint floaters: {} groups, the schedules assume {}",
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
                layout.home_pieces[home][visit.piece] = {layout.groups[f][k], v + 1};
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
    const uint32_t cols = (grid_x - 1) / 2;
    for (uint32_t g = 0; g < kFloaterHomeBlocks; ++g) {
        const uint32_t row = g / cols;
        const uint32_t band = layout.home_band[layout.block_core(g, 0)];
        if (g == 0 || row != (g - 1) / cols || band != layout.home_band[layout.block_core(g - 1, 0)]) {
            layout.home_strips.emplace_back();
        }
        for (uint32_t k = 0; k < kFloaterGroupCores; ++k) {
            layout.home_strips.back().push_back(layout.block_core(g, k));
        }
    }
    for (auto& strip : layout.home_strips) {
        std::sort(strip.begin(), strip.end());
        uint32_t x0 = grid_x, y0 = grid_y, x1 = 0, y1 = 0;
        for (const uint32_t core : strip) {
            const auto [x, y] = layout.logical_xy(core);
            x0 = std::min(x0, x), y0 = std::min(y0, y), x1 = std::max(x1, x), y1 = std::max(y1, y);
        }
        TT_FATAL(
            (x1 - x0 + 1) * (y1 - y0 + 1) == strip.size(),
            "ring_joint floaters: a K multicast strip of {} cores does not fill ({}, {})..({}, {})",
            strip.size(),
            x0,
            y0,
            x1,
            y1);
        for (const uint32_t core : strip) {
            TT_FATAL(
                layout.home_merge_iter[core] == layout.home_merge_iter[strip.front()],
                "ring_joint floaters: the homes of a K multicast strip merge their first piece at different iterations "
                "({} and {})",
                layout.home_merge_iter[core],
                layout.home_merge_iter[strip.front()]);
        }
    }
    // Narrowest strips first: they have the fewest columns to choose from.
    std::vector<uint32_t> by_width(layout.home_strips.size());
    std::iota(by_width.begin(), by_width.end(), 0);
    std::stable_sort(by_width.begin(), by_width.end(), [&](uint32_t a, uint32_t b) {
        return layout.home_strips[a].size() < layout.home_strips[b].size();
    });
    std::vector<bool> injector_column(grid_x, false);
    for (const uint32_t s : by_width) {
        auto& strip = layout.home_strips[s];
        const uint32_t row = layout.logical_xy(strip.front()).second + s % 2;
        // A free column in this strip's turn of row, else any free column, else the top-left core.
        auto free_in = [&](bool own_row) {
            return std::find_if(strip.begin(), strip.end(), [&](uint32_t core) {
                const auto [x, y] = layout.logical_xy(core);
                return !injector_column[x] && (!own_row || y == row);
            });
        };
        auto injector = free_in(true);
        if (injector == strip.end()) {
            injector = free_in(false);
        }
        if (injector == strip.end()) {
            injector = strip.begin();
        }
        injector_column[layout.logical_xy(*injector).first] = true;
        std::rotate(strip.begin(), injector, injector + 1);
    }
    return layout;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
