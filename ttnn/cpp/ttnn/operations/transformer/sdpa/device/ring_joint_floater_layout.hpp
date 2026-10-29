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
#include <optional>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>

#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_floaters.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Floater groups of four cores (a home block's worth): group core k mirrors home core k of the block it visits.
constexpr uint32_t kFloaterGroupCores = 4;
constexpr uint32_t kFloaterGroups = 5;
// Floater cores.
constexpr uint32_t kFloaterCores = kFloaterGroupCores * kFloaterGroups;
// Home blocks of the layout: per K-split band, its head pairs (2 heads x 2 Q chunks, four consecutive units).
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
    // Per group: whether its cores fill a rectangle, so one of them reads each K chunk and multicasts it to the others
    // (the program's K multicast writes the whole rectangle); a group that does not reads K on every core.
    std::vector<bool> group_multicast;
    // Units per band and head pairs per band: home block g is band g / pairs' units 4 (g % pairs) .. + 3.
    uint32_t total_q_chunks = 0;
    uint32_t pairs = 0;

    // Home block g's core k: unit 4 (g % pairs) + k of band g / pairs (head 2 (g % pairs) + k / 2, Q chunk k % 2).
    uint32_t block_core(uint32_t g, uint32_t k) const {
        return unit_home[(g / pairs) * total_q_chunks + kFloaterGroupCores * (g % pairs) + k];
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
// The homes are where the program without floaters puts them (the K split's dense layout: band y / rows_per_band, unit
// (y % rows_per_band) * grid_x + x), so a floater program moves their data as that program does. The floaters are the
// cores that layout leaves idle, with the gather column's cores below its workers: row runs of them (the gather
// column's core extends its row's run) cut into 1x4 groups from the left, then gather-column runs into 4x1 groups,
// then whatever is left in groups of four that read K on every core. Group core k mirrors the visited block's unit k,
// so expansion pairs are (0, 1) and (2, 3) on both sides. The visits go round-robin over the groups in ring order,
// each home block's two pieces; a group without multicast comes last, among the groups with the fewest visits.
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
        heads % 2 == 0 && (heads / 2) * bands == kFloaterHomeBlocks && total_q_chunks == 2 * heads &&
            gather_workers == 4 && grid_y == 10 && grid_x == 11,
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

    layout.total_q_chunks = total_q_chunks;
    layout.pairs = heads / 2;
    // Homes: the dense K-split layout.
    const uint32_t rows_per_band = (total_q_chunks + grid_x - 1) / grid_x;
    TT_FATAL(
        rows_per_band * bands <= grid_y,
        "ring_joint floaters: {} bands of {} rows do not fit {} rows",
        bands,
        rows_per_band,
        grid_y);
    std::vector<bool> idle(layout.num_cores_all, true);
    for (uint32_t core = 0; core < num_cores; ++core) {
        const uint32_t band = core / grid_x / rows_per_band;
        const uint32_t unit = core / grid_x % rows_per_band * grid_x + core % grid_x;
        if (band < bands && unit < total_q_chunks) {
            layout.home_band[core] = band;
            layout.home_unit[core] = unit;
            layout.unit_home[band * total_q_chunks + unit] = core;
            idle[core] = false;
        }
    }

    // Floater groups: the idle cores. The gather column's core of row y is num_cores + y - gather_workers, from row
    // gather_workers down.
    auto core_at = [&](uint32_t cx, uint32_t cy) -> std::optional<uint32_t> {
        if (cx < grid_x) {
            return cy * grid_x + cx;
        }
        return cy >= gather_workers ? std::optional<uint32_t>(num_cores + cy - gather_workers) : std::nullopt;
    };
    auto take = [&](const std::array<uint32_t, kFloaterGroupCores>& group, bool multicast) {
        for (const uint32_t core : group) {
            idle[core] = false;
        }
        layout.groups.push_back(group);
        layout.group_multicast.push_back(multicast);
    };
    for (uint32_t cy = 0; cy < grid_y; ++cy) {
        std::vector<uint32_t> run;
        for (uint32_t cx = 0; cx <= grid_x; ++cx) {
            const std::optional<uint32_t> core = core_at(cx, cy);
            if (core.has_value() && idle[*core]) {
                run.push_back(*core);
            }
            if (!core.has_value() || !idle[*core] || cx == grid_x) {
                for (uint32_t i = 0; i + kFloaterGroupCores <= run.size(); i += kFloaterGroupCores) {
                    take({run[i], run[i + 1], run[i + 2], run[i + 3]}, true);
                }
                run.clear();
            }
        }
    }
    {
        std::vector<uint32_t> run;
        for (uint32_t cy = gather_workers; cy <= grid_y; ++cy) {
            const bool free = cy < grid_y && idle[*core_at(grid_x, cy)];
            if (free) {
                run.push_back(*core_at(grid_x, cy));
            }
            if (!free) {
                for (uint32_t i = 0; i + kFloaterGroupCores <= run.size(); i += kFloaterGroupCores) {
                    take({run[i], run[i + 1], run[i + 2], run[i + 3]}, true);
                }
                run.clear();
            }
        }
    }
    std::vector<uint32_t> rest;
    for (uint32_t core = 0; core < layout.num_cores_all; ++core) {
        if (idle[core]) {
            rest.push_back(core);
        }
    }
    TT_FATAL(
        rest.size() % kFloaterGroupCores == 0 &&
            layout.groups.size() + rest.size() / kFloaterGroupCores == kFloaterGroups,
        "ring_joint floaters: {} idle cores beside the homes, the schedules assume {}",
        layout.groups.size() * kFloaterGroupCores + rest.size(),
        kFloaterCores);
    for (uint32_t i = 0; i < rest.size(); i += kFloaterGroupCores) {
        take({rest[i], rest[i + 1], rest[i + 2], rest[i + 3]}, false);
    }
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
    return layout;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
