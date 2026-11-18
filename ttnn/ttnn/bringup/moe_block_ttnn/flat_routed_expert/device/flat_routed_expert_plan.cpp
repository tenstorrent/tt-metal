// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_plan.hpp"
#include "../../common/core_grid.hpp"  // (bring-up FIX.15: set_max_grid)

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>

#include <fmt/format.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

namespace ttnn::operations::bringup::moe_block::flat_routed_expert {

namespace plan_detail {
using Core = tt::tt_metal::CoreCoord;

constexpr uint32_t MT_MAX = 4;
// the down cores' h chains: 4 on every shape (flat us at Galaxy loads: GLM-5.3 4 / 7 / 10 chains 533.2 / 532.0 / 531.2,
// K3 3 / 4 / 7 1294.1 / 1294.1 / 1294.3, V4.1-Flash 4 / 7 634.6 / 639.0)
constexpr uint32_t D_CHAINS = 4;
constexpr int NOC_X = 17, NOC_Y = 12;  // Blackhole NoC torus

uint32_t al(uint32_t b) { return (b + 2047) / 2048 * 2048; }

// The first chip this rank drives: the plan is read from it.
tt::tt_metal::distributed::MeshCoordinate flat_reference_coord(tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const auto device_ids = mesh_device.get_device_ids();
    TT_FATAL(!device_ids.empty(), "flat_routed_expert: the mesh device has no local chip");
    return mesh_device.get_view().find_device(device_ids.front());
}
int pmod(int a, int n) { return ((a % n) + n) % n; }
// Python round(): half to even
uint32_t pyround(double v) { return static_cast<uint32_t>(std::nearbyint(v)); }

// Hops from src to dst along one NoC torus: NOC0 travels +x then +y, NOC1 -x then -y (both wrap).
int noc_hops(const Core& src, const Core& dst, int noc) {
    if (noc == 0) {
        return pmod(int(dst.x) - int(src.x), NOC_X) + pmod(int(dst.y) - int(src.y), NOC_Y);
    }
    return pmod(int(src.x) - int(dst.x), NOC_X) + pmod(int(src.y) - int(dst.y), NOC_Y);
}

// (DST tiles the gate/up sub-block may use, row passes?): fp32 DEST holds 4 tiles; row passes (NP <= 2, the x ring
// holds a sub-block's K-blocks) keep the 8-tile rows. With 16-column units (units) a block of 3 / 4 pairs' width runs
// the sub-block in a full-sync fp32 DST (8 tiles: one row tile of up to 8 tiles).
std::pair<uint32_t, bool> gu_dst(uint32_t width_pairs, uint32_t hidden_tiles, uint32_t x_slots, bool units = false) {
    // (width_pairs: the gate/up width in pairs; a units width w is w / 2 rounded up)
    if (width_pairs > 2 && 2 * width_pairs <= 8 && units && hidden_tiles / kKBlk <= x_slots) {
        return {8, true};
    }
    const bool row_passes = width_pairs <= 2 && hidden_tiles / kKBlk <= x_slots;
    return row_passes ? std::pair<uint32_t, bool>{8, true} : std::pair<uint32_t, bool>{4, false};
}

// Greedy nearest (by `noc` hops) path through `cores` (down indices), cut into n chains.
std::vector<std::vector<uint32_t>> chains(
    const std::vector<uint32_t>& down_indices,
    const std::vector<Core>& down,
    const std::function<Core(const Core&)>& phys,
    uint32_t n,
    int noc) {
    std::vector<uint32_t> left = down_indices, path;
    auto first = std::min_element(left.begin(), left.end(), [&](uint32_t a, uint32_t b) {
        const auto key_a = noc == 0 ? std::make_pair(int(down[a].y), int(down[a].x))
                                    : std::make_pair(-int(down[a].y), -int(down[a].x));
        const auto key_b = noc == 0 ? std::make_pair(int(down[b].y), int(down[b].x))
                                    : std::make_pair(-int(down[b].y), -int(down[b].x));
        return key_a < key_b;
    });
    path.push_back(*first);
    left.erase(first);
    while (!left.empty()) {
        const Core cur = phys(down[path.back()]);
        auto nearest = std::min_element(left.begin(), left.end(), [&](uint32_t a, uint32_t b) {
            return noc_hops(cur, phys(down[a]), noc) < noc_hops(cur, phys(down[b]), noc);
        });
        path.push_back(*nearest);
        left.erase(nearest);
    }
    const uint32_t L = path.size() / n;
    std::vector<std::vector<uint32_t>> out;
    for (uint32_t i = 0; i < n; ++i) {
        const auto b = path.begin() + i * L;
        out.emplace_back(b, i + 1 < n ? b + L : path.end());
    }
    return out;
}
}  // namespace plan_detail
using namespace plan_detail;

uint32_t FlatRoutedExpertPlan::down_k_block_tiles(uint32_t columns, uint32_t intermediate_tiles) {
    for (uint32_t k : {8u, 4u, 2u, 1u}) {
        if (k * columns <= 16 && intermediate_tiles % k == 0) {
            return k;
        }
    }
    TT_THROW("flat_routed_expert: no down K-block for {} columns", columns);
}

uint32_t FlatRoutedExpertPlan::rectangle_of(const Core& c) const {
    for (uint32_t i = 0; i < gate_up_rectangles.size(); ++i) {
        const auto [first_x, last_x, first_y, last_y] = gate_up_rectangles[i];
        if (first_x <= c.x && c.x <= last_x && first_y <= c.y && c.y <= last_y) {
            return i;
        }
    }
    TT_THROW("flat_routed_expert: core ({}, {}) in no gate/up rectangle", c.x, c.y);
}

std::vector<Core> FlatRoutedExpertPlan::arena_cores() const {
    std::vector<Core> out = gate_up_cores;
    for (const auto* l : {&down_cores, &relay_cores, &idle_gate_up_cores}) {
        out.insert(out.end(), l->begin(), l->end());
    }
    return out;
}

std::vector<Core> FlatRoutedExpertPlan::work_cores() const {
    std::vector<Core> out = gate_up_cores;
    for (const auto* l : {&down_cores, &relay_cores}) {
        out.insert(out.end(), l->begin(), l->end());
    }
    return out;
}

std::vector<FlatRoutedExpertPlan::FreeL1> FlatRoutedExpertPlan::free_l1(uint32_t arena_end) const {
    auto align_64 = [](uint32_t x) { return (x + 63u) & ~63u; };
    const uint32_t tail = align_64(dynamic_counts_offset + dynamic_counts_bytes);
    std::vector<FreeL1> out;
    auto add = [&](std::vector<Core> cores, std::vector<std::pair<uint32_t, uint32_t>> spans) {
        std::vector<std::pair<uint32_t, uint32_t>> ok;
        for (auto [b, e] : spans) {
            b = align_64(b);
            if (e > b) {
                ok.emplace_back(b, e);
            }
        }
        if (!cores.empty() && !ok.empty()) {
            out.push_back({std::move(cores), std::move(ok)});
        }
    };
    const uint32_t primary_relays = static_cast<uint32_t>(
        gate_up_rectangles.size());  // (relay_cores 0 .. primary_relays - 1: the primaries, then helpers)
    add(gate_up_cores, {{gate_up_arena_bytes, schedule_page_offset}, {tail, arena_end}});
    add(down_cores, {{tail, arena_end}});
    add(std::vector<Core>(
            relay_cores.begin(), relay_cores.begin() + std::min<size_t>(primary_relays, relay_cores.size())),
        {{0, relay_landing_offset}, {relay_arena_bytes, schedule_page_offset}, {tail, arena_end}});
    if (relay_cores.size() > primary_relays) {
        add(std::vector<Core>(relay_cores.begin() + primary_relays, relay_cores.end()),
            {{relay_landing_offset, schedule_page_offset}, {tail, arena_end}});
    }
    add(idle_gate_up_cores, {{0, arena_end}});
    return out;
}

tt::tt_metal::CoreRangeSet rect_ranges(const std::vector<Core>& cores) {
    std::set<std::pair<uint32_t, uint32_t>> left;
    for (const auto& c : cores) {
        left.insert({c.x, c.y});
    }
    std::vector<std::pair<uint32_t, uint32_t>> order(left.begin(), left.end());
    std::sort(order.begin(), order.end(), [](auto a, auto b) {
        return std::make_pair(a.second, a.first) < std::make_pair(b.second, b.first);
    });
    std::vector<tt::tt_metal::CoreRange> rects;
    for (const auto& [first_x, first_y] : order) {
        if (!left.contains({first_x, first_y})) {
            continue;
        }
        uint32_t last_x = first_x, last_y = first_y;
        while (left.contains({last_x + 1, first_y})) {
            ++last_x;
        }
        auto row_full = [&](uint32_t y) {
            for (uint32_t x = first_x; x <= last_x; ++x) {
                if (!left.contains({x, y})) {
                    return false;
                }
            }
            return true;
        };
        while (row_full(last_y + 1)) {
            ++last_y;
        }
        for (uint32_t x = first_x; x <= last_x; ++x) {
            for (uint32_t y = first_y; y <= last_y; ++y) {
                left.erase({x, y});
            }
        }
        rects.emplace_back(Core(first_x, first_y), Core(last_x, last_y));
    }
    return tt::tt_metal::CoreRangeSet(rects);
}

namespace plan_detail {
// The plan at at most mt_max row tiles a sub-block and hbuf_max down-core h buffers, its arena under `budget` bytes a
// core where it can (the caller checks the arena against the budget).
FlatRoutedExpertPlan make_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const FlatRoutedExpertConfig& config,
    uint32_t budget,
    uint32_t mt_max,
    uint32_t hbuf_max) {
    auto* device = &mesh_device;
    FlatRoutedExpertPlan p;
    p.hidden = config.hidden;
    p.intermediate = config.intermediate;
    p.experts_per_chip = config.experts_per_chip;
    p.global_experts = config.num_global_experts;
    p.max_tokens_per_expert = config.max_tokens;
    TT_FATAL(
        p.hidden % 256 == 0 && p.intermediate % 32 == 0 && p.max_tokens_per_expert % 32 == 0,
        "flat_routed_expert: H % 256, I % 32, m % 32");
    TT_FATAL(
        p.max_tokens_per_expert >= 256,
        "flat_routed_expert: the op is built for >= 256 tokens per expert (helper relays)");
    TT_FATAL(p.experts_per_chip >= 1 && p.experts_per_chip <= 64, "flat_routed_expert: 1..64 local experts");
    p.hidden_tiles = p.hidden / 32;
    p.intermediate_tiles = p.intermediate / 32;
    p.weight_tile_bytes = config.weights_bf8 ? kBf8Tile : 576;
    p.x_tile_bytes = config.layout.x_tile();  // (x after the tilize: bfp8, or bf16 with layout.x_bf16)
    p.h_tile_bytes = config.layout.h_tile();  // (h after gate/up: bfp8, or bf16 with layout.h_bf16)
    p.dram_banks = device->dram_grid_size().x;
    const auto phys = [device](const Core& c) { return device->worker_core_from_logical_core(c); };

    const auto grid = block_grid(device->compute_with_storage_grid_size());
    TT_FATAL(grid.y >= 10, "flat_routed_expert: laid out for a 10-row Blackhole grid (got {} rows)", grid.y);
    constexpr uint32_t row0 = 0, row1 = 9, nrows = 10;
    // The banks' anchors: each DRAM bank's optimal worker core, in columns 0 and E (E = 6 on a p150's 11 x 10, 7 on
    // a Galaxy chip's 12 x 10). Queried before the rectangles are sized: the rectangles stay off the anchor columns
    // and the columns east of them (1, E + 1: the x relays).
    std::vector<Core> bank_anchors;
    {
        const auto by_bank = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(
            tt::tt_metal::NOC::NOC_0, flat_reference_coord(mesh_device));
        for (uint32_t b = 0; b < by_bank.size(); ++b) {
            bank_anchors.push_back(by_bank.at(b));
        }
    }
    uint32_t east0 = 0;
    for (const auto& c : bank_anchors) {
        east0 = std::max<uint32_t>(east0, c.x);
    }
    TT_FATAL(east0 >= 6, "flat_routed_expert: east bank anchors in column {} (expected >= 6)", east0);
    // The gate/up rectangles: the fixed 4 x 10 west (columns 2..5) and 3 x 8 east, or (shaped below) the columns
    // between and beyond the anchor / relay column pairs: west from column 2 up to E - 1, east from E + 2 to the
    // grid's edge (at most 4 wide).
    const uint32_t a_x0 = 2u, b_x0 = east0 + 2;
    TT_FATAL(
        grid.x > b_x0,
        "flat_routed_expert: no column east of the east anchor and relay columns on a {}-wide grid",
        grid.x);
    const uint32_t hb_max = std::min(nrows, 8u);
    p.gate_up_rectangles = {{2, 5, row0, row1}, {8, 10, row0, row0 + hb_max - 1}};
    // even split: the 2 It weight tiles a K row ([gate | up] 16-column units) over at most 64 gate/up cores, the fewest
    // tiles a core that divide them evenly over a multiple of the DRAM banks (one weight slot a bank: every bank holds
    // and serves the same share of each expert's gate/up weights); whole gate / up tile pairs (NP = tiles / 2) where
    // they give the same tiles a core, else 16-column units. GLM-5.3 / -Flash, K2.7: 64 cores x 2 (pairs); K3, M3
    // (I 3072): 64 x 3 units; V4.1-Flash (I 2304): 48 x 3 units (64 cores would leave the slowest with 3 as well);
    // I 640: 40 x 1 unit
    bool found = false;
    const uint32_t tiles = 2 * p.intermediate_tiles, slots = p.dram_banks;
    TT_FATAL(slots >= 1, "flat_routed_expert: no DRAM banks");
    for (uint32_t tiles_per_core = (tiles + 63) / 64; tiles_per_core <= 8 && !found; ++tiles_per_core) {
        if (tiles % tiles_per_core != 0 || (tiles / tiles_per_core) % slots != 0) {
            continue;
        }
        const uint32_t pairs_per_core = tiles_per_core / 2;
        const bool pairs = tiles_per_core % 2 == 0 && p.intermediate_tiles % pairs_per_core == 0 &&
                           (p.intermediate_tiles / pairs_per_core) % slots == 0;
        for (uint32_t x_ring_blocks : {24u, 16u, 12u}) {
            const uint32_t dst = pairs ? gu_dst(pairs_per_core, p.hidden_tiles, x_ring_blocks).first
                                       : gu_dst((tiles_per_core + 1) / 2, p.hidden_tiles, x_ring_blocks, true).first;
            const uint32_t row_tiles = std::min(dst / tiles_per_core, mt_max);
            // (a gate/up core's arena: the weight ring and the x ring)
            if (row_tiles >= 1 && 2 * p.hidden_tiles * tiles_per_core * p.weight_tile_bytes +
                                          x_ring_blocks * row_tiles * kKBlk * p.x_tile_bytes <=
                                      budget) {
                p.gate_up_pairs_per_core = pairs ? pairs_per_core : 0;
                p.units_per_core = pairs ? 0 : tiles_per_core;
                p.half_width_gate_up = !pairs;
                p.x_ring_blocks = x_ring_blocks;
                found = true;
                break;
            }
        }
    }
    TT_FATAL(
        found,
        "flat_routed_expert: no even gate/up split for I {} ({} weight tiles a K row over at most 64 cores, a multiple "
        "of the {} DRAM banks)",
        p.intermediate,
        2 * p.intermediate_tiles,
        slots);
    p.gate_up_weight_tiles_per_row = p.units_per_core ? p.units_per_core : 2 * p.gate_up_pairs_per_core;
    p.h_tiles_per_core = p.units_per_core ? (p.units_per_core + 1) / 2 : p.gate_up_pairs_per_core;
    p.column_groups = p.units_per_core ? 2 * p.intermediate_tiles / p.units_per_core
                                       : p.intermediate_tiles / p.gate_up_pairs_per_core;
    // A split of fewer gate/up cores than the fixed rectangles' 64 (V4.1-Flash: 48), or units not exactly 64: shape the
    // rectangles to the split, so the cores left over do down work
    const bool auto_shape = p.column_groups < 64 || (p.units_per_core && p.column_groups != 64);
    if (auto_shape) {  // rectangles of exactly the split's cores (west_width columns x west_height + east_width columns
                       // x east_height)
        const uint32_t max_west_width = east0 - a_x0, max_east_width = std::min<uint32_t>(4u, uint32_t(grid.x) - b_x0);
        const uint32_t want = p.column_groups;
        bool shaped = false;
        int best = 1 << 30;
        for (uint32_t west_width = 4; west_width <= max_west_width; ++west_width) {
            for (uint32_t east_width = 1; east_width <= max_east_width; ++east_width) {
                for (uint32_t east_height = 1; east_height <= hb_max; ++east_height) {
                    if (want < east_width * east_height || (want - east_width * east_height) % west_width ||
                        (want - east_width * east_height) / west_width > nrows ||
                        want - east_width * east_height == 0) {
                        continue;
                    }
                    const int west_height = (want - east_width * east_height) / west_width;
                    const int imbalance = std::abs(int(west_width) * west_height - int(east_width * east_height));
                    if (imbalance < best) {
                        best = imbalance;
                        // west anchored against the east anchor column; east in p150 columns (shifted by E - 6 below)
                        p.gate_up_rectangles = {
                            {east0 - west_width, east0 - 1, row0, row0 + west_height - 1},
                            {b_x0 - (east0 - 6), b_x0 - (east0 - 6) + east_width - 1, row0, row0 + east_height - 1}};
                        shaped = true;
                    }
                }
            }
        }
        TT_FATAL(shaped, "flat_routed_expert: no rectangle pair of {} gate/up cores in {} rows", want, nrows);
    }
    std::tie(p.gate_up_dst_tiles, p.gate_up_row_passes) =
        gu_dst((p.gate_up_weight_tiles_per_row + 1) / 2, p.hidden_tiles, p.x_ring_blocks, p.units_per_core != 0);
    const uint32_t mt_cap = std::min(p.gate_up_dst_tiles / p.gate_up_weight_tiles_per_row, MT_MAX);
    p.row_tiles_per_sub_block = std::min({mt_cap, mt_max, std::max(1u, p.max_tokens_per_expert / 32)});
    // the pieces a gate/up weight block is read as: the most (1 / 2 / 4) of at least 6 KB each. bfp4 16-column-unit
    // blocks of 3 tiles a row (13.5 KB: K3, V4.1-Flash, M3) as 2 pieces, pair blocks (9 KB: GLM-5.3 / -Flash, K2.7)
    // whole (flat us at Galaxy loads, 1 / 2 / 4 pieces: K3 1328.3 / 1294.1 / 1347.5, V4.1-Flash 634.6 / 609.3 / -,
    // GLM-5.3 532.0 / 569.1 / 596.3, GLM-5.3-Flash 407.4 / 435.6 / -)
    const uint32_t block_bytes = kKBlk * p.gate_up_weight_tiles_per_row * p.weight_tile_bytes;
    p.gate_up_block_pieces = 1;
    for (uint32_t k : {4u, 2u}) {
        if (block_bytes / k >= 6 * 1024 && (kKBlk * p.gate_up_weight_tiles_per_row) % k == 0) {
            p.gate_up_block_pieces = k;
            break;
        }
    }
    // wide gate/up blocks (gwt > 4) with row passes: gate/up in a full-sync DST
    p.gate_up_full_sync = p.gate_up_weight_tiles_per_row > 4 && p.gate_up_row_passes &&
                          p.row_tiles_per_sub_block * p.gate_up_weight_tiles_per_row > 4;
    TT_FATAL(
        p.row_tiles_per_sub_block * p.gate_up_weight_tiles_per_row <= p.gate_up_dst_tiles,
        "flat_routed_expert: sub-block split");
    const uint32_t m_pad = (p.max_tokens_per_expert + p.row_tiles_per_sub_block * 32 - 1) /
                           (p.row_tiles_per_sub_block * 32) * p.row_tiles_per_sub_block * 32;
    p.helper_relays_per_primary = 1;
    p.relay_landing_slots = 2;  // (two landing rings a primary: the helpers feed both)
    // the down weight ring: two whole experts with pinning (the pinned schedule's regions), else 1.5
    p.down_weight_ring_experts = config.pin ? 2.0f : 1.5f;

    // ---- layout ----
    p.bank_anchor_cores = bank_anchors;
    std::set<uint32_t> rcols;
    for (const auto& c : p.bank_anchor_cores) {
        rcols.insert(c.x);
    }
    // The bank anchors in a west column 0 and an east column E (Blackhole p150 11 x 10: E = 6; a Galaxy chip's
    // 12 x 10, one column less harvested: E = 7), the x relays in the columns east of them (1, E + 1). The layout is
    // written for E = 6 and moved east by E - 6: the east gate/up rectangle and the east relays; a wider west block
    // leaves its extra column to the down cores (as are all columns east of the east rectangle).
    const uint32_t east = rcols.size() == 2 ? *rcols.rbegin() : 0;
    TT_FATAL(
        // (bring-up FIX.15, set_max_grid: the 12-column room is the fixed rectangles' - a shaped plan's east rectangle
        // is sized to the columns left, and every rectangle is checked to fit below)
        rcols.size() == 2 && *rcols.begin() == 0 && east >= 6 && (auto_shape || grid.x >= 11 + (east - 6)),
        "flat_routed_expert: laid out for Blackhole grids with the DRAM banks' anchor cores in columns 0 and E "
        "(E >= 6) and room for the east rectangle; got a {} x {} grid with anchors in columns {}",
        grid.x,
        grid.y,
        [&] {  // (as text: a vector has no fmt formatter in every unity batch)
            std::string cols;
            for (const auto c : rcols) {
                cols += (cols.empty() ? "" : ", ") + std::to_string(c);
            }
            return cols;
        }());
    const uint32_t shift = east - 6;
    for (auto& [first_x, last_x, first_y, last_y] : p.gate_up_rectangles) {
        if (first_x > 6) {  // the east rectangle
            first_x += shift;
            last_x += shift;
        }
    }
    for (const auto& [first_x, last_x, first_y, last_y] : p.gate_up_rectangles) {  // (bring-up FIX.15)
        TT_FATAL(
            last_x < grid.x && last_y < grid.y,
            "flat_routed_expert: a gate/up rectangle (columns {}..{}, rows {}..{}) outside the {} x {} grid",
            first_x,
            last_x,
            first_y,
            last_y,
            grid.x,
            grid.y);
    }
    std::vector<Core> gu_all;
    for (const auto& [first_x, last_x, first_y, last_y] : p.gate_up_rectangles) {
        for (uint32_t y = first_y; y <= last_y; ++y) {
            for (uint32_t x = first_x; x <= last_x; ++x) {
                gu_all.emplace_back(x, y);
            }
        }
    }
    std::set<std::pair<uint32_t, uint32_t>> taken;
    for (const auto& c : gu_all) {
        taken.insert({c.x, c.y});
    }
    auto first_free = [&](uint32_t x, std::initializer_list<uint32_t> preferred_rows) {
        for (uint32_t y : preferred_rows) {
            if (y >= row0 && y <= row1 && !taken.contains({x, y})) {
                return Core(x, y);
            }
        }
        for (uint32_t y = row0; y <= row1; ++y) {
            if (!taken.contains({x, y})) {
                return Core(x, y);
            }
        }
        // the column is full: the free core nearest to it
        int best = 1 << 30;
        Core best_core(0, 0);
        for (uint32_t y = row0; y <= row1; ++y) {
            for (uint32_t column_x = 0; column_x < grid.x; ++column_x) {
                const int d = 2 * std::abs(int(column_x) - int(x)) + std::abs(int(y) - int(row0 + row1) / 2);
                if (!taken.contains({column_x, y}) && d < best) {
                    best = d;
                    best_core = Core(column_x, y);
                }
            }
        }
        TT_FATAL(best < (1 << 30), "flat_routed_expert: no free relay core near column {}", x);
        return best_core;
    };
    const uint32_t rx_w = 1, rx_e = east + 1;
    p.relay_cores = {first_free(rx_w, {4, 5, 3, 6, 2, 7}), first_free(rx_e, {3, 4, 2, 5, 1, 6})};
    for (const auto& c : p.relay_cores) {
        taken.insert({c.x, c.y});
    }
    {  // the helpers: relay NR + k is rectangle k's
        Core a = first_free(rx_w, {5, 3, 6, 2, 7, 1, 8, 0, 9});
        Core b = first_free(rx_e, {4, 2, 5, 1, 6, 0, 7, 8, 9});
        p.relay_cores.push_back(a);
        p.relay_cores.push_back(b);
        taken.insert({a.x, a.y});
        taken.insert({b.x, b.y});
    }
    for (uint32_t y = row0; y <= row1; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            if (!taken.contains({x, y})) {
                p.down_cores.emplace_back(x, y);
            }
        }
    }
    {  // the anchor cores and their east neighbours last: a down-core cap takes the other cores first
        std::set<std::pair<uint32_t, uint32_t>> anchors_and_east_neighbours;
        for (const auto& c : p.bank_anchor_cores) {
            anchors_and_east_neighbours.insert({c.x, c.y});
            anchors_and_east_neighbours.insert({c.x + 1, c.y});
        }
        std::stable_partition(p.down_cores.begin(), p.down_cores.end(), [&](const Core& c) {
            return !anchors_and_east_neighbours.contains({c.x, c.y});
        });
    }
    uint32_t down_core_count = p.down_cores.size();
    const uint32_t n_slots = p.dram_banks;  // (weight slot r: bank r)
    p.gate_up_weight_slots = n_slots;
    p.sub_blocks_per_expert = m_pad / (p.row_tiles_per_sub_block * 32);
    p.sub_blocks_per_chip = p.experts_per_chip * p.sub_blocks_per_expert;
    p.gate_up_k_blocks = p.hidden_tiles / kKBlk;
    p.weight_block_tiles = kKBlk * p.gate_up_weight_tiles_per_row;
    p.column_groups_per_weight_slot = p.column_groups / p.gate_up_weight_slots;
    p.gate_up_cores_per_weight_slot = p.column_groups_per_weight_slot;
    TT_FATAL(
        p.column_groups % p.gate_up_weight_slots == 0,
        "flat_routed_expert: {} gate/up column groups over {} weight slots",
        p.column_groups,
        p.gate_up_weight_slots);
    // gate/up weight ring: 2 expert slices (with pinning: the pinned expert's and one cycling)
    p.gate_up_weight_ring_blocks = 2 * p.gate_up_k_blocks;
    // each gate/up core writes its h slice to every chain head, the chains forward it
    p.down_chains = D_CHAINS;
    p.down_output_columns = p.hidden_tiles;
    // even split: the output tile columns over at most 32 down cores, the fewest cores with the fewest columns a core
    // (GLM-5.3 / M3 32 x 6, K2.7 32 x 7, GLM-5.3-Flash 32 x 4, V4.1-Flash 32 x 5, K3 28 x 4; K3 at M 2048, us per
    // expert: 26 down 5 columns max 409, 28 / 30 at 4 384, 32 at 4 400); the rest of the cores are left to the combine
    if (!p.down_cores.empty()) {
        const uint32_t n = std::min<uint32_t>(32u, p.down_cores.size());
        const uint32_t columns_per_core = (p.down_output_columns + n - 1) / n;
        p.down_cores.resize(
            std::min<size_t>(p.down_cores.size(), (p.down_output_columns + columns_per_core - 1) / columns_per_core));
        down_core_count = p.down_cores.size();
    }
    if (down_core_count > p.down_output_columns) {
        // more down cores than output tile columns (small H on a wide grid): keep the first rem_cols (ordered nearest
        // first); the rest stay idle
        p.down_cores.resize(p.down_output_columns);
        down_core_count = p.down_cores.size();
    }
    const uint32_t base_p = p.down_output_columns / down_core_count, extra = p.down_output_columns % down_core_count;
    p.down_columns_per_core.resize(down_core_count);
    p.down_first_columns.resize(down_core_count);
    for (uint32_t d = 0; d < down_core_count; ++d) {
        p.down_columns_per_core[d] = base_p + (d < extra ? 1 : 0);
    }
    for (uint32_t d = 0, first_column = 0; d < down_core_count; ++d) {
        p.down_first_columns[d] = first_column;
        first_column += p.down_columns_per_core[d];
    }
    const uint32_t max_down_columns = *std::max_element(p.down_columns_per_core.begin(), p.down_columns_per_core.end());
    TT_FATAL(
        max_down_columns <= 16 &&
            *std::min_element(p.down_columns_per_core.begin(), p.down_columns_per_core.end()) >= 1,
        "flat_routed_expert: down columns");
    const uint32_t x_bytes = p.row_tiles_per_sub_block * kKBlk * p.x_tile_bytes;
    p.down_h_buffer_tiles = p.intermediate_tiles * p.row_tiles_per_sub_block;  // one down h buffer: a sub-block
    const uint32_t out_tiles = p.row_tiles_per_sub_block * max_down_columns;

    // gate/up cores per weight slot (every gate/up core reads its own blocks on NOC0): each slot's cores by the NOC0
    // hops from its bank's anchor (the bank's optimal worker core, next to its endpoint) to the core, the direction its
    // reads' responses travel (exact on this chip: harvesting- and routing-aware); all pairs (slot, core) taken nearest
    // first rather than round-robin nearest (which left the last slots 15-21 hops)
    std::vector<std::vector<uint32_t>> per_slot(n_slots);
    std::vector<bool> placed(gu_all.size(), false);
    {
        const auto coord = flat_reference_coord(mesh_device);
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> pairs;  // (hops, slot, core)
        for (uint32_t r = 0; r < n_slots; ++r) {
            const Core anchor = bank_anchors.at(r);
            for (uint32_t core_index = 0; core_index < gu_all.size(); ++core_index) {
                pairs.emplace_back(
                    tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                        &mesh_device, coord, anchor, gu_all[core_index], tt::tt_metal::NOC::NOC_0),
                    r,
                    core_index);
            }
        }
        std::sort(pairs.begin(), pairs.end());
        for (const auto& [h, r, core_index] : pairs) {
            if (!placed[core_index] && per_slot[r].size() < p.gate_up_cores_per_weight_slot) {
                per_slot[r].push_back(core_index);
                placed[core_index] = true;
            }
        }
    }
    for (uint32_t r = 0; r < n_slots; ++r) {
        for (uint32_t core_index : per_slot[r]) {
            p.gate_up_cores.push_back(gu_all[core_index]);
        }
    }
    for (uint32_t core_index = 0; core_index < gu_all.size(); ++core_index) {
        if (!placed[core_index]) {
            p.idle_gate_up_cores.push_back(gu_all[core_index]);
        }
    }

    // ---- arena (per-role layout, 2 KB aligned) ----
    p.x_ring_offset = al(p.gate_up_weight_ring_blocks * p.weight_block_tiles * p.weight_tile_bytes);
    p.h_local_offset =
        p.x_ring_offset + al(p.x_ring_blocks * x_bytes);  // gate/up: weight ring, x ring, then h_local and the pack CB
    uint32_t dn_ring_max = 0;
    for (uint32_t d = 0; d < down_core_count; ++d) {
        const uint32_t down_k_tiles =
            FlatRoutedExpertPlan::down_k_block_tiles(p.down_columns_per_core[d], p.intermediate_tiles);
        dn_ring_max = std::max(
            dn_ring_max,
            pyround(p.down_weight_ring_experts * (p.intermediate_tiles / down_k_tiles)) * down_k_tiles *
                p.down_columns_per_core[d]);
    }
    p.down_h_offset = al(dn_ring_max * p.weight_tile_bytes);
    const uint32_t out_bytes = al(2 * out_tiles * kBf8Tile);
    uint32_t dyn_half = 512;  // as the program factory: CB 7 holds 4 halves of the counts / regions / ids
    while (dyn_half < (p.global_experts * 4 + 63) / 64 * 64 + (4 * p.experts_per_chip + 63) / 64 * 64) {
        dyn_half *= 2;
    }
    p.down_h_buffers =
        hbuf_max >= 3 && p.down_h_offset + al(3 * p.down_h_buffer_tiles * p.h_tile_bytes) + out_bytes + 2048 <= budget
            ? 3
            : 2;
    p.down_output_offset = p.down_h_offset + al(p.down_h_buffers * p.down_h_buffer_tiles * p.h_tile_bytes);
    const uint32_t dn_bytes = std::min(p.down_output_offset + out_bytes + 2048, budget);
    TT_FATAL(
        p.down_output_offset + al(max_down_columns * 2048) + 2048 <= budget,
        "flat_routed_expert: down core arena does not fit");
    // gate/up: h_local (CB 3: hbuf x MT x hpc h tiles) and the pack CB (16: one bf16 tile) after the x ring
    p.pack_cb_offset =
        p.h_local_offset + al(p.down_h_buffers * p.row_tiles_per_sub_block * p.h_tiles_per_core * p.h_tile_bytes);
    const uint32_t gu_bytes = p.pack_cb_offset + al(2048);
    {  // (CBs 6 / 7, as the program factory sizes them: schedule_page_bytes, 4 x schedule_scratch_bytes)
        const uint32_t se_max_e = p.experts_per_chip <= 16 ? 16 : (p.experts_per_chip <= 32 ? 32 : 64);
        p.schedule_page_bytes = se_max_e == 16 ? 512 : (se_max_e == 32 ? 1024 : 2048);
        p.dynamic_counts_bytes = 4 * dyn_half;
        p.static_l1_bytes = 0;  // (every CB is in the arena)
    }
    TT_FATAL(gu_bytes <= budget, "flat_routed_expert: gate/up core arena does not fit");
    for (uint32_t t : {32u, 16u, 8u}) {
        if (p.hidden_tiles % t == 0 && t % kKBlk == 0) {
            p.super_block_tiles = t;
            break;
        }
    }
    p.row_segment_bytes = p.super_block_tiles * 64;
    p.super_blocks = p.hidden_tiles / p.super_block_tiles;
    p.super_block_offset = al(kRmChunks * 32 * p.row_segment_bytes);
    p.relay_landing_offset =
        p.super_block_offset + al(kSbSlots * p.row_tiles_per_sub_block * p.super_block_tiles * p.x_tile_bytes);
    // x read once by the helpers: two rectangles full of gate/up cores on the snakes, one helper each - each helper
    // reads / tilizes half of x (super-blocks g % 2 == k) and writes every super-block into both primaries' landing
    // rings (2 slots each); the primaries read nothing and feed only their own snake. x leaves DRAM once and no relay
    // sends more than a primary did before (GLM-5.3 / GLM-Flash / K3 / K2.7 at Galaxy loads 522 / 380 / 1216 / 807 ->
    // 483 / 355 / 1181 / 749 us, never slower at M 32..1024 or under six routings)
    for (uint32_t k = 0; k < 2; ++k) {
        const auto [first_x, last_x, first_y, last_y] = p.gate_up_rectangles[k];
        uint32_t n = 0;
        for (const auto& c : p.gate_up_cores) {
            n += p.rectangle_of(c) == k;
        }
        TT_FATAL(
            n == (last_x - first_x + 1) * (last_y - first_y + 1),
            "flat_routed_expert: gate/up rectangle {} is not full ({} of {} cores): the x snakes need full rectangles",
            k,
            n,
            (last_x - first_x + 1) * (last_y - first_y + 1));
    }
    const uint32_t relay_bytes = p.relay_landing_offset + al(2 * p.relay_landing_slots * p.row_tiles_per_sub_block *
                                                             p.super_block_tiles * p.x_tile_bytes);
    TT_FATAL(relay_bytes <= budget, "flat_routed_expert: relay arena does not fit");
    // CBs 6 / 7 at one offset on every arena core, past every role's bytes (in arena_tiles: the caller's fit to the
    // budget counts them)
    p.schedule_page_offset = al(std::max({gu_bytes, dn_bytes, relay_bytes}));
    p.dynamic_counts_offset = p.schedule_page_offset + al(p.schedule_page_bytes);
    p.arena_tiles = (p.dynamic_counts_offset + al(p.dynamic_counts_bytes) + 2047) / 2048;
    p.gate_up_arena_bytes = gu_bytes;
    p.down_arena_bytes = dn_bytes;
    p.relay_arena_bytes = relay_bytes;
    p.relay_stride = 1 + p.helper_relays_per_primary;
    p.gate_up_weight_region_bytes = p.experts_per_chip * p.gate_up_k_blocks * p.column_groups_per_weight_slot *
                                    p.weight_block_tiles * p.weight_tile_bytes;
    // The down cores' weight reads rotate over the DRAM banks instead of one bank per core (bank-pinned streams ran
    // at 55% of DRAM peak alone, rotating 65%). Bit-identical. Kimi K2.7 / GLM 5.3 us per expert at M 32 / 128: 62.3 /
    // 72.3 -> 58.2 / 68.7, 54.3 / 63.9 -> 51.2 / 61.4; none at M >= 512.
    {
        uint32_t bank_share_tiles = 0;  // tiles: a core's share of one bank, the widest core's
        for (uint32_t d = 0; d < p.down_cores.size(); ++d) {
            const uint32_t down_k_tiles = FlatRoutedExpertPlan::down_k_block_tiles(
                               p.down_columns_per_core[d], p.intermediate_tiles),
                           down_blocks = p.experts_per_chip * (p.intermediate_tiles / down_k_tiles);
            bank_share_tiles = std::max(
                bank_share_tiles,
                (down_blocks + p.dram_banks - 1) / p.dram_banks * down_k_tiles * p.down_columns_per_core[d]);
        }
        p.down_weight_region_bytes = bank_share_tiles * p.weight_tile_bytes;
    }
    p.coordinator_cores.push_back(p.down_cores[0]);

    // ---- down chains: D_CH chains through the down cores ----
    {
        std::vector<uint32_t> down_indices(down_core_count);
        for (uint32_t d = 0; d < down_core_count; ++d) {
            down_indices[d] = d;
        }
        for (const auto& chain : chains(down_indices, p.down_cores, phys, p.down_chains, 1)) {
            p.down_chain_heads.push_back(chain.front());
            for (size_t i = 0; i + 1 < chain.size(); ++i) {
                p.down_chain_successor[chain[i]] = chain[i + 1];
                p.down_chain_predecessor[chain[i + 1]] = chain[i];
            }
        }
    }
    return p;
}
}  // namespace plan_detail

FlatRoutedExpertPlan make_flat_routed_expert_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const FlatRoutedExpertConfig& config) {
    // The arena's budget a core: the allocator's L1 bank, or less (layout.arena_bytes: the MoE block's combine arena).
    // The plan fits it keeping the 128-row sub-blocks first - 3 down-core h buffers, then 2 - and only then halves the
    // sub-block rows (GLM-5.3 at Galaxy loads, flat us uniform / hot: 128 rows + 2 h buffers 531.5 / 540.2, 64 rows + 3
    // 532.0 / 563.2 - and 128 rows + 3 do not fit the block's arena at 8x4 chunk 5120)
    const uint32_t bank = static_cast<uint32_t>(mesh_device.allocator()->get_bank_size(tt::tt_metal::BufferType::L1));
    const int32_t want = config.layout.arena_bytes;
    const uint32_t budget = want > 0 ? std::min<uint32_t>(bank, static_cast<uint32_t>(want)) : bank;
    std::string tried;
    for (uint32_t row_tiles = MT_MAX; row_tiles >= 1; row_tiles /= 2) {
        for (uint32_t hbuf : {3u, 2u}) {
            try {
                auto p = make_plan(mesh_device, config, budget, row_tiles, hbuf);
                if (p.arena_tiles * 2048 <= budget) {
                    return p;
                }
                tried += fmt::format(
                    "{}{} rows {} h buffers: {} B",
                    tried.empty() ? "" : "; ",
                    p.row_tiles_per_sub_block * 32,
                    p.down_h_buffers,
                    p.arena_tiles * 2048);
                if (p.row_tiles_per_sub_block <
                    row_tiles) {  // (fewer rows than row_tiles: the rows a sub-block were already at their least)
                    row_tiles = p.row_tiles_per_sub_block;
                }
            } catch (const std::exception& e) {
                tried += fmt::format(
                    "{}{} rows {} h buffers: {}",
                    tried.empty() ? "" : "; ",
                    row_tiles * 32,
                    hbuf,
                    std::string(e.what()).substr(0, 160));
            }
        }
    }
    TT_THROW("flat_routed_expert: no plan fits the arena's {} B a core ({})", budget, tried);
}
std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const FlatRoutedExpertConfig& c) {
    // (the plan ignores the program-only fields: activation, y formats, fidelity, the combine overlap)
    using Key = std::
        tuple<int, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t, std::vector<uint32_t>>;
    static std::mutex cache_mutex;
    static std::map<Key, std::shared_ptr<const FlatRoutedExpertPlan>> cache;
    const Key key{
        mesh_device.id(),
        static_cast<uint32_t>(mesh_device.shape().mesh_size()),
        c.hidden,
        c.intermediate,
        c.experts_per_chip,
        c.num_global_experts,
        c.max_tokens,
        c.weights_bf8,
        c.pin,
        c.layout.key()};
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto plan = std::make_shared<const FlatRoutedExpertPlan>(make_flat_routed_expert_plan(mesh_device, c));
        log_debug(
            tt::LogOp,
            "flat_routed_expert plan ({}): gate/up row passes {} sub-block rows {}",
            c.layout.str(),
            plan->gate_up_row_passes,
            plan->row_tiles_per_sub_block * 32);
        it = cache.emplace(key, std::move(plan)).first;
    }
    return it->second;
}

}  // namespace ttnn::operations::bringup::moe_block::flat_routed_expert
