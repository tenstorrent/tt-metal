// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdlib>
#include "flat_routed_expert_plan.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <set>

#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::bringup::flat_routed_expert {

namespace plan_detail {
using Core = tt::tt_metal::CoreCoord;

constexpr uint32_t KBLK = kKBlk, MT_MAX = 4, BF8_TILE = kBf8Tile;
// L1 per core for the arena: the Blackhole L1 bank (1427 KB) less the program's static CBs (meta 2 KB, dynamic counts
// 4 x dyn_half, the gate/up 2 KB pack CB: one address range on every core) and the 2 KB done words and 2 KB relay
// scratch words (same address on every core), all below / beside the arena
constexpr uint32_t L1_BANK_BYTES = 1427 * 1024;
constexpr uint32_t GU_L1_BUDGET = 1400 * 1024;  // gate/up core: weight ring + x ring
constexpr uint32_t D_CHAINS = 7, RM_CHUNKS = kRmChunks;
constexpr int NOC_X = 17, NOC_Y = 12;  // Blackhole NoC torus

uint32_t al(uint32_t b) { return (b + 2047) / 2048 * 2048; }
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
// holds a sub-block's K-blocks) keep the 8-tile rows. l1acc (3 subgrids) keeps the bf16 8-tile DST half.
std::pair<uint32_t, bool> gu_dst(uint32_t np, uint32_t ht, uint32_t x_slots, bool fp32) {
    if (!fp32) {
        return {8, false};
    }
    const bool rp = np <= 2 && ht / KBLK <= x_slots;
    return rp ? std::pair<uint32_t, bool>{8, true} : std::pair<uint32_t, bool>{4, false};
}

// Gate/up split for It tile columns: (NP pairs per core, G M-groups); see flat_expert.py _gu_split.
std::optional<std::pair<uint32_t, uint32_t>> gu_split(
    uint32_t it,
    uint32_t ht,
    uint32_t w_tile,
    uint32_t x_slots,
    uint32_t n_readers,
    uint32_t max_cores,
    bool fp32,
    uint32_t x_tile = BF8_TILE) {
    struct Cand {
        uint32_t cores, np, g;
    };
    std::vector<Cand> cands;
    std::optional<std::tuple<int, int, int>> best_key;
    uint32_t best_np = 0, best_g = 0, best_cores = 0;
    for (uint32_t np : {1u, 2u, 3u, 4u}) {
        for (uint32_t g : {1u, 2u, 4u}) {
            const uint32_t ps = it / np;
            const uint32_t dst = gu_dst(np, ht, x_slots, fp32).first;
            if (it % np || ps % n_readers || ps * g > max_cores || std::min(dst / (2 * np), MT_MAX / g) < 1) {
                continue;
            }
            const uint32_t mt_ = g * std::min(dst / (2 * np), MT_MAX / g);
            if (2 * ht * 2 * np * w_tile + x_slots * mt_ * KBLK * x_tile > GU_L1_BUDGET) {
                continue;
            }
            cands.push_back({ps * g, np, g});
            const auto key = std::make_tuple(-int(g), int(ps * g), -int(np));
            if (!best_key || key > *best_key) {
                best_key = key;
                best_np = np;
                best_g = g;
                best_cores = ps * g;
            }
        }
    }
    if (!best_key) {
        return std::nullopt;
    }
    // fewer than 32 gate/up cores (TP4-like): 2 M-groups on twice the cores
    if (best_cores < 32 && max_cores == 64) {
        for (const auto& c : cands) {
            if (c.g == 2 && c.np == best_np && c.cores >= 2 * best_cores) {
                return std::make_pair(c.np, 2u);
            }
        }
    }
    return std::make_pair(best_np, best_g);
}

// Greedy nearest (by `noc` hops) path through `cores` (down indices), cut into n chains.
std::vector<std::vector<uint32_t>> chains(
    const std::vector<uint32_t>& ids,
    const std::vector<Core>& down,
    const std::function<Core(const Core&)>& phys,
    uint32_t n,
    int noc) {
    std::vector<uint32_t> left = ids, path;
    auto first = std::min_element(left.begin(), left.end(), [&](uint32_t a, uint32_t b) {
        const auto ka = noc == 0 ? std::make_pair(int(down[a].y), int(down[a].x))
                                 : std::make_pair(-int(down[a].y), -int(down[a].x));
        const auto kb = noc == 0 ? std::make_pair(int(down[b].y), int(down[b].x))
                                 : std::make_pair(-int(down[b].y), -int(down[b].x));
        return ka < kb;
    });
    path.push_back(*first);
    left.erase(first);
    while (!left.empty()) {
        const Core cur = phys(down[path.back()]);
        auto nxt = std::min_element(left.begin(), left.end(), [&](uint32_t a, uint32_t b) {
            return noc_hops(cur, phys(down[a]), noc) < noc_hops(cur, phys(down[b]), noc);
        });
        path.push_back(*nxt);
        left.erase(nxt);
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

uint32_t FlatRoutedExpertPlan::kd_of(uint32_t pcd, uint32_t it) {
    for (uint32_t k : {8u, 4u, 2u, 1u}) {
        if (k * pcd <= 16 && it % k == 0) {
            return k;
        }
    }
    TT_THROW("flat_routed_expert: no down K-block for {} columns", pcd);
}

uint32_t FlatRoutedExpertPlan::rect_of(const Core& c) const {
    for (uint32_t i = 0; i < rects.size(); ++i) {
        const auto [x0, x1, y0, y1] = rects[i];
        if (x0 <= c.x && c.x <= x1 && y0 <= c.y && c.y <= y1) {
            return i;
        }
    }
    TT_THROW("flat_routed_expert: core ({}, {}) in no gate/up rectangle", c.x, c.y);
}

std::vector<Core> FlatRoutedExpertPlan::arena_cores() const {
    std::vector<Core> out = gu;
    for (const auto* l : {&down, &relays, &readers, &gu_idle}) {
        out.insert(out.end(), l->begin(), l->end());
    }
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
    for (const auto& [x0, y0] : order) {
        if (!left.contains({x0, y0})) {
            continue;
        }
        uint32_t x1 = x0, y1 = y0;
        while (left.contains({x1 + 1, y0})) {
            ++x1;
        }
        auto row_full = [&](uint32_t y) {
            for (uint32_t x = x0; x <= x1; ++x) {
                if (!left.contains({x, y})) {
                    return false;
                }
            }
            return true;
        };
        while (row_full(y1 + 1)) {
            ++y1;
        }
        for (uint32_t x = x0; x <= x1; ++x) {
            for (uint32_t y = y0; y <= y1; ++y) {
                left.erase({x, y});
            }
        }
        rects.emplace_back(Core(x0, y0), Core(x1, y1));
    }
    return tt::tt_metal::CoreRangeSet(rects);
}

FlatRoutedExpertPlan make_flat_routed_expert_plan(tt::tt_metal::IDevice* device, const FlatRoutedExpertConfig& cfg) {
    FlatRoutedExpertPlan p;
    p.H = cfg.hidden;
    p.I = cfg.intermediate;
    p.E = cfg.experts_per_chip;
    p.NG = cfg.num_global_experts;
    p.m = cfg.max_tokens;
    TT_FATAL(p.H % 256 == 0 && p.I % 32 == 0 && p.m % 32 == 0, "flat_routed_expert: H % 256, I % 32, m % 32");
    TT_FATAL(p.m >= 256, "flat_routed_expert: the op is built for >= 256 tokens per expert (helper relays)");
    TT_FATAL(p.E >= 1 && p.E <= 64, "flat_routed_expert: 1..64 local experts");
    p.Ht = p.H / 32;
    p.It = p.I / 32;
    p.w_tile = cfg.weights_bf8 ? BF8_TILE : 576;
    if (cfg.x_bf16) {  // x as bf16 tiles (relay tilize, multicast, gate/up x ring)
        p.x_tile = 2048;
        p.sb_slots = 2;
    }
    if (cfg.h_bf16) {  // h as bf16 tiles (gate/up pack, h exchange, down h_all)
        p.h_tile = 2048;
    }
    p.banks = device->dram_grid_size().x;
    const auto phys = [device](const Core& c) { return device->worker_core_from_logical_core(c); };

    // subgrids: 2 for It <= 16 (TP4-like), 3 for It <= 32 (TP2-like), else 1
    p.nsg = p.It <= 16 ? 2 : (p.It <= 32 ? 3 : 1);
    p.n_rd_sg = p.nsg > 1 ? (p.nsg == 2 ? 8 : 4) : 16;
    // MIMO_FL_ROWS=y0,y1 (probe): plan on grid rows [y0, y1] only, leaving the rest of the chip to another op (e.g.
    // combine_fabric2d's senders / untilizers on rows 0-1, next to the eth cores). One subgrid only.
    // A grid of fewer than 10 rows (dispatch on a row: 12 x 9 on a p150) plans on all of its rows the same way.
    const uint32_t grid_rows = device->compute_with_storage_grid_size().y;
    uint32_t row0 = 0, row1 = std::min(grid_rows, 10u) - 1;
    const bool rows_cap = std::getenv("MIMO_FL_ROWS") != nullptr || grid_rows < 10;
    if (std::getenv("MIMO_FL_ROWS")) {
        TT_FATAL(
            std::sscanf(std::getenv("MIMO_FL_ROWS"), "%u,%u", &row0, &row1) == 2 && row0 <= row1 &&
                row1 < std::min(grid_rows, 10u),
            "MIMO_FL_ROWS: y0,y1");
    }
    if (rows_cap) {
        TT_FATAL(p.nsg == 1, "MIMO_FL_ROWS: one subgrid only (It {})", p.It);
    }
    const uint32_t nrows = row1 - row0 + 1;
    const auto grid = device->compute_with_storage_grid_size();
    // The DRAM-optimal readers (one per bank): columns 0 and E (E = 6 on a p150's 11 x 10, 7 on a Galaxy chip's
    // 12 x 10). Queried before the rectangles are sized, because with the rows capped they stretch to the columns the
    // readers leave. With dispatch on a row (12 x 9 on a p150) the driver's assignment names cores of the dispatch row
    // and throws; the p150's column-dispatch assignment is the same physical placement (rows 0-8 and the first 11
    // columns map identically), and the reader relocation below moves the ones on the missing row.
    std::vector<Core> opt;
    try {
        opt = device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);
    } catch (const std::exception&) {
        TT_FATAL(rows_cap, "flat_routed_expert: no DRAM-optimal worker assignment on a full-height grid");
        opt = {Core(0, 9), Core(0, 0), Core(0, 7), Core(0, 3), Core(6, 9), Core(6, 1), Core(6, 6), Core(6, 4)};
    }
    uint32_t east0 = 0;
    for (const auto& c : opt) {
        east0 = std::max<uint32_t>(east0, c.x);
    }
    TT_FATAL(east0 >= 6, "flat_routed_expert: east DRAM readers in column {} (expected >= 6)", east0);
    // MIMO_FL_RD_SAMECOL (with rows capped): a bank's second reader in its first reader's column (nearest free row)
    // instead of the column east of it, so columns 1 and E + 1 join the gate/up rectangles: 64 gate/up cores fit next
    // to a combine that keeps grid row 0
    const bool rd_samecol = rows_cap && std::getenv("MIMO_FL_RD_SAMECOL") != nullptr;
    // With the rows capped, the gate/up rectangles take every column the readers leave: west from column 2 (1 with
    // same-column readers) up to E - 1, east from E + 2 (E + 1) to the grid's edge (at most 4 wide). On a p150 that is
    // 4 x 3 (5 x 4 same-column); on a Galaxy chip 5 x 3 (6 x 4): its extra column lands west of the east readers.
    // Full grid: the fixed 4 x 3 below. (The east width used to be min(4, grid - 8), which assumed E = 6 and ran a
    // Galaxy chip's east rectangle off the grid.)
    const uint32_t a_x0 = rd_samecol ? 1u : 2u, b_x0 = rd_samecol ? east0 + 1 : east0 + 2;
    TT_FATAL(grid.x > b_x0, "flat_routed_expert: no column east of the east readers on a {}-wide grid", grid.x);
    const uint32_t aw_max = rows_cap ? east0 - a_x0 : 4u;
    const uint32_t bw_cap = rows_cap ? std::min<uint32_t>(rd_samecol ? 5u : 4u, uint32_t(grid.x) - b_x0) : 3u;
    const uint32_t hb_max = rd_samecol ? nrows : std::min(nrows, 8u);
    if (p.nsg == 1) {
        p.rects = {{2, 5, row0, row1}, {8, 10, row0, row0 + std::min(nrows, 8u) - 1}};
    } else if (p.nsg == 2) {
        p.rects = {{2, 5, 0, 3}, {8, 9, 0, 7}};
    } else {
        p.rects = {{2, 5, 0, 3}, {2, 5, 4, 7}, {8, 9, 0, 7}};
    }
    p.gu_fp32 = p.nsg != 3;  // 3 subgrids: bf16 DEST + packer L1 accumulation
    p.gu_l1acc = !p.gu_fp32;
    uint32_t min_rect = 1000;
    for (const auto& [x0, x1, y0, y1] : p.rects) {
        min_rect = std::min(min_rect, (x1 - x0 + 1) * (y1 - y0 + 1));
    }
    // the x ring shrinks (24 -> 16 -> 12 slots) before the gate/up split gives up
    // bfp8 x: the first ring size with a split. bf16 x: the split with the most gate/up cores over all ring sizes
    // (ties: the larger ring); the first-fit choice halved the gate/up cores (NP 2 at 24 slots instead of NP 1 at 16:
    // +57% expert time at large M)
    bool found = false;
    uint32_t best_cores = 0;
    for (uint32_t xs : {24u, 16u, 12u}) {
        const uint32_t gu_cells = rows_cap ? aw_max * nrows + bw_cap * hb_max : 64;  // 64 on the full grid
        const auto split = p.nsg == 1 ? gu_split(p.It, p.Ht, p.w_tile, xs, 16, gu_cells, p.gu_fp32, p.x_tile)
                                      : gu_split(p.It, p.Ht, p.w_tile, xs, p.n_rd_sg, min_rect, p.gu_fp32, p.x_tile);
        if (split) {
            const uint32_t cores = p.It / split->first * split->second;
            if (found && cores <= best_cores) {
                continue;
            }
            std::tie(p.np, p.g) = *split;
            p.x_slots = xs;
            best_cores = cores;
            found = true;
            if (p.x_tile == BF8_TILE) {
                break;
            }
        }
    }
    TT_FATAL(found, "flat_routed_expert: no gate/up split for {} tile columns", p.It);
    if (rows_cap) {  // rectangles of exactly the split's cores (4 columns x hA + bw columns x hB), the rest is down
                     // work
        const uint32_t want = p.It / p.np * p.g;
        bool shaped = false;
        int best = 1 << 30;
        for (uint32_t aw = 4; aw <= aw_max; ++aw) {
            for (uint32_t bw = 1; bw <= bw_cap; ++bw) {
                for (uint32_t hb = 1; hb <= hb_max; ++hb) {
                    if (want < bw * hb || (want - bw * hb) % aw || (want - bw * hb) / aw > nrows ||
                        want - bw * hb == 0) {
                        continue;
                    }
                    const int ha = (want - bw * hb) / aw;
                    const int imb = std::abs(int(aw) * ha - int(bw * hb));
                    if (imb < best) {
                        best = imb;
                        // west anchored against the east readers; east in p150 columns (shifted by E - 6 below)
                        p.rects = {
                            {east0 - aw, east0 - 1, row0, row0 + ha - 1},
                            {b_x0 - (east0 - 6), b_x0 - (east0 - 6) + bw - 1, row0, row0 + hb - 1}};
                        shaped = true;
                    }
                }
            }
        }
        TT_FATAL(shaped, "MIMO_FL_ROWS: no rectangle pair of {} gate/up cores in {} rows", want, nrows);
    }
    std::tie(p.dst_tiles, p.gu_rp) = gu_dst(p.np, p.Ht, p.x_slots, p.gu_fp32);
    const uint32_t mt_cap = p.g * std::min(p.dst_tiles / (2 * p.np), MT_MAX / p.g);
    p.mt = std::min(mt_cap, std::max(p.g, p.m / 32 / p.g * p.g));
    const uint32_t mt_full = p.mt;
    p.rdown = p.Ht > 6 * 26;  // readers compute down columns when down is heavy (> 6 columns per down core)
    if (const char* rd = std::getenv("MIMO_FL_RDOWN")) {  // (perf probe: force the readers' down columns on / off)
        p.rdown = std::atoi(rd) != 0;
    }
    // h units (MIMO_FL_HU = row tiles per unit): one M-group, one subgrid, no reader down columns
    if (const char* hu = std::getenv("MIMO_FL_HU")) {
        p.hu = static_cast<uint32_t>(std::atoi(hu));
    }
    if (p.hu && (p.g != 1 || p.nsg != 1 || p.rdown || p.mt % p.hu != 0)) {
        p.hu = 0;
    }
    // bf16 h: 64-row sub-blocks (256 KB h buffers, 4 of them fit; 128-row ones leave room for 2 and expose the
    // h -> down -> done -> go loop: bf16 x + h at M 2048 483 -> 384 us per expert); with h units the sub-block stays
    if (cfg.h_bf16 && !p.hu) {
        p.mt = std::min(p.mt, 2u / p.g * p.g);
    }
    if (const char* mt = std::getenv("MIMO_FL_MT")) {  // (perf probe: cap the sub-block's row tiles)
        p.mt = std::min(mt_full, static_cast<uint32_t>(std::atoi(mt)) / p.g * p.g);
    }
    // bf16 x + bfp8 h (128-row sub-blocks): gate/up in a full-sync DST (bf16 x + bfp8 h at M 160 / 512 / 2048: 48.9 /
    // 101.6 / 373.1 -> 44.2 / 91.4 / 366.5 us per expert); with 64-row sub-blocks there is one row pass anyway
    p.gu_full_sync = cfg.x_bf16 && p.gu_fp32 && p.gu_rp && p.mt / p.g * 2 * p.np > 4;
    TT_FATAL(p.mt % p.g == 0 && (p.mt / p.g) * 2 * p.np <= p.dst_tiles, "flat_routed_expert: sub-block split");
    p.mtg = p.mt / p.g;
    const uint32_t m_pad = (p.m + p.mt * 32 - 1) / (p.mt * 32) * p.mt * 32;
    p.nh = p.g > 1 ? 2 : 1;
    p.land_slots = p.nh > 1 || p.x_tile != BF8_TILE ? 2 : 3;
    // the down weight ring: two whole experts with pinning (the pinned schedule's regions); bf16 h: 1.5 experts, the
    // unpinned down schedule, so two 512 KB h buffers leave the row-major y out CB its room (else ~1 row tile: the
    // down packer stalls on the y writes)
    p.dring = cfg.pin && !cfg.h_bf16 ? 2.0f : 1.5f;
    if (const char* dr = std::getenv("MIMO_FL_DRING")) {  // (perf probe: the down weight ring in experts)
        p.dring = static_cast<float>(std::atof(dr));
    }

    // ---- layout ----
    p.readers = opt;
    if (rd_samecol) {
        std::set<std::pair<uint32_t, uint32_t>> used;
        for (const auto& c : opt) {
            used.insert({c.x, c.y});
        }
        for (const auto& c : opt) {
            uint32_t best_y = 1000;
            for (uint32_t y = row0; y <= row1; ++y) {
                if (!used.contains({c.x, y}) &&
                    (best_y == 1000 || std::abs(int(y) - int(c.y)) < std::abs(int(best_y) - int(c.y)))) {
                    best_y = y;
                }
            }
            TT_FATAL(best_y != 1000, "MIMO_FL_RD_SAMECOL: no free row for a second reader in column {}", c.x);
            p.readers.emplace_back(c.x, best_y);
            used.insert({c.x, best_y});
        }
    } else {
        for (const auto& c : opt) {
            p.readers.emplace_back(c.x + 1, c.y);
        }
    }
    if (rows_cap) {  // a bank's reader outside the rows moves to the nearest free row of its column
        std::set<std::pair<uint32_t, uint32_t>> used;
        for (const auto& c : p.readers) {
            if (c.y >= row0 && c.y <= row1) {
                used.insert({c.x, c.y});
            }
        }
        for (auto& c : p.readers) {
            if (c.y >= row0 && c.y <= row1) {
                continue;
            }
            uint32_t best_y = 1000;
            for (uint32_t y = row0; y <= row1; ++y) {
                if (!used.contains({c.x, y}) &&
                    (best_y == 1000 || std::abs(int(y) - int(c.y)) < std::abs(int(best_y) - int(c.y)))) {
                    best_y = y;
                }
            }
            TT_FATAL(best_y != 1000, "MIMO_FL_ROWS: no row for a reader in column {}", c.x);
            c = Core(c.x, best_y);
            used.insert({c.x, c.y});
        }
    }
    std::set<uint32_t> rcols;
    for (const auto& c : p.readers) {
        rcols.insert(c.x);
    }
    // DRAM readers: a west column pair 0, 1 and an east pair E, E + 1 (Blackhole p150 11 x 10: E = 6; a Galaxy chip's
    // 12 x 10, one column less harvested: E = 7). The layout is written for E = 6 and moved east by E - 6: the east
    // gate/up rectangle, the east relays and the east reader set; a wider west block leaves its extra column to the
    // down cores (as are all columns east of the east rectangle).
    const uint32_t east = rd_samecol ? (rcols.size() == 2 ? *rcols.rbegin() : 0)
                                     : (rcols.size() == 4 ? *std::next(rcols.begin(), 2) : 0);
    TT_FATAL(
        (rd_samecol ? rcols.size() == 2 && *rcols.begin() == 0 && east >= 6
                    : rcols.size() == 4 && *rcols.begin() == 0 && *std::next(rcols.begin()) == 1 && east >= 6 &&
                          *rcols.rbegin() == east + 1) &&
            grid.x >= 11 + (east - 6) && grid.y >= (rows_cap ? 8u : 10u),
        "flat_routed_expert: laid out for Blackhole grids with DRAM readers in columns 0, 1 and E, E + 1 (E >= 6) and "
        "room for the east rectangle; got a {} x {} grid with readers in columns {}",
        grid.x,
        grid.y,
        std::vector<uint32_t>(rcols.begin(), rcols.end()));
    const uint32_t shift = east - 6;
    for (auto& [x0, x1, y0, y1] : p.rects) {
        if (x0 > 6) {  // the east rectangle
            x0 += shift;
            x1 += shift;
        }
    }
    std::vector<Core> gu_all;
    for (const auto& [x0, x1, y0, y1] : p.rects) {
        for (uint32_t y = y0; y <= y1; ++y) {
            for (uint32_t x = x0; x <= x1; ++x) {
                gu_all.emplace_back(x, y);
            }
        }
    }
    std::set<std::pair<uint32_t, uint32_t>> taken;
    for (const auto* l : {&p.readers, &gu_all}) {
        for (const auto& c : *l) {
            taken.insert({c.x, c.y});
        }
    }
    if (p.nsg > 2) {  // per rectangle, the free cells of the column just west of it nearest its middle row
        for (uint32_t round = 0; round < 1 + p.nh; ++round) {
            for (const auto& [x0, x1, y0, y1] : p.rects) {
                const double yc = (y0 + y1) / 2.0;
                std::vector<uint32_t> ys(grid.y);
                for (uint32_t y = 0; y < grid.y; ++y) {
                    ys[y] = y;
                }
                std::stable_sort(ys.begin(), ys.end(), [&](uint32_t a, uint32_t b) {
                    return std::make_pair(std::abs(a - yc), a) < std::make_pair(std::abs(b - yc), b);
                });
                bool placed = false;
                for (uint32_t y : ys) {
                    if (!taken.contains({x0 - 1, y})) {
                        p.relays.emplace_back(x0 - 1, y);
                        taken.insert({x0 - 1, y});
                        placed = true;
                        break;
                    }
                }
                TT_FATAL(placed, "flat_routed_expert: no relay cell west of a rectangle");
            }
        }
    } else {
        auto first_free = [&](uint32_t x, std::initializer_list<uint32_t> ys) {
            for (uint32_t y : ys) {
                if (y >= row0 && y <= row1 && !taken.contains({x, y})) {
                    return Core(x, y);
                }
            }
            for (uint32_t y = row0; y <= row1; ++y) {
                if (!taken.contains({x, y})) {
                    return Core(x, y);
                }
            }
            // the column is full (same-column readers): the free cell nearest to it
            int best = 1 << 30;
            Core bc(0, 0);
            for (uint32_t y = row0; y <= row1; ++y) {
                for (uint32_t xx = 0; xx < grid.x; ++xx) {
                    const int d = 2 * std::abs(int(xx) - int(x)) + std::abs(int(y) - int(row0 + row1) / 2);
                    if (!taken.contains({xx, y}) && d < best) {
                        best = d;
                        bc = Core(xx, y);
                    }
                }
            }
            TT_FATAL(best < (1 << 30), "flat_routed_expert: no free relay cell near column {}", x);
            return bc;
        };
        const uint32_t rx_w = rd_samecol ? 0 : 1, rx_e = rd_samecol ? east : east + 1;
        p.relays = {first_free(rx_w, {4, 5, 3, 6, 2, 7}), first_free(rx_e, {3, 4, 2, 5, 1, 6})};
        for (const auto& c : p.relays) {
            taken.insert({c.x, c.y});
        }
        for (uint32_t j = 0; j < p.nh; ++j) {  // helpers: relay NR + NR j + k is rectangle k's j-th helper
            const Core a = first_free(rx_w, {5, 3, 6, 2, 7, 1, 8, 0, 9});
            const Core b = first_free(rx_e, {4, 2, 5, 1, 6, 0, 7, 8, 9});
            p.relays.push_back(a);
            p.relays.push_back(b);
            taken.insert({a.x, a.y});
            taken.insert({b.x, b.y});
        }
    }
    // MIMO_FL_COLS=n (probe): down cores only in columns [0, n), e.g. 11 on a 12-column grid to reproduce the 11 x 10
    const uint32_t cols_cap = std::getenv("MIMO_FL_COLS") ? std::atoi(std::getenv("MIMO_FL_COLS")) : grid.x;
    // MIMO_FL_SKIP_COLS="x,x,.." (probe): no down cores in these columns, e.g. 6 on a Galaxy chip (12 x 10, east
    // readers in column 7): the column a p150's 11 x 10 doesn't have, so the plan is the p150's (cores-for-cores) on a
    // Galaxy chip
    std::set<uint32_t> skip_cols;
    if (const char* sc = std::getenv("MIMO_FL_SKIP_COLS")) {
        for (const char* q = sc; *q;) {
            char* end = nullptr;
            skip_cols.insert(static_cast<uint32_t>(std::strtoul(q, &end, 10)));
            q = *end == ',' ? end + 1 : end;
        }
    }
    for (uint32_t y = row0; y <= std::min(row1, uint32_t(grid.y) - 1); ++y) {
        for (uint32_t x = 0; x < std::min(cols_cap, uint32_t(grid.x)); ++x) {
            if (!taken.contains({x, y}) && !skip_cols.contains(x)) {
                p.down.emplace_back(x, y);
            }
        }
    }
    // MIMO_FL_XDOWN="x,y;x,y;.." (probe, with the rows capped): extra down cores outside the rows, e.g. the grid row 0 cells
    // combine leaves free when it runs on row-major y (flat_combine_overlap)
    if (const char* xd = std::getenv("MIMO_FL_XDOWN"); xd && rows_cap) {
        for (const char* q = xd; *q;) {
            uint32_t x = 0, y = 0;
            int n = 0;
            TT_FATAL(std::sscanf(q, "%u,%u%n", &x, &y, &n) == 2, "MIMO_FL_XDOWN: x,y;x,y;..");
            TT_FATAL(
                x < grid.x && y < grid.y && (y < row0 || y > row1),
                "MIMO_FL_XDOWN: {},{} is off the grid or inside the planned rows",
                x,
                y);
            p.down.emplace_back(x, y);
            q += n;
            q += *q == ';';
        }
    }
    if (p.nsg >
        1) {  // subgrid k: rectangle k, its nearest readers, primary relay k + helpers, a share of the down cores
        std::vector<std::pair<double, double>> ctr;
        for (const auto& [x0, x1, y0, y1] : p.rects) {
            ctr.emplace_back((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        }
        auto dist = [&](const Core& c, uint32_t k) {
            return std::abs(c.x - ctr[k].first) + std::abs(c.y - ctr[k].second);
        };
        if (p.nsg == 2) {  // west / east reader columns, down split by relative distance
            std::vector<Core> rd;
            for (const auto& c : p.readers) {
                if (c.x <= 1) {
                    rd.push_back(c);
                }
            }
            for (const auto& c : p.readers) {
                if (c.x == east || c.x == east + 1) {
                    rd.push_back(c);
                }
            }
            p.readers = rd;
            std::vector<Core> od = p.down;
            std::stable_sort(od.begin(), od.end(), [&](const Core& a, const Core& b) {
                return std::make_tuple(dist(a, 0) - dist(a, 1), a.y, a.x) <
                       std::make_tuple(dist(b, 0) - dist(b, 1), b.y, b.x);
            });
            const size_t half = od.size() / 2;
            p.down.assign(od.begin(), od.begin() + 2 * half);
        } else {
            auto nearest = [&](std::vector<Core>& pool, uint32_t k) {
                auto it = std::min_element(pool.begin(), pool.end(), [&](const Core& a, const Core& b) {
                    return std::make_tuple(dist(a, k), a.y, a.x) < std::make_tuple(dist(b, k), b.y, b.x);
                });
                const Core c = *it;
                pool.erase(it);
                return c;
            };
            std::vector<Core> pool = p.readers;
            std::vector<std::vector<Core>> by_sg(p.nsg);
            for (uint32_t i = 0; i < p.n_rd_sg; ++i) {
                for (uint32_t k = 0; k < p.nsg; ++k) {
                    by_sg[k].push_back(nearest(pool, k));
                }
            }
            p.readers.clear();
            for (const auto& l : by_sg) {
                p.readers.insert(p.readers.end(), l.begin(), l.end());
            }
            p.down.insert(p.down.end(), pool.begin(), pool.end());  // unused readers do down work
            const uint32_t nd_sg = p.down.size() / p.nsg;
            std::vector<Core> left_d = p.down;
            std::vector<std::vector<Core>> parts(p.nsg);
            for (uint32_t i = 0; i < nd_sg; ++i) {
                for (uint32_t k = 0; k < p.nsg; ++k) {
                    parts[k].push_back(nearest(left_d, k));
                }
            }
            p.down.clear();
            for (const auto& l : parts) {
                p.down.insert(p.down.end(), l.begin(), l.end());
            }
        }
    }
    if (const char* ndc = std::getenv("MIMO_FL_ND")) {  // (perf probe: cap the down cores; the rest stay idle)
        const uint32_t cap = static_cast<uint32_t>(std::atoi(ndc));
        if (cap > 0 && cap < p.down.size()) {
            p.down.resize(cap);
        }
    }
    uint32_t ND = p.down.size();
    p.nd_sg = ND / p.nsg;
    const uint32_t n_rd = p.readers.size();
    p.n_rd_sg = n_rd / p.nsg;
    p.s = m_pad / (p.mt * 32);
    p.v = p.E * p.s;
    p.nk_gu = p.Ht / KBLK;
    p.slot = KBLK * 2 * p.np;
    p.rg = (p.It / p.np) / p.n_rd_sg;
    p.r_ = p.rg * p.g;
    TT_FATAL((p.It / p.np) % p.n_rd_sg == 0 && p.r_ <= 4, "flat_routed_expert: gate/up pair-sets per reader");
    p.ring_g = 2 * p.nk_gu;
    p.d_ch = p.rdown ? std::min(D_CHAINS, p.n_rd_sg) : D_CHAINS;
    if (const char* dch = std::getenv("MIMO_FL_DCH"); dch && !p.rdown) {  // (perf probe: down chains)
        p.d_ch = static_cast<uint32_t>(std::atoi(dch));
    }
    p.n_rdn = p.rdown ? p.d_ch : 0;
    p.pcd_r = p.rdown ? 6 : 0;
    if (const char* pr = std::getenv("MIMO_FL_PCD_R"); pr && p.rdown) {  // (perf probe: the readers' down columns)
        p.pcd_r = static_cast<uint32_t>(std::atoi(pr));
    }
    p.rem_cols = p.Ht - p.n_rdn * p.pcd_r;
    if (p.nd_sg > p.rem_cols) {
        // more down cores than output tile columns (small H on a wide grid, e.g. a Galaxy chip's 12 x 10): keep each
        // subgrid's first rem_cols (its block is ordered nearest first); the rest stay idle
        std::vector<Core> kept;
        for (uint32_t k = 0; k < p.nsg; ++k) {
            kept.insert(kept.end(), p.down.begin() + k * p.nd_sg, p.down.begin() + k * p.nd_sg + p.rem_cols);
        }
        p.down = kept;
        p.nd_sg = p.rem_cols;
        ND = p.down.size();
    }
    const uint32_t base_p = p.rem_cols / p.nd_sg, extra = p.rem_cols % p.nd_sg;
    p.pcds.resize(ND);
    p.col0s.resize(ND);
    for (uint32_t d = 0; d < ND; ++d) {
        p.pcds[d] = base_p + (p.dl(d) < extra ? 1 : 0);
    }
    for (uint32_t d = 0; d < ND; ++d) {
        uint32_t c0 = 0;
        for (uint32_t e = p.sg_dn(d) * p.nd_sg; e < d; ++e) {
            c0 += p.pcds[e];
        }
        p.col0s[d] = c0;
    }
    const uint32_t pcd = *std::max_element(p.pcds.begin(), p.pcds.end());
    TT_FATAL(pcd <= 16 && *std::min_element(p.pcds.begin(), p.pcds.end()) >= 1, "flat_routed_expert: down columns");
    if (p.rdown) {
        p.kd_r = FlatRoutedExpertPlan::kd_of(p.pcd_r, p.It);
        p.nblk_r = p.It / p.kd_r;
        p.slot_dr = p.kd_r * p.pcd_r;
        p.ring_dr = pyround(p.dring * p.nblk_r);
        p.out_tiles_r = p.mt * p.pcd_r;
    }
    const uint32_t x_bytes = p.mt * KBLK * p.x_tile;
    p.h_tiles = p.It * (p.hu ? p.hu : p.mt);  // one down h buffer: a sub-block, or an h unit
    const uint32_t out_tiles = p.mt * pcd;
    uint32_t rect_min = 1000;
    for (const auto& [x0, x1, y0, y1] : p.rects) {
        rect_min = std::min(rect_min, (x1 - x0 + 1) * (y1 - y0 + 1));
    }
    p.group_rect = p.g == 2 && rect_min >= p.It / p.np;

    // gate/up cores per reader: nearest by the forwarder's NoC (1) hops, balanced
    std::vector<uint32_t> left(gu_all.size());
    for (uint32_t i = 0; i < left.size(); ++i) {
        left[i] = i;
    }
    auto rect_of0 = [&](const Core& c) { return p.rect_of(c); };
    std::vector<std::vector<uint32_t>> per_reader(n_rd);
    for (uint32_t j = 0; j < p.r_; ++j) {
        for (uint32_t r = 0; r < n_rd; ++r) {
            std::vector<uint32_t> cand;
            for (uint32_t ci : left) {
                if ((!p.group_rect || rect_of0(gu_all[ci]) == j % p.g) &&
                    (p.nsg == 1 || rect_of0(gu_all[ci]) == p.sg_rd(r))) {
                    cand.push_back(ci);
                }
            }
            if (cand.empty()) {
                cand = left;
            }
            const Core rp = phys(p.readers[r]);
            const uint32_t best = *std::min_element(cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) {
                return noc_hops(rp, phys(gu_all[a]), 1) < noc_hops(rp, phys(gu_all[b]), 1);
            });
            per_reader[r].push_back(best);
            left.erase(std::find(left.begin(), left.end(), best));
        }
    }
    for (uint32_t r = 0; r < n_rd; ++r) {
        for (uint32_t ci : per_reader[r]) {
            p.gu.push_back(gu_all[ci]);
        }
    }
    for (uint32_t ci : left) {
        p.gu_idle.push_back(gu_all[ci]);
    }

    // ---- arena (per-role layout, 2 KB aligned) ----
    p.x_off = al(p.ring_g * p.slot * p.w_tile);
    if (cfg.h_bf16) {  // 64-row sub-blocks: x blocks are half the size, a deeper x ring fits (more relay prefetch)
        for (uint32_t xs : {32u, 24u}) {
            if (xs > p.x_slots && p.x_off + al(xs * x_bytes) + al(4 * p.mtg * p.np * p.h_tile) <= GU_L1_BUDGET) {
                p.x_slots = xs;
                break;
            }
        }
    }
    p.p_off = p.x_off + al(p.x_slots * x_bytes);
    p.hl_off = p.p_off + (p.gu_l1acc ? al(p.mtg * 2 * p.np * 2048) : 0);  // gate/up h_local (CB 3) after the partials
    p.rd_off = al(p.rd_slots * p.rg * p.slot * p.w_tile);
    uint32_t dn_ring_max = 0;
    for (uint32_t d = 0; d < ND; ++d) {
        const uint32_t kd = FlatRoutedExpertPlan::kd_of(p.pcds[d], p.It);
        dn_ring_max = std::max(dn_ring_max, pyround(p.dring * (p.It / kd)) * kd * p.pcds[d]);
    }
    p.h_off = al(std::max(dn_ring_max * p.w_tile, p.rd_off + (p.rdown ? p.ring_dr * p.slot_dr * p.w_tile : 0)));
    const uint32_t out_bytes = al(2 * out_tiles * BF8_TILE);
    uint32_t dyn_half = 512;  // as the program factory: CB 7 holds 4 halves of the counts / regions / ids
    while (dyn_half < (p.NG * 4 + 63) / 64 * 64 + (4 * p.E + 63) / 64 * 64) {
        dyn_half *= 2;
    }
    // bfp8 x and h: the layout the research builder (FlatExpert) shares, budgeted against the whole bank as before;
    // bf16 x / h: the exact budget (and h_local in the arena), which the larger tiles need
    const bool exact = cfg.x_bf16 || cfg.h_bf16 || p.hu;
    p.hl_in_arena = exact;
    const uint32_t L1_BANK = exact ? L1_BANK_BYTES - (2048 + 4 * dyn_half + 2048) - 2048 - 2048 : L1_BANK_BYTES;
    p.hbuf = p.h_off + al(3 * p.h_tiles * p.h_tile) + out_bytes + 2048 <= L1_BANK ? 3 : 2;
    // bf16 h: a 4th h buffer when it fits (the h -> down -> done -> go loop is latency-bound: time per sub-block ~ loop
    // latency / HBUF)
    if (cfg.h_bf16 && p.h_off + al(4 * p.h_tiles * p.h_tile) + out_bytes + 2048 <= L1_BANK) {
        p.hbuf = 4;
    }
    if (p.hu) {  // h units: as many unit buffers as fit (<= 8: the gather semaphores on the down heads)
        uint32_t cap = 8;
        if (const char* hb = std::getenv("MIMO_FL_HBUF")) {  // (perf probe: cap the h unit buffers)
            cap = std::min(cap, static_cast<uint32_t>(std::atoi(hb)));
        }
        p.hbuf = 2;
        for (uint32_t n = cap; n > 2; --n) {
            if (p.h_off + al(n * p.h_tiles * p.h_tile) + out_bytes + 2048 <= L1_BANK) {
                p.hbuf = n;
                break;
            }
        }
    }
    p.hl_slots = p.hu ? 3 : p.hbuf;
    p.o_off = p.h_off + al(p.hbuf * p.h_tiles * p.h_tile);
    // (bf16 h: two h buffers leave the row-major y out CB less than out_bytes; it takes what is left, >= 1 row tile)
    const uint32_t dn_bytes = std::min(p.o_off + out_bytes + 2048, L1_BANK);
    TT_FATAL(p.o_off + al(pcd * 2048) + 2048 <= L1_BANK, "flat_routed_expert: down core arena does not fit");
    const uint32_t gu_bytes = p.hl_off + (p.hl_in_arena ? al(p.hl_slots * p.mtg * p.np * p.h_tile) : 0);
    TT_FATAL(gu_bytes <= L1_BANK, "flat_routed_expert: gate/up core arena does not fit");
    for (uint32_t t : {32u, 16u, 8u}) {
        if (p.Ht % t == 0 && t % KBLK == 0) {
            p.sbt = t;
            break;
        }
    }
    p.seg = p.sbt * 64;
    p.nsb = p.Ht / p.sbt;
    p.sb_off = al(RM_CHUNKS * 32 * p.seg);
    p.land_off = p.sb_off + al(p.sb_slots * p.mt * p.sbt * p.x_tile);
    const uint32_t relay_bytes = p.land_off + al(p.nh * p.land_slots * p.mt * p.sbt * p.x_tile);
    TT_FATAL(relay_bytes <= L1_BANK, "flat_routed_expert: relay arena does not fit");
    p.arena_tiles = std::max({gu_bytes, dn_bytes, relay_bytes, p.rd_off}) / 2048;
    p.vstride = 1 + p.nh;
    p.region_bytes = p.E * p.nk_gu * p.rg * p.slot * p.w_tile;
    p.wd_region = p.E * p.It * pcd * p.w_tile;
    p.wr_region = p.rdown ? p.E * p.It * p.pcd_r * p.w_tile : 0;
    for (uint32_t k = 0; k < p.nsg; ++k) {
        p.coords.push_back(p.down[k * p.nd_sg]);
    }

    // ---- down chains: per subgrid D_CH chains, each tail a reader when readers compute down columns ----
    for (uint32_t k = 0; k < p.nsg; ++k) {
        std::vector<uint32_t> ids;
        for (uint32_t d = 0; d < ND; ++d) {
            if (p.sg_dn(d) == k) {
                ids.push_back(d);
            }
        }
        for (const auto& seg : chains(ids, p.down, phys, p.d_ch, 1)) {
            p.d_heads.push_back(seg.front());
            if (p.rdown) {
                const uint32_t tail = seg.back();
                std::optional<uint32_t> best;
                for (uint32_t r = 0; r < n_rd; ++r) {
                    const bool used = std::any_of(p.rdn.begin(), p.rdn.end(), [r](auto t) { return t.first == r; });
                    if (used || p.sg_rd(r) != k) {
                        continue;
                    }
                    if (!best || noc_hops(phys(p.down[tail]), phys(p.readers[r]), 1) <
                                     noc_hops(phys(p.down[tail]), phys(p.readers[*best]), 1)) {
                        best = r;
                    }
                }
                TT_FATAL(best.has_value(), "flat_routed_expert: no reader for a down chain tail");
                p.rdn.emplace_back(*best, tail);
            }
            for (size_t i = 0; i + 1 < seg.size(); ++i) {
                p.d_succ[seg[i]] = seg[i + 1];
                p.d_pred[seg[i + 1]] = seg[i];
            }
        }
    }
    return p;
}

}  // namespace ttnn::operations::bringup::flat_routed_expert
