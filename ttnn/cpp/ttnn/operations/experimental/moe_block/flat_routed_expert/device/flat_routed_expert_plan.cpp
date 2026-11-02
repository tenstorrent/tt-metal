// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_routed_expert_plan.hpp"
#include "flat_drisc.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>

#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

namespace plan_detail {
using Core = tt::tt_metal::CoreCoord;

constexpr uint32_t MT_MAX = 4;
// L1 per core for the arena: the Blackhole L1 bank (1427 KB) less the program's static CBs (meta 2 KB, dynamic counts
// 4 x dyn_half, the gate/up 2 KB pack CB: one address range on every core) and the 2 KB done words and 2 KB relay
// scratch words (same address on every core), all below / beside the arena
constexpr uint32_t L1_BANK_BYTES = 1427 * 1024;
constexpr uint32_t GU_L1_BUDGET = 1400 * 1024;  // gate/up core: weight ring + x ring
constexpr uint32_t D_CHAINS = 7;
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
// holds a sub-block's K-blocks) keep the 8-tile rows. With 16-column units (units) a block of 3 / 4 pairs' width runs
// the sub-block in a full-sync fp32 DST (8 tiles: one row tile of up to 8 tiles).
std::pair<uint32_t, bool> gu_dst(uint32_t np, uint32_t ht, uint32_t x_slots, bool units = false) {
    // (np: the gate/up width in pairs; a units width w is w / 2 rounded up)
    if (np > 2 && 2 * np <= 8 && units && ht / kKBlk <= x_slots) {
        return {8, true};
    }
    const bool rp = np <= 2 && ht / kKBlk <= x_slots;
    return rp ? std::pair<uint32_t, bool>{8, true} : std::pair<uint32_t, bool>{4, false};
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
    for (const auto* l : {&down, &relays, &gu_idle}) {
        out.insert(out.end(), l->begin(), l->end());
    }
    out.push_back(launcher);  // (the other reader cells are down cores, relays or idle)
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

FlatRoutedExpertPlan make_flat_routed_expert_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const FlatRoutedExpertConfig& cfg) {
    auto* device = &mesh_device;
    const FlatLayoutOptions& lo = cfg.layout;
    FlatRoutedExpertPlan p;
    p.H = cfg.hidden;
    p.I = cfg.intermediate;
    p.E = cfg.experts_per_chip;
    p.NG = cfg.num_global_experts;
    p.m = cfg.max_tokens;
    TT_FATAL(p.H % 256 == 0 && p.I % 32 == 0 && p.m % 32 == 0, "flat_routed_expert: H % 256, I % 32, m % 32");
    TT_FATAL(p.m >= 256, "flat_routed_expert: the op is built for >= 256 tokens per expert (helper relays)");
    TT_FATAL(p.E >= 1 && p.E <= 64, "flat_routed_expert: 1..64 local experts");
    TT_FATAL(
        lo.x_helpers < 0 || lo.x_helpers == 1,
        "flat layout x_helpers {}: one helper relay per rectangle",
        lo.x_helpers);
    p.Ht = p.H / 32;
    p.It = p.I / 32;
    p.w_tile = cfg.weights_bf8 ? kBf8Tile : 576;
    p.banks = device->dram_grid_size().x;
    const auto phys = [device](const Core& c) { return device->worker_core_from_logical_core(c); };

    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(grid.y >= 10, "flat_routed_expert: laid out for a 10-row Blackhole grid (got {} rows)", grid.y);
    constexpr uint32_t row0 = 0, row1 = 9, nrows = 10;
    // The DRAM-optimal readers (one per bank): columns 0 and E (E = 6 on a p150's 11 x 10, 7 on a Galaxy chip's
    // 12 x 10). Queried before the rectangles are sized: shaped rectangles stretch to the columns the readers leave.
    std::vector<Core> opt;
    {
        const auto by_bank = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(
            tt::tt_metal::NOC::NOC_0, flat_reference_coord(mesh_device));
        for (uint32_t b = 0; b < by_bank.size(); ++b) {
            opt.push_back(by_bank.at(b));
        }
    }
    uint32_t east0 = 0;
    for (const auto& c : opt) {
        east0 = std::max<uint32_t>(east0, c.x);
    }
    TT_FATAL(east0 >= 6, "flat_routed_expert: east DRAM readers in column {} (expected >= 6)", east0);
    // The gate/up rectangles: the fixed 4 x 10 west (columns 2..5) and 3 x 8 east, or (shaped below) every column the
    // readers leave: west from column 2 up to E - 1, east from E + 2 to the grid's edge (at most 4 wide).
    const uint32_t a_x0 = 2u, b_x0 = east0 + 2;
    TT_FATAL(grid.x > b_x0, "flat_routed_expert: no column east of the east readers on a {}-wide grid", grid.x);
    const uint32_t hb_max = std::min(nrows, 8u);
    p.rects = {{2, 5, row0, row1}, {8, 10, row0, row0 + hb_max - 1}};
    // even split: the 2 It weight tiles a K row ([gate | up] 16-column units) over at most 64 gate/up cores, the fewest
    // tiles a core that divide them evenly over a multiple of the 16 weight slots; whole gate / up tile pairs (NP =
    // tiles / 2) where they give the same tiles a core, else 16-column units. GLM-5.3 / -Flash, K2.7: 64 cores x 2
    // (pairs); K3, M3 (I 3072): 64 x 3 units; V4.1-Flash (I 2304): 48 x 3 units (64 cores would leave the slowest with
    // 3 as well)
    bool found = false;
    const uint32_t tiles = 2 * p.It;
    for (uint32_t per = (tiles + 63) / 64; per <= 8 && !found; ++per) {
        if (tiles % per != 0 || (tiles / per) % 16 != 0) {
            continue;
        }
        const uint32_t np = per / 2;
        const bool pairs = per % 2 == 0 && p.It % np == 0 && (p.It / np) % 16 == 0;
        for (uint32_t xs : {24u, 16u, 12u}) {
            const uint32_t dst = pairs ? gu_dst(np, p.Ht, xs).first : gu_dst((per + 1) / 2, p.Ht, xs, true).first;
            const uint32_t mt_ = std::min(dst / per, MT_MAX);
            if (mt_ >= 1 && 2 * p.Ht * per * p.w_tile + xs * mt_ * kKBlk * kBf8Tile <= GU_L1_BUDGET) {
                p.np = pairs ? np : 0;
                p.nu = pairs ? 0 : per;
                p.hg = !pairs;
                p.g = 1;
                p.x_slots = xs;
                found = true;
                break;
            }
        }
    }
    TT_FATAL(
        found,
        "flat_routed_expert: no even gate/up split for I {} ({} weight tiles a K row over at most 64 cores, a multiple "
        "of 16)",
        p.I,
        2 * p.It);
    p.gwt = p.nu ? p.nu : 2 * p.np;
    p.hpc = p.nu ? (p.nu + 1) / 2 : p.np;
    p.ngrp = p.nu ? 2 * p.It / p.nu : p.It / p.np;
    // A split of fewer gate/up cores than the fixed rectangles' 64 (V4.1-Flash: 48), or units not exactly 64: shape the
    // rectangles to the split, so the cells left over do down work (the down cores are trimmed below)
    const bool auto_shape = p.ngrp < 64 || (p.nu && p.ngrp != 64);
    if (auto_shape) {  // rectangles of exactly the split's cores (aw columns x hA + bw columns x hB)
        const uint32_t aw_max = east0 - a_x0, bw_cap = std::min<uint32_t>(4u, uint32_t(grid.x) - b_x0);
        const uint32_t want = p.ngrp;
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
        if (!shaped && p.nu) {  // (no exact pair: the smallest pair covering the split, extra cells idle)
            int best_cells = 1 << 30;
            for (uint32_t aw = 4; aw <= aw_max; ++aw) {
                for (uint32_t bw = 1; bw <= bw_cap; ++bw) {
                    for (uint32_t hb = 1; hb <= hb_max; ++hb) {
                        for (uint32_t ha = 1; ha <= nrows; ++ha) {
                            const int cells = int(aw * ha + bw * hb);
                            if (cells < int(want) || cells > best_cells ||
                                (cells == best_cells && std::abs(int(aw * ha) - int(bw * hb)) >= best)) {
                                continue;
                            }
                            best_cells = cells;
                            best = std::abs(int(aw * ha) - int(bw * hb));
                            p.rects = {
                                {east0 - aw, east0 - 1, row0, row0 + ha - 1},
                                {b_x0 - (east0 - 6), b_x0 - (east0 - 6) + bw - 1, row0, row0 + hb - 1}};
                            shaped = true;
                        }
                    }
                }
            }
        }
        TT_FATAL(shaped, "flat_routed_expert: no rectangle pair of {} gate/up cores in {} rows", want, nrows);
    }
    std::tie(p.dst_tiles, p.gu_rp) = gu_dst((p.gwt + 1) / 2, p.Ht, p.x_slots, p.nu != 0);
    const uint32_t mt_cap = std::min(p.dst_tiles / p.gwt, MT_MAX);
    p.mt = std::min(mt_cap, std::max(1u, p.m / 32));
    // DRISC only: every gate/up weight slot from the DRAM cores (pure, tiles in switch mode); no Tensix weight readers
    // (FlatDriscTopology::of fails on a device without programmable DRAM cores)
    const FlatDriscTopology drisc = FlatDriscTopology::of(mesh_device, lo.drisc_tiles);
    p.drisc_tiles = lo.drisc_tiles;
    // layout.rows_hint (expected rows per expert, <= 320): 64-row sub-blocks for the plans with 128-row ones and
    // gate/up row passes - one pass a sub-block, so the next sub-block's x streams in during the whole pass and the
    // drain after the last weights is halved (GLM-5.3 E 8 M 32 / 128 / 256: 408 / 516 / 642 -> 403 / 498 / 632 us)
    if (lo.rows_hint > 0 && lo.rows_hint <= 320 && p.gu_rp && p.mt == 4) {
        p.mt = 2;
    }
    // wide gate/up blocks (gwt > 4) with row passes: gate/up in a full-sync DST
    p.gu_full_sync = p.gwt > 4 && p.gu_rp && p.mt * p.gwt > 4;
    TT_FATAL(p.mt * p.gwt <= p.dst_tiles, "flat_routed_expert: sub-block split");
    // gate/up K blocks per handshake (row passes only): 2 (GLM-5.3 E 8 M 1024 / 2048 -2.7 / -3.4%, DSv4.1 -1.2 /
    // -1.3%, neutral below M 512 and for K3 / K2.7 / M3; 4 regressed M3 at 2048)
    if (p.gu_rp && !p.gu_full_sync) {
        p.gu_kg = std::min(2u, p.Ht / kKBlk);
    }
    p.mtg = p.mt;
    const uint32_t m_pad = (p.m + p.mt * 32 - 1) / (p.mt * 32) * p.mt * 32;
    p.nh = 1;
    p.land_slots = 2;  // (two landing rings a primary: the helpers feed both)
    // the down weight ring: two whole experts with pinning (the pinned schedule's regions), else 1.5
    p.dring = cfg.pin ? 2.0f : 1.5f;

    // ---- layout ----
    p.readers = opt;
    for (const auto& c : opt) {
        p.readers.emplace_back(c.x + 1, c.y);
    }
    std::set<uint32_t> rcols;
    for (const auto& c : p.readers) {
        rcols.insert(c.x);
    }
    // DRAM readers: a west column pair 0, 1 and an east pair E, E + 1 (Blackhole p150 11 x 10: E = 6; a Galaxy chip's
    // 12 x 10, one column less harvested: E = 7). The layout is written for E = 6 and moved east by E - 6: the east
    // gate/up rectangle, the east relays and the east reader set; a wider west block leaves its extra column to the
    // down cores (as are all columns east of the east rectangle).
    const uint32_t east = rcols.size() == 4 ? *std::next(rcols.begin(), 2) : 0;
    TT_FATAL(
        rcols.size() == 4 && *rcols.begin() == 0 && *std::next(rcols.begin()) == 1 && east >= 6 &&
            *rcols.rbegin() == east + 1 && grid.x >= 11 + (east - 6),
        "flat_routed_expert: laid out for Blackhole grids with DRAM readers in columns 0, 1 and E, E + 1 (E >= 6) and "
        "room for the east rectangle; got a {} x {} grid with readers in columns {}",
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
    // one weight slot per reader cell, every one on the DRAM cores; the reader cells but the launcher's go to the
    // relays / down cores
    p.drisc_n = p.readers.size();
    TT_FATAL(p.drisc_n <= 16, "flat DRISC: {} weight slots > 16", p.drisc_n);
    p.launcher = p.readers[0];
    for (uint32_t r = 1; r < p.drisc_n; ++r) {
        taken.erase({p.readers[r].x, p.readers[r].y});
    }
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
        // the column is full: the free cell nearest to it
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
    const uint32_t rx_w = 1, rx_e = east + 1;
    p.relays = {first_free(rx_w, {4, 5, 3, 6, 2, 7}), first_free(rx_e, {3, 4, 2, 5, 1, 6})};
    for (const auto& c : p.relays) {
        taken.insert({c.x, c.y});
    }
    {  // the helpers: relay NR + k is rectangle k's
        Core a = first_free(rx_w, {5, 3, 6, 2, 7, 1, 8, 0, 9});
        Core b = first_free(rx_e, {4, 2, 5, 1, 6, 0, 7, 8, 9});
        p.relays.push_back(a);
        p.relays.push_back(b);
        taken.insert({a.x, a.y});
        taken.insert({b.x, b.y});
    }
    for (uint32_t y = row0; y <= row1; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            if (!taken.contains({x, y})) {
                p.down.emplace_back(x, y);
            }
        }
    }
    {  // the freed reader cells last (a down-core cap keeps the usual cells first)
        std::set<std::pair<uint32_t, uint32_t>> rd;
        for (const auto& c : p.readers) {
            rd.insert({c.x, c.y});
        }
        std::stable_partition(p.down.begin(), p.down.end(), [&](const Core& c) { return !rd.contains({c.x, c.y}); });
    }
    uint32_t ND = p.down.size();
    const uint32_t n_rd = p.readers.size();
    p.n_rd_sg = n_rd;
    p.s = m_pad / (p.mt * 32);
    p.v = p.E * p.s;
    p.nk_gu = p.Ht / kKBlk;
    p.slot = kKBlk * p.gwt;
    p.rg = p.ngrp / p.n_rd_sg;
    p.r_ = p.rg * p.g;
    TT_FATAL(p.ngrp % p.n_rd_sg == 0 && p.r_ <= 4u, "flat_routed_expert: gate/up pair-sets per reader");
    // gate/up weight ring: 2 expert slices (with pinning: the pinned expert's and one cycling)
    p.ring_g = 2 * p.nk_gu;
    // 16-column units (40 down cores on V4.1-Flash): 4 chains. Each gate/up core writes its h slice to every chain
    // head, the chains forward it; at E 24 M 1024 bfp8 h 3 / 4 / 5 / 7 / 10 chains: 211.7 / 211.6 / 213.4 / 226.6 /
    // 236.1 us per expert (h writes off: 210.3), M 80 / 256 flat
    p.d_ch = p.nu ? 4 : D_CHAINS;
    p.rem_cols = p.Ht;
    // even split: the output tile columns over at most 32 down cores, the fewest cores with the fewest columns a core
    // (GLM-5.3 / M3 32 x 6, K2.7 32 x 7, GLM-5.3-Flash 32 x 4, V4.1-Flash 32 x 5, K3 28 x 4); the rest of the cells are
    // left to the combine
    {
        // with shaped rectangles, down cores under the east rectangle (its columns, below it) are slow (far from the
        // h relays, in the east gate/up's traffic): drop them, then keep the fewest cores with the same most tile
        // columns per core (K3 at M 2048, us per expert: 26 down 5 columns max 409, 28 / 30 at 4 384, 32 at 4 400, 34
        // / 42 into columns 8+ 443 / 408)
        const uint32_t east_x0 = b_x0 - (east0 - 6);
        std::vector<Core> kept;
        for (const auto& c : p.down) {
            if (!auto_shape || c.x < east_x0) {
                kept.push_back(c);
            }
        }
        if (!kept.empty()) {
            const uint32_t n = std::min<uint32_t>(32u, kept.size());
            const uint32_t per = (p.rem_cols + n - 1) / n;
            kept.resize(std::min<size_t>(kept.size(), (p.rem_cols + per - 1) / per));
            p.down = kept;
            ND = p.down.size();
        }
    }
    if (ND > p.rem_cols) {
        // more down cores than output tile columns (small H on a wide grid): keep the first rem_cols (ordered nearest
        // first); the rest stay idle
        p.down.resize(p.rem_cols);
        ND = p.down.size();
    }
    const uint32_t base_p = p.rem_cols / ND, extra = p.rem_cols % ND;
    p.pcds.resize(ND);
    p.col0s.resize(ND);
    for (uint32_t d = 0; d < ND; ++d) {
        p.pcds[d] = base_p + (d < extra ? 1 : 0);
    }
    for (uint32_t d = 0, c0 = 0; d < ND; ++d) {
        p.col0s[d] = c0;
        c0 += p.pcds[d];
    }
    const uint32_t pcd = *std::max_element(p.pcds.begin(), p.pcds.end());
    TT_FATAL(pcd <= 16 && *std::min_element(p.pcds.begin(), p.pcds.end()) >= 1, "flat_routed_expert: down columns");
    const uint32_t x_bytes = p.mt * kKBlk * kBf8Tile;
    p.h_tiles = p.It * p.mt;  // one down h buffer: a sub-block
    const uint32_t out_tiles = p.mt * pcd;

    // gate/up cores per weight slot: each slot's cores by the hops from the DRAM tile that actually sends it (switch
    // mode: a bank's second slot from its NOC1-endpoint tile, on NOC0; one-tile choices: both from the bank's tile),
    // all pairs (slot, core) taken nearest first rather than round-robin nearest (which left the last slots 15-21
    // hops). Galaxy loads GLM-5.3 / GLM-Flash / K3 / K2.7 525 / 382 / 1232 / 808 -> 522 / 381 / 1216 / 808 us.
    std::vector<std::vector<uint32_t>> per_reader(n_rd);
    std::vector<bool> placed(gu_all.size(), false);
    {
        std::vector<std::tuple<int, uint32_t, uint32_t>> pairs;  // (hops, slot, core)
        std::vector<Core> worker_noc0;
        for (const auto& c : gu_all) {
            worker_noc0.push_back(flat_worker_noc0(mesh_device, c));
        }
        for (uint32_t r = 0; r < n_rd; ++r) {
            const auto [sp, snoc] = drisc.sender(r);
            for (uint32_t ci = 0; ci < gu_all.size(); ++ci) {
                pairs.emplace_back(noc_hops(Core(sp), worker_noc0[ci], int(snoc)), r, ci);
            }
        }
        std::sort(pairs.begin(), pairs.end());
        for (const auto& [h, r, ci] : pairs) {
            if (!placed[ci] && per_reader[r].size() < p.r_) {
                per_reader[r].push_back(ci);
                placed[ci] = true;
            }
        }
    }
    for (uint32_t r = 0; r < n_rd; ++r) {
        for (uint32_t ci : per_reader[r]) {
            p.gu.push_back(gu_all[ci]);
        }
    }
    for (uint32_t ci = 0; ci < gu_all.size(); ++ci) {
        if (!placed[ci]) {
            p.gu_idle.push_back(gu_all[ci]);
        }
    }

    // ---- arena (per-role layout, 2 KB aligned) ----
    p.x_off = al(p.ring_g * p.slot * p.w_tile);
    p.hl_off = p.x_off + al(p.x_slots * x_bytes);  // gate/up: weight ring, x ring (h_local is a static CB)
    uint32_t dn_ring_max = 0;
    for (uint32_t d = 0; d < ND; ++d) {
        const uint32_t kd = FlatRoutedExpertPlan::kd_of(p.pcds[d], p.It);
        dn_ring_max = std::max(dn_ring_max, pyround(p.dring * (p.It / kd)) * kd * p.pcds[d]);
    }
    p.h_off = al(dn_ring_max * p.w_tile);
    const uint32_t out_bytes = al(2 * out_tiles * kBf8Tile);
    uint32_t dyn_half = 512;  // as the program factory: CB 7 holds 4 halves of the counts / regions / ids
    while (dyn_half < (p.NG * 4 + 63) / 64 * 64 + (4 * p.E + 63) / 64 * 64) {
        dyn_half *= 2;
    }
    p.hbuf = p.h_off + al(3 * p.h_tiles * kBf8Tile) + out_bytes + 2048 <= L1_BANK_BYTES ? 3 : 2;
    p.o_off = p.h_off + al(p.hbuf * p.h_tiles * kBf8Tile);
    const uint32_t dn_bytes = std::min(p.o_off + out_bytes + 2048, L1_BANK_BYTES);
    TT_FATAL(p.o_off + al(pcd * 2048) + 2048 <= L1_BANK_BYTES, "flat_routed_expert: down core arena does not fit");
    const uint32_t gu_bytes = p.hl_off;
    {  // (the program factory's static CBs, an upper bound: CB 9 counted whether or not the schedule pull is on)
        const uint32_t se_max_e = p.E <= 16 ? 16 : (p.E <= 32 ? 32 : 64);
        const uint32_t meta_bytes = se_max_e == 16 ? 512 : (se_max_e == 32 ? 1024 : 2048);
        const auto a64 = [](uint32_t b) { return (b + 63) / 64 * 64; };
        p.static_l1_bytes = a64(meta_bytes) + a64(4 * dyn_half) + 1024 + 2048 + a64(p.hbuf * p.mtg * p.hpc * kBf8Tile);
    }
    TT_FATAL(gu_bytes <= L1_BANK_BYTES, "flat_routed_expert: gate/up core arena does not fit");
    for (uint32_t t : {32u, 16u, 8u}) {
        if (p.Ht % t == 0 && t % kKBlk == 0) {
            p.sbt = t;
            break;
        }
    }
    p.seg = p.sbt * 64;
    p.nsb = p.Ht / p.sbt;
    p.sb_off = al(kRmChunks * 32 * p.seg);
    p.land_off = p.sb_off + al(kSbSlots * p.mt * p.sbt * kBf8Tile);
    // x read once by the helpers: two rectangles full of gate/up cores on the snakes, one helper each - each helper
    // reads / tilizes half of x (super-blocks g % 2 == k) and writes every super-block into both primaries' landing
    // rings (2 slots each); the primaries read nothing and feed only their own snake. x leaves DRAM once and no relay
    // sends more than a primary did before (GLM-5.3 / GLM-Flash / K3 / K2.7 at Galaxy loads 522 / 380 / 1216 / 807 ->
    // 483 / 355 / 1181 / 749 us, never slower at M 32..1024 or under six routings)
    for (uint32_t k = 0; k < 2; ++k) {
        const auto [x0, x1, y0, y1] = p.rects[k];
        uint32_t n = 0;
        for (const auto& c : p.gu) {
            n += p.rect_of(c) == k;
        }
        TT_FATAL(
            n == (x1 - x0 + 1) * (y1 - y0 + 1),
            "flat_routed_expert: gate/up rectangle {} is not full ({} of {} cells): the x snakes need full rectangles",
            k,
            n,
            (x1 - x0 + 1) * (y1 - y0 + 1));
    }
    p.ti_off = p.land_off + al(2 * p.land_slots * p.mt * p.sbt * kBf8Tile);
    const uint32_t relay_bytes = p.ti_off;
    TT_FATAL(relay_bytes <= L1_BANK_BYTES, "flat_routed_expert: relay arena does not fit");
    p.arena_tiles = (std::max({gu_bytes, dn_bytes, relay_bytes}) + 2047) / 2048;
    p.gu_bytes = gu_bytes;
    p.dn_bytes = dn_bytes;
    p.relay_bytes = relay_bytes;
    p.vstride = 1 + p.nh;
    p.region_bytes = p.E * p.nk_gu * p.rg * p.slot * p.w_tile;
    // The down cores' weight reads rotate over the DRAM banks instead of one bank per core (bank-pinned streams ran
    // at 55% of DRAM peak alone, rotating 65%). Bit-identical. Kimi K2.7 / GLM 5.3 us per expert at M 32 / 128: 62.3 /
    // 72.3 -> 58.2 / 68.7, 54.3 / 63.9 -> 51.2 / 61.4; none at M >= 512.
    {
        uint32_t sub = 0;  // tiles: a core's share of one bank, the widest core's
        for (uint32_t d = 0; d < p.down.size(); ++d) {
            const uint32_t kd = FlatRoutedExpertPlan::kd_of(p.pcds[d], p.It), nb = p.E * (p.It / kd);
            sub = std::max(sub, (nb + p.banks - 1) / p.banks * kd * p.pcds[d]);
        }
        p.wd_rot_region = sub * p.w_tile;
    }
    p.coords.push_back(p.down[0]);

    // ---- down chains: D_CH chains through the down cores ----
    {
        std::vector<uint32_t> ids(ND);
        for (uint32_t d = 0; d < ND; ++d) {
            ids[d] = d;
        }
        for (const auto& seg : chains(ids, p.down, phys, p.d_ch, 1)) {
            p.d_heads.push_back(seg.front());
            for (size_t i = 0; i + 1 < seg.size(); ++i) {
                p.d_succ[seg[i]] = seg[i + 1];
                p.d_pred[seg[i + 1]] = seg[i];
            }
        }
    }
    return p;
}

std::shared_ptr<const FlatRoutedExpertPlan> flat_routed_expert_plan(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const FlatRoutedExpertConfig& c) {
    // (the plan ignores the program-only fields: activation, y formats, fidelity, the combine overlap)
    using Key = std::
        tuple<int, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t, std::vector<uint32_t>>;
    static std::mutex mu;
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
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it == cache.end()) {
        auto plan = std::make_shared<const FlatRoutedExpertPlan>(make_flat_routed_expert_plan(mesh_device, c));
        log_debug(
            tt::LogOp,
            "flat_routed_expert plan ({}): gate/up row passes {} K group {} sub-block rows {}",
            c.layout.str(),
            plan->gu_rp,
            plan->gu_kg,
            plan->mt * 32);
        it = cache.emplace(key, std::move(plan)).first;
    }
    return it->second;
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
