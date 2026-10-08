// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert {

namespace plan_detail {
using Core = tt::tt_metal::CoreCoord;

constexpr uint32_t KBLK = kKBlk, MT_MAX = 4, BF8_TILE = kBf8Tile, H_TILE = kBf8Tile;
constexpr uint32_t L1_BANK = 1427 * 1024;       // usable L1 per core for the arena (Blackhole)
constexpr uint32_t GU_L1_BUDGET = 1400 * 1024;  // gate/up core: weight ring + x ring
constexpr uint32_t D_CHAINS = 7, RM_CHUNKS = kRmChunks, SB_SLOTS = kSbSlots;
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
    uint32_t it, uint32_t ht, uint32_t w_tile, uint32_t x_slots, uint32_t n_readers, uint32_t max_cores, bool fp32) {
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
            if (2 * ht * 2 * np * w_tile + x_slots * mt_ * KBLK * BF8_TILE > GU_L1_BUDGET) {
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
    p.banks = device->dram_grid_size().x;
    const auto phys = [device](const Core& c) { return device->worker_core_from_logical_core(c); };

    // subgrids: 2 for It <= 16 (TP4-like), 3 for It <= 32 (TP2-like), else 1
    p.nsg = p.It <= 16 ? 2 : (p.It <= 32 ? 3 : 1);
    p.n_rd_sg = p.nsg > 1 ? (p.nsg == 2 ? 8 : 4) : 16;
    if (p.nsg == 1) {
        p.rects = {{2, 5, 0, 9}, {8, 10, 0, 7}};
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
    bool found = false;
    for (uint32_t xs : {24u, 16u, 12u}) {
        const auto split = p.nsg == 1 ? gu_split(p.It, p.Ht, p.w_tile, xs, 16, 64, p.gu_fp32)
                                      : gu_split(p.It, p.Ht, p.w_tile, xs, p.n_rd_sg, min_rect, p.gu_fp32);
        if (split) {
            std::tie(p.np, p.g) = *split;
            p.x_slots = xs;
            found = true;
            break;
        }
    }
    TT_FATAL(found, "flat_routed_expert: no gate/up split for {} tile columns", p.It);
    std::tie(p.dst_tiles, p.gu_rp) = gu_dst(p.np, p.Ht, p.x_slots, p.gu_fp32);
    const uint32_t mt_cap = p.g * std::min(p.dst_tiles / (2 * p.np), MT_MAX / p.g);
    p.mt = std::min(mt_cap, std::max(p.g, p.m / 32 / p.g * p.g));
    TT_FATAL(p.mt % p.g == 0 && (p.mt / p.g) * 2 * p.np <= p.dst_tiles, "flat_routed_expert: sub-block split");
    p.mtg = p.mt / p.g;
    const uint32_t m_pad = (p.m + p.mt * 32 - 1) / (p.mt * 32) * p.mt * 32;
    p.rdown = p.Ht > 6 * 26;  // readers compute down columns when down is heavy (> 6 columns per down core)
    p.nh = p.g > 1 ? 2 : 1;
    p.land_slots = p.nh > 1 ? 2 : 3;
    p.dring = cfg.pin ? 2.0f : 1.5f;

    // ---- layout ----
    const auto grid = device->compute_with_storage_grid_size();
    const auto opt = device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);
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
            *rcols.rbegin() == east + 1 && grid.x >= 11 + (east - 6) && grid.y >= 10,
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
                if (!taken.contains({x, y})) {
                    return Core(x, y);
                }
            }
            TT_THROW("flat_routed_expert: no free relay cell in column {}", x);
        };
        p.relays = {first_free(1, {4, 5, 3, 6, 2, 7}), first_free(east + 1, {3, 4, 2, 5, 1, 6})};
        for (const auto& c : p.relays) {
            taken.insert({c.x, c.y});
        }
        for (uint32_t j = 0; j < p.nh; ++j) {  // helpers: relay NR + NR j + k is rectangle k's j-th helper
            const Core a = first_free(1, {5, 3, 6, 2, 7, 1, 8, 0, 9});
            const Core b = first_free(east + 1, {4, 2, 5, 1, 6, 0, 7, 8, 9});
            p.relays.push_back(a);
            p.relays.push_back(b);
            taken.insert({a.x, a.y});
            taken.insert({b.x, b.y});
        }
    }
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            if (!taken.contains({x, y})) {
                p.down.emplace_back(x, y);
            }
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
    p.n_rdn = p.rdown ? p.d_ch : 0;
    p.pcd_r = p.rdown ? 6 : 0;
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
    const uint32_t x_bytes = p.mt * KBLK * BF8_TILE;
    p.h_tiles = p.It * p.mt;
    // the out region also holds the reader tails' out CB (pcd_r columns), wider than the down cores' when the down
    // cores are many (a Galaxy chip's 12 x 10 at H 6144: pcd 5 < pcd_r 6)
    const uint32_t out_tiles = p.mt * std::max(pcd, p.pcd_r);
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
    p.p_off = p.x_off + al(p.x_slots * x_bytes);
    const uint32_t gu_bytes = p.p_off + (p.gu_l1acc ? al(p.mtg * 2 * p.np * 2048) : 0);
    p.rd_off = al(p.rd_slots * p.rg * p.slot * p.w_tile);
    uint32_t dn_ring_max = 0;
    for (uint32_t d = 0; d < ND; ++d) {
        const uint32_t kd = FlatRoutedExpertPlan::kd_of(p.pcds[d], p.It);
        dn_ring_max = std::max(dn_ring_max, pyround(p.dring * (p.It / kd)) * kd * p.pcds[d]);
    }
    p.h_off = al(std::max(dn_ring_max * p.w_tile, p.rd_off + (p.rdown ? p.ring_dr * p.slot_dr * p.w_tile : 0)));
    const uint32_t out_bytes = al(2 * out_tiles * BF8_TILE);
    p.hbuf = p.h_off + al(3 * p.h_tiles * H_TILE) + out_bytes + 2048 <= L1_BANK ? 3 : 2;
    p.o_off = p.h_off + al(p.hbuf * p.h_tiles * H_TILE);
    const uint32_t dn_bytes = p.o_off + out_bytes + 2048;
    for (uint32_t t : {32u, 16u, 8u}) {
        if (p.Ht % t == 0 && t % KBLK == 0) {
            p.sbt = t;
            break;
        }
    }
    p.seg = p.sbt * 64;
    p.nsb = p.Ht / p.sbt;
    p.sb_off = al(RM_CHUNKS * 32 * p.seg);
    p.land_off = p.sb_off + al(SB_SLOTS * p.mt * p.sbt * BF8_TILE);
    const uint32_t relay_bytes = p.land_off + al(p.nh * p.land_slots * p.mt * p.sbt * BF8_TILE);
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

}  // namespace ttnn::operations::experimental::deepseek_prefill::flat_routed_expert
