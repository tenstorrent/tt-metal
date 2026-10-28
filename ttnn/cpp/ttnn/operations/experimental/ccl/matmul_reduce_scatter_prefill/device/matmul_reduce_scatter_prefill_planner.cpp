// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_prefill_planner.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include <fmt/format.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill::detail {

namespace {

// The search's own outcomes: a candidate that misses the L1 budget, and a geometry that cannot be laid out (no port
// strip, too few cores). solve_plan turns whichever reaches it into the op's TT_THROW.
struct L1Miss : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct GeometryMiss : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// bf16 tile: finished / arrival / zero slots of the staging tensor
constexpr uint64_t kStageTileBytes = tt::tile_size(tt::DataFormat::Float16_b);
// fp32 tile: cb::PARTIAL_HOLD's page
constexpr uint64_t kHoldTileBytes = tt::tile_size(tt::DataFormat::Float32);
constexpr uint32_t kDepthStream = 3;  // DEPTH_IN0 / DEPTH_IN1 of the stream regime
constexpr std::array<uint32_t, 2> kResidentStreamDepths = {3, 2};
constexpr uint32_t kSuperBlock = 2;       // K-windows per DEST-resident pack
constexpr uint32_t kSuperBlockExtra = 2;  // look-ahead windows beyond a super-block
constexpr uint32_t kDepthHold = 2;        // fp32 hold slots when some entry lags
constexpr uint32_t kZoneSets = 2;         // chunks' arrivals the zones hold at once
constexpr uint32_t kMaxAutoChunks = 256;
constexpr uint32_t kKbMax = 16;
constexpr uint32_t kKbMaxChunked = 32;
constexpr uint32_t kKbMinResident = 8;
constexpr uint32_t kDestTilesFp32FullSync = 8;  // subblock area cap
// chunk_cost_us() fit
constexpr double kCostFpuEff = 0.93;
constexpr double kCostStepUs = 1.4;
constexpr double kCostStepStreamUs = 2.1;
constexpr double kCostHandoffUs = 0.29;
constexpr double kCostDramGbps = 200.0;
constexpr double kCostFloorUs = 8.0;
constexpr double kClockMhz = 1350.0;

size_t dir_idx(Direction d) { return static_cast<size_t>(d); }

Role role_of(uint32_t p, uint32_t b, uint32_t h, uint32_t G, ScheduleKind kind) {
    if (b == p) {
        return {Direction::Fin, false, 0};
    }
    if (kind == ScheduleKind::Line) {
        if (b > p) {
            return {Direction::Fwd, p == 0, b - p};
        }
        return {Direction::Bwd, p == G - 1, p - b};
    }
    const uint32_t df = (b + G - p) % G;
    const uint32_t db = G - df;
    if (kind == ScheduleKind::RingUnbalanced) {
        const uint32_t nf = G / 2;
        const uint32_t nb = G - 1 - nf;
        if (df <= nf) {
            return {Direction::Fwd, df == nf, df};
        }
        return {Direction::Bwd, db == nb, db};
    }
    const uint32_t H = G / 2;
    if (h == 0) {
        if (df <= H) {
            return {Direction::Fwd, df == H, df};
        }
        return {Direction::Bwd, db == H - 1, db};
    }
    if (df <= H - 1) {
        return {Direction::Fwd, df == H - 1, df};
    }
    return {Direction::Bwd, db == H, db};
}

struct ChipWalk {
    std::vector<std::array<Role, 2>> roles;
    std::array<std::vector<uint32_t>, 2> entries;
};

// build_schedule + expand_waves
ChipWalk build_schedule(uint32_t p, uint32_t G, ScheduleKind kind, bool mirror, uint32_t waves) {
    std::vector<std::array<Role, 2>> roles(G);
    for (uint32_t b = 0; b < G; ++b) {
        for (uint32_t h = 0; h < 2; ++h) {
            roles[b][h] = role_of(p, b, h, G, kind);
        }
    }
    auto sort_key = [&](uint32_t b, uint32_t h) {
        if (!mirror) {
            const uint32_t key = std::max(roles[b][0].distance, roles[b][1].distance);
            return std::make_tuple(-static_cast<int64_t>(key), roles[b][0].dir == Direction::Fwd ? 0 : 1, b);
        }
        const Direction first = h == 0 ? Direction::Fwd : Direction::Bwd;
        return std::make_tuple(-static_cast<int64_t>(roles[b][h].distance), roles[b][h].dir == first ? 0 : 1, b);
    };
    ChipWalk out;
    out.roles.resize(static_cast<size_t>(G) * waves);
    for (uint32_t b = 0; b < G; ++b) {
        for (uint32_t w = 0; w < waves; ++w) {
            out.roles[(b * waves) + w] = roles[b];
        }
    }
    for (uint32_t h = 0; h < 2; ++h) {
        std::vector<uint32_t> others;
        for (uint32_t b = 0; b < G; ++b) {
            if (b != p) {
                others.push_back(b);
            }
        }
        std::sort(
            others.begin(), others.end(), [&](uint32_t x, uint32_t y) { return sort_key(x, h) < sort_key(y, h); });
        others.push_back(p);
        for (uint32_t b : others) {
            for (uint32_t w = 0; w < waves; ++w) {
                out.entries[h].push_back((b * waves) + w);
            }
        }
    }
    return out;
}

uint32_t index_of(const std::vector<uint32_t>& v, uint32_t x) {
    const auto it = std::find(v.begin(), v.end(), x);
    TT_FATAL(it != v.end(), "matmul_reduce_scatter_prefill: block {} missing from a walk", x);
    return static_cast<uint32_t>(it - v.begin());
}

uint32_t hold_depth(const Transport& t) {
    for (const auto& sc : t.chips) {
        for (uint32_t h = 0; h < 2; ++h) {
            for (const auto& ef : entry_finish_plan(sc, h)) {
                if (ef.lag != 0) {
                    return kDepthHold;
                }
            }
        }
    }
    return 1;
}

std::array<uint32_t, 2> window_depths(bool resident, uint32_t depth) {
    return resident ? std::array<uint32_t, 2>{depth, depth} : std::array<uint32_t, 2>{kDepthStream, kDepthStream};
}

// Only one wave per block is ever planned (ENTRIES_PER_BLOCK 1): the whole per-core block.
WavePlan wave_plan(uint32_t core_m, uint32_t core_n) {
    WavePlan wp;
    wp.n = 1;
    wp.win_m = core_m;
    wp.win_n = core_n;
    wp.P = core_m * core_n;
    wp.descs = {WaveDesc{0, 0, core_m, core_n}};
    return wp;
}

struct Geo {
    uint32_t core_m = 0;
    uint32_t core_n = 0;
    uint32_t tA = 0;
    uint32_t tW = 0;
    int32_t scatter_dim = -2;
    std::array<std::array<uint32_t, 2>, 2> slots{};  // [resident][dir]
    std::array<uint32_t, 2> hold_depths{};           // [resident]
    std::array<WavePlan, 2> waves_of;                // [resident]
    uint32_t zone_sets = 1;
};

uint64_t footprint(
    uint32_t kb,
    bool resident,
    uint32_t depth,
    uint32_t Kt,
    const Geo& geo,
    const std::optional<std::array<uint32_t, 2>>& depths = std::nullopt) {
    const WavePlan& wv = geo.waves_of[resident ? 1 : 0];
    const uint64_t P = wv.P;
    const bool rA = resident && geo.scatter_dim == -1;
    const bool rW = resident && geo.scatter_dim == -2;
    const auto [dA, dW] = depths.has_value() ? *depths : window_depths(resident, depth);
    const uint64_t a = uint64_t{geo.tA} * (rA ? uint64_t{geo.core_m} * Kt : uint64_t{dA} * wv.win_m * kb);
    const uint64_t w = uint64_t{geo.tW} * (rW ? uint64_t{Kt} * geo.core_n : uint64_t{dW} * kb * wv.win_n);
    const uint64_t hold = kHoldTileBytes * geo.hold_depths[resident ? 1 : 0] * P;
    const auto& sl = geo.slots[resident ? 1 : 0];
    const uint64_t staging = kStageTileBytes * ((kDepthHandoff + uint64_t{geo.zone_sets} * (sl[0] + sl[1])) * P);
    return a + w + hold + staging + kStageTileBytes;
}

struct KBlock {
    bool resident = false;
    uint32_t kb = 0;
    uint32_t depth = 0;  // resident only
};

KBlock choose_k_block(uint32_t Kt, uint64_t budget, uint32_t kb_max, const Geo& geo) {
    auto divisors = [Kt](uint32_t cap) {
        std::vector<uint32_t> out;
        for (uint32_t d = std::min(Kt, cap); d >= 1; --d) {
            if (Kt % d == 0) {
                out.push_back(d);
            }
        }
        for (uint32_t d = cap + 1; d <= Kt; ++d) {
            if (Kt % d == 0) {
                out.push_back(d);
            }
        }
        return out;
    };
    for (uint32_t depth : kResidentStreamDepths) {
        for (uint32_t kb : divisors(kb_max * geo.waves_of[1].n)) {
            if (kb < std::min(Kt, kKbMinResident)) {
                continue;
            }
            if (footprint(kb, true, depth, Kt, geo) <= budget) {
                return {true, kb, depth};
            }
        }
    }
    for (uint32_t kb : divisors(kb_max)) {
        if (footprint(kb, false, kResidentStreamDepths[0], Kt, geo) <= budget) {
            return {false, kb, 0};
        }
    }
    throw L1Miss("matmul_reduce_scatter_prefill: even kb=1 exceeds the L1 budget (regime DRAM-landing is deferred)");
}

GridChoice choose_grid(uint32_t gx, uint32_t gy, uint32_t block_m, uint32_t block_n, uint32_t L) {
    std::optional<std::pair<std::tuple<bool, uint64_t, int64_t, uint64_t, uint64_t, bool>, GridChoice>> best;
    for (Strip strip : {Strip::Row, Strip::Col}) {
        const uint32_t strip_len = strip == Strip::Row ? gx : gy;
        if (2 * L > strip_len) {
            continue;
        }
        const uint32_t rx = strip == Strip::Row ? gx : gx - 1;
        const uint32_t ry = strip == Strip::Row ? gy - 1 : gy;
        for (Orient orient : {Orient::MOnX, Orient::MOnY}) {
            const uint32_t ext_m = orient == Orient::MOnX ? rx : ry;
            const uint32_t ext_n = orient == Orient::MOnX ? ry : rx;
            GridChoice c;
            c.strip = strip;
            c.orient = orient;
            c.core_m = tt::div_up(block_m, std::min(ext_m, block_m));
            c.grid_m = tt::div_up(block_m, c.core_m);
            c.core_n = tt::div_up(block_n, std::min(ext_n, block_n));
            c.grid_n = tt::div_up(block_n, c.core_n);
            c.subblock = choose_subblock(c.core_m, c.core_n);
            const uint64_t ragged =
                (uint64_t{c.grid_m} * c.core_m * c.grid_n * c.core_n) - (uint64_t{block_m} * block_n);
            const auto key = std::make_tuple(
                strip != Strip::Row,
                uint64_t{c.core_m} * c.core_n,
                -static_cast<int64_t>(c.subblock[0] * c.subblock[1]),
                ragged,
                uint64_t{c.grid_m} * c.grid_n,
                strip != Strip::Row);
            if (!best.has_value() || key < best->first) {
                best = std::make_pair(key, c);
            }
        }
    }
    if (!best.has_value()) {
        throw GeometryMiss(fmt::format(
            "matmul_reduce_scatter_prefill: no transport strip of the {}x{} grid fits {} ports", gx, gy, 2 * L));
    }
    return best->second;
}

std::vector<std::pair<uint32_t, uint32_t>> chunk_candidates(uint32_t block_m, uint32_t block_n, int32_t scatter_dim) {
    const bool inner_n = scatter_dim == -1;
    std::set<std::pair<uint32_t, uint32_t>> seen;
    std::vector<std::pair<uint32_t, uint32_t>> out;
    const uint64_t limit = std::min<uint64_t>(kMaxAutoChunks, uint64_t{block_m} * block_n);
    for (uint32_t total = 1; total <= limit; ++total) {
        std::vector<std::pair<uint32_t, uint32_t>> pairs;
        for (uint32_t d = 1; d <= total; ++d) {
            if (total % d == 0) {
                pairs.emplace_back(total / d, d);
            }
        }
        std::stable_sort(pairs.begin(), pairs.end(), [inner_n](const auto& x, const auto& y) {
            return (inner_n ? x.second : x.first) > (inner_n ? y.second : y.first);
        });
        for (const auto& [n_m, n_n] : pairs) {
            if (n_m > block_m || n_n > block_n) {
                continue;
            }
            if (!seen.insert({tt::div_up(block_m, n_m), tt::div_up(block_n, n_n)}).second) {
                continue;
            }
            out.emplace_back(n_m, n_n);
        }
    }
    return out;
}

Plan plan_chunk_geometry(uint32_t n_m_req, uint32_t n_n_req, const PlanInputs& in) {
    Plan pl;
    pl.chunk_m = tt::div_up(in.block_m, n_m_req);
    pl.chunk_n = tt::div_up(in.block_n, n_n_req);
    pl.n_m = tt::div_up(in.block_m, pl.chunk_m);
    pl.n_n = tt::div_up(in.block_n, pl.chunk_n);
    const uint32_t chunks = pl.n_m * pl.n_n;
    pl.zone_sets = chunks == 1 ? 1 : std::max(1u, std::min(kZoneSets, chunks));
    pl.grid = choose_grid(in.gx, in.gy, pl.chunk_m, pl.chunk_n, in.L);
    const GridChoice& gd = pl.grid;
    if (in.kind == ScheduleKind::RingBalanced && gd.grid_m * gd.grid_n < 2) {
        throw GeometryMiss("matmul_reduce_scatter_prefill: a balanced ring needs >= 2 matmul cores (two halves)");
    }
    HalfSplit mirror_split = HalfSplit::K;
    if (in.kind == ScheduleKind::RingBalanced) {
        if (in.scatter_dim == -1 && gd.grid_n >= 2) {
            mirror_split = HalfSplit::N;
        } else if (in.scatter_dim == -2 && gd.grid_m >= 2) {
            mirror_split = HalfSplit::M;
        }
    }
    Geo geo;
    geo.core_m = gd.core_m;
    geo.core_n = gd.core_n;
    geo.tA = in.tA;
    geo.tW = in.tW;
    geo.scatter_dim = in.scatter_dim;
    geo.zone_sets = pl.zone_sets;
    std::array<Transport, 2> sched_of;
    for (uint32_t r = 0; r < 2; ++r) {
        geo.waves_of[r] = wave_plan(gd.core_m, gd.core_n);
        sched_of[r] = plan_transport(in.G, in.kind, r == 1 && mirror_split != HalfSplit::K, geo.waves_of[r].n);
        geo.hold_depths[r] = hold_depth(sched_of[r]);
        geo.slots[r] = sched_of[r].slots;
    }
    const KBlock kblk = choose_k_block(in.Kt, in.budget, chunks == 1 ? kKbMax : kKbMaxChunked, geo);
    pl.resident = kblk.resident;
    pl.kb = kblk.kb;
    // super_block_plan
    const auto base = window_depths(kblk.resident, kblk.depth);
    pl.super_block = 1;
    pl.depths = base;
    if ((in.Kt / kblk.kb) % kSuperBlock == 0) {
        const uint32_t d = kSuperBlock + kSuperBlockExtra;
        const std::array<uint32_t, 2> deep = {std::max(d, base[0]), std::max(d, base[1])};
        if (footprint(kblk.kb, kblk.resident, kblk.depth, in.Kt, geo, deep) <= in.budget) {
            pl.super_block = kSuperBlock;
            pl.depths = deep;
        }
    }
    const uint32_t r = kblk.resident ? 1 : 0;
    pl.depth_hold = geo.hold_depths[r];
    pl.half_split = kblk.resident ? mirror_split : HalfSplit::K;
    pl.sched = sched_of[r];
    pl.waves = geo.waves_of[r];
    pl.P = pl.waves.P;
    pl.footprint = footprint(kblk.kb, kblk.resident, kblk.depth, in.Kt, geo, pl.depths);
    return pl;
}

double chunk_cost_us(const Plan& pl, const PlanInputs& in) {
    const double C = static_cast<double>(pl.n_m) * pl.n_n;
    const double cores = static_cast<double>(pl.grid.grid_m) * pl.grid.grid_n;
    const uint32_t k_steps = in.Kt / pl.kb;  // K-blocks per entry (kb divides Kt)
    const double steps = C * in.G * pl.waves.n * k_steps;
    const double step_us = pl.resident ? kCostStepUs : kCostStepStreamUs;
    const double compute =
        (C * in.G * pl.P * pl.waves.n * in.Kt * tt::constants::TILE_WIDTH / kClockMhz / kCostFpuEff) +
        (step_us * steps);
    const double a_bytes = static_cast<double>(in.Mt) * in.Kt * in.tA;
    const double w_bytes = static_cast<double>(in.Kt) * in.Nt * in.tW;
    double traffic = 0.0;
    if (in.scatter_dim == -1) {
        traffic = (pl.resident ? a_bytes : in.G * static_cast<double>(pl.n_n) * a_bytes) + (pl.n_m * w_bytes);
    } else {
        traffic = (pl.resident ? w_bytes : in.G * static_cast<double>(pl.n_m) * w_bytes) + (pl.n_n * a_bytes);
    }
    const double dram = traffic / (kCostDramGbps * 1e3);
    const double transport = kCostHandoffUs * C * (in.G - 1) * cores / in.L;
    return kCostFloorUs + std::max({compute, dram, transport});
}

// The unchunked plan, or nothing when it misses the L1 budget (the chunk search takes over).
std::optional<Plan> plan_if_fits(const PlanInputs& in) {
    try {
        return plan_chunk_geometry(1, 1, in);
    } catch (const L1Miss&) {
        return std::nullopt;
    }
}

}  // namespace

Transport plan_transport(uint32_t G, ScheduleKind kind, bool mirror, uint32_t waves) {
    const bool ring = kind != ScheduleKind::Line;
    mirror = mirror && kind == ScheduleKind::RingBalanced;
    std::vector<ChipWalk> sch;
    sch.reserve(G);
    for (uint32_t p = 0; p < G; ++p) {
        sch.push_back(build_schedule(p, G, kind, mirror, waves));
    }
    auto up = [&](uint32_t p, Direction d) -> std::optional<uint32_t> {
        const int64_t q = d == Direction::Fwd ? int64_t{p} - 1 : int64_t{p} + 1;
        if (ring) {
            return static_cast<uint32_t>((q + G) % G);
        }
        if (q >= 0 && q < G) {
            return static_cast<uint32_t>(q);
        }
        return std::nullopt;
    };

    Transport t;
    t.chips.resize(G);
    for (uint32_t p = 0; p < G; ++p) {
        ChipSchedule& sc = t.chips[p];
        sc.roles = sch[p].roles;
        sc.entries = sch[p].entries;
        for (uint32_t h = 0; h < 2; ++h) {
            for (Direction d : {Direction::Fwd, Direction::Bwd}) {
                const auto u = up(p, d);
                std::vector<uint32_t> lst;
                for (uint32_t b : sc.entries[h]) {
                    const Role& r = sc.roles[b][h];
                    if (r.dir == d && !r.is_start) {
                        TT_FATAL(
                            u.has_value() && sch[*u].roles[b][h].dir == d,
                            "matmul_reduce_scatter_prefill: schedule ({}, {}, {}, {})",
                            p,
                            b,
                            h,
                            dir_idx(d));
                        lst.push_back(b);
                    } else if (r.dir == Direction::Fin && u.has_value() && sch[*u].roles[b][h].dir == d) {
                        lst.push_back(b);
                    }
                }
                std::vector<uint32_t> seq;
                if (u.has_value()) {
                    for (uint32_t b : sch[*u].entries[h]) {
                        if (sch[*u].roles[b][h].dir == d) {
                            seq.push_back(b);
                        }
                    }
                }
                auto ss = seq;
                auto sl = lst;
                std::sort(ss.begin(), ss.end());
                std::sort(sl.begin(), sl.end());
                TT_FATAL(
                    ss == sl, "matmul_reduce_scatter_prefill: arrival lists disagree ({}, {}, {})", p, h, dir_idx(d));
                auto& thr = sc.thresholds[h][dir_idx(d)];
                auto& prod = sc.produced_at[h][dir_idx(d)];
                for (uint32_t i = 0; i < seq.size(); ++i) {
                    thr[seq[i]] = i + 1;  // every unit carries its increment
                    prod[seq[i]] = index_of(sch[*u].entries[h], seq[i]);
                }
                if (mirror) {
                    for (uint32_t b : lst) {
                        if (sc.roles[b][h].dir != Direction::Fin) {
                            TT_FATAL(
                                prod.at(b) < index_of(sc.entries[h], b),
                                "matmul_reduce_scatter_prefill: mirrored schedule deadlock ({}, {}, {}, {})",
                                p,
                                b,
                                h,
                                dir_idx(d));
                        }
                    }
                }
                sc.arrivals[h][dir_idx(d)] = std::move(lst);
            }
        }
    }
    for (size_t d = 0; d < 2; ++d) {
        uint32_t m = 1;
        for (const auto& sc : t.chips) {
            for (uint32_t h = 0; h < 2; ++h) {
                m = std::max(m, static_cast<uint32_t>(sc.arrivals[h][d].size()));
            }
        }
        t.slots[d] = m;
    }
    return t;
}

std::vector<EntryFinish> entry_finish_plan(const ChipSchedule& sc, uint32_t half) {
    const auto& entries = sc.entries[half];
    std::vector<EntryFinish> out;
    out.reserve(entries.size());
    for (uint32_t e = 0; e < entries.size(); ++e) {
        const uint32_t b = entries[e];
        const Role& r = sc.roles[b][half];
        uint32_t mask = 0;
        if (r.dir != Direction::Fin && !r.is_start) {
            mask = r.dir == Direction::Fwd ? 1 : 2;
        } else if (r.dir == Direction::Fin) {
            for (size_t d = 0; d < 2; ++d) {
                const auto& arr = sc.arrivals[half][d];
                if (std::find(arr.begin(), arr.end(), b) != arr.end()) {
                    mask |= 1u << d;
                }
            }
        }
        bool lag = false;
        for (size_t d = 0; d < 2; ++d) {
            if ((mask & (1u << d)) != 0 && sc.produced_at[half][d].at(b) >= e) {
                lag = true;
            }
        }
        out.push_back({mask, (lag && e + 1 < entries.size()) ? 1u : 0u});
    }
    return out;
}

std::array<uint32_t, 2> choose_subblock(uint32_t rows, uint32_t cols) {
    std::array<uint32_t, 2> best = {1, 1};
    for (uint32_t h = 1; h <= rows; ++h) {
        if (rows % h != 0) {
            continue;
        }
        for (uint32_t w = 1; w <= cols; ++w) {
            if (cols % w != 0 || h * w > kDestTilesFp32FullSync) {
                continue;
            }
            if (h * w > best[0] * best[1] || (h * w == best[0] * best[1] && w > best[1])) {
                best = {h, w};
            }
        }
    }
    return best;
}

uint32_t staging_tiles(const Plan& plan) {
    return (kDepthHandoff + (plan.zone_sets * (plan.sched.slots[0] + plan.sched.slots[1]))) * plan.P;
}

namespace {

Plan solve_plan_uncached(const PlanInputs& in) {
    if (auto unchunked = plan_if_fits(in)) {
        return std::move(*unchunked);  // whatever fits unchunked keeps its plan
    }
    std::optional<std::pair<double, Plan>> best;
    for (const auto& [n_m, n_n] : chunk_candidates(in.block_m, in.block_n, in.scatter_dim)) {
        if (n_m * n_n == 1) {
            continue;
        }
        std::optional<Plan> cand;
        try {
            cand = plan_chunk_geometry(n_m, n_n, in);
        } catch (const L1Miss&) {
            continue;
        } catch (const GeometryMiss&) {
            continue;
        }
        const double cost = chunk_cost_us(*cand, in);
        if (!best.has_value() || cost < best->first) {
            best = std::make_pair(cost, std::move(*cand));
        }
    }
    if (!best.has_value()) {
        throw L1Miss(fmt::format(
            "matmul_reduce_scatter_prefill: no split of the {}x{}-tile block into <= {} chunks fits L1",
            in.block_m,
            in.block_n,
            kMaxAutoChunks));
    }
    return std::move(best->second);
}

}  // namespace

const Plan& solve_plan(const PlanInputs& in) {
    // every field of PlanInputs: the solve is a pure function of them
    const auto key = std::make_tuple(
        in.G,
        in.kind,
        in.L,
        in.Mt,
        in.Kt,
        in.Nt,
        in.block_m,
        in.block_n,
        in.gx,
        in.gy,
        in.tA,
        in.tW,
        in.scatter_dim,
        in.budget);
    static std::mutex solved_mutex;
    static std::map<std::remove_const_t<decltype(key)>, Plan> solved;
    const std::lock_guard<std::mutex> lock(solved_mutex);
    auto it = solved.find(key);
    if (it == solved.end()) {
        try {
            it = solved.emplace(key, solve_plan_uncached(in)).first;  // a refusal throws and caches nothing
        } catch (const L1Miss& miss) {
            TT_THROW("{}", miss.what());
        } catch (const GeometryMiss& miss) {
            TT_THROW("{}", miss.what());
        }
    }
    return it->second;
}

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill::detail
