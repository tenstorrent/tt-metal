// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_planner.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt_stl/assert.hpp>

#include "matmul_reduce_scatter_device_operation_types.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

namespace {

// Stream regime: A / W K-block windows (one computing, one landing, one read ahead by a sender).
constexpr uint32_t kStreamDepthIn0 = 3;
constexpr uint32_t kStreamDepthIn1 = 3;
// Resident regime: depths the streamed operand's window may take, preferred first.
constexpr std::array<uint32_t, 2> kResidentStreamDepths{3, 2};
// DEST-resident K: K-windows per pack, and the extra look-ahead windows the streamed CBs then need.
constexpr uint32_t kSuperBlock = 2;
constexpr uint32_t kSuperBlockExtra = 2;
// fp32 partial slots when some entry lags (lag 1 + current); else 1.
constexpr uint32_t kDepthHoldLagged = 2;
constexpr uint32_t kZoneSets = 2;  // chunks whose arrivals the zones hold at once (chunked plans)
constexpr uint32_t kMaxAutoChunks = 256;
constexpr uint32_t kKbMax = 16;                 // K-block cap of an unchunked plan
constexpr uint32_t kKbMaxChunked = 32;          // K-block cap of a chunked plan
constexpr uint32_t kKbMinResident = 8;          // smallest K-block the resident regime accepts
constexpr uint32_t kDestTilesFp32FullSync = 8;  // subblock area cap
// chunk_cost_us() constants (least-squares fit of traced configs); they only rank the auto split's candidates.
constexpr double kCostFpuEff = 0.93;
constexpr double kCostStepUs = 1.4;
constexpr double kCostStepStreamUs = 2.1;
constexpr double kCostHandoffUs = 0.29;
constexpr double kCostDramGbps = 200;
constexpr double kCostFloorUs = 8.0;
constexpr double kClockMhz = 1350;
// HiFi2 matmul cycles per tile per K tile
constexpr double kCyclesPerTileK = 32;
constexpr uint64_t kHoldTileBytes = tt::tile_size(tt::DataFormat::Float32);     // fp32 partial hold tile
constexpr uint64_t kStageTileBytes = tt::tile_size(tt::DataFormat::Float16_b);  // bf16 staging / zero tile

Role role_of(uint32_t p, uint32_t b, uint32_t h, uint32_t G, ScheduleKind kind) {
    if (b == p) {
        return {RoleKind::FIN, false, 0};
    }
    if (kind == ScheduleKind::LINE) {
        if (b > p) {
            return {RoleKind::FWD, p == 0, b - p};
        }
        return {RoleKind::BWD, p == G - 1, p - b};
    }
    const uint32_t df = (b + G - p) % G;
    const uint32_t db = G - df;
    if (kind == ScheduleKind::RING_UNBALANCED) {
        const uint32_t nf = G / 2;
        const uint32_t nb = G - 1 - nf;
        if (df <= nf) {
            return {RoleKind::FWD, df == nf, df};
        }
        return {RoleKind::BWD, db == nb, db};
    }
    const uint32_t H = G / 2;
    if (h == 0) {
        if (df <= H) {
            return {RoleKind::FWD, df == H, df};
        }
        return {RoleKind::BWD, db == H - 1, db};
    }
    if (df <= H - 1) {
        return {RoleKind::FWD, df == H - 1, df};
    }
    return {RoleKind::BWD, db == H, db};
}

// build_schedule: roles and the entry order per half.
ChipSchedule build_chip_schedule(uint32_t p, uint32_t G, ScheduleKind kind, bool mirror) {
    ChipSchedule sc;
    sc.roles.resize(G);
    for (uint32_t b = 0; b < G; ++b) {
        for (uint32_t h = 0; h < 2; ++h) {
            sc.roles[b][h] = role_of(p, b, h, G, kind);
        }
    }
    for (uint32_t h = 0; h < 2; ++h) {
        std::vector<std::tuple<int64_t, int, uint32_t>> keyed;
        for (uint32_t b = 0; b < G; ++b) {
            if (b == p) {
                continue;
            }
            if (!mirror) {
                const uint32_t key = std::max(sc.roles[b][0].distance, sc.roles[b][1].distance);
                keyed.emplace_back(-static_cast<int64_t>(key), sc.roles[b][0].kind == RoleKind::FWD ? 0 : 1, b);
            } else {
                const RoleKind first = h == 0 ? RoleKind::FWD : RoleKind::BWD;
                keyed.emplace_back(
                    -static_cast<int64_t>(sc.roles[b][h].distance), sc.roles[b][h].kind == first ? 0 : 1, b);
            }
        }
        std::sort(keyed.begin(), keyed.end());
        for (const auto& k : keyed) {
            sc.entries[h].push_back(std::get<2>(k));
        }
        sc.entries[h].push_back(p);
    }
    return sc;
}

uint32_t index_in(const std::vector<uint32_t>& v, uint32_t x) {
    const auto it = std::find(v.begin(), v.end(), x);
    TT_FATAL(it != v.end(), "matmul_reduce_scatter: block {} missing from a schedule", x);
    return static_cast<uint32_t>(std::distance(v.begin(), it));
}

uint32_t hold_depth_of(const TransportPlan& tp) {
    for (const auto& sc : tp.chips) {
        for (uint32_t h = 0; h < 2; ++h) {
            for (const auto& f : entry_finish_plan(sc, h)) {
                if (f.lag) {
                    return kDepthHoldLagged;
                }
            }
        }
    }
    return 1;
}

std::array<uint32_t, 2> window_depths(bool resident, uint32_t depth) {
    if (resident) {
        return {depth, depth};
    }
    return {kStreamDepthIn0, kStreamDepthIn1};
}

// Per-regime geometry the footprint needs (index 0 = stream, 1 = resident).
struct Geometry {
    uint32_t Kt = 0;
    uint32_t core_m = 0;
    uint32_t core_n = 0;
    uint64_t tA = 0;
    uint64_t tW = 0;
    int32_t scatter_dim = -2;
    std::array<std::array<uint32_t, 2>, 2> slots{};  // [resident][dir]
    std::array<uint32_t, 2> hold_depths{};           // [resident]
    uint32_t zone_sets = 1;
};

uint64_t core_footprint(uint32_t kb, bool resident, const std::array<uint32_t, 2>& depths, const Geometry& g) {
    const uint64_t P = static_cast<uint64_t>(g.core_m) * g.core_n;
    const bool rA = resident && g.scatter_dim == -1;
    const bool rW = resident && g.scatter_dim == -2;
    const uint64_t a = g.tA * (rA ? static_cast<uint64_t>(g.core_m) * g.Kt : uint64_t{depths[0]} * g.core_m * kb);
    const uint64_t w = g.tW * (rW ? static_cast<uint64_t>(g.Kt) * g.core_n : uint64_t{depths[1]} * kb * g.core_n);
    const uint64_t hold = kHoldTileBytes * g.hold_depths[resident ? 1 : 0] * P;
    const auto& sl = g.slots[resident ? 1 : 0];
    const uint64_t staging = kStageTileBytes * (kDepthHandoff + uint64_t{g.zone_sets} * (sl[0] + sl[1])) * P;
    return a + w + hold + staging + kStageTileBytes;  // + the zero tile
}

// Preference order: the largest divisor <= cap first, then smaller ones, then (no fit below the cap) larger.
std::vector<uint32_t> kblock_divisors(uint32_t Kt, uint32_t cap) {
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
}

struct KBlockChoice {
    bool resident = false;
    uint32_t kb = 0;
    uint32_t depth = 0;
};

KBlockChoice choose_k_block(uint32_t Kt, uint64_t budget, uint32_t kb_max, const Geometry& g) {
    for (const uint32_t depth : kResidentStreamDepths) {
        for (const uint32_t kb : kblock_divisors(Kt, kb_max)) {
            if (kb < std::min(Kt, kKbMinResident)) {
                continue;
            }
            if (core_footprint(kb, true, window_depths(true, depth), g) <= budget) {
                return {true, kb, depth};
            }
        }
    }
    for (const uint32_t kb : kblock_divisors(Kt, kb_max)) {
        if (core_footprint(kb, false, window_depths(false, 0), g) <= budget) {
            return {false, kb, 0};
        }
    }
    throw NotImplementedRefusal(
        "matmul_reduce_scatter: even kb=1 exceeds the L1 budget (regime DRAM-landing is deferred)");
}

GridPlan choose_grid(uint32_t gx, uint32_t gy, uint32_t block_m, uint32_t block_n, uint32_t L) {
    using Key = std::tuple<bool, uint32_t, int64_t, uint64_t, uint32_t, bool>;
    std::optional<std::pair<Key, GridPlan>> best;
    for (const bool row : {true, false}) {
        const uint32_t strip_len = row ? gx : gy;
        if (2 * L > strip_len) {
            continue;
        }
        const uint32_t rect_x = row ? gx : gx - 1;
        const uint32_t rect_y = row ? gy - 1 : gy;
        for (const bool m_on_x : {true, false}) {
            const uint32_t ext_m = m_on_x ? rect_x : rect_y;
            const uint32_t ext_n = m_on_x ? rect_y : rect_x;
            TT_FATAL(ext_m > 0 && ext_n > 0, "matmul_reduce_scatter: the {}x{} grid leaves no matmul cores", gx, gy);
            const uint32_t core_m = tt::div_up(block_m, std::min(ext_m, block_m));
            const uint32_t grid_m = tt::div_up(block_m, core_m);
            const uint32_t core_n = tt::div_up(block_n, std::min(ext_n, block_n));
            const uint32_t grid_n = tt::div_up(block_n, core_n);
            const auto sb = choose_subblock(core_m, core_n);
            const uint64_t ragged = (uint64_t{grid_m} * core_m * grid_n * core_n) - (uint64_t{block_m} * block_n);
            const Key key{!row, core_m * core_n, -static_cast<int64_t>(sb[0] * sb[1]), ragged, grid_m * grid_n, !row};
            if (!best.has_value() || key < best->first) {
                best = std::pair{
                    key,
                    GridPlan{
                        .row_strip = row,
                        .m_on_x = m_on_x,
                        .grid_m = grid_m,
                        .grid_n = grid_n,
                        .core_m = core_m,
                        .core_n = core_n,
                        .sb_h = sb[0],
                        .sb_w = sb[1]}};
            }
        }
    }
    if (!best.has_value()) {
        throw std::invalid_argument(
            fmt::format("matmul_reduce_scatter: no transport strip of the {}x{} grid fits {} ports", gx, gy, 2 * L));
    }
    return best->second;
}

// plan_chunk_geometry: grid, regime and schedule for one chunk.
Plan plan_chunk(uint32_t n_m_req, uint32_t n_n_req, const PlanInputs& in) {
    Plan plan;
    plan.chunk_m = tt::div_up(in.block_m, n_m_req);
    plan.chunk_n = tt::div_up(in.block_n, n_n_req);
    plan.n_m = tt::div_up(in.block_m, plan.chunk_m);  // no empty trailing chunk
    plan.n_n = tt::div_up(in.block_n, plan.chunk_n);
    const uint32_t num_chunks = plan.n_m * plan.n_n;
    plan.zone_sets = num_chunks == 1 ? 1 : std::max(1u, std::min(kZoneSets, num_chunks));
    plan.grid = choose_grid(in.grid_x, in.grid_y, plan.chunk_m, plan.chunk_n, in.num_links);
    const GridPlan& gd = plan.grid;
    if (in.kind == ScheduleKind::RING_BALANCED && gd.grid_m * gd.grid_n < 2) {
        throw std::invalid_argument("matmul_reduce_scatter: a balanced ring needs >= 2 matmul cores (two halves)");
    }
    HalfSplit mirror_split = HalfSplit::K;
    if (in.kind == ScheduleKind::RING_BALANCED) {
        if (in.scatter_dim == -1 && gd.grid_n >= 2) {
            mirror_split = HalfSplit::N;
        } else if (in.scatter_dim == -2 && gd.grid_m >= 2) {
            mirror_split = HalfSplit::M;
        }
    }
    Geometry g{
        .Kt = in.Kt,
        .core_m = gd.core_m,
        .core_n = gd.core_n,
        .tA = in.a_tile_bytes,
        .tW = in.w_tile_bytes,
        .scatter_dim = in.scatter_dim,
        .zone_sets = plan.zone_sets};
    std::array<TransportPlan, 2> sched;
    for (uint32_t r = 0; r < 2; ++r) {
        sched[r] = plan_transport(in.group_size, in.kind, r == 1 && mirror_split != HalfSplit::K);
        g.slots[r] = sched[r].slots;
        g.hold_depths[r] = hold_depth_of(sched[r]);
    }
    const KBlockChoice kc = choose_k_block(in.Kt, in.l1_budget, num_chunks == 1 ? kKbMax : kKbMaxChunked, g);
    // super_block_plan
    const auto base = window_depths(kc.resident, kc.depth);
    plan.super_block = 1;
    std::array<uint32_t, 2> depths = base;
    if ((in.Kt / kc.kb) % kSuperBlock == 0) {
        const uint32_t d = kSuperBlock + kSuperBlockExtra;
        const std::array<uint32_t, 2> deep{std::max(d, base[0]), std::max(d, base[1])};
        if (core_footprint(kc.kb, kc.resident, deep, g) <= in.l1_budget) {
            plan.super_block = kSuperBlock;
            depths = deep;
        }
    }
    const uint32_t r = kc.resident ? 1 : 0;
    plan.resident = kc.resident;
    plan.kb = kc.kb;
    plan.depth_a = depths[0];
    plan.depth_w = depths[1];
    plan.depth_hold = g.hold_depths[r];
    plan.half_split = kc.resident ? mirror_split : HalfSplit::K;
    plan.P = gd.core_m * gd.core_n;
    plan.slots_fwd = sched[r].slots[0];
    plan.slots_bwd = sched[r].slots[1];
    plan.footprint = core_footprint(kc.kb, kc.resident, depths, g);
    return plan;
}

// plan_chunk, or nothing when the chunk misses the L1 budget (the Python's NotImplementedError from choose_k_block).
std::optional<Plan> plan_chunk_if_fits(uint32_t n_m_req, uint32_t n_n_req, const PlanInputs& in) {
    try {
        return plan_chunk(n_m_req, n_n_req, in);
    } catch (const NotImplementedRefusal&) {
        return std::nullopt;
    }
}

double chunk_cost_us(const Plan& pl, const PlanInputs& in) {
    const double C = static_cast<double>(pl.n_m) * pl.n_n;
    const double cores = static_cast<double>(pl.grid.grid_m) * pl.grid.grid_n;
    const double G = in.group_size;
    const uint32_t k_steps = in.Kt / pl.kb;  // K-steps per entry (kb divides Kt)
    const double steps = C * G * k_steps;
    const double step_us = pl.resident ? kCostStepUs : kCostStepStreamUs;
    const double compute = (C * G * pl.P * in.Kt * kCyclesPerTileK / kClockMhz / kCostFpuEff) + (step_us * steps);
    const double a_bytes = static_cast<double>(in.Mt) * in.Kt * in.a_tile_bytes;
    const double w_bytes = static_cast<double>(in.Kt) * in.Nt * in.w_tile_bytes;
    double traffic = 0;
    if (in.scatter_dim == -1) {
        traffic = (pl.resident ? a_bytes : G * pl.n_n * a_bytes) + (pl.n_m * w_bytes);
    } else {
        traffic = (pl.resident ? w_bytes : G * pl.n_m * w_bytes) + (pl.n_n * a_bytes);
    }
    const double dram = traffic / (kCostDramGbps * 1e3);
    const double transport = kCostHandoffUs * C * (G - 1) * cores / in.num_links;
    return kCostFloorUs + std::max({compute, dram, transport});
}

// chunk_candidates: (n_m, n_n) in the auto search order, distinct chunk grids only.
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
        std::stable_sort(pairs.begin(), pairs.end(), [&](const auto& x, const auto& y) {
            return (inner_n ? x.second : x.first) > (inner_n ? y.second : y.first);
        });
        for (const auto& [n_m, n_n] : pairs) {
            if (n_m > block_m || n_n > block_n) {
                continue;
            }
            if (!seen.emplace(tt::div_up(block_m, n_m), tt::div_up(block_n, n_n)).second) {
                continue;
            }
            out.emplace_back(n_m, n_n);
        }
    }
    return out;
}

using PlanKey = std::tuple<
    uint32_t,
    uint8_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    int32_t,
    uint64_t>;

}  // namespace

uint32_t staging_tiles(const Plan& plan) {
    return (kDepthHandoff + plan.zone_sets * (plan.slots_fwd + plan.slots_bwd)) * plan.P;
}

std::array<uint32_t, 2> choose_subblock(uint32_t rows, uint32_t cols) {
    std::array<uint32_t, 2> best{1, 1};
    for (uint32_t h = 1; h <= rows; ++h) {
        if (rows % h != 0) {
            continue;
        }
        for (uint32_t w = 1; w <= cols; ++w) {
            if (cols % w != 0 || h * w > kDestTilesFp32FullSync) {
                continue;
            }
            const uint32_t area = best[0] * best[1];
            if (h * w > area || (h * w == area && w > best[1])) {
                best = {h, w};
            }
        }
    }
    return best;
}

TransportPlan plan_transport(uint32_t group_size, ScheduleKind kind, bool mirror) {
    const uint32_t G = group_size;
    const bool ring = kind != ScheduleKind::LINE;
    mirror = mirror && kind == ScheduleKind::RING_BALANCED;
    TransportPlan tp;
    tp.chips.reserve(G);
    for (uint32_t p = 0; p < G; ++p) {
        tp.chips.push_back(build_chip_schedule(p, G, kind, mirror));
    }
    const std::vector<ChipSchedule> sch = tp.chips;
    auto up = [&](uint32_t p, uint32_t d) -> std::optional<uint32_t> {
        const int64_t q = d == 0 ? int64_t{p} - 1 : int64_t{p} + 1;
        if (ring) {
            return static_cast<uint32_t>((q + G) % G);
        }
        if (q >= 0 && q < G) {
            return static_cast<uint32_t>(q);
        }
        return std::nullopt;
    };
    tp.slots = {1, 1};
    for (uint32_t p = 0; p < G; ++p) {
        ChipSchedule& sc = tp.chips[p];
        for (uint32_t h = 0; h < 2; ++h) {
            for (uint32_t d = 0; d < 2; ++d) {
                const auto dir = static_cast<RoleKind>(d);
                const auto u = up(p, d);
                std::vector<uint32_t> lst;
                for (const uint32_t b : sc.entries[h]) {
                    const Role& r = sc.roles[b][h];
                    if (r.kind == dir && !r.is_start) {
                        TT_FATAL(
                            u.has_value() && sch[*u].roles[b][h].kind == dir,
                            "matmul_reduce_scatter: schedule {} {} {} {}",
                            p,
                            b,
                            h,
                            d);
                        lst.push_back(b);
                    } else if (r.kind == RoleKind::FIN && u.has_value() && sch[*u].roles[b][h].kind == dir) {
                        lst.push_back(b);
                    }
                }
                std::vector<uint32_t> seq;
                if (u.has_value()) {
                    for (const uint32_t b : sch[*u].entries[h]) {
                        if (sch[*u].roles[b][h].kind == dir) {
                            seq.push_back(b);
                        }
                    }
                }
                auto sorted_seq = seq;
                auto sorted_lst = lst;
                std::sort(sorted_seq.begin(), sorted_seq.end());
                std::sort(sorted_lst.begin(), sorted_lst.end());
                TT_FATAL(sorted_seq == sorted_lst, "matmul_reduce_scatter: arrivals mismatch {} {} {}", p, h, d);
                auto& thr = sc.thresholds[h][d];
                auto& prod = sc.produced_at[h][d];
                for (uint32_t i = 0; i < seq.size(); ++i) {
                    thr[seq[i]] = i + 1;  // every unit carries an increment
                    prod[seq[i]] = index_in(sch[*u].entries[h], seq[i]);
                }
                if (mirror) {
                    for (const uint32_t b : lst) {
                        if (sc.roles[b][h].kind != RoleKind::FIN) {
                            TT_FATAL(
                                prod.at(b) < index_in(sc.entries[h], b),
                                "matmul_reduce_scatter: mirrored schedule deadlock {} {} {} {}",
                                p,
                                b,
                                h,
                                d);
                        }
                    }
                }
                tp.slots[d] = std::max(tp.slots[d], static_cast<uint32_t>(lst.size()));
                sc.arrivals[h][d] = std::move(lst);
            }
        }
    }
    return tp;
}

std::vector<EntryFinish> entry_finish_plan(const ChipSchedule& schedule, uint32_t half) {
    const auto& entries = schedule.entries[half];
    std::vector<EntryFinish> out;
    out.reserve(entries.size());
    for (uint32_t e = 0; e < entries.size(); ++e) {
        const uint32_t b = entries[e];
        const Role& r = schedule.roles[b][half];
        uint32_t mask = 0;
        if ((r.kind == RoleKind::FWD || r.kind == RoleKind::BWD) && !r.is_start) {
            mask = r.kind == RoleKind::FWD ? 1 : 2;
        } else if (r.kind == RoleKind::FIN) {
            for (uint32_t d = 0; d < 2; ++d) {
                const auto& arr = schedule.arrivals[half][d];
                if (std::find(arr.begin(), arr.end(), b) != arr.end()) {
                    mask |= 1u << d;
                }
            }
        }
        bool lag = false;
        for (uint32_t d = 0; d < 2; ++d) {
            if ((mask >> d) & 1u) {
                lag = lag || schedule.produced_at[half][d].at(b) >= e;
            }
        }
        out.push_back({.mask = mask, .lag = lag && e + 1 < entries.size()});
    }
    return out;
}

Plan solve_plan(const PlanInputs& in) {
    if (std::optional<Plan> unchunked = plan_chunk_if_fits(1, 1, in)) {
        return *unchunked;  // whatever fits unchunked keeps its plan
    }
    static std::mutex cache_mutex;
    static std::map<PlanKey, std::pair<uint32_t, uint32_t>> cache;
    const PlanKey key{
        in.group_size,
        static_cast<uint8_t>(in.kind),
        in.num_links,
        in.Mt,
        in.Kt,
        in.Nt,
        in.block_m,
        in.block_n,
        in.grid_x,
        in.grid_y,
        in.a_tile_bytes,
        in.w_tile_bytes,
        in.scatter_dim,
        in.l1_budget};
    {
        const std::lock_guard<std::mutex> lock(cache_mutex);
        const auto it = cache.find(key);
        if (it != cache.end()) {
            return plan_chunk(it->second.first, it->second.second, in);
        }
    }
    std::optional<std::pair<double, Plan>> best;
    for (const auto& [n_m, n_n] : chunk_candidates(in.block_m, in.block_n, in.scatter_dim)) {
        if (n_m * n_n == 1) {
            continue;
        }
        std::optional<Plan> cand;
        try {
            cand = plan_chunk_if_fits(n_m, n_n, in);
        } catch (const std::invalid_argument&) {
            continue;
        }
        if (!cand.has_value()) {
            continue;
        }
        const double cost = chunk_cost_us(*cand, in);
        if (!best.has_value() || cost < best->first) {
            best = std::pair{cost, *cand};
        }
    }
    if (!best.has_value()) {
        throw NotImplementedRefusal(fmt::format(
            "matmul_reduce_scatter: no split of the {}x{}-tile block into <= {} chunks fits L1",
            in.block_m,
            in.block_n,
            kMaxAutoChunks));
    }
    const std::lock_guard<std::mutex> lock(cache_mutex);
    cache[key] = {best->second.n_m, best->second.n_n};
    return best->second;
}

}  // namespace ttnn::operations::matmul_reduce_scatter
