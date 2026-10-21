// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_planner.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "matmul_reduce_scatter_device_operation_types.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

namespace {

constexpr std::array<RoleKind, 2> kPlannerDirs = {RoleKind::Fwd, RoleKind::Bwd};
constexpr uint32_t kCostCyclesPerTileK = 32;  // HiFi2 matmul cycles per output tile per K tile (chunk_cost_us)

uint32_t dir_index(RoleKind d) { return static_cast<uint32_t>(d); }

// _role: (direction, is_start, distance-to-owner) of block b's half h at chip p.
Role planner_role(uint32_t p, uint32_t b, uint32_t h, uint32_t G, ScheduleKind kind) {
    if (b == p) {
        return Role{RoleKind::Fin, false, 0};
    }
    if (kind == ScheduleKind::Line) {
        if (b > p) {
            return Role{RoleKind::Fwd, p == 0, b - p};
        }
        return Role{RoleKind::Bwd, p == G - 1, p - b};
    }
    const uint32_t df = (b + G - p) % G;
    const uint32_t db = G - df;
    if (kind == ScheduleKind::RingUnbalanced) {
        const uint32_t nf = G / 2;
        const uint32_t nb = G - 1 - nf;
        if (df <= nf) {
            return Role{RoleKind::Fwd, df == nf, df};
        }
        return Role{RoleKind::Bwd, db == nb, db};
    }
    const uint32_t H = G / 2;
    if (h == 0) {
        if (df <= H) {
            return Role{RoleKind::Fwd, df == H, df};
        }
        return Role{RoleKind::Bwd, db == H - 1, db};
    }
    if (df <= H - 1) {
        return Role{RoleKind::Fwd, df == H - 1, df};
    }
    return Role{RoleKind::Bwd, db == H, db};
}

// build_schedule: roles and the per-half entry orders of chip p.
ChipSchedule planner_build_schedule(uint32_t p, uint32_t G, ScheduleKind kind, bool mirror) {
    ChipSchedule sc;
    sc.roles.resize(G);
    for (uint32_t b = 0; b < G; ++b) {
        for (uint32_t h = 0; h < 2; ++h) {
            sc.roles[b][h] = planner_role(p, b, h, G, kind);
        }
    }
    for (uint32_t h = 0; h < 2; ++h) {
        auto sort_key = [&](uint32_t b) {
            if (!mirror) {
                const uint32_t key = std::max(sc.roles[b][0].distance, sc.roles[b][1].distance);
                return std::make_tuple(-static_cast<int64_t>(key), sc.roles[b][0].kind == RoleKind::Fwd ? 0 : 1, b);
            }
            const RoleKind first = h == 0 ? RoleKind::Fwd : RoleKind::Bwd;
            return std::make_tuple(
                -static_cast<int64_t>(sc.roles[b][h].distance), sc.roles[b][h].kind == first ? 0 : 1, b);
        };
        std::vector<uint32_t> others;
        for (uint32_t b = 0; b < G; ++b) {
            if (b != p) {
                others.push_back(b);
            }
        }
        std::stable_sort(
            others.begin(), others.end(), [&](uint32_t x, uint32_t y) { return sort_key(x) < sort_key(y); });
        others.push_back(p);
        sc.entries[h] = std::move(others);
    }
    return sc;
}

uint32_t index_of(const std::vector<uint32_t>& v, uint32_t value) {
    const auto it = std::find(v.begin(), v.end(), value);
    TT_FATAL(it != v.end(), "matmul_reduce_scatter: block {} missing from a schedule", value);
    return static_cast<uint32_t>(it - v.begin());
}

bool contains(const std::vector<uint32_t>& v, uint32_t value) {
    return std::find(v.begin(), v.end(), value) != v.end();
}

// The per-regime geometry footprint() reads (index 0: stream, 1: resident).
struct FootprintGeo {
    uint32_t core_m = 0;
    uint32_t core_n = 0;
    uint32_t p = 0;
    uint32_t a_tile_bytes = 0;
    uint32_t w_tile_bytes = 0;
    uint32_t hold_tile_bytes = 0;
    uint32_t stage_tile_bytes = 0;
    int32_t scatter_dim = -2;
    std::array<std::array<uint32_t, 2>, 2> slots{};
    std::array<uint32_t, 2> hold_depths{};
    uint32_t zone_sets = 1;
};

std::pair<uint32_t, uint32_t> window_depths(bool resident, uint32_t depth) {
    return resident ? std::pair{depth, depth} : std::pair{kDepthIn0, kDepthIn1};
}

uint64_t footprint(
    uint32_t kb,
    bool resident,
    uint32_t kt,
    const FootprintGeo& geo,
    std::optional<uint32_t> depth,
    std::optional<std::pair<uint32_t, uint32_t>> depths) {
    const uint32_t r = resident ? 1 : 0;
    const bool r_a = resident && geo.scatter_dim == -1;
    const bool r_w = resident && geo.scatter_dim == -2;
    const auto [d_a, d_w] = depths.value_or(window_depths(resident, depth.value_or(kResidentStreamDepths[0])));
    const uint64_t a = uint64_t{geo.a_tile_bytes} * (r_a ? uint64_t{geo.core_m} * kt : uint64_t{d_a} * geo.core_m * kb);
    const uint64_t w = uint64_t{geo.w_tile_bytes} * (r_w ? uint64_t{kt} * geo.core_n : uint64_t{d_w} * kb * geo.core_n);
    const uint64_t hold = uint64_t{geo.hold_tile_bytes} * geo.hold_depths[r] * geo.p;
    const uint64_t staging = uint64_t{geo.stage_tile_bytes} *
                             (kDepthHandoff + uint64_t{geo.zone_sets} * (geo.slots[r][0] + geo.slots[r][1])) * geo.p;
    return a + w + hold + staging + geo.stage_tile_bytes;
}

std::vector<uint32_t> kb_divisors(uint32_t kt, uint32_t cap) {
    std::vector<uint32_t> out;
    for (uint32_t d = std::min(kt, cap); d >= 1; --d) {
        if (kt % d == 0) {
            out.push_back(d);
        }
    }
    for (uint32_t d = cap + 1; d <= kt; ++d) {
        if (kt % d == 0) {
            out.push_back(d);
        }
    }
    return out;
}

struct KBlockChoice {
    bool resident = false;
    uint32_t kb = 0;
    std::optional<uint32_t> depth;
};

KBlockChoice choose_k_block(uint32_t kt, uint64_t budget, uint32_t kb_max, const FootprintGeo& geo) {
    for (const uint32_t depth : kResidentStreamDepths) {
        for (const uint32_t kb : kb_divisors(kt, kb_max)) {
            if (kb < std::min(kt, kKbMinResident)) {
                continue;
            }
            if (footprint(kb, true, kt, geo, depth, std::nullopt) <= budget) {
                return {true, kb, depth};
            }
        }
    }
    for (const uint32_t kb : kb_divisors(kt, kb_max)) {
        if (footprint(kb, false, kt, geo, std::nullopt, std::nullopt) <= budget) {
            return {false, kb, std::nullopt};
        }
    }
    throw NotImplementedRefusal(
        "matmul_reduce_scatter: even kb=1 exceeds the L1 budget (regime DRAM-landing is deferred)");
}

GridPlan choose_grid(uint32_t gx, uint32_t gy, uint32_t block_m, uint32_t block_n, uint32_t links) {
    std::optional<std::tuple<bool, uint32_t, int64_t, uint32_t, uint32_t, bool>> best_key;
    GridPlan best;
    for (const bool row_strip : {true, false}) {
        const uint32_t strip_len = row_strip ? gx : gy;
        if (2 * links > strip_len) {
            continue;
        }
        const uint32_t rect_x = row_strip ? gx : gx - 1;
        const uint32_t rect_y = row_strip ? gy - 1 : gy;
        for (const bool m_on_x : {true, false}) {
            const uint32_t ext_m = m_on_x ? rect_x : rect_y;
            const uint32_t ext_n = m_on_x ? rect_y : rect_x;
            const uint32_t core_m = tt::div_up(block_m, std::min(ext_m, block_m));
            const uint32_t grid_m = tt::div_up(block_m, core_m);
            const uint32_t core_n = tt::div_up(block_n, std::min(ext_n, block_n));
            const uint32_t grid_n = tt::div_up(block_n, core_n);
            const Subblock sb = choose_subblock(core_m, core_n);
            const uint32_t ragged = (grid_m * core_m * grid_n * core_n) - (block_m * block_n);
            const auto key = std::make_tuple(
                !row_strip, core_m * core_n, -static_cast<int64_t>(sb.h * sb.w), ragged, grid_m * grid_n, !row_strip);
            if (!best_key.has_value() || key < *best_key) {
                best_key = key;
                best = GridPlan{row_strip, m_on_x, grid_m, grid_n, core_m, core_n, sb};
            }
        }
    }
    if (!best_key.has_value()) {
        throw std::invalid_argument(fmt::format(
            "matmul_reduce_scatter: no transport strip of the {}x{} grid fits {} ports", gx, gy, 2 * links));
    }
    return best;
}

// plan_chunk_geometry: grid, regime and schedule of one chunk.
ChunkPlan plan_chunk_geometry(uint32_t n_m_req, uint32_t n_n_req, const PlanInputs& in) {
    ChunkPlan plan;
    plan.chunk_m = tt::div_up(in.block_m, n_m_req);
    plan.chunk_n = tt::div_up(in.block_n, n_n_req);
    plan.n_m = tt::div_up(in.block_m, plan.chunk_m);
    plan.n_n = tt::div_up(in.block_n, plan.chunk_n);
    const uint32_t num_chunks = plan.n_m * plan.n_n;
    plan.zone_sets = num_chunks == 1 ? 1 : std::max(1u, std::min(kZoneSets, num_chunks));
    plan.grid = choose_grid(in.grid_x, in.grid_y, plan.chunk_m, plan.chunk_n, in.num_links);
    const GridPlan& gd = plan.grid;
    if (in.kind == ScheduleKind::RingBalanced && gd.grid_m * gd.grid_n < 2) {
        throw std::invalid_argument("matmul_reduce_scatter: a balanced ring needs >= 2 matmul cores (two halves)");
    }
    HalfSplit mirror_split = HalfSplit::K;
    if (in.kind == ScheduleKind::RingBalanced) {
        if (in.scatter_dim == -1 && gd.grid_n >= 2) {
            mirror_split = HalfSplit::N;
        } else if (in.scatter_dim == -2 && gd.grid_m >= 2) {
            mirror_split = HalfSplit::M;
        }
    }
    plan.p = gd.core_m * gd.core_n;
    FootprintGeo geo{
        .core_m = gd.core_m,
        .core_n = gd.core_n,
        .p = plan.p,
        .a_tile_bytes = in.a_tile_bytes,
        .w_tile_bytes = in.w_tile_bytes,
        .hold_tile_bytes = in.hold_tile_bytes,
        .stage_tile_bytes = in.stage_tile_bytes,
        .scatter_dim = in.scatter_dim,
        .zone_sets = plan.zone_sets,
    };
    for (uint32_t r = 0; r < 2; ++r) {
        const TransportPlan transport = plan_transport(in.group_size, in.kind, r == 1 && mirror_split != HalfSplit::K);
        geo.slots[r] = transport.slots;
        geo.hold_depths[r] = hold_depth(transport);
    }
    const uint32_t kb_max = num_chunks == 1 ? kKbMax : kKbMaxChunked;
    const KBlockChoice choice = choose_k_block(in.kt, in.budget, kb_max, geo);
    plan.resident = choice.resident;
    plan.kb = choice.kb;
    // super_block_plan
    auto depths = window_depths(plan.resident, choice.depth.value_or(kResidentStreamDepths[0]));
    plan.super_block = 1;
    if (kSuperBlock > 1 && (in.kt / plan.kb) % kSuperBlock == 0) {
        const uint32_t d = kSuperBlock + kSuperBlockExtra;
        const std::pair<uint32_t, uint32_t> deep{std::max(d, depths.first), std::max(d, depths.second)};
        if (footprint(plan.kb, plan.resident, in.kt, geo, choice.depth, deep) <= in.budget) {
            plan.super_block = kSuperBlock;
            depths = deep;
        }
    }
    const uint32_t r = plan.resident ? 1 : 0;
    plan.depth_a = depths.first;
    plan.depth_w = depths.second;
    plan.depth_hold = geo.hold_depths[r];
    plan.half_split = plan.resident ? mirror_split : HalfSplit::K;
    plan.slots = geo.slots[r];
    plan.footprint = footprint(plan.kb, plan.resident, in.kt, geo, choice.depth, depths);
    return plan;
}

// plan_chunk_geometry, or nullopt when its footprint misses the L1 budget (the Python's NotImplementedError).
std::optional<ChunkPlan> plan_if_fits(uint32_t n_m, uint32_t n_n, const PlanInputs& in) {
    try {
        return plan_chunk_geometry(n_m, n_n, in);
    } catch (const NotImplementedRefusal&) {
        return std::nullopt;
    }
}

double chunk_cost_us(const ChunkPlan& plan, const PlanInputs& in) {
    const double chunks = static_cast<double>(plan.n_m) * plan.n_n;
    const double cores = static_cast<double>(plan.grid.grid_m) * plan.grid.grid_n;
    const double G = in.group_size;
    const uint32_t k_steps = in.kt / plan.kb;  // K-blocks per entry (kb divides Kt)
    const double steps = chunks * G * k_steps;
    const double step_us = plan.resident ? kCostStepUs : kCostStepStreamUs;
    const double compute =
        (chunks * G * plan.p * in.kt * kCostCyclesPerTileK / kClockMhz / kCostFpuEff) + (step_us * steps);
    const double a_bytes = static_cast<double>(in.mt) * in.kt * in.a_tile_bytes;
    const double w_bytes = static_cast<double>(in.kt) * in.nt * in.w_tile_bytes;
    double traffic = 0;
    if (in.scatter_dim == -1) {
        traffic = (plan.resident ? a_bytes : G * plan.n_n * a_bytes) + (plan.n_m * w_bytes);
    } else {
        traffic = (plan.resident ? w_bytes : G * plan.n_m * w_bytes) + (plan.n_n * a_bytes);
    }
    const double dram = traffic / (kCostDramGbps * 1e3);
    const double transport = kCostHandoffUs * chunks * (G - 1) * cores / in.num_links;
    return kCostFloorUs + std::max({compute, dram, transport});
}

// chunk_candidates: fewest chunks first, more splits along the inner axis first, distinct chunk grids only.
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
            const std::pair<uint32_t, uint32_t> key{tt::div_up(block_m, n_m), tt::div_up(block_n, n_n)};
            if (!seen.insert(key).second) {
                continue;
            }
            out.emplace_back(n_m, n_n);
        }
    }
    return out;
}

}  // namespace

TransportPlan plan_transport(uint32_t group_size, ScheduleKind kind, bool mirror) {
    const uint32_t G = group_size;
    const bool ring = kind != ScheduleKind::Line;
    mirror = mirror && kind == ScheduleKind::RingBalanced;
    std::vector<ChipSchedule> sch;
    sch.reserve(G);
    for (uint32_t p = 0; p < G; ++p) {
        sch.push_back(planner_build_schedule(p, G, kind, mirror));
    }
    auto up = [&](uint32_t p, RoleKind d) -> std::optional<uint32_t> {
        const int64_t q = d == RoleKind::Fwd ? int64_t{p} - 1 : int64_t{p} + 1;
        if (ring) {
            return static_cast<uint32_t>((q + G) % G);
        }
        if (q >= 0 && q < G) {
            return static_cast<uint32_t>(q);
        }
        return std::nullopt;
    };

    TransportPlan out;
    for (uint32_t p = 0; p < G; ++p) {
        ChipSchedule& sc = sch[p];
        for (uint32_t h = 0; h < 2; ++h) {
            for (const RoleKind d : kPlannerDirs) {
                const uint32_t di = dir_index(d);
                const auto u = up(p, d);
                std::vector<uint32_t> lst;
                for (const uint32_t b : sc.entries[h]) {
                    const Role& r = sc.roles[b][h];
                    if (r.kind == d && !r.is_start) {
                        TT_FATAL(
                            u.has_value() && sch[*u].roles[b][h].kind == d,
                            "matmul_reduce_scatter: schedule ({}, {}, {}, {})",
                            p,
                            b,
                            h,
                            di);
                        lst.push_back(b);
                    } else if (r.kind == RoleKind::Fin && u.has_value() && sch[*u].roles[b][h].kind == d) {
                        lst.push_back(b);
                    }
                }
                std::vector<uint32_t> seq;
                if (u.has_value()) {
                    for (const uint32_t b : sch[*u].entries[h]) {
                        if (sch[*u].roles[b][h].kind == d) {
                            seq.push_back(b);
                        }
                    }
                }
                auto seq_sorted = seq;
                auto lst_sorted = lst;
                std::sort(seq_sorted.begin(), seq_sorted.end());
                std::sort(lst_sorted.begin(), lst_sorted.end());
                TT_FATAL(seq_sorted == lst_sorted, "matmul_reduce_scatter: arrivals mismatch ({}, {}, {})", p, h, di);
                auto& thresholds = sc.thresholds[h][di];
                auto& produced_at = sc.produced_at[h][di];
                for (uint32_t i = 0; i < seq.size(); ++i) {
                    thresholds[seq[i]] = i + 1;
                    produced_at[seq[i]] = index_of(sch[*u].entries[h], seq[i]);
                }
                if (mirror) {
                    for (const uint32_t b : lst) {
                        if (sc.roles[b][h].kind != RoleKind::Fin) {
                            TT_FATAL(
                                produced_at.at(b) < index_of(sc.entries[h], b),
                                "matmul_reduce_scatter: mirrored schedule deadlock ({}, {}, {}, {})",
                                p,
                                b,
                                h,
                                di);
                        }
                    }
                }
                sc.arrivals[h][di] = std::move(lst);
            }
        }
    }
    for (const RoleKind d : kPlannerDirs) {
        const uint32_t di = dir_index(d);
        uint32_t most = 0;
        for (const ChipSchedule& sc : sch) {
            for (uint32_t h = 0; h < 2; ++h) {
                most = std::max(most, static_cast<uint32_t>(sc.arrivals[h][di].size()));
            }
        }
        out.slots[di] = std::max(1u, most);
    }
    out.chips = std::move(sch);
    return out;
}

std::vector<EntryFinish> entry_finish_plan(const ChipSchedule& schedule, uint32_t half) {
    const auto& entries = schedule.entries[half];
    std::vector<EntryFinish> out;
    out.reserve(entries.size());
    for (uint32_t e_idx = 0; e_idx < entries.size(); ++e_idx) {
        const uint32_t b = entries[e_idx];
        const Role& r = schedule.roles[b][half];
        uint32_t mask = 0;
        if (r.kind != RoleKind::Fin && !r.is_start) {
            mask = r.kind == RoleKind::Fwd ? 1 : 2;
        } else if (r.kind == RoleKind::Fin) {
            mask = (contains(schedule.arrivals[half][0], b) ? 1u : 0u) |
                   (contains(schedule.arrivals[half][1], b) ? 2u : 0u);
        }
        bool lag = false;
        for (const RoleKind d : kPlannerDirs) {
            const uint32_t di = dir_index(d);
            if ((mask & (1u << di)) != 0 && schedule.produced_at[half][di].at(b) >= e_idx) {
                lag = true;
            }
        }
        out.push_back(EntryFinish{mask, lag && e_idx + 1 < entries.size()});
    }
    return out;
}

uint32_t hold_depth(const TransportPlan& transport) {
    for (const ChipSchedule& sc : transport.chips) {
        for (uint32_t h = 0; h < 2; ++h) {
            for (const EntryFinish& f : entry_finish_plan(sc, h)) {
                if (f.lag) {
                    return kDepthHold;
                }
            }
        }
    }
    return 1;
}

Subblock choose_subblock(uint32_t core_m, uint32_t core_n) {
    Subblock best;
    for (uint32_t h = 1; h <= core_m; ++h) {
        if (core_m % h != 0) {
            continue;
        }
        for (uint32_t w = 1; w <= core_n; ++w) {
            if (core_n % w != 0 || h * w > kDestTilesFp32FullSync) {
                continue;
            }
            const uint32_t area = h * w;
            const uint32_t best_area = best.h * best.w;
            if (area > best_area || (area == best_area && w > best.w)) {
                best = Subblock{h, w};
            }
        }
    }
    return best;
}

ChunkPlan solve_plan(const PlanInputs& inputs) {
    TT_FATAL(
        inputs.block_m > 0 && inputs.block_n > 0 && inputs.num_links > 0,
        "matmul_reduce_scatter: empty block ({}x{} tiles, {} links)",
        inputs.block_m,
        inputs.block_n,
        inputs.num_links);
    if (auto unchunked = plan_if_fits(1, 1, inputs)) {
        return *unchunked;  // whatever fits unchunked keeps its plan
    }
    std::optional<std::pair<double, ChunkPlan>> best;
    for (const auto& [n_m, n_n] : chunk_candidates(inputs.block_m, inputs.block_n, inputs.scatter_dim)) {
        if (n_m * n_n == 1) {
            continue;
        }
        std::optional<ChunkPlan> cand;
        try {
            cand = plan_if_fits(n_m, n_n, inputs);
        } catch (const std::invalid_argument&) {
            continue;
        }
        if (!cand.has_value()) {
            continue;
        }
        const double cost = chunk_cost_us(*cand, inputs);
        if (!best.has_value() || cost < best->first) {
            best = std::pair{cost, *cand};
        }
    }
    if (!best.has_value()) {
        throw NotImplementedRefusal(fmt::format(
            "matmul_reduce_scatter: no split of the {}x{}-tile block into <= {} chunks fits L1",
            inputs.block_m,
            inputs.block_n,
            kMaxAutoChunks));
    }
    return best->second;
}

std::vector<McastSenderPlan> plan_mcast_senders(
    SenderPlacement placement,
    uint32_t grid_m,
    uint32_t grid_n,
    bool a_two_parts,
    bool w_two_parts,
    ttnn::kernel_lib::host::Mcast1DShape a_shape,
    ttnn::kernel_lib::host::Mcast1DShape w_shape) {
    // both placements are diagonal: A line gm's sender at gn = (start + gm) % grid_n, W line gn's at
    // gm = (start + gn) % grid_m
    using CoreSet = std::set<std::pair<uint32_t, uint32_t>>;
    auto cores = [&](bool is_a, uint32_t start) {
        CoreSet out;
        if (is_a) {
            for (uint32_t gm = 0; gm < grid_m; ++gm) {
                out.emplace(gm, (start + gm) % grid_n);
            }
        } else {
            for (uint32_t gn = 0; gn < grid_n; ++gn) {
                out.emplace((start + gn) % grid_m, gn);
            }
        }
        return out;
    };
    struct Family {
        const char* prefix;
        ttnn::kernel_lib::host::Mcast1DShape shape;
        bool active;
        uint32_t line_len;
        bool is_a;
        bool second;
    };
    const std::array<Family, 4> families = {{
        {"a", a_shape, true, grid_n, true, false},
        {"a2", a_shape, a_two_parts, grid_n, true, true},
        {"b", w_shape, true, grid_m, false, false},
        {"b2", w_shape, w_two_parts, grid_m, false, true},
    }};
    constexpr auto mode = ttnn::kernel_lib::host::Mcast1DSenderPlacement::Diagonal;
    CoreSet taken;
    std::vector<McastSenderPlan> plan;
    for (const Family& f : families) {
        if (!f.active) {
            plan.push_back(McastSenderPlan{f.prefix, f.shape, std::nullopt, mode});
            continue;
        }
        uint32_t start = f.second ? f.line_len / 2 : 0;
        if (placement == SenderPlacement::DiagonalDisjoint && !f.is_a) {
            const uint32_t nominal = start;
            std::optional<std::pair<size_t, uint32_t>> best_key;
            for (uint32_t s0 = 0; s0 < f.line_len; ++s0) {
                size_t overlap = 0;
                for (const auto& c : cores(false, s0)) {
                    overlap += taken.count(c);
                }
                const std::pair<size_t, uint32_t> key{overlap, (s0 + f.line_len - nominal) % f.line_len};
                if (!best_key.has_value() || key < *best_key) {
                    best_key = key;
                    start = s0;
                }
            }
        }
        const CoreSet placed = cores(f.is_a, start);
        taken.insert(placed.begin(), placed.end());
        plan.push_back(McastSenderPlan{f.prefix, f.shape, start, mode});
    }
    return plan;
}

}  // namespace ttnn::operations::matmul_reduce_scatter
