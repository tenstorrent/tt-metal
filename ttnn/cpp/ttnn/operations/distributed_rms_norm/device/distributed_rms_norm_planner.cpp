// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "distributed_rms_norm_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt_stl/indestructible.hpp>

#include "kernels/distributed_rms_norm_args.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::distributed_rms_norm::detail {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataType;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;

namespace {

constexpr uint32_t FWD = ::distributed_rms_norm::dir::FWD;
constexpr uint32_t BWD = ::distributed_rms_norm::dir::BWD;
constexpr uint32_t OPP[2] = {BWD, FWD};
constexpr uint32_t BH_ODD_BANKS = 0xAAAAAAAAu;  // banks 1, 3, ..., 31 on NOC0

// The Python's NotImplementedError inside the direct schedule: caught to fall back to relay.
struct OffPortRoute : std::runtime_error {
    using std::runtime_error::runtime_error;
};

int64_t pos_mod(int64_t a, int64_t m) { return ((a % m) + m) % m; }

uint32_t datum_bytes(DataType dtype) {
    return dtype == DataType::BFLOAT8_B ? 1u : tt::tt_metal::tile_size(dtype) / tt::constants::TILE_HW;
}

uint32_t largest_divisor_at_most(uint32_t val, uint32_t bound) {
    for (uint32_t d = std::min(bound, val); d > 1; --d) {
        if (val % d == 0) {
            return d;
        }
    }
    return 1;
}

uint32_t landing_words_bucket(uint32_t words) {
    uint32_t b = LANDING_MIN_WORDS;
    while (b < words) {
        b *= 2;
    }
    return b;
}

// ---- groups / links / schedule ----

std::vector<std::vector<MeshCoordinate>> build_groups(const MeshDevice& mesh, uint32_t cluster_axis) {
    const auto& shape = mesh.shape();
    const uint32_t rows = shape[0];
    const uint32_t cols = shape[1];
    std::vector<std::vector<MeshCoordinate>> groups;
    if (cluster_axis == 0) {
        for (uint32_t c = 0; c < cols; ++c) {
            auto& grp = groups.emplace_back();
            for (uint32_t r = 0; r < rows; ++r) {
                grp.emplace_back(r, c);
            }
        }
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            auto& grp = groups.emplace_back();
            for (uint32_t c = 0; c < cols; ++c) {
                grp.emplace_back(r, c);
            }
        }
    }
    return groups;
}

std::optional<int> eth_dir(const MeshDevice& mesh, const MeshCoordinate& a, const MeshCoordinate& b) {
    const auto d = tt::tt_fabric::get_eth_forwarding_direction(mesh.get_fabric_node_id(a), mesh.get_fabric_node_id(b));
    if (!d.has_value()) {
        return std::nullopt;
    }
    return static_cast<int>(*d);
}

uint32_t count_usable_links(const MeshDevice& mesh, uint32_t cluster_axis, bool ring) {
    std::optional<uint32_t> best;
    for (const auto& grp : build_groups(mesh, cluster_axis)) {
        const size_t g = grp.size();
        std::vector<std::pair<MeshCoordinate, MeshCoordinate>> hops;
        for (size_t i = 0; i + 1 < g; ++i) {
            hops.emplace_back(grp[i], grp[i + 1]);
        }
        if (ring && g > 0) {
            hops.emplace_back(grp.back(), grp.front());
        }
        for (const auto& [a, b] : hops) {
            const auto na = mesh.get_fabric_node_id(a);
            const auto nb = mesh.get_fabric_node_id(b);
            for (const auto n :
                 {tt::tt_fabric::get_forwarding_link_indices(mesh, na, nb).size(),
                  tt::tt_fabric::get_forwarding_link_indices(mesh, nb, na).size()}) {
                best = std::min(best.value_or(std::numeric_limits<uint32_t>::max()), static_cast<uint32_t>(n));
            }
        }
    }
    return best.value_or(0);
}

// _exchange_plan of the Python, one kind.
void build_schedule(
    const MeshDevice& mesh,
    const std::vector<std::vector<MeshCoordinate>>& groups,
    bool ring,
    ExchangeMode kind,
    std::map<MeshCoordinate, ChipSchedule>& chips) {
    const bool two_d = is_2d_fabric();
    for (const auto& grp : groups) {
        const auto G = static_cast<int64_t>(grp.size());
        auto nxt_of = [&](int64_t p) -> std::optional<int64_t> {
            if (ring || p < G - 1) {
                return pos_mod(p + 1, G);
            }
            return std::nullopt;
        };
        auto prv_of = [&](int64_t p) -> std::optional<int64_t> {
            if (ring || p > 0) {
                return pos_mod(p - 1, G);
            }
            return std::nullopt;
        };
        for (int64_t p = 0; p < G; ++p) {
            ChipSchedule ch;
            ch.position = static_cast<uint32_t>(p);
            if (auto n = nxt_of(p)) {
                ch.next = grp[*n];
            }
            if (auto v = prv_of(p)) {
                ch.prev = grp[*v];
            }
            chips.insert_or_assign(grp[p], std::move(ch));
        }
        auto chip = [&](int64_t p) -> ChipSchedule& { return chips.at(grp[p]); };
        auto hdr = [&](int64_t p, uint32_t d, const DestHeader& key) -> uint32_t {
            auto& lst = chip(p).dests[d];
            auto it = std::find(lst.begin(), lst.end(), key);
            if (it == lst.end()) {
                lst.push_back(key);
                return static_cast<uint32_t>(lst.size() - 1);
            }
            return static_cast<uint32_t>(it - lst.begin());
        };
        auto first_hop = [&](int64_t s, int64_t t) -> uint32_t {
            const auto dt = eth_dir(mesh, grp[s], grp[t]);
            for (const auto& [d, nb] : {std::pair{FWD, nxt_of(s)}, std::pair{BWD, prv_of(s)}}) {
                if (nb.has_value() && dt == eth_dir(mesh, grp[s], grp[*nb])) {
                    return d;
                }
            }
            throw OffPortRoute("direct exchange: route leaves through no port");
        };
        auto mcast_code = [&](int64_t s, uint32_t d, uint32_t rng) -> std::pair<int64_t, uint32_t> {
            const auto nb = d == FWD ? nxt_of(s) : prv_of(s);
            TT_FATAL(nb.has_value(), "distributed_rms_norm: multicast port without a neighbour");
            int dirn = 0;
            if (two_d) {
                const auto e = eth_dir(mesh, grp[s], grp[*nb]);
                TT_FATAL(e.has_value(), "distributed_rms_norm: no forwarding direction to the neighbour");
                dirn = *e;
            }
            return {*nb, (static_cast<uint32_t>(dirn + 1) << 8) | rng};
        };
        auto hop = [&](int64_t s, int64_t t, uint32_t d) -> uint32_t {
            return static_cast<uint32_t>(d == FWD ? pos_mod(t - s, G) : pos_mod(s - t, G));
        };
        // (s, d) -> destination positions, in insertion order (the Python dict's order).
        std::vector<std::pair<std::pair<int64_t, uint32_t>, std::vector<int64_t>>> targets;
        auto target = [&](int64_t s, uint32_t d) -> std::vector<int64_t>& {
            for (auto& [k, v] : targets) {
                if (k.first == s && k.second == d) {
                    return v;
                }
            }
            return targets.emplace_back(std::pair{s, d}, std::vector<int64_t>{}).second;
        };
        auto target_get = [&](int64_t s, uint32_t d) -> const std::vector<int64_t>* {
            for (const auto& [k, v] : targets) {
                if (k.first == s && k.second == d) {
                    return &v;
                }
            }
            return nullptr;
        };

        if (kind == ExchangeMode::RELAY) {
            for (int64_t p = 0; p < G; ++p) {
                std::vector<int64_t> fwd;
                std::vector<int64_t> bwd;
                if (ring) {
                    const int64_t kf = G / 2;
                    const int64_t kb = G - 1 - kf;
                    for (int64_t i = 0; i < kf; ++i) {
                        fwd.push_back(pos_mod(p - i, G));
                    }
                    for (int64_t i = 0; i < kb; ++i) {
                        bwd.push_back(pos_mod(p + i, G));
                    }
                } else {
                    if (p < G - 1) {
                        for (int64_t i = 0; i < p + 1; ++i) {
                            fwd.push_back(p - i);
                        }
                    }
                    if (p > 0) {
                        for (int64_t i = 0; i < G - p; ++i) {
                            bwd.push_back(p + i);
                        }
                    }
                }
                for (const auto& [d, lst, peer] :
                     {std::tuple{FWD, &fwd, nxt_of(p)}, std::tuple{BWD, &bwd, prv_of(p)}}) {
                    if (lst->empty()) {
                        continue;
                    }
                    const uint32_t h = hdr(p, d, DestHeader{grp[*peer], 0, 1});
                    auto& sends = chip(p).sends[d];
                    sends.clear();
                    for (size_t i = 0; i < lst->size(); ++i) {
                        sends.push_back(SendItem{static_cast<uint32_t>((*lst)[i]), h, static_cast<uint32_t>(i)});
                    }
                    auto& entries = chip(*peer).entries[d];
                    entries.clear();
                    for (const int64_t q : *lst) {
                        entries.push_back(static_cast<uint32_t>(q));
                    }
                    target(p, d) = {*peer};
                }
            }
        } else if (kind == ExchangeMode::DIRECT) {
            for (int64_t s = 0; s < G; ++s) {
                target(s, FWD).clear();
                target(s, BWD).clear();
                if (ring) {
                    for (int64_t t = 0; t < G; ++t) {
                        if (t != s) {
                            target(s, first_hop(s, t)).push_back(t);
                        }
                    }
                } else {
                    for (const uint32_t d : {FWD, BWD}) {
                        const int64_t k = d == FWD ? G - 1 - s : s;
                        std::vector<int64_t> ts;
                        for (int64_t i = 1; i <= k; ++i) {
                            ts.push_back(d == FWD ? s + i : s - i);
                        }
                        for (const int64_t t : ts) {
                            if (first_hop(s, t) != d) {
                                throw OffPortRoute("direct exchange (line): the fabric routes off the line");
                            }
                        }
                        target(s, d) = ts;
                    }
                }
                for (const uint32_t d : {FWD, BWD}) {
                    std::vector<int64_t> order = target(s, d);
                    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                        return hop(s, a, d) > hop(s, b, d);  // farthest first
                    });
                    auto& sends = chip(s).sends[d];
                    sends.clear();
                    for (const int64_t t : order) {
                        sends.push_back(
                            SendItem{static_cast<uint32_t>(s), hdr(s, d, DestHeader{grp[t], 0, hop(s, t, d)}), 0});
                    }
                }
            }
        } else {
            for (int64_t s = 0; s < G; ++s) {
                std::vector<int64_t> hf;
                std::vector<int64_t> hb;
                if (ring) {
                    const int64_t kf = G / 2;
                    for (int64_t i = 1; i <= kf; ++i) {
                        hf.push_back(pos_mod(s + i, G));
                    }
                    for (int64_t i = 1; i < G - kf; ++i) {
                        hb.push_back(pos_mod(s - i, G));
                    }
                } else {
                    for (int64_t t = s + 1; t < G; ++t) {
                        hf.push_back(t);
                    }
                    for (int64_t t = s - 1; t >= 0; --t) {
                        hb.push_back(t);
                    }
                }
                for (const auto& [d, ts] : {std::pair{FWD, &hf}, std::pair{BWD, &hb}}) {
                    target(s, d) = *ts;
                    if (!ts->empty()) {
                        const auto [nb, code] = mcast_code(s, d, static_cast<uint32_t>(ts->size()));
                        chip(s).sends[d] = {
                            SendItem{static_cast<uint32_t>(s), hdr(s, d, DestHeader{grp[nb], code, 1}), 0}};
                    }
                }
            }
        }
        if (kind != ExchangeMode::RELAY) {
            for (int64_t t = 0; t < G; ++t) {
                for (const uint32_t d : {FWD, BWD}) {
                    std::vector<int64_t> srcs;
                    for (int64_t s = 0; s < G; ++s) {
                        const auto* ts = target_get(s, d);
                        if (s != t && ts != nullptr && std::find(ts->begin(), ts->end(), t) != ts->end()) {
                            srcs.push_back(s);
                        }
                    }
                    std::stable_sort(
                        srcs.begin(), srcs.end(), [&](int64_t a, int64_t b) { return hop(a, t, d) < hop(b, t, d); });
                    auto& entries = chip(t).entries[d];
                    entries.clear();
                    for (const int64_t s : srcs) {
                        entries.push_back(static_cast<uint32_t>(s));
                    }
                }
            }
        }
        for (const auto& [k, ts] : targets) {
            if (!ts.empty()) {
                chip(k.first).n_wait[k.second] = static_cast<uint32_t>(ts.size());
            }
        }
        if (kind == ExchangeMode::MCAST) {
            for (int64_t t = 0; t < G; ++t) {
                for (const uint32_t d : {FWD, BWD}) {
                    const auto srcs = chip(t).entries[d];
                    if (srcs.empty()) {
                        continue;
                    }
                    const uint32_t back = OPP[d];
                    uint32_t max_hop = 0;
                    for (const uint32_t s : srcs) {
                        max_hop = std::max(max_hop, hop(s, t, d));
                    }
                    const auto [nb, code] = mcast_code(t, back, max_hop);
                    const uint32_t h = hdr(t, back, DestHeader{grp[nb], code, 1});
                    chip(t).ready_out[back].push_back(ReadyOut{grp[srcs.front()], d, h});
                }
            }
        } else {
            for (const auto& [k, ts] : targets) {
                const auto [s, d] = k;
                for (const int64_t t : ts) {
                    uint32_t back = OPP[d];
                    if (kind == ExchangeMode::DIRECT) {
                        back = first_hop(t, s);
                        if (!ring && back != OPP[d]) {
                            throw OffPortRoute("direct exchange (line): ready off the line");
                        }
                    }
                    const uint32_t h = hdr(t, back, DestHeader{grp[s], 0, hop(t, s, back)});
                    chip(t).ready_out[back].push_back(ReadyOut{grp[s], d, h});
                }
            }
        }
    }
}

// ---- local plan ----

Geometry compute_geometry(
    const PlanInputs& in,
    uint32_t chunk_cap,
    uint32_t S,
    bool sharded,
    bool combine,
    uint32_t G_w,
    bool masked,
    bool gamma_resident) {
    Geometry g;
    g.W = in.shape.back();
    g.Wt = tt::div_up(g.W, tt::constants::TILE_WIDTH);
    if (sharded) {
        g.C = largest_divisor_at_most(S, std::min(chunk_cap, S));
        g.num_w_chunks = S / g.C;
    } else {
        g.num_w_chunks = tt::div_up(S, chunk_cap);
        g.C = tt::div_up(S, g.num_w_chunks);
    }
    g.S = S;
    g.Wp = g.num_w_chunks * g.C;
    g.WS = sharded ? S : g.Wp;
    g.has_gamma = in.gamma_dtype.has_value();
    g.T_in = tt::tt_metal::tile_size(in.input_dtype);
    g.T_out = g.T_in;
    g.T_g = g.has_gamma ? tt::tt_metal::tile_size(*in.gamma_dtype) : 0;
    g.es = datum_bytes(in.input_dtype);
    g.es_g = g.has_gamma ? datum_bytes(*in.gamma_dtype) : 0;
    g.gamma_is_tile = g.has_gamma && in.gamma_is_tile;
    g.interm_fp32 = in.fp32_acc;
    g.T_i = tt::tt_metal::tile_size(in.fp32_acc ? DataType::FLOAT32 : DataType::BFLOAT16);
    g.sharded = sharded;
    g.combine = combine;
    g.G_w = G_w;
    g.masked = masked;
    g.G = in.group_size;
    g.gamma_resident = g.has_gamma && gamma_resident;
    g.out_native = sharded && in.out_native;
    return g;
}

std::pair<uint32_t, uint32_t> wave_split(uint32_t block_rows) {
    const uint32_t cap =
        std::max(1u, std::min({block_rows, NUM_WAVES, ::distributed_rms_norm::sem::MAX_WAVES_PER_BLOCK}));
    const uint32_t nwpb = largest_divisor_at_most(block_rows, cap);
    return {block_rows / nwpb, nwpb};
}

uint64_t footprint_bytes(const Geometry& g, uint32_t R, uint32_t x_depth) {
    uint64_t sum = 0;
    for (const CbEntry& cb :
         cb_inventory(g, R, x_depth, tt::DataFormat::Invalid, tt::DataFormat::Invalid, tt::DataFormat::Invalid)) {
        sum += static_cast<uint64_t>(cb.page_size) * cb.pages;
    }
    return sum;
}

uint32_t max_block_rows(const Geometry& g, uint32_t x_depth, uint32_t cap, int64_t budget) {
    uint32_t best = 0;
    for (uint32_t r = 1; r <= cap; ++r) {  // not strictly monotone through wave_split: scan
        if (static_cast<int64_t>(footprint_bytes(g, r, x_depth)) <= budget) {
            best = r;
        }
    }
    return best;
}

std::optional<std::pair<uint32_t, uint32_t>> choose_block_rows(const Geometry& g, uint32_t rows_max, int64_t budget) {
    if (g.sharded) {
        const uint32_t r = max_block_rows(g, 1, rows_max, budget);
        if (r >= 1) {
            return std::pair{r, 1u};
        }
        return std::nullopt;
    }
    const uint32_t cap = std::max(1u, tt::div_up(rows_max, MIN_BLOCKS_PER_CORE));
    const uint32_t r1 = max_block_rows(g, 1, cap, budget);
    if (r1 >= rows_max) {
        return std::pair{r1, 1u};
    }
    const uint32_t r2 = max_block_rows(g, 2, cap, budget);
    if (r2 >= 1) {
        return std::pair{r2, 2u};
    }
    if (r1 >= 1) {
        return std::pair{r1, 1u};
    }
    return std::nullopt;
}

struct Blocking {
    Geometry g;
    uint32_t R = 0;
    uint32_t x_depth = 1;
};

std::optional<Blocking> search_blocking(
    const PlanInputs& in,
    uint32_t rows_max,
    int64_t budget,
    uint32_t S,
    bool sharded,
    bool combine,
    uint32_t G_w,
    bool masked,
    bool gamma_resident) {
    std::optional<Blocking> fallback;
    for (uint32_t cap = W_CHUNK_TILES; cap >= 1; cap /= 2) {
        const Geometry g = compute_geometry(in, cap, S, sharded, combine, G_w, masked, gamma_resident);
        if (const auto choice = choose_block_rows(g, rows_max, budget)) {
            if (choice->second == 2 || sharded || choice->first >= rows_max) {
                return Blocking{g, choice->first, choice->second};
            }
            if (!fallback.has_value()) {
                fallback = Blocking{g, choice->first, choice->second};
            }
        }
    }
    return fallback;
}

std::optional<Blocking> choose_blocking(
    const PlanInputs& in,
    uint32_t rows_max,
    int64_t budget,
    uint32_t S,
    bool sharded,
    bool combine,
    uint32_t G_w,
    bool masked) {
    // Resident gamma first; streamed gamma when no blocking fits with it.
    auto choice = search_blocking(in, rows_max, budget, S, sharded, combine, G_w, masked, true);
    if (!choice.has_value() && in.gamma_dtype.has_value()) {
        choice = search_blocking(in, rows_max, budget, S, sharded, combine, G_w, masked, false);
    }
    return choice;
}

std::vector<CoreWork> group_work(
    const std::vector<CoreCoord>& cores,
    uint32_t group_size,
    uint32_t S,
    uint32_t Wt,
    const std::vector<std::pair<uint32_t, uint32_t>>& row_ranges) {
    std::vector<CoreWork> work;
    for (size_t gi = 0; gi < row_ranges.size(); ++gi) {
        const auto [start, rows] = row_ranges[gi];
        const size_t first = gi * group_size;
        const size_t last = std::min(cores.size(), first + group_size);
        for (size_t j = first; j < last; ++j) {
            const auto idx = static_cast<uint32_t>(j - first);
            const uint32_t w0 = idx * S;
            if (w0 >= Wt || rows == 0) {
                throw RegimeRefusal("distributed_rms_norm: a core with no valid data is not supported");
            }
            const uint32_t V = std::min(S, Wt - w0);
            work.push_back(CoreWork{cores[j], start, rows, w0, V, idx, cores[first]});
        }
    }
    return work;
}

struct LocalPlan {
    Regime regime = Regime::R1;
    std::vector<CoreWork> work;
    Blocking blocking;
    uint32_t rows_max = 0;
};

std::optional<LocalPlan> finish_plan(
    const PlanInputs& in,
    Regime regime,
    std::vector<CoreWork> work,
    uint32_t S,
    uint32_t rows_max,
    bool sharded,
    int64_t budget) {
    uint32_t max_index = 0;
    for (const auto& w : work) {
        max_index = std::max(max_index, w.index);
    }
    const uint32_t G_w = max_index + 1;
    const bool combine = G_w > 1;
    const uint32_t W = in.shape.back();
    const bool masked = (W % tt::constants::TILE_WIDTH != 0) && (sharded || in.input_dtype == DataType::BFLOAT8_B);
    auto choice = choose_blocking(in, rows_max, budget, S, sharded, combine, G_w, masked);
    if (!choice.has_value()) {
        return std::nullopt;
    }
    return LocalPlan{regime, std::move(work), *choice, rows_max};
}

CoreRangeSet set_of_rects(const std::vector<std::array<uint32_t, 4>>& rects) {
    std::vector<CoreRange> ranges;
    ranges.reserve(rects.size());
    for (const auto& r : rects) {
        ranges.emplace_back(CoreCoord(r[0], r[1]), CoreCoord(r[2], r[3]));
    }
    return CoreRangeSet(ranges);
}

LocalPlan select_plan(const PlanInputs& in, const std::vector<CoreCoord>& available, int64_t budget) {
    const uint32_t W = in.shape.back();
    const uint32_t Wt = tt::div_up(W, tt::constants::TILE_WIDTH);
    uint32_t lead = 1;
    for (size_t i = 0; i + 2 < in.shape.size(); ++i) {
        lead *= in.shape[i];
    }
    const uint32_t T = lead * tt::div_up(in.shape[in.shape.size() - 2], tt::constants::TILE_HEIGHT);
    TT_FATAL(T > 0 && Wt > 0, "distributed_rms_norm: the input has no tile-rows");
    const TensorMemoryLayout ml = in.input_layout;

    if (ml == TensorMemoryLayout::HEIGHT_SHARDED || ml == TensorMemoryLayout::WIDTH_SHARDED) {
        if (in.input_shard_rects.empty() || !in.input_l1) {
            throw RegimeRefusal("distributed_rms_norm: sharded inputs must carry a 2-D L1 shard spec");
        }
        const bool height = ml == TensorMemoryLayout::HEIGHT_SHARDED;
        if (!in.shard_row_major && !height) {
            throw RegimeRefusal("distributed_rms_norm: only ROW_MAJOR shard orientation is supported");
        }
        const auto [shard_h, shard_w] = in.shard_shape;
        TT_FATAL(
            shard_h % tt::constants::TILE_HEIGHT == 0 && shard_w % tt::constants::TILE_WIDTH == 0,
            "TILE shards are tile-multiples");
        const uint32_t Hsh = shard_h / tt::constants::TILE_HEIGHT;
        const uint32_t S = shard_w / tt::constants::TILE_WIDTH;
        TT_FATAL(Hsh > 0 && S > 0, "distributed_rms_norm: empty shard shape");
        const CoreRangeSet grid = set_of_rects(in.input_shard_rects);
        std::vector<CoreWork> work;
        uint32_t rows_max = 0;
        if (height) {
            if (S < Wt) {
                throw RegimeRefusal("distributed_rms_norm: height shard spec does not cover the row width");
            }
            const uint32_t n = tt::div_up(T, Hsh);
            const auto cores = tt::tt_metal::corerange_to_cores(grid, n, in.shard_row_major);
            if (cores.size() < n) {
                throw RegimeRefusal("distributed_rms_norm: height shard spec does not cover the tensor");
            }
            std::vector<std::pair<uint32_t, uint32_t>> ranges;
            for (uint32_t i = 0; i < n; ++i) {
                ranges.emplace_back(i * Hsh, std::min(Hsh, T - (i * Hsh)));
                rows_max = std::max(rows_max, ranges.back().second);
            }
            work = group_work(cores, 1, S, Wt, ranges);
        } else {
            std::vector<CoreCoord> cores;
            for (const auto& r : grid.ranges()) {
                for (auto y = r.start_coord.y; y <= r.end_coord.y; ++y) {
                    for (auto x = r.start_coord.x; x <= r.end_coord.x; ++x) {
                        cores.emplace_back(x, y);
                    }
                }
            }
            std::stable_sort(cores.begin(), cores.end(), [](const CoreCoord& a, const CoreCoord& b) {
                return std::tie(a.y, a.x) < std::tie(b.y, b.x);
            });
            const uint32_t n = tt::div_up(Wt, S);
            if (Hsh < T || n > cores.size()) {
                throw RegimeRefusal("distributed_rms_norm: width shard spec does not cover the tensor");
            }
            cores.resize(n);
            work = group_work(cores, n, S, Wt, {{0, T}});
            rows_max = T;
        }
        for (const auto& w : work) {
            if (std::find(available.begin(), available.end(), w.core) == available.end()) {
                throw RegimeRefusal("distributed_rms_norm: a shard core collides with a port core");
            }
        }
        auto plan = finish_plan(
            in,
            height ? Regime::R1_HEIGHT_SHARDED : Regime::R3_WIDTH_SHARDED,
            std::move(work),
            S,
            rows_max,
            true,
            budget);
        if (!plan.has_value()) {
            throw RegimeRefusal("distributed_rms_norm: the shard does not fit one resident tile-row in L1");
        }
        return std::move(*plan);
    }
    if (ml != TensorMemoryLayout::INTERLEAVED) {
        throw RegimeRefusal(fmt::format("distributed_rms_norm: memory_layout {} is not built", ml));
    }

    // R1: rows split over every available core.
    const auto n1 = static_cast<uint32_t>(std::min<size_t>(T, available.size()));
    TT_FATAL(n1 > 0, "distributed_rms_norm: no compute cores left after the port cores");
    const uint32_t base1 = T / n1;
    const uint32_t extra1 = T % n1;
    std::vector<CoreWork> r1_work;
    uint32_t start = 0;
    for (uint32_t i = 0; i < n1; ++i) {
        const uint32_t rows = base1 + (i < extra1 ? 1 : 0);
        r1_work.push_back(CoreWork{available[i], start, rows, 0, Wt, 0, available[i]});
        start += rows;
    }
    TT_FATAL(start == T, "distributed_rms_norm: R1 split does not cover the tile-rows");
    const size_t r1_cores = r1_work.size();
    auto r1 = finish_plan(in, Regime::R1, std::move(r1_work), Wt, base1 + (extra1 != 0 ? 1 : 0), false, budget);

    // R3-i layout.
    const auto num_cores = static_cast<uint32_t>(available.size());
    const uint32_t groups_target = tt::div_up(T, R3_ROWS_PER_GROUP);
    const uint32_t G_w_target = std::max(1u, std::min(R3_MAX_GROUP_CORES, num_cores / groups_target));
    const uint32_t S = std::min(std::max(R3_MIN_SLICE_TILES, tt::div_up(Wt, G_w_target)), Wt);
    const uint32_t G_w = tt::div_up(Wt, S);
    const uint32_t G_h = std::max(1u, std::min(T, num_cores / G_w));
    const bool wants_r3 = !r1.has_value() ||
                          (G_w >= R3_MIN_GROUP_CORES && static_cast<size_t>(G_w) * G_h >= R3_OCCUPANCY_GAIN * r1_cores);
    if (wants_r3 && G_w > 1) {
        const uint32_t base = T / G_h;
        const uint32_t extra = T % G_h;
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        uint32_t s = 0;
        for (uint32_t gi = 0; gi < G_h; ++gi) {
            const uint32_t rows = base + (gi < extra ? 1 : 0);
            ranges.emplace_back(s, rows);
            s += rows;
        }
        const std::vector<CoreCoord> cores(available.begin(), available.begin() + (static_cast<size_t>(G_w) * G_h));
        auto r3 = finish_plan(
            in,
            Regime::R3_INTERLEAVED,
            group_work(cores, G_w, S, Wt, ranges),
            S,
            base + (extra != 0 ? 1 : 0),
            false,
            budget);
        if (r3.has_value()) {
            return std::move(*r3);
        }
    }
    if (!r1.has_value()) {
        throw RegimeRefusal(fmt::format("distributed_rms_norm: W={} does not fit a resident tile-row in L1", W));
    }
    return std::move(*r1);
}

// 2L port cores among the cores no shard occupies: lowest y first, then nearest the grid's x-centre.
std::vector<std::array<CoreCoord, 2>> place_ports(const PlanInputs& in) {
    const uint32_t L = in.num_links;
    const uint32_t n = 2 * L;
    const int64_t x0 = std::max<int64_t>(0, (static_cast<int64_t>(in.grid_x) / 2) - L);
    const double centre = static_cast<double>(x0) + ((static_cast<double>(n) - 1.0) / 2.0);
    auto busy = [&](uint32_t x, uint32_t y) {
        for (const auto* rects : {&in.input_shard_rects, &in.output_shard_rects}) {
            for (const auto& r : *rects) {
                if (x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3]) {
                    return true;
                }
            }
        }
        return false;
    };
    std::vector<CoreCoord> cand;
    for (uint32_t y = 0; y < in.grid_y; ++y) {
        for (uint32_t x = 0; x < in.grid_x; ++x) {
            if (!busy(x, y)) {
                cand.emplace_back(x, y);
            }
        }
    }
    std::stable_sort(cand.begin(), cand.end(), [&](const CoreCoord& a, const CoreCoord& b) {
        const double da = std::abs(static_cast<double>(a.x) - centre);
        const double db = std::abs(static_cast<double>(b.x) - centre);
        return std::tie(a.y, da, a.x) < std::tie(b.y, db, b.x);
    });
    if (cand.size() < n) {
        throw RegimeRefusal(fmt::format(
            "distributed_rms_norm: the shard grid leaves {} free cores, {} port cores are needed", cand.size(), n));
    }
    cand.resize(n);
    std::sort(cand.begin(), cand.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return std::tie(a.y, a.x) < std::tie(b.y, b.x);
    });
    std::vector<std::array<CoreCoord, 2>> ports(L);
    for (uint32_t l = 0; l < L; ++l) {
        ports[l][FWD] = cand[(2 * l) + FWD];
        ports[l][BWD] = cand[(2 * l) + BWD];
    }
    return ports;
}

// Cores merged into rectangles: x-runs per row, then merged.
CoreRangeSet merged_set(std::vector<CoreCoord> cores) {
    std::sort(cores.begin(), cores.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return std::tie(a.y, a.x) < std::tie(b.y, b.x);
    });
    std::vector<CoreRange> runs;
    for (const auto& c : cores) {
        if (!runs.empty() && runs.back().end_coord.y == c.y && runs.back().end_coord.x + 1 == c.x) {
            runs.back().end_coord.x = c.x;
        } else {
            runs.emplace_back(c, c);
        }
    }
    if (runs.empty()) {
        return CoreRangeSet();
    }
    return CoreRangeSet(runs).merge_ranges();
}

std::vector<uint32_t> esw_offsets_of(const Geometry& g, const std::vector<const CoreWork*>& roots) {
    const size_t n = roots.size();
    uint64_t rows = 0;
    for (const auto* r : roots) {
        rows += r->rows;
    }
    const uint64_t x_bytes = rows * g.Wt * g.T_in;
    std::vector<uint32_t> offs(n, 0);
    if (g.sharded || x_bytes < ESW_MIN_BYTES || n < ESW_CLASSES) {
        return offs;
    }
    std::vector<size_t> bounds;
    for (uint32_t c = 1; c < ESW_CLASSES; ++c) {
        bounds.push_back(static_cast<size_t>(
            std::lround(static_cast<double>(c) * static_cast<double>(n) / static_cast<double>(ESW_CLASSES))));
    }
    for (size_t i = 0; i < n; ++i) {
        offs[i] = static_cast<uint32_t>(std::count_if(bounds.begin(), bounds.end(), [&](size_t b) { return i >= b; }));
    }
    return offs;
}

uint32_t wave_n_g(uint32_t wv, uint32_t rows, uint32_t off, uint32_t R, uint32_t Rw, uint32_t nwpb) {
    if (wv < off) {
        return 0;
    }
    const uint32_t b = (wv - off) / nwpb;
    const uint32_t k = (wv - off) % nwpb;
    if (static_cast<uint64_t>(b) * R >= rows) {
        return 0;
    }
    const uint32_t rb = std::min(rows - (b * R), R);
    const uint32_t first = k * Rw;
    return first >= rb ? 0 : std::min(Rw, rb - first);
}

// The exchange layout of a local plan (exchange_layout of the Python).
void fill_exchange_layout(Plan& plan, uint32_t L, uint32_t G) {
    const uint32_t R = plan.block_rows;
    const auto [Rw, nwpb] = wave_split(R);
    plan.wave_rows = Rw;
    plan.waves_per_block = nwpb;
    const auto roots = plan.roots();
    plan.esw_offsets = esw_offsets_of(plan.g, roots);
    uint32_t num_waves = 0;
    for (size_t i = 0; i < roots.size(); ++i) {
        num_waves = std::max(num_waves, plan.esw_offsets[i] + (tt::div_up(roots[i]->rows, R) * nwpb));
    }
    plan.num_waves = num_waves;
    uint32_t max_seg = 4;
    for (uint32_t l = 0; l < L; ++l) {
        uint32_t n_l = 0;
        for (size_t gi = 0; gi < roots.size(); ++gi) {
            n_l += (gi % L == l) ? 1 : 0;
        }
        max_seg = std::max(max_seg, n_l * Rw * ::distributed_rms_norm::RECORD_BYTES);
    }
    plan.seg_stride = tt::round_up(max_seg, SEG_ALIGN);
    uint32_t K = num_waves;
    if (landing_bytes_for(static_cast<uint32_t>(static_cast<uint64_t>(num_waves) * G * plan.seg_stride / 4)) >
        PORT_LANDING_MAX_BYTES) {
        K = std::min(num_waves, std::max({LANDING_RING_WAVES, nwpb, 2u}));
    }
    plan.landing_waves = K;
    plan.landing_words = static_cast<uint32_t>(static_cast<uint64_t>(K) * G * plan.seg_stride / 4);
    plan.landing_alloc_words = landing_words_bucket(plan.landing_words);
}

void fill_wave_extents(Plan& plan, uint32_t L) {
    const auto roots = plan.roots();
    const uint32_t R = plan.block_rows;
    const uint32_t Rw = plan.wave_rows;
    const uint32_t nwpb = plan.waves_per_block;
    std::vector<uint64_t> wave_rows(plan.num_waves, 0);
    for (uint32_t wv = 0; wv < plan.num_waves; ++wv) {
        for (size_t i = 0; i < roots.size(); ++i) {
            wave_rows[wv] += wave_n_g(wv, roots[i]->rows, plan.esw_offsets[i], R, Rw, nwpb);
        }
    }
    plan.rows_before.assign(plan.num_waves + 1, 0);
    for (uint32_t o = 1; o <= plan.num_waves; ++o) {
        plan.rows_before[o] = plan.rows_before[o - 1] + wave_rows[o - 1];
    }
    plan.seg_wave.assign(L, std::vector<uint32_t>(plan.num_waves, 0));
    for (uint32_t l = 0; l < L; ++l) {
        for (uint32_t wv = 0; wv < plan.num_waves; ++wv) {
            uint32_t ext = 0;
            for (size_t gi = 0; gi < roots.size(); ++gi) {
                if (gi % L == l && wave_n_g(wv, roots[gi]->rows, plan.esw_offsets[gi], R, Rw, nwpb) > 0) {
                    ext =
                        std::max(ext, static_cast<uint32_t>((gi / L) + 1) * Rw * ::distributed_rms_norm::RECORD_BYTES);
                }
            }
            plan.seg_wave[l][wv] = ext;
        }
    }
}

std::unique_ptr<Plan> solve(const PlanInputs& in) {
    const uint32_t L = in.num_links;
    const uint32_t G = in.group_size;
    TT_FATAL(L >= 1 && G >= 2, "distributed_rms_norm: the plan needs a group of >= 2 devices and >= 1 link");
    auto ports = place_ports(in);
    std::vector<CoreCoord> port_list;
    for (const auto& p : ports) {
        port_list.push_back(p[FWD]);
        port_list.push_back(p[BWD]);
    }
    std::vector<CoreCoord> available;
    for (uint32_t y = 0; y < in.grid_y; ++y) {
        for (uint32_t x = 0; x < in.grid_x; ++x) {
            const CoreCoord c(x, y);
            if (std::find(port_list.begin(), port_list.end(), c) == port_list.end()) {
                available.push_back(c);
            }
        }
    }

    auto plan = std::make_unique<Plan>();
    int64_t reserve = LANDING_L1_RESERVE;
    bool fits = false;
    for (uint32_t attempt = 0; attempt < PLAN_ATTEMPTS; ++attempt) {
        LocalPlan local;
        try {
            local = select_plan(in, available, static_cast<int64_t>(in.l1_budget) - reserve);
        } catch (const RegimeRefusal&) {
            if (reserve == LANDING_L1_RESERVE) {
                throw;
            }
            throw RegimeRefusal(fmt::format(
                "distributed_rms_norm: the local plan does not fit L1 next to a {} B exchange landing (long sequence)",
                reserve));
        }
        plan->regime = local.regime;
        plan->work = std::move(local.work);
        plan->g = local.blocking.g;
        plan->block_rows = local.blocking.R;
        plan->x_depth = local.blocking.x_depth;
        plan->rows_max = local.rows_max;
        fill_exchange_layout(*plan, L, G);
        const uint32_t need = landing_bytes_for(plan->landing_words);
        if (need > PORT_LANDING_MAX_BYTES) {
            throw RegimeRefusal(fmt::format(
                "distributed_rms_norm: the exchange landing needs {} B per port core (> {}) even as a ring of {} wave "
                "slots",
                need,
                PORT_LANDING_MAX_BYTES,
                plan->landing_waves));
        }
        if (need <= reserve) {
            fits = true;
            break;
        }
        reserve = need;  // re-plan against the landing this plan really needs
    }
    if (!fits) {
        throw RegimeRefusal("distributed_rms_norm: the local plan and the exchange landing do not fit L1 together");
    }
    fill_wave_extents(*plan, L);

    const uint64_t out_bytes = static_cast<uint64_t>(plan->g.Wt) * plan->g.T_out * [&] {
        uint32_t lead = 1;
        for (size_t i = 0; i + 2 < in.shape.size(); ++i) {
            lead *= in.shape[i];
        }
        return lead * tt::div_up(in.shape[in.shape.size() - 2], tt::constants::TILE_HEIGHT);
    }();
    plan->out_split = {0, 0, 0};
    if (!plan->g.out_native && out_bytes >= OUT_NOC_SPLIT_MIN_BYTES && in.out_dram_interleaved && in.blackhole) {
        plan->out_split = {BH_DRAM_SPLIT_X, BH_ODD_BANKS, BH_ODD_BANKS};
    }

    std::vector<CoreCoord> compute;
    compute.reserve(plan->work.size());
    for (const auto& w : plan->work) {
        compute.push_back(w.core);
    }
    plan->compute_cores = merged_set(compute);
    plan->port_cores = merged_set(port_list);
    compute.insert(compute.end(), port_list.begin(), port_list.end());
    plan->all_cores = merged_set(compute);
    plan->ports = std::move(ports);
    return plan;
}

ttsl::hash::hash_t inputs_hash(const PlanInputs& in) {
    std::vector<uint64_t> words(in.shape.begin(), in.shape.end());
    for (const auto* rects : {&in.input_shard_rects, &in.output_shard_rects}) {
        words.push_back(rects->size());
        for (const auto& r : *rects) {
            words.insert(words.end(), r.begin(), r.end());
        }
    }
    words.insert(
        words.end(),
        {static_cast<uint64_t>(in.input_dtype),
         in.gamma_dtype.has_value() ? static_cast<uint64_t>(*in.gamma_dtype) + 1 : 0,
         static_cast<uint64_t>(in.gamma_is_tile),
         static_cast<uint64_t>(in.input_layout),
         static_cast<uint64_t>(in.input_l1),
         in.shard_shape[0],
         in.shard_shape[1],
         static_cast<uint64_t>(in.shard_row_major),
         static_cast<uint64_t>(in.out_native),
         static_cast<uint64_t>(in.out_dram_interleaved),
         static_cast<uint64_t>(in.blackhole),
         static_cast<uint64_t>(in.fp32_acc),
         in.group_size,
         in.num_links,
         in.grid_x,
         in.grid_y,
         in.l1_budget});
    return ttsl::hash::hash_objects_with_default_seed(words);
}

std::vector<std::array<uint32_t, 4>> rects_of(const CoreRangeSet& set) {
    std::vector<std::array<uint32_t, 4>> rects;
    for (const auto& r : set.ranges()) {
        rects.push_back(
            {static_cast<uint32_t>(r.start_coord.x),
             static_cast<uint32_t>(r.start_coord.y),
             static_cast<uint32_t>(r.end_coord.x),
             static_cast<uint32_t>(r.end_coord.y)});
    }
    return rects;
}

bool same_placement(const tt::tt_metal::MemoryConfig& out_mc, const tt::tt_metal::MemoryConfig& in_mc) {
    if (out_mc.memory_layout() != in_mc.memory_layout()) {
        return false;
    }
    if (in_mc.memory_layout() == TensorMemoryLayout::INTERLEAVED) {
        return true;
    }
    const auto& a = out_mc.shard_spec();
    const auto& b = in_mc.shard_spec();
    return out_mc.buffer_type() == in_mc.buffer_type() && a.has_value() && b.has_value() && a->shape == b->shape &&
           a->grid == b->grid && a->orientation == b->orientation;
}

CoreRangeSet whole_grid(const MeshDevice& mesh) {
    const auto grid = mesh.compute_with_storage_grid_size();
    return CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
}

}  // namespace

std::vector<const CoreWork*> Plan::roots() const {
    std::vector<const CoreWork*> out;
    for (const auto& w : work) {
        if (w.index == 0) {
            out.push_back(&w);
        }
    }
    return out;
}

bool is_2d_fabric() {
    switch (tt::tt_fabric::GetFabricConfig()) {
        case tt::tt_fabric::FabricConfig::FABRIC_2D:
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X:
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y:
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY: return true;
        default: return false;
    }
}

bool fabric_wraps_axis(uint32_t cluster_axis) {
    switch (tt::tt_fabric::GetFabricConfig()) {
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY:
        case tt::tt_fabric::FabricConfig::FABRIC_1D_RING: return cluster_axis == 0 || cluster_axis == 1;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X: return cluster_axis == 1;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y: return cluster_axis == 0;
        default: return false;
    }
}

uint32_t usable_links(MeshDevice& mesh, uint32_t cluster_axis, bool ring) {
    using Key = std::tuple<int, uint32_t, uint32_t, uint32_t, bool>;
    static ttsl::Indestructible<std::map<Key, uint32_t>> memo;
    static std::mutex mutex;
    const Key key{mesh.id(), mesh.shape()[0], mesh.shape()[1], cluster_axis, ring};
    const std::lock_guard lock(mutex);
    auto& m = memo.get();
    if (auto it = m.find(key); it != m.end()) {
        return it->second;
    }
    const uint32_t n = count_usable_links(mesh, cluster_axis, ring);
    m.emplace(key, n);
    return n;
}

const MeshExchange& mesh_exchange(MeshDevice& mesh, uint32_t cluster_axis, bool ring, ExchangeMode mode) {
    using Key = std::tuple<int, uint32_t, uint32_t, uint32_t, bool, ExchangeMode>;
    static ttsl::Indestructible<std::map<Key, std::unique_ptr<MeshExchange>>> memo;
    static std::mutex mutex;
    const Key key{mesh.id(), mesh.shape()[0], mesh.shape()[1], cluster_axis, ring, mode};
    const std::lock_guard lock(mutex);
    auto& m = memo.get();
    if (auto it = m.find(key); it != m.end()) {
        return *it->second;
    }
    auto ex = std::make_unique<MeshExchange>();
    const auto groups = build_groups(mesh, cluster_axis);
    for (const auto& grp : groups) {
        for (const auto& c : grp) {
            ex->nodes.insert_or_assign(c, mesh.get_fabric_node_id(c));
        }
    }
    if (mode == ExchangeMode::DIRECT) {
        try {
            build_schedule(mesh, groups, ring, ExchangeMode::DIRECT, ex->chips);
        } catch (const OffPortRoute&) {
            ex->chips.clear();
            build_schedule(mesh, groups, ring, ExchangeMode::RELAY, ex->chips);
        }
    } else {
        build_schedule(mesh, groups, ring, mode, ex->chips);
    }
    const auto& ref = *ex;
    m.emplace(key, std::move(ex));
    return ref;
}

const ExchangeSemaphores& exchange_semaphores(MeshDevice& mesh, uint32_t group_size) {
    struct Entry {
        std::vector<tt::tt_metal::GlobalSemaphore> sems;
        ExchangeSemaphores addrs;
    };
    static ttsl::Indestructible<std::map<int, Entry>> memo;
    static std::mutex mutex;
    const std::lock_guard lock(mutex);
    auto& entry = memo.get()[mesh.id()];
    const uint32_t want = group_size + 1;  // ready + one arrival counter per slot
    if (entry.sems.size() < want) {
        const CoreRangeSet crs = whole_grid(mesh);
        while (entry.sems.size() < want) {
            entry.sems.push_back(ttnn::global_semaphore::create_global_semaphore(&mesh, crs, 0));
        }
        // Every chip's counters are 0 before any peer's first increment.
        tt::tt_metal::distributed::Synchronize(mesh, std::nullopt);
        entry.addrs.ready_addr =
            static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(entry.sems[0]));
        entry.addrs.arrival_addrs.clear();
        for (size_t i = 1; i < entry.sems.size(); ++i) {
            entry.addrs.arrival_addrs.push_back(
                static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(entry.sems[i])));
        }
    }
    return entry.addrs;
}

PlanInputs plan_inputs(const DistributedRmsNormParams& attrs, const DistributedRmsNormInputs& t, const Tensor& output) {
    auto* mesh = t.input.device();
    TT_FATAL(mesh != nullptr, "distributed_rms_norm: the input must be on a mesh device");
    // Persistent semaphores first: their L1 words bound this call's CB region too.
    exchange_semaphores(*mesh, attrs.group_size);

    PlanInputs in;
    const auto& shape = t.input.logical_shape();
    for (size_t i = 0; i < shape.rank(); ++i) {
        in.shape.push_back(shape[static_cast<int32_t>(i)]);
    }
    in.input_dtype = t.input.dtype();
    if (t.gamma.has_value()) {
        in.gamma_dtype = t.gamma->dtype();
        in.gamma_is_tile = t.gamma->layout() == tt::tt_metal::Layout::TILE;
    }
    const auto& in_mc = t.input.memory_config();
    const auto& out_mc = output.memory_config();
    in.input_layout = in_mc.memory_layout();
    in.input_l1 = in_mc.buffer_type() == tt::tt_metal::BufferType::L1;
    if (in_mc.memory_layout() != TensorMemoryLayout::INTERLEAVED && in_mc.shard_spec().has_value()) {
        const auto& spec = *in_mc.shard_spec();
        in.shard_shape = spec.shape;
        in.shard_row_major = spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
        in.input_shard_rects = rects_of(spec.grid);
    }
    if (out_mc.memory_layout() != TensorMemoryLayout::INTERLEAVED && out_mc.shard_spec().has_value()) {
        in.output_shard_rects = rects_of(out_mc.shard_spec()->grid);
    }
    in.out_native = in_mc.memory_layout() != TensorMemoryLayout::INTERLEAVED && same_placement(out_mc, in_mc);
    in.out_dram_interleaved = out_mc.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                              out_mc.buffer_type() == tt::tt_metal::BufferType::DRAM;
    in.blackhole = mesh->arch() == tt::ARCH::BLACKHOLE;
    in.fp32_acc = attrs.compute_config.fp32_dest_acc_en;
    in.group_size = attrs.group_size;
    in.num_links = attrs.num_links;
    const auto grid = mesh->compute_with_storage_grid_size();
    in.grid_x = grid.x;
    in.grid_y = grid.y;

    // The CB region starts at the allocator base and ends below the lowest L1 buffer (every tensor this call knows,
    // the persistent semaphores), and is no larger than the largest free block. As in the Python, the plan is solved
    // against the L1 left before the exchange landing exists (the solve reserves the landing's bytes itself): when
    // the call's landing is the lowest L1 buffer, the region it occupies counts as free, so the entry (landing not
    // yet allocated) and the program key (landing allocated) solve the same plan. The landing is refused unless it
    // is exactly the size that plan needs, so the CBs stay below it.
    const auto base = mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    uint64_t budget = mesh->allocator()->get_statistics(tt::tt_metal::BufferType::L1).largest_free_block_bytes;
    if (const auto lowest = mesh->lowest_occupied_compute_l1_address(); lowest.has_value() && *lowest > base) {
        const tt::tt_metal::Buffer* landing = t.scratch.has_value() ? t.scratch->buffer() : nullptr;
        if (landing != nullptr && landing->buffer_type() == tt::tt_metal::BufferType::L1 &&
            landing->address() == *lowest) {  // smuggled-rta-ok: L1 budget of the plan, never a kernel arg
            budget = *lowest + landing->aligned_size_per_bank() - base;
        } else {
            budget = std::min<uint64_t>(budget, *lowest - base);
        }
    }
    in.l1_budget = static_cast<uint32_t>(std::min<uint64_t>(budget, std::numeric_limits<uint32_t>::max()));
    return in;
}

const Plan& solve_plan(const PlanInputs& inputs) {
    static ttsl::Indestructible<std::map<ttsl::hash::hash_t, std::vector<std::pair<PlanInputs, std::unique_ptr<Plan>>>>>
        memo;
    static std::mutex mutex;
    const auto h = inputs_hash(inputs);
    const std::lock_guard lock(mutex);
    auto& bucket = memo.get()[h];
    for (const auto& [key, plan] : bucket) {
        if (key == inputs) {
            return *plan;
        }
    }
    auto plan = solve(inputs);
    const Plan& ref = *plan;
    bucket.emplace_back(inputs, std::move(plan));
    return ref;
}

ttsl::hash::hash_t plan_hash(const Plan& plan) {
    const Geometry& g = plan.g;
    std::vector<uint64_t> words{
        static_cast<uint64_t>(plan.regime),
        plan.block_rows,
        plan.x_depth,
        plan.rows_max,
        g.has_gamma,
        g.W,
        g.Wt,
        g.S,
        g.C,
        g.num_w_chunks,
        g.Wp,
        g.WS,
        g.T_in,
        g.T_out,
        g.T_g,
        g.es,
        g.es_g,
        g.gamma_is_tile,
        g.interm_fp32,
        g.T_i,
        g.sharded,
        g.combine,
        g.G_w,
        g.masked,
        g.G,
        g.gamma_resident,
        g.out_native,
        plan.wave_rows,
        plan.waves_per_block,
        plan.num_waves,
        plan.landing_waves,
        plan.seg_stride,
        plan.landing_words,
        plan.landing_alloc_words,
        plan.out_split[0],
        plan.out_split[1],
        plan.out_split[2],
        plan.work.size()};
    for (const auto& w : plan.work) {
        words.insert(
            words.end(), {w.core.x, w.core.y, w.start_tile_row, w.rows, w.w0, w.V, w.index, w.root.x, w.root.y});
    }
    for (const auto& p : plan.ports) {
        words.insert(words.end(), {p[FWD].x, p[FWD].y, p[BWD].x, p[BWD].y});
    }
    words.insert(words.end(), plan.esw_offsets.begin(), plan.esw_offsets.end());
    words.insert(words.end(), plan.rows_before.begin(), plan.rows_before.end());
    for (const auto& l : plan.seg_wave) {
        words.insert(words.end(), l.begin(), l.end());
    }
    return ttsl::hash::hash_objects_with_default_seed(words);
}

std::vector<CbEntry> cb_inventory(
    const Geometry& g,
    uint32_t block_rows,
    uint32_t x_depth,
    tt::DataFormat x_format,
    tt::DataFormat output_format,
    tt::DataFormat gamma_format) {
    const uint32_t R = block_rows;
    const auto [Rw, nwpb] = wave_split(R);
    const tt::DataFormat fi = g.interm_fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t T_i = g.T_i;
    const tt::DataFormat scaler_format = tt::tt_metal::datatype_to_dataformat_converter(DataType::BFLOAT16);
    namespace cb = ::distributed_rms_norm::cb;
    std::vector<CbEntry> inv{
        {cb::REDUCE_SCALER, scaler_format, tt::tt_metal::tile_size(DataType::BFLOAT16), g.masked ? 2u : 1u},
        {cb::SUMSQ_PARTIAL, fi, T_i, Rw},
        {cb::INV_RMS, fi, T_i, R},
        {cb::DEV_PARTIAL, fi, T_i, nwpb * Rw},
        {cb::STATS_GATHER, fi, T_i, g.fin_tpr() * Rw},
        {cb::STATS_RECORD, tt::DataFormat::UInt32, ::distributed_rms_norm::RECORD_BYTES, (g.G + 1) * Rw},
    };
    if (!g.sharded) {
        inv.push_back({cb::X_TILES, x_format, g.T_in, x_depth * R * g.WS});
    }
    if (!g.out_native) {
        inv.push_back({cb::OUTPUT, output_format, g.T_out, STREAM_DEPTH * g.C});
    }
    if (g.has_gamma) {
        // resident window or stream ring
        inv.push_back({cb::GAMMA_TILES, gamma_format, g.T_g, g.gamma_resident ? g.Wp : STREAM_DEPTH * g.C});
        inv.push_back({cb::X_GAMMA, fi, T_i, R * g.C});
    }
    if (g.combine) {
        inv.push_back({cb::PARTIAL_COL, fi, T_i, R});
        inv.push_back({cb::GATHER, fi, T_i, nwpb * g.G_w * Rw});
        inv.push_back({cb::RMS_ROOT, fi, T_i, nwpb * Rw});
    } else if (g.fin_compact()) {
        inv.push_back({cb::RMS_ROOT, fi, T_i, nwpb * Rw});  // R1 roots re-lay their own inv_rms too
    }
    if (g.masked) {
        inv.push_back({cb::SUMSQ_LAST, fi, T_i, Rw});
        inv.push_back({cb::SUMSQ_ACC, fi, T_i, Rw});
    }
    return inv;
}

uint32_t landing_bytes_for(uint32_t landing_words) { return landing_words_bucket(landing_words) * 4; }

tt::tt_metal::TensorSpec landing_spec(MeshDevice& mesh, uint32_t landing_alloc_words) {
    const auto grid = mesh.compute_with_storage_grid_size();
    const tt::tt_metal::MemoryConfig mc(
        TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(whole_grid(mesh), {1, landing_alloc_words}, tt::tt_metal::ShardOrientation::ROW_MAJOR));
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({static_cast<uint32_t>(grid.x * grid.y), landing_alloc_words}),
        tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), mc));
}

}  // namespace ttnn::operations::distributed_rms_norm::detail
