// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_rms_norm_planner.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>

#include <fmt/format.h>

#include <internal/mesh_device.hpp>
#include <tt-metalium/mesh/allocator.hpp>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/discovery/hal.hpp>
#include <tt-metalium/utils/math.hpp>
#include <tt-metalium/mesh/mesh_device.hpp>
#include <tt-metalium/memory/tensor/tensor_types.hpp>

#include "kernels/all_gather_rms_norm_args.hpp"

namespace ttnn::operations::experimental::ccl::all_gather_rms_norm::planner {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::distributed::MeshDevice;

namespace {

constexpr uint32_t kTile = tt::constants::TILE_WIDTH;

uint32_t plan_datum_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::BFLOAT16: return 2;
        case DataType::BFLOAT8_B: return 1;
        default: TT_THROW("all_gather_rms_norm: unsupported dtype {}", dtype);
    }
}

uint32_t plan_cb_round(uint64_t bytes) { return static_cast<uint32_t>(tt::round_up(bytes, uint64_t{CB_ALIGN})); }

std::vector<std::vector<ttnn::MeshCoordinate>> plan_build_groups(const MeshDevice& mesh, uint32_t cluster_axis) {
    const auto& shape = mesh.shape();
    const uint32_t rows = shape[0];
    const uint32_t cols = shape.dims() > 1 ? shape[1] : 1;
    std::vector<std::vector<ttnn::MeshCoordinate>> groups;
    if (cluster_axis == 0) {
        for (uint32_t c = 0; c < cols; ++c) {
            auto& g = groups.emplace_back();
            for (uint32_t r = 0; r < rows; ++r) {
                g.emplace_back(r, c);
            }
        }
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            auto& g = groups.emplace_back();
            for (uint32_t c = 0; c < cols; ++c) {
                g.emplace_back(r, c);
            }
        }
    }
    return groups;
}

uint32_t plan_pymod(int64_t a, int64_t m) { return static_cast<uint32_t>(((a % m) + m) % m); }

void plan_push_mc(std::vector<uint64_t>& key, const tt::tt_metal::MemoryConfig& mc) {
    key.push_back(static_cast<uint64_t>(mc.memory_layout()));
    key.push_back(static_cast<uint64_t>(mc.buffer_type()));
    if (mc.shard_spec().has_value()) {
        const auto& ss = *mc.shard_spec();
        key.push_back(ss.shape[0]);
        key.push_back(ss.shape[1]);
        key.push_back(static_cast<uint64_t>(ss.orientation));
        for (const auto& r : ss.grid.ranges()) {
            key.push_back(r.start_coord.x);
            key.push_back(r.start_coord.y);
            key.push_back(r.end_coord.x);
            key.push_back(r.end_coord.y);
        }
    }
    key.push_back(0xFFFFFFFFu);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Geometry and work split
// ---------------------------------------------------------------------------------------------------------------------

bool Geometry::aligned() const { return W_local % kTile == 0; }

Geometry derive_geometry(const ttnn::Shape& shape, uint32_t G) {
    TT_FATAL(shape.rank() >= 2, "all_gather_rms_norm: input rank must be >= 2, got {}", shape.rank());
    Geometry geo;
    geo.W_local = shape[-1];
    geo.M = shape[-2];
    uint32_t num_images = 1;
    for (int32_t i = 0; i + 2 < static_cast<int32_t>(shape.rank()); ++i) {
        num_images *= shape[i];
    }
    geo.num_images = num_images;
    geo.R = num_images * tt::div_up(geo.M, kTile);
    geo.Wlt = tt::div_up(geo.W_local, kTile);
    geo.G = G;
    geo.W = G * geo.W_local;
    geo.Wt = tt::div_up(geo.W, kTile);
    geo.w_valid_last = geo.W_local - (kTile * (geo.Wlt - 1));
    return geo;
}

uint32_t Split::last_rows() const {
    if (last_crt != 0) {
        return last_crt;
    }
    return big == n_groups ? rows_hi : rows_hi - 1;
}

uint32_t Split::n_active() const { return n_groups * k; }

uint32_t Split::member_width(uint32_t m, uint32_t Wlt) const {
    const int64_t rest = static_cast<int64_t>(Wlt) - (static_cast<int64_t>(m) * cw_nom);
    return static_cast<uint32_t>(std::max<int64_t>(0, std::min<int64_t>(cw_nom, rest)));
}

uint32_t stats_k_cap(uint32_t R, uint32_t G, uint32_t nstats) {
    const uint64_t denom = uint64_t{G} * R * nstats * agrn_args::RECORD_BYTES;
    TT_FATAL(denom > 0, "all_gather_rms_norm: empty slice (R={})", R);
    return static_cast<uint32_t>(STATS_TABLE_CAP / denom);
}

Split select_split(uint32_t R, uint32_t Wlt, uint32_t Nw, uint32_t G, uint32_t k_cap) {
    bool found = false;
    uint64_t best_cost = 0;
    uint32_t best_k = 1;
    const uint32_t k_hi = std::max(1u, std::min(Wlt, k_cap));
    for (uint32_t k = 1; k <= k_hi; ++k) {
        const uint32_t groups = std::min(Nw / k, R);
        if (groups == 0) {
            break;
        }
        const uint64_t cost =
            uint64_t{tt::div_up(R, groups)} * (tt::div_up(Wlt, k) + (COMB_TILE_COST_FUSED * tt::div_up(G * k, kTile)));
        if (!found || cost < best_cost) {
            found = true;
            best_cost = cost;
            best_k = k;
        }
    }
    TT_FATAL(found, "all_gather_rms_norm: no worker core for the slice ({} workers, {} tile-rows)", Nw, R);
    Split s;
    s.k = tt::div_up(Wlt, tt::div_up(Wlt, best_k));
    s.n_groups = std::min(Nw / s.k, R);
    TT_FATAL(s.n_groups > 0, "all_gather_rms_norm: no row-group fits {} workers", Nw);
    s.rows_hi = tt::div_up(R, s.n_groups);
    s.big = R - (s.n_groups * (s.rows_hi - 1));
    s.cw_nom = tt::div_up(Wlt, s.k);
    s.last_crt = 0;
    return s;
}

std::vector<CoreCoord> ordered_shard_cores(const CoreRangeSet& grid, bool row_major) {
    std::vector<CoreCoord> cores;
    for (const auto& r : grid.ranges()) {
        for (auto y = r.start_coord.y; y <= r.end_coord.y; ++y) {
            for (auto x = r.start_coord.x; x <= r.end_coord.x; ++x) {
                cores.emplace_back(x, y);
            }
        }
    }
    std::sort(cores.begin(), cores.end(), [row_major](const CoreCoord& a, const CoreCoord& b) {
        return row_major ? std::tie(a.y, a.x) < std::tie(b.y, b.x) : std::tie(a.x, a.y) < std::tie(b.x, b.y);
    });
    cores.erase(std::unique(cores.begin(), cores.end()), cores.end());
    return cores;
}

Split select_sharded_split(const Geometry& geo, const Tensor& input, std::vector<CoreCoord>& shard_order) {
    const auto& mc = input.memory_config();
    TT_FATAL(mc.shard_spec().has_value(), "all_gather_rms_norm: a sharded input needs a shard spec");
    const auto& ss = *mc.shard_spec();
    const uint32_t sh = ss.shape[0];
    const uint32_t sw = ss.shape[1];
    if (sh % kTile != 0 || sw % kTile != 0) {
        throw std::runtime_error(
            fmt::format("all_gather_rms_norm: TILE shard {}x{} is not a whole number of tiles", sh, sw));
    }
    const uint32_t unit_h = sh / kTile;
    const uint32_t unit_w = sw / kTile;
    const auto order = ordered_shard_cores(ss.grid, ss.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
    if (mc.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        if (unit_w != geo.Wlt) {
            throw std::runtime_error(
                fmt::format("all_gather_rms_norm: HEIGHT shard width {} != the padded slice width", sw));
        }
        const uint32_t n_h = tt::div_up(geo.R, unit_h);
        if (order.size() < n_h) {
            throw std::runtime_error(
                fmt::format("all_gather_rms_norm: {} row shards but only {} shard-grid cores", n_h, order.size()));
        }
        Split s{
            .k = 1,
            .n_groups = n_h,
            .rows_hi = unit_h,
            .big = n_h,
            .cw_nom = geo.Wlt,
            .last_crt = geo.R - ((n_h - 1) * unit_h)};
        shard_order.assign(order.begin(), order.begin() + n_h);
        return s;
    }
    if (mc.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        if (unit_h < geo.R) {
            throw std::runtime_error(
                fmt::format("all_gather_rms_norm: WIDTH shard height {} does not cover the {} tile-rows", sh, geo.R));
        }
        const uint32_t k = tt::div_up(geo.Wlt, unit_w);
        if (order.size() < k) {
            throw std::runtime_error(
                fmt::format("all_gather_rms_norm: {} W shards but only {} shard-grid cores", k, order.size()));
        }
        Split s{.k = k, .n_groups = 1, .rows_hi = geo.R, .big = 1, .cw_nom = unit_w, .last_crt = geo.R};
        shard_order.assign(order.begin(), order.begin() + k);
        return s;
    }
    throw std::runtime_error(fmt::format("all_gather_rms_norm: memory layout {} is not built", mc.memory_layout()));
}

// ---------------------------------------------------------------------------------------------------------------------
// Chunk walk (host replica of agrn_port_common.hpp for_each_packet: counts only)
// ---------------------------------------------------------------------------------------------------------------------

namespace {
struct PlanWalkState {
    uint32_t count = 0, runs = 0, pages = 0;
};

void plan_walk_add(const Walk& w, PlanWalkState& st, uint32_t n) {
    if (w.split_runs == 0) {
        if (st.runs == w.scatter_runs || st.pages + n > w.run_pages) {
            ++st.count;
            st.runs = 0;
            st.pages = 0;
        }
        ++st.runs;
        st.pages += n;
        return;
    }
    while (n > 0) {
        if (st.runs == w.scatter_runs || st.pages == w.run_pages) {
            ++st.count;
            st.runs = 0;
            st.pages = 0;
        }
        const uint32_t m = std::min(n, w.run_pages - st.pages);
        ++st.runs;
        st.pages += m;
        n -= m;
    }
}
}  // namespace

uint32_t Walk::chunks(uint32_t q, uint32_t part, uint32_t first_bank, uint32_t stride) const {
    const uint32_t n_lanes = shard_w != 0 ? rows_eff : nb;
    if (first_bank >= n_lanes) {
        return 0;
    }
    const uint32_t nl = (n_lanes - first_bank + stride - 1) / stride;
    const uint32_t lo = part == 2 ? nl / 2 : 0;
    const uint32_t hi = part == 1 ? nl / 2 : nl;
    PlanWalkState st;
    const uint32_t base = q * slot_stride;
    for (uint32_t i = lo; i < hi; ++i) {
        if (i == nl / 2 && st.runs != 0) {
            ++st.count;
            st.runs = 0;
            st.pages = 0;
        }
        if (shard_w != 0) {
            uint32_t jj = 0;
            while (jj < cols_eff) {
                const uint32_t gc = base + jj;
                const uint32_t seg_end = std::min(cols_eff, ((gc / shard_w + 1) * shard_w) - base);
                const uint32_t n = std::min(seg_end - jj, run_pages);
                plan_walk_add(*this, st, n);
                jj += n;
            }
            continue;
        }
        const uint32_t b = first_bank + (i * stride);
        for (uint32_t rr = 0; rr < rows_eff; ++rr) {
            const uint32_t rb = base + (rr * row_stride);
            uint32_t jj = plan_pymod(static_cast<int64_t>(b) - static_cast<int64_t>(rb % nb), nb);
            while (jj < cols_eff) {
                const uint32_t n = std::min(((cols_eff - 1 - jj) / nb) + 1, run_pages);
                plan_walk_add(*this, st, n);
                jj += n * nb;
            }
        }
    }
    return st.count + (st.runs != 0 ? 1 : 0);
}

// ---------------------------------------------------------------------------------------------------------------------
// Mesh plan
// ---------------------------------------------------------------------------------------------------------------------

std::optional<uint32_t> usable_links(MeshDevice& mesh, uint32_t cluster_axis, bool ring) {
    std::optional<uint32_t> best;
    for (const auto& grp : plan_build_groups(mesh, cluster_axis)) {
        const size_t G = grp.size();
        std::vector<std::pair<ttnn::MeshCoordinate, ttnn::MeshCoordinate>> hops;
        for (size_t i = 0; i + 1 < G; ++i) {
            hops.emplace_back(grp[i], grp[i + 1]);
        }
        if (ring && G > 0) {
            hops.emplace_back(grp[G - 1], grp[0]);
        }
        for (const auto& [a, b] : hops) {
            const auto na = mesh.get_fabric_node_id(a);
            const auto nb = mesh.get_fabric_node_id(b);
            for (const auto& [x, y] : {std::pair{na, nb}, std::pair{nb, na}}) {
                const auto n = static_cast<uint32_t>(tt::tt_fabric::get_forwarding_link_indices(mesh, x, y).size());
                best = best.has_value() ? std::min(*best, n) : n;
            }
        }
    }
    return best;
}

const MeshPlan& build_mesh_plan(
    MeshDevice& mesh, uint32_t cluster_axis, bool ring, uint32_t num_links, const std::vector<CoreCoord>& excluded) {
    static std::map<std::vector<uint64_t>, MeshPlan> cache;
    std::vector<CoreCoord> excl = excluded;
    std::sort(excl.begin(), excl.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return std::tie(a.x, a.y) < std::tie(b.x, b.y);
    });
    excl.erase(std::unique(excl.begin(), excl.end()), excl.end());
    std::vector<uint64_t> key{static_cast<uint64_t>(mesh.id()), cluster_axis, ring ? 1u : 0u, num_links};
    for (size_t i = 0; i < mesh.shape().dims(); ++i) {
        key.push_back(mesh.shape()[i]);
    }
    for (const auto& c : excl) {
        key.push_back((uint64_t{c.x} << 32) | c.y);
    }
    if (auto it = cache.find(key); it != cache.end()) {
        return it->second;
    }

    MeshPlan plan;
    plan.groups = plan_build_groups(mesh, cluster_axis);
    const auto G = static_cast<uint32_t>(plan.groups.front().size());
    plan.G = G;
    plan.ring = ring;
    plan.num_links = num_links;
    plan.grid = mesh.compute_with_storage_grid_size();
    const CoreCoord grid = plan.grid;

    std::vector<CoreCoord> allowed;  // sorted by (x, y), as Python's sorted(allowed)
    for (uint32_t x = 0; x < grid.x; ++x) {
        for (uint32_t y = 0; y < grid.y; ++y) {
            const CoreCoord c(x, y);
            if (!std::binary_search(excl.begin(), excl.end(), c, [](const CoreCoord& a, const CoreCoord& b) {
                    return std::tie(a.x, a.y) < std::tie(b.x, b.y);
                })) {
                allowed.push_back(c);
            }
        }
    }
    TT_FATAL(
        allowed.size() >= 2 * num_links,
        "all_gather_rms_norm: {} cores outside the input shard grid, {} port cores needed",
        allowed.size(),
        2 * num_links);

    for (const auto& grp : plan.groups) {
        for (uint32_t p = 0; p < G; ++p) {
            ChipPlan ch;
            ch.p = p;
            if (ring || p < G - 1) {
                ch.next = grp[(p + 1) % G];
            }
            if (ring || p > 0) {
                ch.prev = grp[(p + G - 1) % G];
            }
            plan.chips.emplace(grp[p], std::move(ch));
        }
    }
    for (auto& [coord, ch] : plan.chips) {
        for (uint32_t d = 0; d < 2; ++d) {
            const auto& peer = d == 0 ? ch.next : ch.prev;
            if (!peer.has_value()) {
                continue;
            }
            auto links = tt::tt_fabric::get_forwarding_link_indices(
                mesh, mesh.get_fabric_node_id(coord), mesh.get_fabric_node_id(*peer));
            if (links.size() < num_links) {
                throw std::invalid_argument(fmt::format(
                    "all_gather_rms_norm: {} -> {} has {} usable link(s), {} requested",
                    coord,
                    *peer,
                    links.size(),
                    num_links));
            }
            links.resize(num_links);
            ch.links[d] = std::move(links);
        }
    }

    // Port placement: per link the free worker with the fewest NoC1 hops to the link's Ethernet core (ties: logical
    // row, then column); a port without a peer takes the worker nearest NoC column 0.
    for (auto& [coord, ch] : plan.chips) {
        tt::tt_metal::IDevice* device = tt::tt_metal::internal::get_device(mesh, coord);
        std::vector<CoreCoord> taken;
        auto is_taken = [&taken](const CoreCoord& c) {
            return std::find(taken.begin(), taken.end(), c) != taken.end();
        };
        std::array<std::vector<std::optional<CoreCoord>>, 2> ports;
        for (uint32_t d = 0; d < 2; ++d) {
            const auto& peer = d == 0 ? ch.next : ch.prev;
            for (uint32_t l = 0; l < num_links; ++l) {
                std::optional<CoreCoord> core;
                if (peer.has_value()) {
                    const CoreCoord eth = tt::tt_fabric::get_forwarding_eth_core(
                        mesh, mesh.get_fabric_node_id(coord), mesh.get_fabric_node_id(*peer), ch.links[d][l]);
                    const auto closest = tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(
                        *device, eth, tt::tt_metal::NOC::NOC_1);
                    std::optional<std::tuple<uint32_t, size_t, size_t>> best;
                    for (const auto& c : allowed) {
                        if (is_taken(c)) {
                            continue;
                        }
                        const uint32_t hops = closest.distance_in_noc_hops +
                                              tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                                                  device, c, closest.logical_coord, tt::tt_metal::NOC::NOC_1);
                        const auto score = std::make_tuple(hops, c.y, c.x);
                        if (!best.has_value() || score < *best) {
                            best = score;
                            core = c;
                        }
                    }
                }
                ports[d].push_back(core);
                if (core.has_value()) {
                    taken.push_back(*core);
                }
            }
        }
        for (uint32_t d = 0; d < 2; ++d) {
            for (auto& core : ports[d]) {
                if (!core.has_value()) {
                    std::optional<std::pair<size_t, size_t>> best;
                    for (const auto& c : allowed) {
                        if (is_taken(c)) {
                            continue;
                        }
                        const CoreCoord v = device->worker_core_from_logical_core(c);
                        const auto score = std::make_pair(v.x, v.y);
                        if (!best.has_value() || score < *best) {
                            best = score;
                            core = c;
                        }
                    }
                    TT_FATAL(core.has_value(), "all_gather_rms_norm: no free core for a port on {}", coord);
                    taken.push_back(*core);
                }
                ch.ports[d].push_back(*core);
            }
        }
    }
    return cache.emplace(std::move(key), std::move(plan)).first->second;
}

std::pair<Schedule, Schedule> schedule(uint32_t p, uint32_t G, bool ring, bool balance) {
    Schedule fwd;
    Schedule bwd;
    const auto pi = static_cast<int64_t>(p);
    if (ring && balance && G % 2 == 0 && G >= 4) {
        const uint32_t h = G / 2;
        for (uint32_t i = 0; i < h; ++i) {
            fwd.emplace_back(plan_pymod(pi - i, G), i == h - 1 ? 1u : 0u);
            bwd.emplace_back(plan_pymod(pi + i, G), i == h - 1 ? 2u : 0u);
        }
        return {fwd, bwd};
    }
    if (ring) {
        const uint32_t kf = G / 2;
        const uint32_t kb = G - 1 - kf;
        for (uint32_t i = 0; i < kf; ++i) {
            fwd.emplace_back(plan_pymod(pi - i, G), 0u);
        }
        for (uint32_t i = 0; i < kb; ++i) {
            bwd.emplace_back(plan_pymod(pi + i, G), 0u);
        }
        return {fwd, bwd};
    }
    if (p < G - 1) {
        for (uint32_t i = 0; i <= p; ++i) {
            fwd.emplace_back(p - i, 0u);
        }
    }
    if (p > 0) {
        for (uint32_t i = 0; i < G - p; ++i) {
            bwd.emplace_back(p + i, 0u);
        }
    }
    return {fwd, bwd};
}

// ---------------------------------------------------------------------------------------------------------------------
// L1 room, staging, the program plan
// ---------------------------------------------------------------------------------------------------------------------

std::pair<uint32_t, uint32_t> l1_cb_room(MeshDevice& mesh) {
    const auto base = static_cast<uint32_t>(mesh.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    const auto lowest = mesh.lowest_occupied_compute_l1_address();
    return {base, static_cast<uint32_t>(lowest.has_value() ? *lowest : mesh.l1_size_per_core())};
}

tt::tt_metal::TensorSpec staging_spec(const Geometry& geo, DataType dtype) {
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, geo.G * geo.R * kTile, geo.Wlt * kTile}),
        tt::tt_metal::TensorLayout(
            dtype,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)));
}

const ProgramPlan& solve_program_plan(
    const AllGatherRmsNormParams& attrs, const AllGatherRmsNormInputs& tensors, const MeshPlan& mesh_plan) {
    static std::map<std::vector<uint64_t>, ProgramPlan> cache;
    const Tensor& input = tensors.input;
    MeshDevice& mesh = *input.device();
    const tt::tt_metal::MemoryConfig& in_mc = input.memory_config();
    const tt::tt_metal::MemoryConfig& out_mc =
        tensors.output.has_value() ? tensors.output->memory_config() : attrs.output_memory_config;
    const DataType x_dtype = input.dtype();
    const DataType out_dtype = tensors.output.has_value() ? tensors.output->dtype() : x_dtype;
    const bool has_gamma = tensors.gamma.has_value();
    const bool has_beta = tensors.beta.has_value();
    const DataType g_dtype = has_gamma ? tensors.gamma->dtype() : DataType::BFLOAT16;
    const auto payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());

    uint32_t out_page = 0;
    if (tensors.output.has_value()) {
        out_page = static_cast<uint32_t>(tensors.output->buffer()->aligned_page_size());
    } else {
        const uint32_t align = out_mc.buffer_type() == BufferType::DRAM ? tt::tt_metal::hal::get_dram_alignment()
                                                                        : tt::tt_metal::hal::get_l1_alignment();
        out_page = tt::round_up(tt::tt_metal::tile_size(out_dtype), align);
    }
    const auto [l1_base, l1_lowest] = l1_cb_room(mesh);

    std::vector<uint64_t> key{
        reinterpret_cast<uintptr_t>(&mesh_plan),
        attrs.is_layer_norm ? 1u : 0u,
        attrs.fp32_dest_acc_en ? 1u : 0u,
        static_cast<uint64_t>(x_dtype),
        static_cast<uint64_t>(out_dtype),
        has_gamma ? 1u : 0u,
        has_beta ? 1u : 0u,
        static_cast<uint64_t>(g_dtype),
        out_page,
        payload,
        l1_base,
        l1_lowest};
    for (size_t i = 0; i < input.logical_shape().rank(); ++i) {
        key.push_back(input.logical_shape()[static_cast<int32_t>(i)]);
    }
    plan_push_mc(key, in_mc);
    plan_push_mc(key, out_mc);
    if (auto it = cache.find(key); it != cache.end()) {
        return it->second;
    }

    ProgramPlan pl;
    const uint32_t G = mesh_plan.G;
    const uint32_t L = mesh_plan.num_links;
    const auto gx = static_cast<uint32_t>(mesh_plan.grid.x);
    const auto gy = static_cast<uint32_t>(mesh_plan.grid.y);
    const uint32_t n_ports = 2 * L;
    TT_FATAL(gx * gy > n_ports, "all_gather_rms_norm: grid {}x{} has no worker beside {} ports", gx, gy, n_ports);
    const uint32_t Nw = (gx * gy) - n_ports;
    const bool is_ln = attrs.is_layer_norm;
    const Geometry geo = derive_geometry(input.logical_shape(), G);
    pl.geo = geo;
    pl.nstats = is_ln ? 3 : 1;
    uint32_t k_cap = P_CAP / G;
    if (is_ln) {
        k_cap = std::min(k_cap, kTile / G);
    }
    k_cap = std::max(1u, std::min(k_cap, stats_k_cap(geo.R, G, pl.nstats)));
    pl.x_sharded = in_mc.is_sharded();
    if (pl.x_sharded) {
        pl.split = select_sharded_split(geo, input, pl.shard_order);
    } else {
        pl.split = select_split(geo.R, geo.Wlt, Nw, G, k_cap);
    }
    const Split& split = pl.split;
    const uint32_t k = split.k;
    pl.pre_combine = pl.x_sharded && in_mc.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
    pl.kt = pl.pre_combine ? 1 : k;
    pl.P = G * pl.kt;
    pl.gather_tiles = tt::div_up(pl.P, kTile);
    pl.gt_pc = pl.pre_combine ? tt::div_up(k, kTile) : 0;
    pl.gt_max = std::max(pl.gather_tiles, pl.gt_pc);
    pl.rpo = pl.pre_combine ? tt::div_up(geo.R, k) : 0;
    pl.n_owners = pl.pre_combine ? tt::div_up(geo.R, pl.rpo) : 0;
    pl.nfinal = is_ln ? 2 : 1;
    pl.crt_max = split.rows_hi;
    pl.cw_max = split.cw_nom;
    const uint64_t slot_bytes = uint64_t{geo.R} * pl.kt * pl.nstats * agrn_args::RECORD_BYTES;
    const uint64_t table_bytes = G * slot_bytes;
    TT_FATAL(
        table_bytes <= STATS_TABLE_CAP,
        "all_gather_rms_norm: stats table {} B exceeds STATS_TABLE_CAP {} B (R={}, k={}, nstats={}); a row-windowed "
        "stats round (R_WIN) is deferred",
        table_bytes,
        STATS_TABLE_CAP,
        geo.R,
        k,
        pl.nstats);
    pl.slot_bytes = static_cast<uint32_t>(slot_bytes);
    pl.table_bytes = static_cast<uint32_t>(table_bytes);

    pl.is_ru = !geo.aligned();
    const bool fp32_dest = attrs.fp32_dest_acc_en;
    const DataType acc_dtype = fp32_dest ? DataType::FLOAT32 : DataType::BFLOAT16;
    const uint32_t T_acc = tt::tt_metal::tile_size(acc_dtype);
    const uint32_t T_in = tt::tt_metal::tile_size(x_dtype);
    const uint32_t T32 = tt::tt_metal::tile_size(DataType::FLOAT32);
    pl.out_page = out_page;
    const uint32_t g_eb = plan_datum_bytes(g_dtype);
    const uint32_t gamma_page = 2 * tt::constants::FACE_HEIGHT * tt::constants::FACE_WIDTH * g_eb;
    const bool x_is_f32 = x_dtype == DataType::FLOAT32;
    pl.mask_tail = pl.x_sharded && !geo.aligned();

    if (pl.pre_combine) {
        uint64_t per_row = 0;
        if (is_ln) {
            per_row = (uint64_t{(pl.nstats * (1 + (2 * pl.gt_max))) + 6 + pl.gt_max + 2} * T32) + (pl.cw_max * T_acc);
        } else {
            per_row = (uint64_t{1 + (2 * pl.gt_max) + 2} * T32) + (pl.mask_tail ? pl.cw_max * T_acc : 0);
        }
        const auto cap =
            static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(geo.R, ROW_CHUNK_L1_BUDGET / per_row)));
        uint32_t row_chunk = 1;
        for (uint32_t d = 1; d <= cap; ++d) {
            if (geo.R % d == 0) {
                row_chunk = d;
            }
        }
        pl.row_chunk = row_chunk;
        pl.rpo = tt::round_up(pl.rpo, row_chunk);
        pl.n_owners = tt::div_up(geo.R, pl.rpo);
    } else {
        pl.row_chunk = pl.crt_max;
    }
    pl.xg_pre = XG_PRE;

    // destination banks: the DRAM banks (DRAM / sharded output, R_U staging) or the L1 banks (L1 interleaved output)
    pl.out_sharded = out_mc.is_sharded();
    if (pl.is_ru || pl.out_sharded || out_mc.buffer_type() == BufferType::DRAM) {
        const CoreCoord dram = mesh.dram_grid_size();
        pl.nb = static_cast<uint32_t>(dram.x * dram.y);
    } else {
        pl.nb = mesh.allocator()->get_num_banks(BufferType::L1);
    }
    TT_FATAL(out_page > 0 && pl.nb > 0, "all_gather_rms_norm: empty output page or no destination bank");
    pl.run_pages = std::max(1u, payload / out_page);
    pl.stats_payload = payload;

    pl.out_in_place = false;
    if (pl.x_sharded && pl.out_sharded && !pl.is_ru && in_mc.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
        out_mc.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED && split.rows_hi == 1) {
        const auto& o_ss = *out_mc.shard_spec();
        const auto o_order =
            ordered_shard_cores(o_ss.grid, o_ss.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        pl.out_in_place = o_ss.shape[0] == kTile && o_order.size() >= pl.shard_order.size() &&
                          std::equal(pl.shard_order.begin(), pl.shard_order.end(), o_order.begin());
    }

    Walk& walk = pl.walk;
    walk.nb = pl.nb;
    walk.run_pages = pl.run_pages;
    walk.scatter_runs = SCATTER_RUNS;
    walk.split_runs = SPLIT_RUNS;
    if (pl.is_ru) {
        walk.rows_eff = 1;
        walk.cols_eff = walk.row_stride = walk.slot_stride = geo.R * geo.Wlt;
        walk.shard_w = 0;
    } else {
        walk.rows_eff = geo.R;
        walk.cols_eff = geo.Wlt;
        walk.row_stride = geo.Wt;
        walk.slot_stride = geo.Wlt;
        if (pl.out_sharded) {
            walk.shard_w = out_mc.shard_spec()->shape[1] / kTile;
        } else if (pl.nb > L1I_BANK_WALK_MAX_NB) {
            walk.shard_w = 1;  // L1 interleaved output: row-major walk, one page per run
        } else {
            walk.shard_w = 0;
        }
    }
    pl.balance = mesh_plan.ring && G % 2 == 0 && G >= 4 && geo.R * geo.Wlt >= BALANCE_MIN_TILES;

    pl.src_max = 1;
    if (pl.is_ru) {
        for (uint32_t oc = 0; oc < geo.Wt; ++oc) {
            std::vector<std::pair<uint32_t, uint32_t>> pages;
            for (uint32_t c = 0; c < kTile; ++c) {
                const uint32_t g = (oc * kTile) + c;
                if (g >= geo.W) {
                    break;
                }
                const uint32_t q = g / geo.W_local;
                const std::pair<uint32_t, uint32_t> pg{q, (g - (q * geo.W_local)) / kTile};
                if (std::find(pages.begin(), pages.end(), pg) == pages.end()) {
                    pages.push_back(pg);
                }
            }
            pl.src_max = std::max(pl.src_max, static_cast<uint32_t>(pages.size()));
        }
    }

    pl.stage_on = L <= STAGE_MAX_LANES && (walk.shard_w > 0 || pl.nb <= STAGE_MAX_NB);
    pl.own_bands = 1;  // OWN_BANDS == 1
    pl.band_need.assign(pl.own_bands, 0);
    pl.band_row_end.assign(pl.own_bands, 0);
    {
        uint32_t row = 0;
        for (uint32_t g = 0; g < split.n_groups; ++g) {
            const uint32_t b = g * pl.own_bands / split.n_groups;
            row += g < split.big ? split.rows_hi : split.rows_hi - 1;
            if (g == split.n_groups - 1) {
                row = geo.R;
            }
            pl.band_need[b] += k;
            pl.band_row_end[b] = row;
        }
    }

    // ---- the static CB region without the own-entry staging (Python _static_cb_bytes(make_cbs(0, 0))) ----
    uint64_t fixed = 0;
    auto add_cb = [&fixed](uint64_t bytes) { fixed += plan_cb_round(bytes); };
    const uint32_t rc = pl.row_chunk;
    const uint32_t blk = pl.crt_max * pl.cw_max;
    if (!pl.x_sharded) {
        add_cb(uint64_t{blk} * T_in);
    }
    add_cb(tt::tt_metal::tile_size(DataType::BFLOAT16));
    add_cb(uint64_t{rc} * pl.nstats * T32);
    add_cb(uint64_t{rc} * pl.nstats * agrn_args::RECORD_BYTES);
    add_cb(uint64_t{rc} * pl.nstats * pl.gt_max * T32);
    add_cb(uint64_t{rc} * pl.nstats * pl.gt_max * T32);
    add_cb(uint64_t{rc} * T_acc);
    if (!pl.out_in_place) {
        add_cb(uint64_t{blk} * out_page);
    }
    add_cb(uint64_t{2} * BATCH_CHUNKS * pl.run_pages * out_page);
    add_cb(uint64_t{2} * BATCH_CHUNKS * agrn_args::PORT_DESC_BYTES);
    add_cb(uint64_t{pl.table_bytes / agrn_args::RECORD_BYTES} * agrn_args::RECORD_BYTES);
    if (has_gamma) {
        add_cb(uint64_t{pl.cw_max} * gamma_page);
        add_cb((uint64_t{pl.cw_max + 2} * 128) + 64);
    }
    if (has_beta) {
        add_cb(uint64_t{pl.cw_max} * gamma_page);
    }
    if (is_ln || pl.mask_tail) {
        const DataType sq_dtype = (is_ln && pl.x_sharded && !x_is_f32) ? DataType::BFLOAT16 : acc_dtype;
        add_cb(uint64_t{pl.mask_tail ? pl.cw_max : 1u} * T_acc);
        add_cb(uint64_t{rc} * pl.cw_max * tt::tt_metal::tile_size(sq_dtype));
    }
    if (is_ln) {
        add_cb(uint64_t{rc} * T32);
        add_cb(uint64_t{rc} * T32);
        add_cb(uint64_t{pl.pre_combine ? pl.gt_pc : pl.gather_tiles} * T32);
        add_cb(uint64_t{rc} * T32);
        add_cb(uint64_t{rc} * T32);
        add_cb(uint64_t{rc} * T_acc);
        add_cb(uint64_t{rc} * pl.gt_max * T32);
        if (pl.pre_combine) {
            add_cb(uint64_t{pl.gather_tiles} * T32);
            add_cb(uint64_t{rc} * T32);
        }
        if (!x_is_f32 && fp32_dest && !pl.x_sharded) {
            add_cb(uint64_t{blk} * T32);
        }
    } else {
        add_cb(uint64_t{rc} * T_acc);
    }
    if (pl.is_ru) {
        add_cb(uint64_t{pl.src_max + 1} * out_page);
    }
    if (pl.pre_combine) {
        add_cb(uint64_t{pl.rpo} * k * pl.nstats * agrn_args::RECORD_BYTES);
        add_cb(uint64_t{geo.R} * pl.nfinal * agrn_args::RECORD_BYTES);
        add_cb(uint64_t{rc} * T32);
    }
    if (uint64_t{l1_base} + fixed > l1_lowest) {
        throw UnsupportedAxisValue(fmt::format(
            "all_gather_rms_norm: the op's circular buffers do not fit next to the L1 buffers: CB region [{}, {}) ({} "
            "B "
            "per core) reaches the lowest L1 buffer at {} (output {} B pages, {}); use a DRAM output or free L1 for "
            "this shape",
            l1_base,
            uint64_t{l1_base} + fixed,
            fixed,
            l1_lowest,
            out_page,
            out_mc));
    }
    pl.fixed_cb_bytes = static_cast<uint32_t>(fixed);
    const int64_t stage_budget = std::min<int64_t>(
        STAGE_L1_BUDGET,
        static_cast<int64_t>(l1_lowest) - static_cast<int64_t>(l1_base) - static_cast<int64_t>(fixed) - CB_ALIGN);
    pl.stage_budget = static_cast<uint32_t>(std::max<int64_t>(0, stage_budget));

    // ---- own-entry staging plans (per chip: the own slot's walk depends on p) ----
    const uint32_t stage_cp = pl.run_pages;
    const uint32_t n_active = split.n_active();
    auto stage_plan = [&](uint32_t p) -> std::optional<StagePlan> {
        if (!pl.stage_on) {
            return std::nullopt;
        }
        StagePlan sp;
        const uint32_t base = p * walk.slot_stride;
        std::vector<std::vector<uint32_t>> lane_pages;
        for (uint32_t l = 0; l < L; ++l) {
            std::vector<uint32_t> pages;
            if (walk.shard_w != 0) {
                for (uint32_t u = l; u < walk.rows_eff; u += L) {
                    pages.push_back(walk.cols_eff);
                }
            } else {
                uint32_t off = 0;
                for (uint32_t b = l; b < pl.nb; b += L) {
                    uint32_t c = 0;
                    for (uint32_t rr = 0; rr < walk.rows_eff; ++rr) {
                        const uint32_t rb = base + (rr * walk.row_stride);
                        const uint32_t j0 =
                            plan_pymod(static_cast<int64_t>(b) - static_cast<int64_t>(rb % pl.nb), pl.nb);
                        c += j0 < walk.cols_eff ? ((walk.cols_eff - 1 - j0) / pl.nb) + 1 : 0;
                    }
                    pages.push_back(c);
                    sp.unit_off.at(b) = off;
                    off += c;
                }
            }
            lane_pages.push_back(std::move(pages));
        }
        if (stage_budget <= 0) {
            return std::nullopt;
        }
        const uint64_t max_slots = static_cast<uint64_t>(stage_budget) / (uint64_t{stage_cp} * out_page);
        if (max_slots == 0) {
            return std::nullopt;
        }
        const uint64_t cap = (max_slots * n_active / L) * stage_cp;
        uint32_t max_tot = 0;
        for (const auto& pages : lane_pages) {
            StageLane lane;
            while (lane.n_st < pages.size() && lane.tot + pages[lane.n_st] <= cap) {
                lane.tot += pages[lane.n_st];
                ++lane.n_st;
            }
            max_tot = std::max(max_tot, lane.tot);
            sp.lanes.push_back(lane);
        }
        if (max_tot == 0) {
            return std::nullopt;
        }
        for (const auto& lane : sp.lanes) {
            sp.nch_max = std::max(sp.nch_max, tt::div_up(lane.tot, stage_cp));
        }
        const uint32_t slots = tt::div_up(L * sp.nch_max, n_active);
        sp.max_pages = slots * stage_cp;
        return sp;
    };
    pl.stage_max_pages = 0;
    for (const auto& [coord, ch] : mesh_plan.chips) {
        auto sp = stage_plan(ch.p);
        if (sp.has_value()) {
            pl.stage_max_pages = std::max(pl.stage_max_pages, sp->max_pages);
        }
        pl.stage_plans.emplace(coord, std::move(sp));
    }

    return cache.emplace(std::move(key), std::move(pl)).first->second;
}

}  // namespace ttnn::operations::experimental::ccl::all_gather_rms_norm::planner
