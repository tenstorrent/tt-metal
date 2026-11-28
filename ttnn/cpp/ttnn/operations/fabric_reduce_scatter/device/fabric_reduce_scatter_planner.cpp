// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fabric_reduce_scatter_planner.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

#include <fmt/format.h>

#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/indestructible.hpp>

namespace ttnn::operations::fabric_reduce_scatter::plan {

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;

// Python random.Random(1)'s search cannot be reproduced bit for bit; any decomposition the search accepts is valid.
constexpr uint32_t kHamiltonianSeed = 1;
constexpr uint32_t kHamiltonianTries = 20000;

struct Hamiltonian {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<std::vector<uint32_t>> nbrs;
    std::vector<uint32_t> path;
    std::vector<bool> used;
    std::mt19937 rng{kHamiltonianSeed};

    uint32_t free_degree(uint32_t n) const {
        uint32_t k = 0;
        for (uint32_t m : nbrs[n]) {
            k += used[m] ? 0 : 1;
        }
        return k;
    }

    bool dfs() {
        const auto total = static_cast<uint32_t>(nbrs.size());
        if (path.size() == total) {
            const auto& last = nbrs[path.back()];
            return std::find(last.begin(), last.end(), path.front()) != last.end();
        }
        std::vector<uint32_t> cand;
        for (uint32_t n : nbrs[path.back()]) {
            if (!used[n]) {
                cand.push_back(n);
            }
        }
        std::shuffle(cand.begin(), cand.end(), rng);
        std::stable_sort(
            cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) { return free_degree(a) < free_degree(b); });
        for (uint32_t n : cand) {
            path.push_back(n);
            used[n] = true;
            if (dfs()) {
                return true;
            }
            path.pop_back();
            used[n] = false;
        }
        return false;
    }
};

// Python hamiltonian_decomposition: two edge-disjoint Hamiltonian cycles of an R x C torus (row-major node ids).
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> hamiltonian_decomposition(uint32_t rows, uint32_t cols) {
    Hamiltonian h;
    h.rows = rows;
    h.cols = cols;
    const uint32_t total = rows * cols;
    h.nbrs.resize(total);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            h.nbrs[(r * cols) + c] = {
                (((r + 1) % rows) * cols) + c,
                (((r + rows - 1) % rows) * cols) + c,
                (r * cols) + ((c + 1) % cols),
                (r * cols) + ((c + cols - 1) % cols)};
        }
    }
    for (uint32_t t = 0; t < kHamiltonianTries; ++t) {
        h.path = {0};
        h.used.assign(total, false);
        h.used[0] = true;
        if (!h.dfs()) {
            continue;
        }
        std::set<std::pair<uint32_t, uint32_t>> a_edges;
        for (uint32_t i = 0; i < total; ++i) {
            const uint32_t u = h.path[i];
            const uint32_t v = h.path[(i + 1) % total];
            a_edges.emplace(std::min(u, v), std::max(u, v));
        }
        std::vector<std::vector<uint32_t>> comp(total);
        bool ok = true;
        for (uint32_t n = 0; n < total && ok; ++n) {
            for (uint32_t m : h.nbrs[n]) {
                if (!a_edges.contains({std::min(n, m), std::max(n, m)})) {
                    comp[n].push_back(m);
                }
            }
            ok = comp[n].size() == 2;
        }
        if (!ok) {
            continue;
        }
        std::vector<uint32_t> cyc_b{0};
        std::optional<uint32_t> prev;
        uint32_t cur = 0;
        while (true) {
            const uint32_t x = comp[cur][0];
            const uint32_t y = comp[cur][1];
            const uint32_t nxt = (!prev.has_value() || x != *prev) ? x : y;
            if (nxt == 0) {
                break;
            }
            cyc_b.push_back(nxt);
            prev = cur;
            cur = nxt;
            if (cyc_b.size() > total) {
                break;
            }
        }
        if (cyc_b.size() == total) {
            return {h.path, cyc_b};
        }
    }
    throw std::invalid_argument(fmt::format("no Hamiltonian decomposition found for a {}x{} torus", rows, cols));
}

using Taken = std::set<std::pair<uint32_t, uint32_t>>;  // (x, y)

bool is_taken(const Taken& taken, const CoreCoord& c) {
    return taken.contains({static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)});
}
void take(Taken& taken, const CoreCoord& c) { taken.emplace(static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)); }

// Python _worker_nearest_noc1: the free worker with the fewest NoC1 hops to the Ethernet core; ties to the lower
// row, then the lower column.
std::optional<CoreCoord> worker_nearest_noc1(
    tt::tt_metal::IDevice* device, const CoreCoord& eth, const CoreCoord& grid, const Taken& taken) {
    const auto closest =
        tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(*device, eth, tt::tt_metal::NOC::NOC_1);
    std::optional<std::tuple<uint32_t, size_t, size_t>> best;
    std::optional<CoreCoord> core;
    for (size_t x = 0; x < grid.x; ++x) {
        for (size_t y = 0; y < grid.y; ++y) {
            const CoreCoord c(x, y);
            if (is_taken(taken, c)) {
                continue;
            }
            const uint32_t hops =
                closest.distance_in_noc_hops + tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                                                   device, c, closest.logical_coord, tt::tt_metal::NOC::NOC_1);
            const auto score = std::make_tuple(hops, y, x);
            if (!best.has_value() || score < *best) {
                best = score;
                core = c;
            }
        }
    }
    return core;
}

// Python _worker_below(noc_x=0): the free worker in the lowest NoC column, then the lowest row.
std::optional<CoreCoord> worker_below(const CoreCoord& grid, const Taken& taken) {
    for (size_t x = 0; x < grid.x; ++x) {
        for (size_t y = 0; y < grid.y; ++y) {
            const CoreCoord c(x, y);
            if (!is_taken(taken, c)) {
                return c;
            }
        }
    }
    return std::nullopt;
}

uint32_t walk(const Plan& plan, uint32_t chip, uint32_t ring, int steps) {
    for (int i = 0; i < std::abs(steps); ++i) {
        const auto& rg = plan.chips[chip].rings[ring];
        const auto next = steps > 0 ? rg.next : rg.prev;
        TT_FATAL(next.has_value(), "fabric_reduce_scatter: walked off the end of a line");
        chip = *next;
    }
    return chip;
}

std::unique_ptr<Plan> solve(MeshDevice& mesh_device, const PlanInputs& in) {
    const auto shape = mesh_device.shape();
    const auto rows = static_cast<uint32_t>(shape[0]);
    const auto cols = static_cast<uint32_t>(shape[1]);
    const uint32_t n_chips = rows * cols;
    auto plan = std::make_unique<Plan>();
    plan->num_links = in.num_links;
    plan->chips.resize(n_chips);

    std::vector<std::vector<uint32_t>> groups;
    std::vector<std::vector<std::vector<uint32_t>>> ring_lists;
    if (in.scheme == Scheme::DUAL_CYCLES) {
        if (in.cluster_axis.has_value() || in.topology != tt::tt_fabric::Topology::Ring || rows < 3 || cols < 3) {
            throw std::invalid_argument(
                "fabric_all_gather: dual_cycles needs cluster_axis=None, Ring, and a torus with both sides >= 3");
        }
        std::vector<uint32_t> all(n_chips);
        for (uint32_t i = 0; i < n_chips; ++i) {
            all[i] = i;
        }
        groups.push_back(all);
        auto [a, b] = hamiltonian_decomposition(rows, cols);
        ring_lists.push_back({a, b});
        plan->ring = true;
    } else {
        if (in.cluster_axis == 0u) {
            for (uint32_t c = 0; c < cols; ++c) {
                std::vector<uint32_t> g;
                g.reserve(rows);
                for (uint32_t r = 0; r < rows; ++r) {
                    g.push_back((r * cols) + c);
                }
                groups.push_back(g);
            }
        } else if (in.cluster_axis == 1u) {
            for (uint32_t r = 0; r < rows; ++r) {
                std::vector<uint32_t> g;
                g.reserve(cols);
                for (uint32_t c = 0; c < cols; ++c) {
                    g.push_back((r * cols) + c);
                }
                groups.push_back(g);
            }
        } else {
            std::vector<uint32_t> snake;
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t k = 0; k < cols; ++k) {
                    const uint32_t c = r % 2 == 0 ? k : cols - 1 - k;
                    snake.push_back((r * cols) + c);
                }
            }
            groups.push_back(snake);
        }
        for (const auto& g : groups) {
            ring_lists.push_back({g});
        }
        plan->ring = in.topology == tt::tt_fabric::Topology::Ring && groups[0].size() > 2;
    }
    const auto G = static_cast<uint32_t>(groups[0].size());
    TT_FATAL(G >= 2, "a group needs at least two chips");
    plan->group_size = G;
    plan->n_rings = static_cast<uint32_t>(ring_lists[0].size());
    const bool ring = plan->ring;

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        auto sorted = groups[gi];
        std::sort(sorted.begin(), sorted.end());
        for (uint32_t i = 0; i < G; ++i) {
            plan->chips[sorted[i]].out = i;
        }
        for (const auto& cyc : ring_lists[gi]) {
            for (uint32_t p = 0; p < G; ++p) {
                RingPosition rp;
                rp.p = p;
                if (ring || p < G - 1) {
                    rp.next = cyc[(p + 1) % G];
                }
                if (ring || p > 0) {
                    rp.prev = cyc[(p + G - 1) % G];
                }
                // fabric_all_gather_py._schedule (balance off): whether each direction sends anything
                if (ring) {
                    rp.ag_sends[FWD] = G / 2 > 0;
                    rp.ag_sends[BWD] = G - 1 - (G / 2) > 0;
                } else {
                    rp.ag_sends[FWD] = p < G - 1;
                    rp.ag_sends[BWD] = p > 0;
                }
                plan->chips[cyc[p]].rings.push_back(std::move(rp));
            }
        }
    }

    for (uint32_t i = 0; i < n_chips; ++i) {
        const auto node = mesh_device.get_fabric_node_id(MeshCoordinate(i / cols, i % cols));
        plan->fabric_mesh_ids.push_back(*node.mesh_id);
        plan->fabric_chip_ids.push_back(node.chip_id);
    }
    auto node = [&](uint32_t chip) { return mesh_device.get_fabric_node_id(MeshCoordinate(chip / cols, chip % cols)); };
    auto coord_str = [&](uint32_t chip) { return fmt::format("({}, {})", chip / cols, chip % cols); };

    // links for every neighbour a chip sends to; every hop must be direct
    for (uint32_t chip = 0; chip < n_chips; ++chip) {
        for (uint32_t j = 0; j < plan->chips[chip].rings.size(); ++j) {
            for (uint32_t d : {FWD, BWD}) {
                auto& rg = plan->chips[chip].rings[j];
                const auto peer = d == FWD ? rg.next : rg.prev;
                if (!peer.has_value() || !(rg.ag_sends[d] || needs_ready(*plan, chip, j, static_cast<Direction>(d)))) {
                    continue;
                }
                if (tt::tt_fabric::get_neighbor_eth_directions(mesh_device, node(chip), node(*peer)).empty()) {
                    throw std::invalid_argument(fmt::format(
                        "fabric_all_gather: {} -> {} is not a direct link", coord_str(chip), coord_str(*peer)));
                }
                auto links = tt::tt_fabric::get_forwarding_link_indices(mesh_device, node(chip), node(*peer));
                if (links.size() < in.num_links) {
                    throw std::invalid_argument(fmt::format(
                        "fabric_all_gather: {} -> {} has {} usable link(s), {} requested",
                        coord_str(chip),
                        coord_str(*peer),
                        links.size(),
                        in.num_links));
                }
                links.resize(in.num_links);
                rg.links[d] = std::move(links);
            }
        }
    }

    const CoreCoord grid = mesh_device.compute_with_storage_grid_size();
    Placement placement = in.placement;
    if (placement == Placement::AUTO && std::getenv("TT_METAL_EMULE_MODE") != nullptr) {
        placement = Placement::SIMPLE;  // the emulator runs no Ethernet cores and has no NoC timing
    }
    for (uint32_t chip = 0; chip < n_chips; ++chip) {
        auto& ch = plan->chips[chip];
        Taken taken;
        std::vector<std::optional<CoreCoord>> ports;
        auto* device = mesh_device.get_device(chip / cols, chip % cols);
        for (const auto& rg : ch.rings) {
            for (uint32_t d : {FWD, BWD}) {
                const auto peer = d == FWD ? rg.next : rg.prev;
                for (uint32_t l = 0; l < in.num_links; ++l) {
                    std::optional<CoreCoord> core;
                    if (placement == Placement::AUTO && !rg.links[d].empty()) {
                        const CoreCoord eth = tt::tt_fabric::get_forwarding_eth_core(
                            mesh_device, node(chip), node(*peer), rg.links[d][l]);
                        core = worker_nearest_noc1(device, eth, grid, taken);
                    }
                    if (core.has_value()) {
                        take(taken, *core);
                    }
                    ports.push_back(core);
                }
            }
        }
        for (auto& core : ports) {  // receive-only ports, or simple placement: any free core
            if (!core.has_value()) {
                core = worker_below(grid, taken);
                if (core.has_value()) {
                    take(taken, *core);
                }
            }
        }
        // copy cores: one per link, from the middle row down, then up
        const auto mid = static_cast<uint32_t>(grid.y / 2);
        std::vector<uint32_t> row_order;
        for (uint32_t y = mid; y < grid.y; ++y) {
            row_order.push_back(y);
        }
        for (uint32_t y = 0; y < mid; ++y) {
            row_order.push_back(y);
        }
        std::vector<CoreCoord> copy;
        for (uint32_t l = 0; l < in.num_links; ++l) {
            for (uint32_t y : row_order) {
                bool found = false;
                for (size_t x = 0; x < grid.x; ++x) {
                    const CoreCoord c(x, y);
                    if (!is_taken(taken, c)) {
                        copy.push_back(c);
                        take(taken, c);
                        found = true;
                        break;
                    }
                }
                if (found) {
                    break;
                }
            }
        }
        const bool missing_port = std::any_of(ports.begin(), ports.end(), [](const auto& c) { return !c.has_value(); });
        if (copy.size() < in.num_links || missing_port) {
            throw std::invalid_argument(fmt::format(
                "fabric_all_gather: the core grid has too few cores for {} ports + copy cores", ports.size()));
        }
        for (const auto& c : ports) {
            ch.ports.push_back(*c);
        }
        ch.copy = copy;

        // _with_final_cores: two final cores per lane
        auto free_in_row = [&](uint32_t row) {
            for (size_t x = 0; x < grid.x; ++x) {
                const CoreCoord c(x, row);
                if (!is_taken(taken, c)) {
                    take(taken, c);
                    return c;
                }
            }
            for (size_t y = 0; y < grid.y; ++y) {
                for (size_t x = 0; x < grid.x; ++x) {
                    const CoreCoord c(x, y);
                    if (!is_taken(taken, c)) {
                        take(taken, c);
                        return c;
                    }
                }
            }
            throw std::invalid_argument(
                "fabric_reduce_scatter: the core grid has too few cores for two final cores per lane");
        };
        for (uint32_t j = 0; j < plan->n_rings; ++j) {
            for (uint32_t l = 0; l < in.num_links; ++l) {
                const auto row = static_cast<uint32_t>(ch.copy[l].y);
                ch.finals.push_back(j == 0 ? ch.copy[l] : free_in_row(row));
                ch.finals.push_back(free_in_row(row));
            }
        }
    }
    return plan;
}

}  // namespace

bool needs_ready(const Plan& plan, uint32_t chip, uint32_t ring, Direction dir) {
    const auto& rg = plan.chips[chip].rings[ring];
    const auto peer = dir == FWD ? rg.next : rg.prev;
    const uint32_t opp = dir == FWD ? BWD : FWD;
    return peer.has_value() && plan.chips[*peer].rings[ring].ag_sends[opp];
}

PortSchedule rs_schedule(const Plan& plan, uint32_t chip, uint32_t ring, Direction dir) {
    const auto& rg = plan.chips[chip].rings[ring];
    const uint32_t G = plan.group_size;
    const uint32_t p = rg.p;
    const auto peer = dir == FWD ? rg.next : rg.prev;
    PortSchedule out;
    if (!peer.has_value()) {
        return out;
    }
    std::vector<int> steps;
    uint32_t lead = 0;
    if (plan.ring) {
        const uint32_t a = G / 2;
        const auto n = static_cast<int>(dir == FWD ? a : G - 1 - a);
        if (dir == FWD) {
            for (int k = n; k > 0; --k) {
                steps.push_back(k);
            }
        } else {
            for (int k = -n; k < 0; ++k) {
                steps.push_back(k);
            }
        }
        lead = 1;
    } else {
        if (dir == FWD) {
            for (int k = static_cast<int>(G - 1 - p); k > 0; --k) {
                steps.push_back(k);
            }
        } else {
            for (int k = -static_cast<int>(p); k < 0; ++k) {
                steps.push_back(k);
            }
        }
        const auto upstream = dir == FWD ? rg.prev : rg.next;
        lead = upstream.has_value() ? 0 : static_cast<uint32_t>(steps.size());
    }
    for (int k : steps) {
        const uint32_t blk = plan.chips[walk(plan, chip, ring, k)].out;
        out.blocks.emplace_back(blk, (dir == BWD && k == -1) ? G : blk);
    }
    out.lead = std::min(lead, static_cast<uint32_t>(out.blocks.size()));
    return out;
}

uint32_t chunks_per_shard(
    uint32_t shard_pages, uint32_t first_bank, uint32_t stride, uint32_t num_banks, uint32_t run_pages, uint32_t part) {
    TT_FATAL(stride > 0 && run_pages > 0 && num_banks > 0, "fabric_reduce_scatter: zero stride, run or bank count");
    std::vector<uint32_t> banks;
    for (uint32_t b = first_bank; b < num_banks; b += stride) {
        banks.push_back(b);
    }
    const auto nb = static_cast<uint32_t>(banks.size());
    uint32_t total = 0;
    for (uint32_t slot = 0; slot < nb; ++slot) {
        if ((part == 1 && slot >= nb / 2) || (part == 2 && slot < nb / 2)) {
            continue;
        }
        const uint32_t b = banks[slot];
        const uint32_t per_bank = b < shard_pages ? (shard_pages - b + num_banks - 1) / num_banks : 0;
        total += (per_bank + run_pages - 1) / run_pages;
    }
    return total;
}

const Plan& get_plan(MeshDevice& mesh_device, const PlanInputs& inputs) {
    using Key = std::tuple<
        uint32_t,
        uint32_t,
        uint32_t,
        std::vector<std::pair<uint32_t, uint32_t>>,
        std::optional<uint32_t>,
        tt::tt_fabric::Topology,
        uint32_t,
        Scheme,
        Placement>;
    static ttsl::Indestructible<std::map<Key, std::unique_ptr<Plan>>> cache;
    static std::mutex mutex;
    const auto shape = mesh_device.shape();
    const auto rows = static_cast<uint32_t>(shape[0]);
    const auto cols = static_cast<uint32_t>(shape[1]);
    std::vector<std::pair<uint32_t, uint32_t>> nodes;
    nodes.reserve(rows * cols);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const auto n = mesh_device.get_fabric_node_id(MeshCoordinate(r, c));
            nodes.emplace_back(*n.mesh_id, n.chip_id);
        }
    }
    Key key{
        static_cast<uint32_t>(mesh_device.id()),
        rows,
        cols,
        std::move(nodes),
        inputs.cluster_axis,
        inputs.topology,
        inputs.num_links,
        inputs.scheme,
        inputs.placement};
    const std::lock_guard<std::mutex> lock(mutex);
    auto& map = cache.get();
    auto it = map.find(key);
    if (it == map.end()) {
        it = map.emplace(std::move(key), solve(mesh_device, inputs)).first;
    }
    return *it->second;
}

}  // namespace ttnn::operations::fabric_reduce_scatter::plan
