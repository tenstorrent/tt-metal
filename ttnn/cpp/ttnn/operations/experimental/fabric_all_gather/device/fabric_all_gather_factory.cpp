// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fabric_all_gather program factory.
//
// Per chip, per ring, per ring direction (forward = toward the next chip of the ring) and per link there is one link
// worker core: a reader (NCRISC, NoC0) that reads this chip's own shard from the input and then the shards it relays
// from the output, and a sender (BRISC, NoC1) that sends every chunk one fabric hop into the same pages of the
// neighbour's output. One copy core per link writes this chip's own shard into its own output. Every link worker is
// placed as close as the core grid allows to the Ethernet core of its link.
//
// Rings: an axis line / ring (cluster_axis 0 or 1), a snake over the whole mesh (cluster_axis None), or -- on a torus
// with both sides >= 3 -- two edge-disjoint Hamiltonian cycles over the whole mesh, each owning half of the banks, so
// every chip uses all four neighbours. Even rings are balanced: the shard opposite each chip goes half one way, half
// the other.

#include "fabric_all_gather_factory.hpp"
#include "kernels/fabric_all_gather_walk.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/mesh_ring_plan.hpp"

namespace ttnn::operations::experimental::fabric_all_gather {

namespace CMAKE_UNIQUE_NAMESPACE {

constexpr uint32_t kCbChunks = tt::CBIndex::c_0;
constexpr uint32_t kCbMeta = tt::CBIndex::c_1;        // reader's metadata landing slot
constexpr uint32_t kCbMetaWriter = tt::CBIndex::c_2;  // sender's / copy writer's (the two RISCs read concurrently)
constexpr uint32_t kMetaBytes = 64;
constexpr uint32_t kIncEvery = 8;           // one fused write + increment per this many chunks
constexpr uint32_t kCbBudget = 112 * 1024;  // chunk CB per core: two halves of `group` chunks
constexpr uint32_t kMaxGroup = 8;
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/fabric_all_gather/device/kernels/";

using Coord = ttnn::MeshCoordinate;

struct Entry {
    uint32_t rank = 0;
    uint32_t part = 0;  // 0 whole shard, 1 first half of the link worker's banks, 2 second half
};

// One chip's view of one ring.
struct RingView {
    std::optional<Coord> next;  // forward peer
    std::optional<Coord> prev;  // backward peer
    std::vector<Entry> fwd;     // sent toward next, in the order they arrive downstream; entry 0 is this chip's own
    std::vector<Entry> bwd;
};

struct LinkWorker {
    tt::tt_metal::CoreCoord core;  // logical
    bool connected = false;        // has a fabric connection (sends data or a ready)
    uint32_t link_index = 0;       // fabric link index (routing plane) toward its peer
};

struct Plan {
    uint32_t group_size = 0;
    uint32_t num_rings = 0;
    uint32_t num_links = 0;
    std::vector<uint32_t> rank;                // output rank, by linear mesh index
    std::vector<std::vector<RingView>> rings;  // [linear mesh index][ring]
    // [linear mesh index][(ring * 2 + direction) * num_links + link]
    std::vector<std::vector<LinkWorker>> workers;
    std::vector<std::vector<tt::tt_metal::CoreCoord>> copy_cores;  // [linear mesh index][copy core]
};

uint32_t linear(const Coord& c, uint32_t cols) { return c[0] * cols + c[1]; }

uint32_t worker_slot(uint32_t ring, uint32_t dir, uint32_t link, uint32_t num_links) {
    return (ring * 2 + dir) * num_links + link;
}

// (positions sent forward, positions sent backward), own first, each with its part.
std::pair<std::vector<Entry>, std::vector<Entry>> schedule(uint32_t p, uint32_t G, bool ring) {
    std::vector<Entry> fwd, bwd;
    if (ring && G % 2 == 0 && G >= 4) {  // balanced: the opposite shard goes half each way
        const uint32_t h = G / 2;
        for (uint32_t i = 0; i < h; ++i) {
            fwd.push_back({(p + G - i) % G, i == h - 1 ? 1u : 0u});
            bwd.push_back({(p + i) % G, i == h - 1 ? 2u : 0u});
        }
    } else if (ring) {
        const uint32_t kf = G / 2;
        const uint32_t kb = G - 1 - kf;
        for (uint32_t i = 0; i < kf; ++i) {
            fwd.push_back({(p + G - i) % G, 0});
        }
        for (uint32_t i = 0; kb > 0 && i < kb; ++i) {
            bwd.push_back({(p + i) % G, 0});
        }
    } else {
        if (p < G - 1) {
            for (uint32_t i = 0; i <= p; ++i) {
                fwd.push_back({p - i, 0});
            }
        }
        if (p > 0) {
            for (uint32_t i = 0; i < G - p; ++i) {
                bwd.push_back({p + i, 0});
            }
        }
    }
    return {fwd, bwd};
}

// Two edge-disjoint Hamiltonian cycles covering every edge of an R x C torus (R, C >= 3): randomized Warnsdorff
// search for cycle A, accepted when the complementary edges form one cycle B. Deterministic (fixed seed).
std::pair<std::vector<Coord>, std::vector<Coord>> hamiltonian_decomposition(uint32_t R, uint32_t C) {
    const uint32_t N = R * C;
    auto nbrs = [&](uint32_t n) {
        const uint32_t r = n / C, c = n % C;
        return std::array<uint32_t, 4>{
            ((r + 1) % R) * C + c, ((r + R - 1) % R) * C + c, r * C + (c + 1) % C, r * C + (c + C - 1) % C};
    };
    auto edge = [&](uint32_t a, uint32_t b) { return std::min(a, b) * N + std::max(a, b); };
    std::mt19937 rng(1);
    for (int attempt = 0; attempt < 20000; ++attempt) {
        std::vector<uint32_t> path{0};
        std::vector<bool> used(N, false);
        used[0] = true;
        uint64_t budget = 200000;
        std::function<bool()> dfs = [&]() -> bool {
            if (budget-- == 0) {
                return false;
            }
            if (path.size() == N) {
                const auto nb = nbrs(path.back());
                return std::find(nb.begin(), nb.end(), path[0]) != nb.end();
            }
            std::vector<uint32_t> cand;
            for (uint32_t m : nbrs(path.back())) {
                if (!used[m] && std::find(cand.begin(), cand.end(), m) == cand.end()) {
                    cand.push_back(m);
                }
            }
            std::shuffle(cand.begin(), cand.end(), rng);
            auto degree = [&](uint32_t m) {
                uint32_t d = 0;
                for (uint32_t k : nbrs(m)) {
                    d += used[k] ? 0 : 1;
                }
                return d;
            };
            std::stable_sort(cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) { return degree(a) < degree(b); });
            for (uint32_t m : cand) {
                path.push_back(m);
                used[m] = true;
                if (dfs()) {
                    return true;
                }
                path.pop_back();
                used[m] = false;
            }
            return false;
        };
        if (!dfs()) {
            continue;
        }
        std::set<uint64_t> a_edges;
        for (uint32_t i = 0; i < N; ++i) {
            a_edges.insert(edge(path[i], path[(i + 1) % N]));
        }
        std::vector<std::vector<uint32_t>> comp(N);
        bool ok = true;
        for (uint32_t n = 0; n < N && ok; ++n) {
            for (uint32_t m : nbrs(n)) {
                if (!a_edges.contains(edge(n, m)) && std::find(comp[n].begin(), comp[n].end(), m) == comp[n].end()) {
                    comp[n].push_back(m);
                }
            }
            ok = comp[n].size() == 2;
        }
        if (!ok) {
            continue;
        }
        std::vector<uint32_t> cyc_b{0};
        uint32_t prev = N, cur = 0;
        while (true) {
            const uint32_t nxt = comp[cur][0] != prev ? comp[cur][0] : comp[cur][1];
            if (nxt == 0) {
                break;
            }
            cyc_b.push_back(nxt);
            prev = cur;
            cur = nxt;
            if (cyc_b.size() > N) {
                break;
            }
        }
        if (cyc_b.size() != N) {
            continue;
        }
        auto to_coords = [&](const std::vector<uint32_t>& v) {
            std::vector<Coord> out;
            for (uint32_t n : v) {
                out.push_back(Coord(n / C, n % C));
            }
            return out;
        };
        return {to_coords(path), to_coords(cyc_b)};
    }
    TT_THROW("fabric_all_gather: no Hamiltonian decomposition found for a {}x{} torus", R, C);
}

std::vector<tt::tt_metal::CoreCoord> cores_of(const CoreRangeSet& set) {
    std::vector<tt::tt_metal::CoreCoord> cores;
    for (const auto& range : set.ranges()) {
        for (auto y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (auto x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                cores.emplace_back(x, y);
            }
        }
    }
    std::sort(
        cores.begin(), cores.end(), [](const auto& a, const auto& b) { return a.y != b.y ? a.y < b.y : a.x < b.x; });
    return cores;
}

Plan build_plan(
    const FabricAllGatherParams& args, const Tensor& input_tensor, MeshDevice* mesh_device, const CoreRangeSet& grid) {
    const auto shape = mesh_device->shape();
    const uint32_t R = shape[0], C = shape[1];
    Plan plan;
    plan.rank.assign(R * C, 0);
    plan.rings.assign(R * C, {});
    plan.workers.assign(R * C, {});
    plan.copy_cores.assign(R * C, {});
    auto node = [&](const Coord& c) { return mesh_device->get_fabric_node_id(c); };
    const bool is_2d = tt::tt_fabric::is_2d_fabric_config(args.fabric_config);
    std::array<tt::tt_fabric::Topology, 2> axis_topology{
        tt::tt_fabric::Topology::Linear, tt::tt_fabric::Topology::Linear};
    for (uint32_t axis = 0; axis < 2; ++axis) {
        if (shape[axis] > 1) {
            axis_topology[axis] = ::ttnn::ccl::get_axis_topology(input_tensor, args.fabric_config, axis);
        }
    }

    // Groups (row-major members; output rank = index) and the ring orders over them.
    std::vector<std::vector<Coord>> groups;
    std::vector<std::vector<std::vector<Coord>>> ring_orders;
    bool ring = false;
    if (!args.linearized_mesh_ring) {
        const uint32_t axis = args.cluster_axis;
        const uint32_t other = 1 - axis;
        ring = axis_topology[axis] == tt::tt_fabric::Topology::Ring && shape[axis] > 2;
        for (uint32_t o = 0; o < shape[other]; ++o) {
            std::vector<Coord> members;
            for (uint32_t i = 0; i < shape[axis]; ++i) {
                members.push_back(axis == 0 ? Coord(i, o) : Coord(o, i));
            }
            groups.push_back(members);
            ring_orders.push_back({members});
        }
    } else {
        std::vector<Coord> members;
        for (uint32_t r = 0; r < R; ++r) {
            for (uint32_t c = 0; c < C; ++c) {
                members.emplace_back(r, c);
            }
        }
        groups.push_back(members);
        const bool torus = is_2d && axis_topology[0] == tt::tt_fabric::Topology::Ring &&
                           axis_topology[1] == tt::tt_fabric::Topology::Ring && R >= 3 && C >= 3;
        if (torus) {
            auto [a, b] = hamiltonian_decomposition(R, C);
            ring_orders.push_back({a, b});
            ring = true;
        } else {
            const uint32_t probe_links = args.num_links.value_or(1);
            const auto resolved = ttnn::operations::ccl::common::resolve_mesh_ring_plan(
                input_tensor, std::nullopt, probe_links, axis_topology, true, "fabric_all_gather", true);
            TT_FATAL(resolved.has_value(), "fabric_all_gather: no direct-neighbour snake over the {}x{} mesh", R, C);
            std::vector<Coord> order;
            for (uint32_t t = 0; t < R * C; ++t) {
                order.push_back(
                    ttnn::operations::ccl::common::snake_ring_coordinate(t, shape, resolved->plan.orientation));
            }
            ring_orders.push_back({order});
            ring = resolved->topology == tt::tt_fabric::Topology::Ring;
        }
    }
    plan.group_size = groups[0].size();
    plan.num_rings = ring_orders[0].size();
    const uint32_t G = plan.group_size;

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        for (uint32_t i = 0; i < G; ++i) {
            plan.rank[linear(groups[gi][i], C)] = i;
        }
        for (const auto& cyc : ring_orders[gi]) {
            for (uint32_t p = 0; p < G; ++p) {
                RingView view;
                if (ring || p + 1 < G) {
                    view.next = cyc[(p + 1) % G];
                }
                if (ring || p > 0) {
                    view.prev = cyc[(p + G - 1) % G];
                }
                auto [fwd, bwd] = schedule(p, G, ring);
                for (auto& e : fwd) {
                    e.rank = plan.rank[linear(cyc[e.rank], C)];
                }
                for (auto& e : bwd) {
                    e.rank = plan.rank[linear(cyc[e.rank], C)];
                }
                view.fwd = view.next ? fwd : std::vector<Entry>{};
                view.bwd = view.prev ? bwd : std::vector<Entry>{};
                plan.rings[linear(cyc[p], C)].push_back(view);
            }
        }
    }

    // A worker's peer link worker (the opposite direction on the peer) sends to this chip iff it has entries; then
    // this worker sends it a ready.
    auto peer_of = [](const RingView& v, uint32_t dir) { return dir == 0 ? v.next : v.prev; };
    auto sends_of = [](const RingView& v, uint32_t dir) -> const std::vector<Entry>& {
        return dir == 0 ? v.fwd : v.bwd;
    };
    auto needs_ready = [&](const Coord& c, uint32_t j, uint32_t dir) {
        const auto peer = peer_of(plan.rings[linear(c, C)][j], dir);
        return peer.has_value() && !sends_of(plan.rings[linear(*peer, C)][j], 1 - dir).empty();
    };

    // Links: requested, else the fewest any used hop offers.
    uint32_t num_links = args.num_links.value_or(UINT32_MAX);
    for (uint32_t li = 0; li < R * C; ++li) {
        const Coord c(li / C, li % C);
        for (uint32_t j = 0; j < plan.num_rings; ++j) {
            for (uint32_t dir = 0; dir < 2; ++dir) {
                const auto peer = peer_of(plan.rings[li][j], dir);
                if (!peer) {
                    continue;
                }
                const auto links = tt::tt_fabric::get_forwarding_link_indices(node(c), node(*peer));
                TT_FATAL(
                    !links.empty(),
                    "fabric_all_gather: {} -> {} is not a direct fabric neighbour (fabric config {})",
                    c,
                    *peer,
                    args.fabric_config);
                if (args.num_links.has_value()) {
                    TT_FATAL(
                        links.size() >= *args.num_links,
                        "fabric_all_gather: {} -> {} has {} usable link(s), {} requested",
                        c,
                        *peer,
                        links.size(),
                        *args.num_links);
                } else {
                    num_links = std::min<uint32_t>(num_links, links.size());
                }
            }
        }
    }
    plan.num_links = num_links;
    const uint32_t L = num_links;

    // Placement: every connected link worker as close as the grid allows to its link's Ethernet core, then the
    // receive-only link workers and one copy core per link on free cores.
    const auto allowed = cores_of(grid);
    for (uint32_t li = 0; li < R * C; ++li) {
        const Coord c(li / C, li % C);
        auto* device = mesh_device->get_device(c);
        auto& workers = plan.workers[li];
        workers.assign(plan.num_rings * 2 * L, {});
        std::set<std::pair<size_t, size_t>> taken;
        auto take = [&](const tt::tt_metal::CoreCoord& core) { taken.insert({core.x, core.y}); };
        auto is_free = [&](const tt::tt_metal::CoreCoord& core) { return !taken.contains({core.x, core.y}); };
        for (uint32_t j = 0; j < plan.num_rings; ++j) {
            for (uint32_t dir = 0; dir < 2; ++dir) {
                const auto& view = plan.rings[li][j];
                const auto peer = peer_of(view, dir);
                const bool connected = peer.has_value() && (!sends_of(view, dir).empty() || needs_ready(c, j, dir));
                if (!connected) {
                    continue;
                }
                const auto links = tt::tt_fabric::get_forwarding_link_indices(node(c), node(*peer));
                for (uint32_t l = 0; l < L; ++l) {
                    const auto eth = tt::tt_fabric::get_forwarding_eth_core(node(c), node(*peer), links[l]);
                    const auto closest = tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(
                                             *device, eth, tt::tt_metal::NOC::NOC_1)
                                             .logical_coord;
                    std::optional<tt::tt_metal::CoreCoord> best;
                    uint32_t best_hops = UINT32_MAX;
                    for (const auto& cand : allowed) {
                        if (!is_free(cand)) {
                            continue;
                        }
                        const uint32_t hops = cand == closest
                                                  ? 0
                                                  : tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                                                        device, cand, closest, tt::tt_metal::NOC::NOC_1);
                        if (hops < best_hops) {
                            best_hops = hops;
                            best = cand;
                        }
                    }
                    TT_FATAL(best.has_value(), "fabric_all_gather: the core grid {} has too few cores", grid);
                    take(*best);
                    workers[worker_slot(j, dir, l, L)] = LinkWorker{*best, true, links[l]};
                }
            }
        }
        for (uint32_t s = 0; s < workers.size(); ++s) {
            if (workers[s].connected) {
                continue;
            }
            auto it = std::find_if(allowed.begin(), allowed.end(), is_free);
            TT_FATAL(it != allowed.end(), "fabric_all_gather: the core grid {} has too few cores", grid);
            take(*it);
            workers[s].core = *it;
        }
        // copy cores: from the middle row of the grid downward (away from the Ethernet row), then upward
        std::vector<tt::tt_metal::CoreCoord> order(allowed.begin(), allowed.end());
        const auto mid_y = order[order.size() / 2].y;
        std::stable_partition(order.begin(), order.end(), [&](const auto& core) { return core.y >= mid_y; });
        for (uint32_t l = 0; l < L; ++l) {
            auto it = std::find_if(order.begin(), order.end(), is_free);
            TT_FATAL(it != order.end(), "fabric_all_gather: the core grid {} has too few cores", grid);
            take(*it);
            plan.copy_cores[li].push_back(*it);
        }
    }
    return plan;
}

// A = stripes, B_max = pages per stripe (see kernels/fabric_all_gather_walk.hpp).
FabricAllGatherGeometry derive_geometry(
    const FabricAllGatherParams& args, const FabricAllGatherInputs& tensor_args, bool selected_batch) {
    const auto& input = tensor_args.input_tensor;
    const auto& shape = input.padded_shape();
    const int32_t rank = static_cast<int32_t>(shape.rank());
    const int32_t dim = args.dim;
    const bool tile = input.layout() == Layout::TILE;
    const auto tile_spec = tile ? input.tensor_spec().tile() : tt::tt_metal::Tile();
    TT_FATAL(
        tile || dim < rank - 1,
        "fabric_all_gather: a ROW_MAJOR gather along the innermost dim (partial pages) is not supported");
    auto page_extent = [&](int32_t i) -> uint32_t {
        uint32_t e = (i == 0 && selected_batch) ? 1u : shape[i];
        if (tile && i == rank - 1) {
            return e / tile_spec.get_width();
        }
        if (tile && i == rank - 2) {
            return e / tile_spec.get_height();
        }
        if (!tile && i == rank - 1) {
            return 1;
        }
        return e;
    };
    FabricAllGatherGeometry g;
    g.num_stripes = 1;
    for (int32_t i = 0; i < dim; ++i) {
        g.num_stripes *= page_extent(i);
    }
    g.stripe_pages_max = 1;
    for (int32_t i = dim; i < rank; ++i) {
        g.stripe_pages_max *= page_extent(i);
    }
    const uint32_t num_pages = input.buffer()->num_pages();
    g.pages_per_slot = selected_batch ? num_pages / shape[0] : num_pages;
    TT_FATAL(
        g.num_stripes * g.stripe_pages_max == g.pages_per_slot,
        "fabric_all_gather: page geometry {} x {} does not cover the {} pages of a slot (shape {}, dim {})",
        g.num_stripes,
        g.stripe_pages_max,
        g.pages_per_slot,
        shape,
        dim);
    g.gather_dim_size = shape[dim];
    g.group_size = args.num_devices;
    if (tensor_args.has_gathered_prefix_metadata()) {
        const uint32_t full_global = shape[dim] * args.num_devices;
        TT_FATAL(
            full_global % args.gathered_slab_global == 0,
            "fabric_all_gather: gathered_slab_global {} must divide the full gathered extent {}",
            args.gathered_slab_global,
            full_global);
        const uint32_t slabs = full_global / args.gathered_slab_global;
        TT_FATAL(
            g.stripe_pages_max % slabs == 0,
            "fabric_all_gather: {} pages per stripe do not split into {} slabs",
            g.stripe_pages_max,
            slabs);
        g.pages_per_slab = g.stripe_pages_max / slabs;
    }
    return g;
}

uint32_t host_active_stripe_pages(const FabricAllGatherParams& args, const FabricAllGatherGeometry& g) {
    if (!args.gathered_dim_size.has_value()) {
        return g.stripe_pages_max;
    }
    const uint64_t active_local = *args.gathered_dim_size / args.num_devices;
    const uint64_t pages = active_local * g.stripe_pages_max;
    TT_FATAL(
        pages % g.gather_dim_size == 0,
        "fabric_all_gather: gathered_dim_size {} does not cover whole pages",
        *args.gathered_dim_size);
    return static_cast<uint32_t>(pages / g.gather_dim_size);
}

uint32_t host_slot_base(const FabricAllGatherParams& args, const FabricAllGatherGeometry& g) {
    return args.input_batch_index.value_or(0) * g.pages_per_slot;
}

std::vector<uint32_t> geometry_block(
    const FabricAllGatherParams& args, const FabricAllGatherInputs& tensor_args, const FabricAllGatherGeometry& g) {
    const auto& shape = tensor_args.input_tensor.padded_shape();
    return {
        tensor_args.has_gathered_prefix_metadata() ? tensor_args.gathered_prefix_tensor->buffer()->address() : 0u,
        host_active_stripe_pages(args, g),
        g.num_stripes,
        g.stripe_pages_max,
        g.group_size,
        args.gathered_slab_global,
        shape[args.dim] * args.num_devices,
        g.pages_per_slab};
}

std::vector<uint32_t> reader_common_args(
    const FabricAllGatherParams& args,
    const FabricAllGatherInputs& tensor_args,
    const Tensor& output,
    const FabricAllGatherGeometry& g,
    uint32_t arrival_addr) {
    std::vector<uint32_t> v{
        tensor_args.input_tensor.buffer()->address(),
        output.buffer()->address(),
        arrival_addr,
        tensor_args.has_batch_index_metadata() ? tensor_args.input_batch_index_tensor->buffer()->address() : 0u,
        args.batch_slot_num_layers,
        args.batch_slot_layer_idx,
        host_slot_base(args, g),
        g.pages_per_slot};
    auto geo = geometry_block(args, tensor_args, g);
    v.insert(v.end(), geo.begin(), geo.end());
    return v;
}

std::vector<uint32_t> writer_common_args(
    const FabricAllGatherParams& args,
    const FabricAllGatherInputs& tensor_args,
    const Tensor& output,
    const FabricAllGatherGeometry& g,
    uint32_t arrival_addr,
    uint32_t ready_addr) {
    std::vector<uint32_t> v{output.buffer()->address(), arrival_addr, ready_addr, 0, 0, 0, 0, 0};
    auto geo = geometry_block(args, tensor_args, g);
    v.insert(v.end(), geo.begin(), geo.end());
    return v;
}

CoreRangeSet resolve_available_worker_cores(MeshDevice* mesh_device, const FabricAllGatherParams& args) {
    TT_FATAL(
        mesh_device->get_active_sub_device_manager_id() == args.subdevice_manager_id,
        "fabric_all_gather active subdevice manager changed from {} to {} during operation dispatch",
        args.subdevice_manager_id,
        mesh_device->get_active_sub_device_manager_id());
    const auto subdevice_id = args.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto subdevice_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    TT_FATAL(
        subdevice_cores.contains(args.resolved_worker_core_grid),
        "fabric_all_gather resolved worker grid {} must be fully contained in TENSIX subdevice {} cores {}",
        args.resolved_worker_core_grid,
        subdevice_id,
        subdevice_cores);
    return args.resolved_worker_core_grid;
}

void validate_semaphore_core_coverage(
    const tt::tt_metal::GlobalSemaphore& semaphore, const CoreRangeSet& required_cores, const char* name) {
    const auto attributes = semaphore.attribute_values();
    const auto& cores = std::get<0>(attributes);
    TT_FATAL(
        cores.contains(required_cores),
        "fabric_all_gather {} cores {} must cover all of the op's cores {}",
        name,
        cores,
        required_cores);
}

CoreRangeSet to_range_set(const std::vector<tt::tt_metal::CoreCoord>& cores) {
    std::set<CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.insert(CoreRange(c, c));
    }
    return CoreRangeSet(ranges);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

using namespace CMAKE_UNIQUE_NAMESPACE;

FabricAllGatherFactory::cached_mesh_workload_t FabricAllGatherFactory::create_mesh_workload(
    const FabricAllGatherParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const FabricAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const auto& input = tensor_args.input_tensor;
    auto* mesh_device = input.device();
    const auto shape = mesh_device->shape();
    const uint32_t C = shape[1];
    const auto subdevice_id = args.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = resolve_available_worker_cores(mesh_device, args);

    const Plan plan = build_plan(args, input, mesh_device, available_cores);
    const bool selected_batch = args.input_batch_index.has_value() || tensor_args.has_batch_index_metadata();
    const auto geometry = derive_geometry(args, tensor_args, selected_batch);

    // Semaphores: arrival counter (data_valid) and ready counter (fence), at one address on every chip.
    const bool has_l1_small = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL) > 0;
    const auto sem_buffer_type = has_l1_small ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    const bool external = args.ready_semaphore.has_value();
    auto ready_sem =
        external ? *args.ready_semaphore
                 : ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    auto arrival_sem =
        external ? *args.data_valid_semaphore
                 : ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    if (!external) {
        // every chip's counters must be zero before any chip's first call can increment a neighbour's
        ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);
    }

    const uint32_t page_bytes = input.buffer()->aligned_page_size();
    const uint32_t packet_bytes = static_cast<uint32_t>(args.packet_size);
    const uint32_t run_pages = std::max<uint32_t>(1, packet_bytes / page_bytes);
    const uint32_t chunk_bytes = run_pages * page_bytes;
    const uint32_t group = std::max<uint32_t>(1, std::min<uint32_t>(kMaxGroup, kCbBudget / (2 * chunk_bytes)));
    const uint32_t num_banks = mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const uint32_t L = plan.num_links;
    const uint32_t stride = plan.num_rings * L;
    TT_FATAL(
        stride <= num_banks,
        "fabric_all_gather: {} rings x {} links exceed {} DRAM banks",
        plan.num_rings,
        L,
        num_banks);
    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const bool batch_meta = tensor_args.has_batch_index_metadata();
    const bool prefix_meta = tensor_args.has_gathered_prefix_metadata();
    const bool input_interleaved = input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;

    auto append_accessor = [](std::vector<uint32_t>& ct, const Tensor& t) {
        tt::tt_metal::TensorAccessorArgs(*t.buffer()).append_to(ct);
    };
    auto node = [&](const Coord& c) { return mesh_device->get_fabric_node_id(c); };

    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        const uint32_t li = linear(coord, C);
        tt::tt_metal::Program program{};
        const auto& rings = plan.rings[li];
        const auto& workers = plan.workers[li];
        const auto& copy = plan.copy_cores[li];

        std::vector<tt::tt_metal::CoreCoord> port_cores;
        for (const auto& w : workers) {
            port_cores.push_back(w.core);
        }
        std::vector<tt::tt_metal::CoreCoord> all_cores = port_cores;
        all_cores.insert(all_cores.end(), copy.begin(), copy.end());
        const auto port_set = to_range_set(port_cores);
        const auto copy_set = to_range_set(copy);
        const auto all_set = to_range_set(all_cores);

        tt::tt_metal::CreateCircularBuffer(
            program,
            all_set,
            tt::tt_metal::CircularBufferConfig(2 * group * chunk_bytes, {{kCbChunks, data_format}})
                .set_page_size(kCbChunks, page_bytes));
        for (const uint32_t meta_cb : {kCbMeta, kCbMetaWriter}) {
            tt::tt_metal::CreateCircularBuffer(
                program,
                all_set,
                tt::tt_metal::CircularBufferConfig(kMetaBytes, {{meta_cb, tt::DataFormat::UInt32}})
                    .set_page_size(meta_cb, kMetaBytes));
        }

        const Tensor& batch_t = batch_meta ? *tensor_args.input_batch_index_tensor : output_tensor;
        const Tensor& prefix_t = prefix_meta ? *tensor_args.gathered_prefix_tensor : output_tensor;
        std::vector<uint32_t> base_ct{kCbChunks, kCbMeta, page_bytes, run_pages, num_banks, group, kIncEvery};
        std::vector<uint32_t> reader_ct = base_ct;
        reader_ct.push_back(batch_meta ? 1u : 0u);
        reader_ct.push_back(prefix_meta ? 1u : 0u);
        reader_ct.push_back(input_interleaved ? 1u : 0u);
        append_accessor(reader_ct, input);
        append_accessor(reader_ct, output_tensor);
        append_accessor(reader_ct, batch_t);
        append_accessor(reader_ct, prefix_t);
        std::vector<uint32_t> writer_ct = base_ct;
        writer_ct[1] = kCbMetaWriter;
        writer_ct.push_back(prefix_meta ? 1u : 0u);
        append_accessor(writer_ct, output_tensor);
        append_accessor(writer_ct, prefix_t);

        auto reader_cfg = [&](const std::vector<uint32_t>& ct) {
            return tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = ct};
        };
        auto writer_cfg = [&](const std::vector<uint32_t>& ct) {
            return tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = ct};
        };
        const std::string dir(kKernelDir);
        const auto reader_id =
            tt::tt_metal::CreateKernel(program, dir + "fabric_all_gather_reader.cpp", port_set, reader_cfg(reader_ct));
        const auto sender_id =
            tt::tt_metal::CreateKernel(program, dir + "fabric_all_gather_sender.cpp", port_set, writer_cfg(writer_ct));
        const auto copy_reader_id =
            tt::tt_metal::CreateKernel(program, dir + "fabric_all_gather_reader.cpp", copy_set, reader_cfg(reader_ct));
        const auto copy_writer_id = tt::tt_metal::CreateKernel(
            program, dir + "fabric_all_gather_copy_writer.cpp", copy_set, writer_cfg(writer_ct));

        const auto reader_common =
            reader_common_args(args, tensor_args, output_tensor, geometry, arrival_sem.address());
        const auto writer_common =
            writer_common_args(args, tensor_args, output_tensor, geometry, arrival_sem.address(), ready_sem.address());
        tt::tt_metal::SetCommonRuntimeArgs(program, reader_id, reader_common);
        tt::tt_metal::SetCommonRuntimeArgs(program, sender_id, writer_common);
        tt::tt_metal::SetCommonRuntimeArgs(program, copy_reader_id, reader_common);
        tt::tt_metal::SetCommonRuntimeArgs(program, copy_writer_id, writer_common);

        auto pack = [](const std::vector<Entry>& entries) {
            std::vector<uint32_t> v;
            for (const auto& e : entries) {
                v.push_back(e.rank | (e.part << 16));
            }
            return v;
        };
        for (uint32_t j = 0; j < plan.num_rings; ++j) {
            for (uint32_t d = 0; d < 2; ++d) {
                const auto& view = rings[j];
                const auto peer = d == 0 ? view.next : view.prev;
                const auto& sends = d == 0 ? view.fwd : view.bwd;
                const auto upstream = d == 0 ? view.prev : view.next;
                std::array<uint32_t, 3> up_parts{0, 0, 0};
                if (upstream) {
                    const auto& up_view = plan.rings[linear(*upstream, C)][j];
                    for (const auto& e : (d == 0 ? up_view.fwd : up_view.bwd)) {
                        up_parts[e.part]++;
                    }
                }
                const bool send_ready =
                    peer.has_value() &&
                    !(d == 0 ? plan.rings[linear(*peer, C)][j].bwd : plan.rings[linear(*peer, C)][j].fwd).empty();
                for (uint32_t l = 0; l < L; ++l) {
                    const auto& w = workers[worker_slot(j, d, l, L)];
                    const uint32_t first = j * L + l;
                    auto reader_args = std::vector<uint32_t>{first, stride, 0, static_cast<uint32_t>(sends.size())};
                    const auto packed = pack(sends);
                    reader_args.insert(reader_args.end(), packed.begin(), packed.end());
                    tt::tt_metal::SetRuntimeArgs(program, reader_id, w.core, reader_args);

                    std::vector<uint32_t> sender_args{first, stride, 0, send_ready ? 1u : 0u};
                    if (peer) {
                        auto* peer_device = mesh_device->get_device(*peer);
                        const auto& peer_workers = plan.workers[linear(*peer, C)];
                        const auto ready_core =
                            peer_device->worker_core_from_logical_core(peer_workers[worker_slot(j, 1 - d, l, L)].core);
                        const auto data_core =
                            peer_device->worker_core_from_logical_core(peer_workers[worker_slot(j, d, l, L)].core);
                        const auto pn = node(*peer);
                        sender_args.insert(
                            sender_args.end(),
                            {static_cast<uint32_t>(ready_core.x),
                             static_cast<uint32_t>(ready_core.y),
                             static_cast<uint32_t>(data_core.x),
                             static_cast<uint32_t>(data_core.y),
                             static_cast<uint32_t>(*pn.mesh_id),
                             pn.chip_id});
                    } else {
                        sender_args.insert(sender_args.end(), {0, 0, 0, 0, 0, 0});
                    }
                    sender_args.insert(sender_args.end(), {up_parts[0], up_parts[1], up_parts[2]});
                    sender_args.push_back(static_cast<uint32_t>(sends.size()));
                    sender_args.insert(sender_args.end(), packed.begin(), packed.end());
                    if (w.connected) {
                        tt::tt_fabric::append_fabric_connection_rt_args(
                            node(coord), node(*peer), w.link_index, program, w.core, sender_args);
                    }
                    tt::tt_metal::SetRuntimeArgs(program, sender_id, w.core, sender_args);
                }
            }
        }
        const uint32_t my_rank = plan.rank[li];
        const uint32_t J = copy.size();
        for (uint32_t jj = 0; jj < J; ++jj) {
            tt::tt_metal::SetRuntimeArgs(
                program, copy_reader_id, copy[jj], std::vector<uint32_t>{jj, J, 0, 1, my_rank});
            tt::tt_metal::SetRuntimeArgs(program, copy_writer_id, copy[jj], std::vector<uint32_t>{jj, J, my_rank});
        }

        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(program));
        shared_variables.emplace(
            ttnn::MeshCoordinateRange(coord),
            shared_variables_t{
                .reader_kernel_id = reader_id,
                .sender_kernel_id = sender_id,
                .copy_reader_kernel_id = copy_reader_id,
                .copy_writer_kernel_id = copy_writer_id,
                .has_ports = true,
                .has_copy = J > 0,
                .ready_sem = ready_sem,
                .arrival_sem = arrival_sem,
                .worker_core_range = all_set,
                .geometry = geometry});
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void FabricAllGatherFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const FabricAllGatherParams& args,
    const FabricAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared = cached_workload.shared_variables.at(coordinate_range);
        if (args.ready_semaphore.has_value() &&
            (shared.ready_sem.address() != args.ready_semaphore->address() ||
             shared.arrival_sem.address() != args.data_valid_semaphore->address())) {
            validate_semaphore_core_coverage(*args.ready_semaphore, shared.worker_core_range, "ready_semaphore");
            validate_semaphore_core_coverage(
                *args.data_valid_semaphore, shared.worker_core_range, "data_valid_semaphore");
            shared.ready_sem = *args.ready_semaphore;
            shared.arrival_sem = *args.data_valid_semaphore;
        }
        const auto reader_common =
            reader_common_args(args, tensor_args, output_tensor, shared.geometry, shared.arrival_sem.address());
        const auto writer_common = writer_common_args(
            args,
            tensor_args,
            output_tensor,
            shared.geometry,
            shared.arrival_sem.address(),
            shared.ready_sem.address());
        auto patch = [&](tt::tt_metal::KernelHandle id, const std::vector<uint32_t>& values) {
            auto& common = GetCommonRuntimeArgs(program, id);
            for (size_t i = 0; i < values.size(); ++i) {
                common[i] = values[i];
            }
        };
        patch(shared.reader_kernel_id, reader_common);
        patch(shared.sender_kernel_id, writer_common);
        if (shared.has_copy) {
            patch(shared.copy_reader_kernel_id, reader_common);
            patch(shared.copy_writer_kernel_id, writer_common);
        }
    }
}

}  // namespace ttnn::operations::experimental::fabric_all_gather
