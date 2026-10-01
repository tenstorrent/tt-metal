// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fabric_all_gather program factory (terms: kernels/fabric_all_gather_chunk_walk.hpp).
//
// build_gather_plan decides the rings, every link worker's send list and every core's placement; the kernels only
// walk what they are given. Rings: an axis ring or line (cluster_axis 0 / 1), a snake over the whole mesh
// (cluster_axis None), or on a torus with both sides >= 3 two edge-disjoint Hamiltonian cycles that each own half of
// the banks, so every chip uses all four neighbours.

#include "fabric_all_gather_factory.hpp"
#include "kernels/fabric_all_gather_chunk_walk.hpp"

#include <algorithm>
#include <array>
#include <functional>
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

namespace chunk_walk = ::ttnn::operations::experimental::fabric_all_gather::chunk_walk;

constexpr uint32_t kChunkCbIndex = tt::CBIndex::c_0;
constexpr uint32_t kReaderMetadataCbIndex = tt::CBIndex::c_1;  // the reader's metadata landing slot
constexpr uint32_t kWriterMetadataCbIndex =
    tt::CBIndex::c_2;  // the sender's / copy writer's (the RISCs read concurrently)
constexpr uint32_t kMetadataCbBytes = 64;
constexpr uint32_t kRunCbIndex = tt::CBIndex::c_3;  // copy cores, non-interleaved input: reader -> writer run list
constexpr uint32_t kRunCbPageBytes = 16;
constexpr uint32_t kChunkCbBudgetBytes = 112 * 1024;  // chunk CB per core: two batches
constexpr uint32_t kMaxChunksPerBatch = 8;
constexpr uint32_t kForward = 0;
constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/fabric_all_gather/device/kernels/";

using Coord = ttnn::MeshCoordinate;

// One chip's place on one ring.
struct ChipOnRing {
    std::optional<Coord> next_chip;      // forward neighbour
    std::optional<Coord> previous_chip;  // backward neighbour
    // Send lists (packed chunk_walk send entries) of this chip's forward and backward link workers, in the order the
    // shards arrive downstream; entry 0 is this chip's own shard.
    std::vector<uint32_t> forward_send_list;
    std::vector<uint32_t> backward_send_list;

    const std::optional<Coord>& neighbour(uint32_t direction) const {
        return direction == kForward ? next_chip : previous_chip;
    }
    const std::optional<Coord>& upstream(uint32_t direction) const {
        return direction == kForward ? previous_chip : next_chip;
    }
    const std::vector<uint32_t>& send_list(uint32_t direction) const {
        return direction == kForward ? forward_send_list : backward_send_list;
    }
};

struct LinkWorker {
    tt::tt_metal::CoreCoord core;        // logical
    bool has_fabric_connection = false;  // sends data or a ready signal
    uint32_t fabric_link_index = 0;      // routing plane toward its downstream
};

struct GatherPlan {
    uint32_t num_ranks = 0;
    uint32_t num_rings = 0;
    uint32_t num_links = 0;
    uint32_t mesh_cols = 0;
    std::vector<uint32_t> rank_of_chip;                            // [linear chip index]
    std::vector<std::vector<ChipOnRing>> chip_on_ring;             // [linear chip index][ring]
    std::vector<std::vector<LinkWorker>> link_workers;             // [linear chip index][link_worker_index(...)]
    std::vector<std::vector<tt::tt_metal::CoreCoord>> copy_cores;  // [linear chip index][link]
};

uint32_t linear_chip_index(const Coord& chip, uint32_t mesh_cols) { return chip[0] * mesh_cols + chip[1]; }

uint32_t link_worker_index(uint32_t ring, uint32_t direction, uint32_t link, uint32_t num_links) {
    return (ring * 2 + direction) * num_links + link;
}

// Send lists of the chip at `position` of a ring of `num_ranks` chips, as (forward, backward). Entries hold ring
// POSITIONS here; build_gather_plan replaces them with ranks.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> send_lists_at_ring_position(
    uint32_t position, uint32_t num_ranks, bool closed_ring) {
    std::vector<uint32_t> forward, backward;
    const uint32_t G = num_ranks;
    if (closed_ring && G % 2 == 0 && G >= 4) {
        // balanced: G/2 shards each way; the opposite shard (the last entry each way) goes half each way
        const uint32_t per_direction = G / 2;
        for (uint32_t i = 0; i < per_direction; ++i) {
            const bool last = i == per_direction - 1;
            forward.push_back(chunk_walk::make_send_entry(
                (position + G - i) % G, last ? chunk_walk::kFirstHalf : chunk_walk::kWholeShard));
            backward.push_back(chunk_walk::make_send_entry(
                (position + i) % G, last ? chunk_walk::kSecondHalf : chunk_walk::kWholeShard));
        }
    } else if (closed_ring) {
        const uint32_t num_forward = G / 2;
        const uint32_t num_backward = G - 1 - num_forward;
        for (uint32_t i = 0; i < num_forward; ++i) {
            forward.push_back(chunk_walk::make_send_entry((position + G - i) % G, chunk_walk::kWholeShard));
        }
        for (uint32_t i = 0; i < num_backward; ++i) {
            backward.push_back(chunk_walk::make_send_entry((position + i) % G, chunk_walk::kWholeShard));
        }
    } else {  // open line: every shard travels to both ends
        if (position < G - 1) {
            for (uint32_t i = 0; i <= position; ++i) {
                forward.push_back(chunk_walk::make_send_entry(position - i, chunk_walk::kWholeShard));
            }
        }
        if (position > 0) {
            for (uint32_t i = 0; i < G - position; ++i) {
                backward.push_back(chunk_walk::make_send_entry(position + i, chunk_walk::kWholeShard));
            }
        }
    }
    return {forward, backward};
}

// Two edge-disjoint Hamiltonian cycles covering every edge of a rows x cols torus (both >= 3): randomized Warnsdorff
// search for cycle A, accepted when the complementary edges form one cycle B. Deterministic (fixed seed).
std::pair<std::vector<Coord>, std::vector<Coord>> hamiltonian_decomposition(uint32_t rows, uint32_t cols) {
    const uint32_t num_chips = rows * cols;
    auto torus_neighbours = [&](uint32_t chip) {
        const uint32_t r = chip / cols, c = chip % cols;
        return std::array<uint32_t, 4>{
            ((r + 1) % rows) * cols + c,
            ((r + rows - 1) % rows) * cols + c,
            r * cols + (c + 1) % cols,
            r * cols + (c + cols - 1) % cols};
    };
    auto edge_key = [&](uint32_t a, uint32_t b) {
        return static_cast<uint64_t>(std::min(a, b)) * num_chips + std::max(a, b);
    };
    std::mt19937 rng(1);
    for (int attempt = 0; attempt < 20000; ++attempt) {
        std::vector<uint32_t> cycle_a{0};
        std::vector<bool> visited(num_chips, false);
        visited[0] = true;
        uint64_t search_budget = 200000;
        std::function<bool()> extend = [&]() -> bool {
            if (search_budget == 0) {
                return false;  // exhausted: every branch above gives up too
            }
            --search_budget;
            if (cycle_a.size() == num_chips) {
                const auto closing = torus_neighbours(cycle_a.back());
                return std::find(closing.begin(), closing.end(), cycle_a[0]) != closing.end();
            }
            std::vector<uint32_t> candidates;
            for (uint32_t next : torus_neighbours(cycle_a.back())) {
                if (!visited[next] && std::find(candidates.begin(), candidates.end(), next) == candidates.end()) {
                    candidates.push_back(next);
                }
            }
            std::shuffle(candidates.begin(), candidates.end(), rng);
            auto unvisited_degree = [&](uint32_t chip) {
                uint32_t degree = 0;
                for (uint32_t n : torus_neighbours(chip)) {
                    degree += visited[n] ? 0 : 1;
                }
                return degree;
            };
            std::stable_sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) {
                return unvisited_degree(a) < unvisited_degree(b);
            });
            for (uint32_t next : candidates) {
                cycle_a.push_back(next);
                visited[next] = true;
                if (extend()) {
                    return true;
                }
                cycle_a.pop_back();
                visited[next] = false;
            }
            return false;
        };
        if (!extend()) {
            continue;
        }
        std::set<uint64_t> cycle_a_edges;
        for (uint32_t i = 0; i < num_chips; ++i) {
            cycle_a_edges.insert(edge_key(cycle_a[i], cycle_a[(i + 1) % num_chips]));
        }
        // every chip must have exactly two edges left over, forming one cycle
        std::vector<std::vector<uint32_t>> leftover_neighbours(num_chips);
        bool two_left_each = true;
        for (uint32_t chip = 0; chip < num_chips && two_left_each; ++chip) {
            for (uint32_t n : torus_neighbours(chip)) {
                if (!cycle_a_edges.contains(edge_key(chip, n)) &&
                    std::find(leftover_neighbours[chip].begin(), leftover_neighbours[chip].end(), n) ==
                        leftover_neighbours[chip].end()) {
                    leftover_neighbours[chip].push_back(n);
                }
            }
            two_left_each = leftover_neighbours[chip].size() == 2;
        }
        if (!two_left_each) {
            continue;
        }
        std::vector<uint32_t> cycle_b{0};
        uint32_t previous = num_chips, current = 0;
        while (true) {
            const uint32_t next = leftover_neighbours[current][0] != previous ? leftover_neighbours[current][0]
                                                                              : leftover_neighbours[current][1];
            if (next == 0) {
                break;
            }
            cycle_b.push_back(next);
            previous = current;
            current = next;
            if (cycle_b.size() > num_chips) {
                break;
            }
        }
        if (cycle_b.size() != num_chips) {
            continue;
        }
        auto to_coords = [&](const std::vector<uint32_t>& chips) {
            std::vector<Coord> coords;
            for (uint32_t chip : chips) {
                coords.push_back(Coord(chip / cols, chip % cols));
            }
            return coords;
        };
        return {to_coords(cycle_a), to_coords(cycle_b)};
    }
    TT_THROW("fabric_all_gather: no Hamiltonian decomposition found for a {}x{} torus", rows, cols);
}

// A link worker's downstream sends back into this chip (the opposite direction) iff its send list is not empty; then
// this link worker sends it a ready signal.
bool sends_ready_signal(const GatherPlan& plan, uint32_t chip_index, uint32_t ring, uint32_t direction) {
    const auto downstream = plan.chip_on_ring[chip_index][ring].neighbour(direction);
    const uint32_t mesh_cols = plan.mesh_cols;
    return downstream.has_value() &&
           !plan.chip_on_ring[linear_chip_index(*downstream, mesh_cols)][ring].send_list(1 - direction).empty();
}

std::vector<tt::tt_metal::CoreCoord> cores_in_row_major_order(const CoreRangeSet& core_ranges) {
    std::vector<tt::tt_metal::CoreCoord> cores;
    for (const auto& range : core_ranges.ranges()) {
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

GatherPlan build_gather_plan(
    const FabricAllGatherParams& args,
    const Tensor& input_tensor,
    MeshDevice* mesh_device,
    const CoreRangeSet& allowed_core_ranges) {
    const auto mesh_shape = mesh_device->shape();
    const uint32_t mesh_rows = mesh_shape[0], mesh_cols = mesh_shape[1];
    const uint32_t num_chips = mesh_rows * mesh_cols;
    GatherPlan plan;
    plan.mesh_cols = mesh_cols;
    plan.rank_of_chip.assign(num_chips, 0);
    plan.chip_on_ring.assign(num_chips, {});
    plan.link_workers.assign(num_chips, {});
    plan.copy_cores.assign(num_chips, {});
    auto fabric_node = [&](const Coord& chip) { return mesh_device->get_fabric_node_id(chip); };
    auto index_of = [&](const Coord& chip) { return linear_chip_index(chip, mesh_cols); };
    const bool is_2d = tt::tt_fabric::is_2d_fabric_config(args.fabric_config);
    std::array<tt::tt_fabric::Topology, 2> axis_topology{
        tt::tt_fabric::Topology::Linear, tt::tt_fabric::Topology::Linear};
    for (uint32_t axis = 0; axis < 2; ++axis) {
        if (mesh_shape[axis] > 1) {
            axis_topology[axis] = ::ttnn::ccl::get_axis_topology(input_tensor, args.fabric_config, axis);
        }
    }

    // Gather groups (members in row-major order; rank = index in the group) and the ring orders over each.
    std::vector<std::vector<Coord>> groups;
    std::vector<std::vector<std::vector<Coord>>> rings_of_group;
    bool closed_ring = false;
    if (!args.linearized_mesh_ring) {
        const uint32_t axis = args.cluster_axis;
        const uint32_t other_axis = 1 - axis;
        closed_ring = axis_topology[axis] == tt::tt_fabric::Topology::Ring && mesh_shape[axis] > 2;
        for (uint32_t o = 0; o < mesh_shape[other_axis]; ++o) {
            std::vector<Coord> members;
            for (uint32_t i = 0; i < mesh_shape[axis]; ++i) {
                members.push_back(axis == 0 ? Coord(i, o) : Coord(o, i));
            }
            groups.push_back(members);
            rings_of_group.push_back({members});
        }
    } else {
        std::vector<Coord> members;
        for (uint32_t r = 0; r < mesh_rows; ++r) {
            for (uint32_t c = 0; c < mesh_cols; ++c) {
                members.emplace_back(r, c);
            }
        }
        groups.push_back(members);
        const bool torus = is_2d && axis_topology[0] == tt::tt_fabric::Topology::Ring &&
                           axis_topology[1] == tt::tt_fabric::Topology::Ring && mesh_rows >= 3 && mesh_cols >= 3;
        if (torus) {
            auto [cycle_a, cycle_b] = hamiltonian_decomposition(mesh_rows, mesh_cols);
            rings_of_group.push_back({cycle_a, cycle_b});
            closed_ring = true;
        } else {
            const uint32_t probe_links = args.num_links.value_or(1);
            const auto resolved = ttnn::operations::ccl::common::resolve_mesh_ring_plan(
                input_tensor, std::nullopt, probe_links, axis_topology, true, "fabric_all_gather", true);
            TT_FATAL(
                resolved.has_value(),
                "fabric_all_gather: no direct-neighbour snake over the {}x{} mesh",
                mesh_rows,
                mesh_cols);
            std::vector<Coord> snake;
            for (uint32_t t = 0; t < num_chips; ++t) {
                snake.push_back(
                    ttnn::operations::ccl::common::snake_ring_coordinate(t, mesh_shape, resolved->plan.orientation));
            }
            rings_of_group.push_back({snake});
            closed_ring = resolved->topology == tt::tt_fabric::Topology::Ring;
        }
    }
    plan.num_ranks = groups[0].size();
    plan.num_rings = rings_of_group[0].size();
    const uint32_t G = plan.num_ranks;

    for (size_t group = 0; group < groups.size(); ++group) {
        for (uint32_t i = 0; i < G; ++i) {
            plan.rank_of_chip[index_of(groups[group][i])] = i;
        }
        for (const auto& ring : rings_of_group[group]) {
            // send entries of a ring position name the chip at that position by its rank
            auto positions_to_ranks = [&](std::vector<uint32_t> send_list) {
                for (auto& entry : send_list) {
                    entry = chunk_walk::make_send_entry(
                        plan.rank_of_chip[index_of(ring[chunk_walk::entry_rank(entry)])],
                        chunk_walk::entry_half(entry));
                }
                return send_list;
            };
            for (uint32_t position = 0; position < G; ++position) {
                ChipOnRing on_ring;
                if (closed_ring || position + 1 < G) {
                    on_ring.next_chip = ring[(position + 1) % G];
                }
                if (closed_ring || position > 0) {
                    on_ring.previous_chip = ring[(position + G - 1) % G];
                }
                auto [forward, backward] = send_lists_at_ring_position(position, G, closed_ring);
                if (on_ring.next_chip) {
                    on_ring.forward_send_list = positions_to_ranks(forward);
                }
                if (on_ring.previous_chip) {
                    on_ring.backward_send_list = positions_to_ranks(backward);
                }
                plan.chip_on_ring[index_of(ring[position])].push_back(on_ring);
            }
        }
    }

    // Links: the requested number, else the fewest any used hop offers.
    uint32_t num_links = args.num_links.value_or(UINT32_MAX);
    for (uint32_t chip_index = 0; chip_index < num_chips; ++chip_index) {
        const Coord chip(chip_index / mesh_cols, chip_index % mesh_cols);
        for (uint32_t ring = 0; ring < plan.num_rings; ++ring) {
            for (uint32_t direction = 0; direction < 2; ++direction) {
                const auto downstream = plan.chip_on_ring[chip_index][ring].neighbour(direction);
                if (!downstream) {
                    continue;
                }
                const auto links =
                    tt::tt_fabric::get_forwarding_link_indices(fabric_node(chip), fabric_node(*downstream));
                TT_FATAL(
                    !links.empty(),
                    "fabric_all_gather: {} -> {} is not a direct fabric neighbour (fabric config {})",
                    chip,
                    *downstream,
                    args.fabric_config);
                if (args.num_links.has_value()) {
                    TT_FATAL(
                        links.size() >= *args.num_links,
                        "fabric_all_gather: {} -> {} has {} usable link(s), {} requested",
                        chip,
                        *downstream,
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

    // Placement: every link worker with a fabric connection as close as the grid allows (NoC1 hops, the sender's NoC)
    // to its link's Ethernet core, then the receive-only link workers and the copy cores on free cores.
    const auto allowed_cores = cores_in_row_major_order(allowed_core_ranges);
    for (uint32_t chip_index = 0; chip_index < num_chips; ++chip_index) {
        const Coord chip(chip_index / mesh_cols, chip_index % mesh_cols);
        auto* device = mesh_device->get_device(chip);
        auto& link_workers = plan.link_workers[chip_index];
        link_workers.assign(plan.num_rings * 2 * L, {});
        std::set<std::pair<size_t, size_t>> taken;
        auto take = [&](const tt::tt_metal::CoreCoord& core) { taken.insert({core.x, core.y}); };
        auto is_free = [&](const tt::tt_metal::CoreCoord& core) { return !taken.contains({core.x, core.y}); };
        for (uint32_t ring = 0; ring < plan.num_rings; ++ring) {
            for (uint32_t direction = 0; direction < 2; ++direction) {
                const auto& on_ring = plan.chip_on_ring[chip_index][ring];
                const auto downstream = on_ring.neighbour(direction);
                const bool has_fabric_connection =
                    downstream.has_value() &&
                    (!on_ring.send_list(direction).empty() || sends_ready_signal(plan, chip_index, ring, direction));
                if (!has_fabric_connection) {
                    continue;
                }
                const auto links =
                    tt::tt_fabric::get_forwarding_link_indices(fabric_node(chip), fabric_node(*downstream));
                for (uint32_t link = 0; link < L; ++link) {
                    const auto eth_core = tt::tt_fabric::get_forwarding_eth_core(
                        fabric_node(chip), fabric_node(*downstream), links[link]);
                    const auto closest = tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(
                                             *device, eth_core, tt::tt_metal::NOC::NOC_1)
                                             .logical_coord;
                    std::optional<tt::tt_metal::CoreCoord> best;
                    uint32_t best_hops = UINT32_MAX;
                    for (const auto& candidate : allowed_cores) {
                        if (!is_free(candidate)) {
                            continue;
                        }
                        const uint32_t hops = candidate == closest
                                                  ? 0
                                                  : tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                                                        device, candidate, closest, tt::tt_metal::NOC::NOC_1);
                        if (hops < best_hops) {
                            best_hops = hops;
                            best = candidate;
                        }
                    }
                    TT_FATAL(
                        best.has_value(), "fabric_all_gather: the core grid {} has too few cores", allowed_core_ranges);
                    take(*best);
                    link_workers[link_worker_index(ring, direction, link, L)] = LinkWorker{*best, true, links[link]};
                    log_debug(
                        tt::LogOp,
                        "fabric_all_gather: chip {} ring {} direction {} link {} worker {} eth {}",
                        chip,
                        ring,
                        direction,
                        links[link],
                        *best,
                        eth_core);
                }
            }
        }
        for (auto& link_worker : link_workers) {
            if (link_worker.has_fabric_connection) {
                continue;
            }
            auto it = std::find_if(allowed_cores.begin(), allowed_cores.end(), is_free);
            TT_FATAL(
                it != allowed_cores.end(),
                "fabric_all_gather: the core grid {} has too few cores",
                allowed_core_ranges);
            take(*it);
            link_worker.core = *it;
        }
        // copy cores: from the middle row of the grid downward (away from the Ethernet row), then upward
        std::vector<tt::tt_metal::CoreCoord> copy_core_order(allowed_cores.begin(), allowed_cores.end());
        const auto middle_y = copy_core_order[copy_core_order.size() / 2].y;
        std::stable_partition(
            copy_core_order.begin(), copy_core_order.end(), [&](const auto& core) { return core.y >= middle_y; });
        // one per link; a non-interleaved input takes two per link, since they also convert the own shard page by
        // page for the link workers (QuietBox, GLM KV cache, 6 KiB: 4 is fastest; 6 wins only at 14 KiB)
        const bool input_interleaved = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
        const uint32_t num_copy = input_interleaved ? L : 2 * L;
        for (uint32_t link = 0; link < num_copy; ++link) {
            auto it = std::find_if(copy_core_order.begin(), copy_core_order.end(), is_free);
            TT_FATAL(
                it != copy_core_order.end(),
                "fabric_all_gather: the core grid {} has too few cores",
                allowed_core_ranges);
            take(*it);
            plan.copy_cores[chip_index].push_back(*it);
            log_debug(tt::LogOp, "fabric_all_gather: chip {} copy core {} at {}", chip, link, *it);
        }
    }
    return plan;
}

// num_stripes and stripe_pages of the shard (see kernels/fabric_all_gather_chunk_walk.hpp).
ShardPageGeometry derive_shard_page_geometry(
    const FabricAllGatherParams& args, const FabricAllGatherInputs& tensor_args, bool selected_batch) {
    const auto& input = tensor_args.input_tensor;
    const auto& shape = input.padded_shape();
    const int32_t tensor_rank = static_cast<int32_t>(shape.rank());
    const int32_t dim = args.dim;
    const bool tile = input.layout() == Layout::TILE;
    const auto tile_spec = tile ? input.tensor_spec().tile() : tt::tt_metal::Tile();
    TT_FATAL(
        tile || dim < tensor_rank - 1,
        "fabric_all_gather: a ROW_MAJOR gather along the innermost dim (partial pages) is not supported");
    // extent of dim i counted in pages
    auto extent_in_pages = [&](int32_t i) -> uint32_t {
        const uint32_t extent = (i == 0 && selected_batch) ? 1u : shape[i];
        if (tile && i == tensor_rank - 1) {
            return extent / tile_spec.get_width();
        }
        if (tile && i == tensor_rank - 2) {
            return extent / tile_spec.get_height();
        }
        if (!tile && i == tensor_rank - 1) {
            return 1;
        }
        return extent;
    };
    ShardPageGeometry geometry;
    geometry.num_stripes = 1;
    for (int32_t i = 0; i < dim; ++i) {
        geometry.num_stripes *= extent_in_pages(i);
    }
    geometry.stripe_pages = 1;
    for (int32_t i = dim; i < tensor_rank; ++i) {
        geometry.stripe_pages *= extent_in_pages(i);
    }
    const uint32_t num_pages = input.buffer()->num_pages();
    geometry.pages_per_slot = selected_batch ? num_pages / shape[0] : num_pages;
    TT_FATAL(
        geometry.num_stripes * geometry.stripe_pages == geometry.pages_per_slot,
        "fabric_all_gather: page geometry {} x {} does not cover the {} pages of a slot (shape {}, dim {})",
        geometry.num_stripes,
        geometry.stripe_pages,
        geometry.pages_per_slot,
        shape,
        dim);
    geometry.local_gather_dim_size = shape[dim];
    geometry.gather_dim_elements_per_page = tile && dim == tensor_rank - 1
                                                ? tile_spec.get_width()
                                                : (tile && dim == tensor_rank - 2 ? tile_spec.get_height() : 1);
    geometry.num_ranks = args.num_devices;
    if (tensor_args.has_gathered_prefix_metadata()) {
        const uint32_t full_extent_global = shape[dim] * args.num_devices;
        TT_FATAL(
            full_extent_global % args.gathered_slab_global == 0,
            "fabric_all_gather: gathered_slab_global {} must divide the full gathered extent {}",
            args.gathered_slab_global,
            full_extent_global);
        TT_FATAL(
            (args.gathered_slab_global / args.num_devices) % geometry.gather_dim_elements_per_page == 0,
            "fabric_all_gather: the per-chip slab {} must be a multiple of the {}-element page extent along dim {}",
            args.gathered_slab_global / args.num_devices,
            geometry.gather_dim_elements_per_page,
            dim);
        const uint32_t num_slabs = full_extent_global / args.gathered_slab_global;
        TT_FATAL(
            geometry.stripe_pages % num_slabs == 0,
            "fabric_all_gather: {} pages per stripe do not split into {} slabs",
            geometry.stripe_pages,
            num_slabs);
        geometry.pages_per_slab = geometry.stripe_pages / num_slabs;
    }
    return geometry;
}

uint32_t host_active_stripe_pages(const FabricAllGatherParams& args, const ShardPageGeometry& geometry) {
    if (!args.gathered_dim_size.has_value()) {
        return geometry.stripe_pages;
    }
    const uint64_t active_local_extent = *args.gathered_dim_size / args.num_devices;
    TT_FATAL(
        active_local_extent % geometry.gather_dim_elements_per_page == 0,
        "fabric_all_gather: the active extent {} per chip must be a multiple of the {}-element page extent (tile "
        "alignment)",
        active_local_extent,
        geometry.gather_dim_elements_per_page);
    const uint64_t pages = active_local_extent * geometry.stripe_pages;
    TT_FATAL(
        pages % geometry.local_gather_dim_size == 0,
        "fabric_all_gather: gathered_dim_size {} does not cover whole pages",
        *args.gathered_dim_size);
    return static_cast<uint32_t>(pages / geometry.local_gather_dim_size);
}

uint32_t host_slot_base_page(const FabricAllGatherParams& args, const ShardPageGeometry& geometry) {
    return args.input_batch_index.value_or(0) * geometry.pages_per_slot;
}

// Common args 8..15 of every kernel (read by read_shard_geometry in kernels/fabric_all_gather_common.hpp).
std::vector<uint32_t> geometry_args(
    const FabricAllGatherParams& args, const FabricAllGatherInputs& tensor_args, const ShardPageGeometry& geometry) {
    return {
        tensor_args.has_gathered_prefix_metadata() ? tensor_args.gathered_prefix_tensor->buffer()->address() : 0u,
        host_active_stripe_pages(args, geometry),
        geometry.num_stripes,
        geometry.stripe_pages,
        geometry.num_ranks,
        args.gathered_slab_global,
        tensor_args.input_tensor.padded_shape()[args.dim] * args.num_devices,
        geometry.pages_per_slab};
}

std::vector<uint32_t> reader_common_args(
    const FabricAllGatherParams& args,
    const FabricAllGatherInputs& tensor_args,
    const Tensor& output,
    const ShardPageGeometry& geometry,
    uint32_t arrival_counter_address) {
    std::vector<uint32_t> common_args{
        tensor_args.input_tensor.buffer()->address(),
        output.buffer()->address(),
        arrival_counter_address,
        tensor_args.has_batch_index_metadata() ? tensor_args.input_batch_index_tensor->buffer()->address() : 0u,
        args.batch_slot_num_layers,
        args.batch_slot_layer_idx,
        host_slot_base_page(args, geometry),
        geometry.pages_per_slot};
    const auto geometry_block = geometry_args(args, tensor_args, geometry);
    common_args.insert(common_args.end(), geometry_block.begin(), geometry_block.end());
    return common_args;
}

std::vector<uint32_t> writer_common_args(
    const FabricAllGatherParams& args,
    const FabricAllGatherInputs& tensor_args,
    const Tensor& output,
    const ShardPageGeometry& geometry,
    uint32_t arrival_counter_address,
    uint32_t ready_counter_address) {
    std::vector<uint32_t> common_args{
        output.buffer()->address(), arrival_counter_address, ready_counter_address, 0, 0, 0, 0, 0};
    const auto geometry_block = geometry_args(args, tensor_args, geometry);
    common_args.insert(common_args.end(), geometry_block.begin(), geometry_block.end());
    return common_args;
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
    for (const auto& core : cores) {
        ranges.insert(CoreRange(core, core));
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
    const uint32_t mesh_cols = mesh_device->shape()[1];
    const auto subdevice_id = args.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = resolve_available_worker_cores(mesh_device, args);

    const GatherPlan plan = build_gather_plan(args, input, mesh_device, available_cores);
    const bool selected_batch = args.input_batch_index.has_value() || tensor_args.has_batch_index_metadata();
    const auto shard_page_geometry = derive_shard_page_geometry(args, tensor_args, selected_batch);

    // Arrival counter (data_valid) and ready counter (fence), at one address on every chip.
    const bool has_l1_small = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL) > 0;
    const auto counter_buffer_type = has_l1_small ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    const bool external_counters = args.ready_semaphore.has_value();
    auto ready_counter = external_counters ? *args.ready_semaphore
                                           : ttnn::global_semaphore::create_global_semaphore(
                                                 mesh_device, available_cores, 0, counter_buffer_type);
    auto arrival_counter = external_counters ? *args.data_valid_semaphore
                                             : ttnn::global_semaphore::create_global_semaphore(
                                                   mesh_device, available_cores, 0, counter_buffer_type);
    if (external_counters) {
        for (const auto& chip_link_workers : plan.link_workers) {
            std::vector<tt::tt_metal::CoreCoord> cores;
            for (const auto& link_worker : chip_link_workers) {
                cores.push_back(link_worker.core);
            }
            validate_semaphore_core_coverage(ready_counter, to_range_set(cores), "ready_semaphore");
            validate_semaphore_core_coverage(arrival_counter, to_range_set(cores), "data_valid_semaphore");
        }
    } else {
        // every chip's counters must be zero before any chip's first call can increment a neighbour's
        ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);
    }

    const uint32_t page_bytes = input.buffer()->aligned_page_size();
    const uint32_t packet_payload_bytes = static_cast<uint32_t>(args.packet_size);
    // a page larger than the payload is a chunk of its own, sent as several packets
    const uint32_t pages_per_chunk = std::max<uint32_t>(1, packet_payload_bytes / page_bytes);
    const uint32_t chunk_bytes = pages_per_chunk * page_bytes;
    const uint32_t chunks_per_batch =
        std::max<uint32_t>(1, std::min<uint32_t>(kMaxChunksPerBatch, kChunkCbBudgetBytes / (2 * chunk_bytes)));
    const uint32_t num_dram_banks = mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const uint32_t L = plan.num_links;
    const uint32_t bank_stride = plan.num_rings * L;
    TT_FATAL(
        bank_stride <= num_dram_banks,
        "fabric_all_gather: {} rings x {} links exceed {} DRAM banks",
        plan.num_rings,
        L,
        num_dram_banks);
    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const bool batch_from_metadata = tensor_args.has_batch_index_metadata();
    const bool prefix_from_metadata = tensor_args.has_gathered_prefix_metadata();
    const bool input_interleaved = input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;

    // Compile-time args, see the kernels. Absent metadata tensors pass the output's accessor as a placeholder.
    const Tensor& batch_index_tensor = batch_from_metadata ? *tensor_args.input_batch_index_tensor : output_tensor;
    const Tensor& prefix_tensor = prefix_from_metadata ? *tensor_args.gathered_prefix_tensor : output_tensor;
    auto append_accessor = [](std::vector<uint32_t>& compile_args, const Tensor& tensor) {
        tt::tt_metal::TensorAccessorArgs(*tensor.buffer()).append_to(compile_args);
    };
    // `staged_semaphore` (per program, one per copy core): blocks of a non-interleaved input's own shard that copy
    // core has written into the output.
    const uint32_t num_copy_cores = plan.copy_cores[0].size();
    auto reader_config = [&](bool copy_core, uint32_t staged_semaphore) {
        std::vector<uint32_t> compile_args{
            kChunkCbIndex,
            kReaderMetadataCbIndex,
            page_bytes,
            pages_per_chunk,
            num_dram_banks,
            chunks_per_batch,
            batch_from_metadata ? 1u : 0u,
            prefix_from_metadata ? 1u : 0u,
            input_interleaved ? 1u : 0u,
            copy_core ? 1u : 0u,
            staged_semaphore,
            num_copy_cores,
            kRunCbIndex};
        append_accessor(compile_args, input);
        append_accessor(compile_args, output_tensor);
        append_accessor(compile_args, batch_index_tensor);
        append_accessor(compile_args, prefix_tensor);
        return tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_args};
    };
    std::vector<uint32_t> sender_compile_args{
        kChunkCbIndex,
        kWriterMetadataCbIndex,
        page_bytes,
        pages_per_chunk,
        num_dram_banks,
        chunks_per_batch,
        prefix_from_metadata ? 1u : 0u,
        packet_payload_bytes};
    append_accessor(sender_compile_args, output_tensor);
    append_accessor(sender_compile_args, prefix_tensor);
    const auto sender_config = tt::tt_metal::DataMovementConfig{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_1_default,
        .compile_args = sender_compile_args};
    auto copy_writer_config = [&](uint32_t staged_semaphore) {
        std::vector<uint32_t> compile_args{
            kChunkCbIndex,
            kWriterMetadataCbIndex,
            page_bytes,
            pages_per_chunk,
            num_dram_banks,
            chunks_per_batch,
            prefix_from_metadata ? 1u : 0u,
            input_interleaved ? 1u : 0u,
            staged_semaphore,
            kRunCbIndex};
        append_accessor(compile_args, output_tensor);
        append_accessor(compile_args, prefix_tensor);
        return tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = compile_args};
    };

    auto fabric_node = [&](const Coord& chip) { return mesh_device->get_fabric_node_id(chip); };
    auto index_of = [&](const Coord& chip) { return linear_chip_index(chip, mesh_cols); };

    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& chip : tensor_coords.coords()) {
        const uint32_t chip_index = index_of(chip);
        tt::tt_metal::Program program{};
        const auto& link_workers = plan.link_workers[chip_index];
        const auto& copy_cores = plan.copy_cores[chip_index];

        std::vector<tt::tt_metal::CoreCoord> link_worker_cores;
        for (const auto& link_worker : link_workers) {
            link_worker_cores.push_back(link_worker.core);
        }
        std::vector<tt::tt_metal::CoreCoord> all_cores = link_worker_cores;
        all_cores.insert(all_cores.end(), copy_cores.begin(), copy_cores.end());
        const auto link_worker_core_set = to_range_set(link_worker_cores);
        const auto copy_core_set = to_range_set(copy_cores);
        const auto all_core_set = to_range_set(all_cores);

        tt::tt_metal::CreateCircularBuffer(
            program,
            all_core_set,
            tt::tt_metal::CircularBufferConfig(2 * chunks_per_batch * chunk_bytes, {{kChunkCbIndex, data_format}})
                .set_page_size(kChunkCbIndex, page_bytes));
        for (const uint32_t metadata_cb : {kReaderMetadataCbIndex, kWriterMetadataCbIndex}) {
            tt::tt_metal::CreateCircularBuffer(
                program,
                all_core_set,
                tt::tt_metal::CircularBufferConfig(kMetadataCbBytes, {{metadata_cb, tt::DataFormat::UInt32}})
                    .set_page_size(metadata_cb, kMetadataCbBytes));
        }

        if (!input_interleaved) {
            tt::tt_metal::CreateCircularBuffer(
                program,
                copy_core_set,
                tt::tt_metal::CircularBufferConfig(
                    2 * chunks_per_batch * kRunCbPageBytes, {{kRunCbIndex, tt::DataFormat::UInt32}})
                    .set_page_size(kRunCbIndex, kRunCbPageBytes));
        }
        // one per copy core, consecutive ids: blocks of the own shard that copy core has converted (non-interleaved
        // input)
        const uint32_t staged_semaphore = tt::tt_metal::CreateSemaphore(program, link_worker_core_set, 0);
        for (uint32_t c = 1; c < num_copy_cores; ++c) {
            TT_FATAL(
                tt::tt_metal::CreateSemaphore(program, link_worker_core_set, 0) == staged_semaphore + c,
                "fabric_all_gather: staged semaphores must have consecutive ids");
        }
        const std::string dir(kKernelDir);
        const auto link_worker_reader_id = tt::tt_metal::CreateKernel(
            program,
            dir + "fabric_all_gather_reader.cpp",
            link_worker_core_set,
            reader_config(false, staged_semaphore));
        const auto link_worker_sender_id = tt::tt_metal::CreateKernel(
            program, dir + "fabric_all_gather_sender.cpp", link_worker_core_set, sender_config);
        const auto copy_core_reader_id = tt::tt_metal::CreateKernel(
            program, dir + "fabric_all_gather_reader.cpp", copy_core_set, reader_config(true, staged_semaphore));
        const auto copy_core_writer_id = tt::tt_metal::CreateKernel(
            program, dir + "fabric_all_gather_copy_writer.cpp", copy_core_set, copy_writer_config(staged_semaphore));

        const auto reader_common =
            reader_common_args(args, tensor_args, output_tensor, shard_page_geometry, arrival_counter.address());
        const auto writer_common = writer_common_args(
            args, tensor_args, output_tensor, shard_page_geometry, arrival_counter.address(), ready_counter.address());
        tt::tt_metal::SetCommonRuntimeArgs(program, link_worker_reader_id, reader_common);
        tt::tt_metal::SetCommonRuntimeArgs(program, link_worker_sender_id, writer_common);
        tt::tt_metal::SetCommonRuntimeArgs(program, copy_core_reader_id, reader_common);
        tt::tt_metal::SetCommonRuntimeArgs(program, copy_core_writer_id, reader_common);

        for (uint32_t ring = 0; ring < plan.num_rings; ++ring) {
            const auto& on_ring = plan.chip_on_ring[chip_index][ring];
            for (uint32_t direction = 0; direction < 2; ++direction) {
                const auto downstream = on_ring.neighbour(direction);
                const auto upstream = on_ring.upstream(direction);
                const auto& send_list = on_ring.send_list(direction);
                const uint32_t upstream_entries =
                    upstream ? plan.chip_on_ring[index_of(*upstream)][ring].send_list(direction).size() : 0;
                const bool send_ready = sends_ready_signal(plan, chip_index, ring, direction);
                for (uint32_t link = 0; link < L; ++link) {
                    const auto& link_worker = link_workers[link_worker_index(ring, direction, link, L)];
                    const uint32_t first_bank = ring * L + link;

                    std::vector<uint32_t> reader_args{first_bank, bank_stride, static_cast<uint32_t>(send_list.size())};
                    reader_args.insert(reader_args.end(), send_list.begin(), send_list.end());
                    tt::tt_metal::SetRuntimeArgs(program, link_worker_reader_id, link_worker.core, reader_args);

                    // the ready signal goes to the downstream's link worker that sends back into this chip; data and
                    // arrival increments go to the downstream's link worker of the same direction
                    tt::tt_metal::CoreCoord ready_target{0, 0}, downstream_worker{0, 0};
                    tt::tt_fabric::FabricNodeId downstream_node = fabric_node(chip);
                    if (downstream) {
                        auto* downstream_device = mesh_device->get_device(*downstream);
                        const auto& downstream_workers = plan.link_workers[index_of(*downstream)];
                        ready_target = downstream_device->worker_core_from_logical_core(
                            downstream_workers[link_worker_index(ring, 1 - direction, link, L)].core);
                        downstream_worker = downstream_device->worker_core_from_logical_core(
                            downstream_workers[link_worker_index(ring, direction, link, L)].core);
                        downstream_node = fabric_node(*downstream);
                    }
                    std::vector<uint32_t> sender_args{
                        first_bank,
                        bank_stride,
                        send_ready ? 1u : 0u,
                        static_cast<uint32_t>(ready_target.x),
                        static_cast<uint32_t>(ready_target.y),
                        static_cast<uint32_t>(downstream_worker.x),
                        static_cast<uint32_t>(downstream_worker.y),
                        static_cast<uint32_t>(*downstream_node.mesh_id),
                        downstream_node.chip_id,
                        upstream_entries,
                        static_cast<uint32_t>(send_list.size())};
                    sender_args.insert(sender_args.end(), send_list.begin(), send_list.end());
                    if (link_worker.has_fabric_connection) {
                        tt::tt_fabric::append_fabric_connection_rt_args(
                            fabric_node(chip),
                            fabric_node(*downstream),
                            link_worker.fabric_link_index,
                            program,
                            link_worker.core,
                            sender_args);
                    }
                    tt::tt_metal::SetRuntimeArgs(program, link_worker_sender_id, link_worker.core, sender_args);
                }
            }
        }
        // Copy core c of n owns banks c, c + n, ... of this chip's shard (interleaved input), or blocks c, c + n, ...
        // of it (non-interleaved input, for_each_input_run); the latter signals every link worker of this chip per
        // block.
        const uint32_t rank = plan.rank_of_chip[chip_index];
        auto* device = mesh_device->get_device(chip);
        std::vector<uint32_t> copy_writer_args{0, num_copy_cores, rank, static_cast<uint32_t>(link_workers.size())};
        for (const auto& link_worker : link_workers) {
            const auto noc = device->worker_core_from_logical_core(link_worker.core);
            copy_writer_args.push_back(noc.x);
            copy_writer_args.push_back(noc.y);
        }
        for (uint32_t c = 0; c < num_copy_cores; ++c) {
            tt::tt_metal::SetRuntimeArgs(
                program,
                copy_core_reader_id,
                copy_cores[c],
                std::vector<uint32_t>{
                    c, num_copy_cores, 1, chunk_walk::make_send_entry(rank, chunk_walk::kWholeShard)});
            copy_writer_args[0] = c;
            tt::tt_metal::SetRuntimeArgs(program, copy_core_writer_id, copy_cores[c], copy_writer_args);
        }

        workload.add_program(ttnn::MeshCoordinateRange(chip), std::move(program));
        shared_variables.emplace(
            ttnn::MeshCoordinateRange(chip),
            shared_variables_t{
                .link_worker_reader_kernel_id = link_worker_reader_id,
                .link_worker_sender_kernel_id = link_worker_sender_id,
                .copy_core_reader_kernel_id = copy_core_reader_id,
                .copy_core_writer_kernel_id = copy_core_writer_id,
                .ready_counter = ready_counter,
                .arrival_counter = arrival_counter,
                .link_worker_cores = link_worker_core_set,
                .shard_page_geometry = shard_page_geometry});
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
            (shared.ready_counter.address() != args.ready_semaphore->address() ||
             shared.arrival_counter.address() != args.data_valid_semaphore->address())) {
            validate_semaphore_core_coverage(*args.ready_semaphore, shared.link_worker_cores, "ready_semaphore");
            validate_semaphore_core_coverage(
                *args.data_valid_semaphore, shared.link_worker_cores, "data_valid_semaphore");
            shared.ready_counter = *args.ready_semaphore;
            shared.arrival_counter = *args.data_valid_semaphore;
        }
        const auto reader_common = reader_common_args(
            args, tensor_args, output_tensor, shared.shard_page_geometry, shared.arrival_counter.address());
        const auto writer_common = writer_common_args(
            args,
            tensor_args,
            output_tensor,
            shared.shard_page_geometry,
            shared.arrival_counter.address(),
            shared.ready_counter.address());
        // only the common args change between calls (addresses, slot base, extent)
        auto patch_common_args = [&](tt::tt_metal::KernelHandle kernel, const std::vector<uint32_t>& values) {
            auto& common = GetCommonRuntimeArgs(program, kernel);
            for (size_t i = 0; i < values.size(); ++i) {
                common[i] = values[i];
            }
        };
        patch_common_args(shared.link_worker_reader_kernel_id, reader_common);
        patch_common_args(shared.link_worker_sender_kernel_id, writer_common);
        patch_common_args(shared.copy_core_reader_kernel_id, reader_common);
        patch_common_args(shared.copy_core_writer_kernel_id, reader_common);
    }
}

}  // namespace ttnn::operations::experimental::fabric_all_gather
