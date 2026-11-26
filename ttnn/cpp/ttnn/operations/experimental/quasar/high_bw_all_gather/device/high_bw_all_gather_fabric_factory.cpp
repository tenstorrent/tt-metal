// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// high_bw_all_gather's program factory.
// Terms: kernels/fabric/fabric_all_gather_chunk_walk.hpp.
//
// build_gather_plan decides the rings, every fabric link worker's outgoing shards and every core's placement; the
// kernels only walk what they are given. Rings: an axis ring or line (cluster_axis 0 / 1), a snake over the whole mesh
// (cluster_axis None), or on a torus with both sides >= 3 two edge-disjoint Hamiltonian cycles that each own half of
// the fabric chunks, so every chip uses all four neighbours. On a 4x4 cylinder (one axis a ring, the other a line)
// three Hamiltonian cycles each own a third, with every edge on at most two of them: the line-end chips (three
// neighbours) then receive on all three. A hop shared by two rings gives each its own fabric links.

#include "high_bw_all_gather_fabric_factory.hpp"
#include "kernels/fabric/fabric_all_gather_chunk_walk.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <internal/mesh_device.hpp>
#include <tt-metalium/workload/legacy/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/mesh_ring_plan.hpp"

namespace ttnn::operations::experimental::quasar::high_bw_all_gather {

namespace CMAKE_UNIQUE_NAMESPACE {

namespace chunk_walk = ::ttnn::operations::experimental::quasar::high_bw_all_gather::chunk_walk;

constexpr uint32_t kFabricChunkCbIndex = tt::CBIndex::c_0;
constexpr uint32_t kReaderMetadataCbIndex = tt::CBIndex::c_1;  // the reader's metadata landing slot
constexpr uint32_t kWriterMetadataCbIndex =
    tt::CBIndex::c_2;  // the sender's / copy writer's (the RISCs read concurrently)
constexpr uint32_t kMetadataCbBytes = 64;
constexpr uint32_t kInputRunCbIndex =
    tt::CBIndex::c_3;  // local copy cores, non-interleaved input: contiguous input runs, reader -> writer
constexpr uint32_t kInputRunCbPageBytes = 16;
constexpr uint32_t kFabricChunkCbBudgetBytes = 112 * 1024;  // chunk CB per core: two CB batches
constexpr uint32_t kMaxChunksPerCbBatch = 8;
constexpr uint32_t kForward = 0;
constexpr const char* kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/quasar/high_bw_all_gather/device/kernels/fabric/";

using Coord = ttnn::MeshCoordinate;

// One chip's place on one ring.
struct ChipOnRing {
    std::optional<Coord> next_chip;      // forward neighbour
    std::optional<Coord> previous_chip;  // backward neighbour
    // Outgoing shards (packed chunk_walk::make_outgoing_shard) of this chip's forward and backward link workers, in
    // the order they arrive downstream; outgoing shard 0 is this chip's own shard.
    std::vector<uint32_t> forward_outgoing_shards;
    std::vector<uint32_t> backward_outgoing_shards;

    const std::optional<Coord>& neighbour(uint32_t direction) const {
        return direction == kForward ? next_chip : previous_chip;
    }
    const std::optional<Coord>& upstream(uint32_t direction) const {
        return direction == kForward ? previous_chip : next_chip;
    }
    const std::vector<uint32_t>& outgoing_shards(uint32_t direction) const {
        return direction == kForward ? forward_outgoing_shards : backward_outgoing_shards;
    }
};

struct FabricLinkWorker {
    tt::tt_metal::CoreCoord core;        // logical
    bool has_fabric_connection = false;  // sends data or the started signal
    uint32_t fabric_link_index = 0;      // routing plane toward its downstream
};

struct GatherPlan {
    uint32_t num_ranks = 0;
    uint32_t num_rings = 0;
    uint32_t num_links = 0;      // per (ring, direction)
    uint32_t links_per_hop = 0;  // num_links * the most (ring, direction)s sharing a hop
    uint32_t mesh_cols = 0;
    std::vector<uint32_t> rank_of_chip;                                  // [linear chip index]
    std::vector<std::vector<ChipOnRing>> chip_on_ring;                   // [linear chip index][ring]
    std::vector<std::vector<FabricLinkWorker>> fabric_link_workers;      // [linear chip index][link_worker_index(...)]
    std::vector<std::vector<tt::tt_metal::CoreCoord>> local_copy_cores;  // [linear chip index][link]

    uint32_t linear_chip_index(const Coord& chip) const { return chip[0] * mesh_cols + chip[1]; }
};

uint32_t fabric_link_worker_index(uint32_t ring, uint32_t direction, uint32_t link, uint32_t num_links) {
    return (ring * 2 + direction) * num_links + link;
}

// Outgoing shards of the chip at `position` of a ring of `num_ranks` chips, as (forward, backward). They hold ring
// POSITIONS here; build_gather_plan replaces them with ranks.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> outgoing_shards_at_ring_position(
    uint32_t position, uint32_t num_ranks, bool closed_ring) {
    std::vector<uint32_t> forward, backward;
    const uint32_t G = num_ranks;
    if (closed_ring && G % 2 == 0 && G >= 4) {
        // balanced: G/2 shards each way; the opposite shard (the last one each way) goes half each way
        const uint32_t per_direction = G / 2;
        for (uint32_t i = 0; i < per_direction; ++i) {
            const bool last = i == per_direction - 1;
            forward.push_back(chunk_walk::make_outgoing_shard(
                (position + G - i) % G, last ? chunk_walk::kFirstChunkHalf : chunk_walk::kWholeChipShard));
            backward.push_back(chunk_walk::make_outgoing_shard(
                (position + i) % G, last ? chunk_walk::kSecondChunkHalf : chunk_walk::kWholeChipShard));
        }
    } else if (closed_ring) {
        const uint32_t num_forward = G / 2;
        const uint32_t num_backward = G - 1 - num_forward;
        for (uint32_t i = 0; i < num_forward; ++i) {
            forward.push_back(chunk_walk::make_outgoing_shard((position + G - i) % G, chunk_walk::kWholeChipShard));
        }
        for (uint32_t i = 0; i < num_backward; ++i) {
            backward.push_back(chunk_walk::make_outgoing_shard((position + i) % G, chunk_walk::kWholeChipShard));
        }
    } else {  // open line: every shard travels to both ends
        if (position < G - 1) {
            for (uint32_t i = 0; i <= position; ++i) {
                forward.push_back(chunk_walk::make_outgoing_shard(position - i, chunk_walk::kWholeChipShard));
            }
        }
        if (position > 0) {
            for (uint32_t i = 0; i < G - position; ++i) {
                backward.push_back(chunk_walk::make_outgoing_shard(position + i, chunk_walk::kWholeChipShard));
            }
        }
    }
    return {forward, backward};
}

// Two edge-disjoint Hamiltonian cycles covering every edge of a rows x cols torus (both >= 3, so a chip's four
// neighbours are distinct): randomized Warnsdorff search for cycle A, accepted when the complementary edges form one
// cycle B. Deterministic (fixed seed).
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
                if (!visited[next]) {
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
                if (!cycle_a_edges.contains(edge_key(chip, n))) {
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
            coords.reserve(chips.size());
            for (uint32_t chip : chips) {
                coords.push_back(Coord(chip / cols, chip % cols));
            }
            return coords;
        };
        return {to_coords(cycle_a), to_coords(cycle_b)};
    }
    TT_THROW("high_bw_all_gather: no Hamiltonian decomposition found for a {}x{} torus", rows, cols);
}

// The search depends only on the torus shape (fixed seed), so one result per shape serves every program: a program
// cache miss for a new shape or dtype no longer searches again (8x4: 0.4 ms, 8x8: ~230 ms).
const std::pair<std::vector<Coord>, std::vector<Coord>>& cached_hamiltonian_decomposition(
    uint32_t rows, uint32_t cols) {
    static std::mutex mutex;
    static std::map<std::pair<uint32_t, uint32_t>, std::pair<std::vector<Coord>, std::vector<Coord>>> cycles_of_shape;
    std::lock_guard lock(mutex);
    auto it = cycles_of_shape.find({rows, cols});
    if (it == cycles_of_shape.end()) {
        it = cycles_of_shape.emplace(std::pair{rows, cols}, hamiltonian_decomposition(rows, cols)).first;
    }
    return it->second;
}

// Three Hamiltonian cycles of the 4x4 cylinder, as {line axis index, ring axis index}; every edge is on at most two of
// them (the 16 edges at a line end on exactly two, 4 middle edges on two, 8 middle edges on one). With each cycle
// owning a third of every shard, the busiest directed edge carries 2 * 7.5 / 3 = 5 shards, the bound of a line-end chip
// (15 shards over three neighbours); a snake carries 7.5. Found by exhaustive search (112 such triples exist); this one
// is symmetric under reflecting either axis.
// clang-format off
constexpr std::array<std::array<std::array<uint8_t, 2>, 16>, 3> kCylinder4x4Cycles{{
    {{{0, 0}, {0, 3}, {0, 2}, {0, 1}, {1, 1}, {2, 1}, {2, 0}, {2, 3},
      {3, 3}, {3, 0}, {3, 1}, {3, 2}, {2, 2}, {1, 2}, {1, 3}, {1, 0}}},
    {{{0, 0}, {0, 1}, {0, 2}, {1, 2}, {2, 2}, {2, 3}, {2, 0}, {3, 0},
      {3, 3}, {3, 2}, {3, 1}, {2, 1}, {1, 1}, {1, 0}, {1, 3}, {0, 3}}},
    {{{0, 0}, {0, 1}, {1, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3},
      {3, 3}, {3, 2}, {2, 2}, {2, 1}, {3, 1}, {3, 0}, {2, 0}, {1, 0}}},
}};
// clang-format on

// Each cycle visits every chip once over direct neighbours (ring steps wrap, line steps do not), and no edge is on more
// than two cycles.
constexpr bool cylinder_4x4_cycles_valid() {
    std::array<uint32_t, 16 * 16> cycles_on_edge{};
    for (const auto& cycle : kCylinder4x4Cycles) {
        std::array<bool, 16> visited{};
        for (size_t i = 0; i < cycle.size(); ++i) {
            const uint32_t line = cycle[i][0], ring = cycle[i][1];
            const uint32_t next_line = cycle[(i + 1) % cycle.size()][0], next_ring = cycle[(i + 1) % cycle.size()][1];
            if (line > 3 || ring > 3 || visited[line * 4 + ring]) {
                return false;
            }
            visited[line * 4 + ring] = true;
            const bool ring_step = line == next_line && ((ring + 1) % 4 == next_ring || (next_ring + 1) % 4 == ring);
            const bool line_step = ring == next_ring && (line + 1 == next_line || next_line + 1 == line);
            if (!ring_step && !line_step) {
                return false;
            }
            const uint32_t a = line * 4 + ring, b = next_line * 4 + next_ring;
            if (++cycles_on_edge[std::min(a, b) * 16 + std::max(a, b)] > 2) {
                return false;
            }
        }
    }
    return true;
}
static_assert(cylinder_4x4_cycles_valid());

// The cycles above on a 4x4 mesh whose `ring_axis` wraps and whose other axis is a line.
std::vector<std::vector<Coord>> cylinder_4x4_cycles(uint32_t ring_axis) {
    std::vector<std::vector<Coord>> cycles;
    for (const auto& stored : kCylinder4x4Cycles) {
        auto& cycle = cycles.emplace_back();
        for (const auto& [line, ring] : stored) {
            cycle.push_back(
                ring_axis == 1 ? Coord(uint32_t{line}, uint32_t{ring}) : Coord(uint32_t{ring}, uint32_t{line}));
        }
    }
    return cycles;
}

// A link worker's downstream sends back into this chip (the opposite direction) iff it has outgoing shards; then this
// link worker signals it that this chip has started the call.
bool signals_started_to_downstream(const GatherPlan& plan, uint32_t chip_index, uint32_t ring, uint32_t direction) {
    const auto downstream = plan.chip_on_ring[chip_index][ring].neighbour(direction);
    return downstream.has_value() &&
           !plan.chip_on_ring[plan.linear_chip_index(*downstream)][ring].outgoing_shards(1 - direction).empty();
}

std::vector<tt::tt_metal::CoreCoord> cores_in_row_major_order(const CoreRangeSet& core_ranges) {
    auto cores = tt::tt_metal::corerange_to_cores(core_ranges, std::nullopt, /*row_wise=*/true);
    std::sort(
        cores.begin(), cores.end(), [](const auto& a, const auto& b) { return a.y != b.y ? a.y < b.y : a.x < b.x; });
    return cores;
}

GatherPlan build_gather_plan(
    const HighBwAllGatherParams& args,
    const Tensor& input_tensor,
    MeshDevice* mesh_device,
    const CoreRangeSet& allowed_core_ranges,
    bool input_interleaved) {
    const auto& mesh_shape = mesh_device->shape();
    const uint32_t mesh_rows = mesh_shape[0], mesh_cols = mesh_shape[1];
    const uint32_t num_chips = mesh_rows * mesh_cols;
    GatherPlan plan;
    plan.mesh_cols = mesh_cols;
    plan.rank_of_chip.assign(num_chips, 0);
    plan.chip_on_ring.assign(num_chips, {});
    plan.fabric_link_workers.assign(num_chips, {});
    plan.local_copy_cores.assign(num_chips, {});
    const bool is_2d = tt::tt_fabric::is_2d_fabric_config(args.fabric_config);
    std::array<tt::tt_fabric::Topology, 2> axis_topology{
        tt::tt_fabric::Topology::Linear, tt::tt_fabric::Topology::Linear};
    for (uint32_t axis = 0; axis < 2; ++axis) {
        if (mesh_shape[axis] > 1) {
            axis_topology[axis] = ::ttnn::ccl::get_axis_topology(input_tensor, args.fabric_config, axis);
        }
    }

    // Gather groups (rank = index in the group) and the ring orders over each. An axis group's members run
    // along the axis; the full-mesh group's run row-major, or column-major with full_mesh_outer_axis 1 (a
    // TP-deduplicated cache with SP on mesh axis 1 holds its shards SP-major).
    std::vector<std::vector<Coord>> groups;
    std::vector<std::vector<std::vector<Coord>>> rings_of_group;
    bool closed_ring = false;
    if (!args.linearized_mesh_ring) {
        const uint32_t axis = args.cluster_axis;
        const uint32_t other_axis = 1 - axis;
        closed_ring = axis_topology[axis] == tt::tt_fabric::Topology::Ring && mesh_shape[axis] > 2;
        for (uint32_t o = 0; o < mesh_shape[other_axis]; ++o) {
            std::vector<Coord> members;
            members.reserve(mesh_shape[axis]);
            for (uint32_t i = 0; i < mesh_shape[axis]; ++i) {
                members.push_back(axis == 0 ? Coord(i, o) : Coord(o, i));
            }
            groups.push_back(members);
            rings_of_group.push_back({members});
        }
    } else {
        std::vector<Coord> members;
        for (uint32_t outer = 0; outer < mesh_shape[args.full_mesh_outer_axis]; ++outer) {
            for (uint32_t inner = 0; inner < mesh_shape[1 - args.full_mesh_outer_axis]; ++inner) {
                members.push_back(args.full_mesh_outer_axis == 0 ? Coord(outer, inner) : Coord(inner, outer));
            }
        }
        groups.push_back(members);
        const bool torus = is_2d && axis_topology[0] == tt::tt_fabric::Topology::Ring &&
                           axis_topology[1] == tt::tt_fabric::Topology::Ring && mesh_rows >= 3 && mesh_cols >= 3;
        // A 4x4 cylinder takes three cycles when every hop on them offers two links (a hop shared by two cycles gives
        // each its own), else the snake.
        const bool cylinder_4x4 =
            is_2d && mesh_rows == 4 && mesh_cols == 4 &&
            (axis_topology[0] == tt::tt_fabric::Topology::Ring) != (axis_topology[1] == tt::tt_fabric::Topology::Ring);
        std::vector<std::vector<Coord>> cylinder_cycles;
        if (cylinder_4x4 && args.num_links.value_or(2) >= 2) {
            cylinder_cycles = cylinder_4x4_cycles(axis_topology[1] == tt::tt_fabric::Topology::Ring ? 1 : 0);
            auto two_links = [&](const Coord& a, const Coord& b) {
                return tt::tt_fabric::get_forwarding_link_indices(
                           *mesh_device, mesh_device->get_fabric_node_id(a), mesh_device->get_fabric_node_id(b))
                           .size() >= 2;
            };
            for (const auto& cycle : cylinder_cycles) {
                for (uint32_t i = 0; i < cycle.size(); ++i) {
                    const auto& a = cycle[i];
                    const auto& b = cycle[(i + 1) % cycle.size()];
                    if (!two_links(a, b) || !two_links(b, a)) {
                        log_info(
                            tt::LogOp,
                            "high_bw_all_gather: {} <-> {} has fewer than two links, the 4x4 cylinder uses the snake",
                            a,
                            b);
                        cylinder_cycles.clear();
                        break;
                    }
                }
                if (cylinder_cycles.empty()) {
                    break;
                }
            }
        }
        if (torus) {
            const auto& [cycle_a, cycle_b] = cached_hamiltonian_decomposition(mesh_rows, mesh_cols);
            rings_of_group.push_back({cycle_a, cycle_b});
            closed_ring = true;
        } else if (!cylinder_cycles.empty()) {
            rings_of_group.push_back(cylinder_cycles);
            closed_ring = true;
        } else {
            const uint32_t probe_links = args.num_links.value_or(1);
            const auto resolved = ttnn::operations::ccl::common::resolve_mesh_ring_plan(
                input_tensor, std::nullopt, probe_links, axis_topology, true, "high_bw_all_gather", true);
            TT_FATAL(
                resolved.has_value(),
                "high_bw_all_gather: no direct-neighbour snake over the {}x{} mesh",
                mesh_rows,
                mesh_cols);
            std::vector<Coord> snake;
            snake.reserve(num_chips);
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
            plan.rank_of_chip[plan.linear_chip_index(groups[group][i])] = i;
        }
        for (const auto& ring : rings_of_group[group]) {
            // outgoing shards of a ring position name the chip at that position by its rank
            auto positions_to_ranks = [&](std::vector<uint32_t> outgoing_shards) {
                for (auto& outgoing_shard : outgoing_shards) {
                    outgoing_shard = chunk_walk::make_outgoing_shard(
                        plan.rank_of_chip[plan.linear_chip_index(
                            ring[chunk_walk::outgoing_shard_rank(outgoing_shard)])],
                        chunk_walk::outgoing_shard_chunk_half(outgoing_shard));
                }
                return outgoing_shards;
            };
            for (uint32_t position = 0; position < G; ++position) {
                ChipOnRing on_ring;
                if (closed_ring || position + 1 < G) {
                    on_ring.next_chip = ring[(position + 1) % G];
                }
                if (closed_ring || position > 0) {
                    on_ring.previous_chip = ring[(position + G - 1) % G];
                }
                auto [forward, backward] = outgoing_shards_at_ring_position(position, G, closed_ring);
                if (on_ring.next_chip) {
                    on_ring.forward_outgoing_shards = positions_to_ranks(forward);
                }
                if (on_ring.previous_chip) {
                    on_ring.backward_outgoing_shards = positions_to_ranks(backward);
                }
                plan.chip_on_ring[plan.linear_chip_index(ring[position])].push_back(on_ring);
            }
        }
    }

    // Links. A directed hop (chip -> downstream) is used by one (ring, direction), or by two on the cylinder; a fabric
    // link takes one worker connection, so each user of a hop gets its own links of it. Links per hop: the requested
    // number, else the fewest any used hop offers; every (ring, direction) gets that / the most users of any hop.
    // forwarding_links[chip][ring * 2 + direction]: its links (empty without a downstream).
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> users_of_hop;  // ring * 2 + direction, in order
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> links_of_hop;
    uint32_t links_per_hop = args.num_links.value_or(UINT32_MAX);
    uint32_t max_users_per_hop = 1;
    for (uint32_t chip_index = 0; chip_index < num_chips; ++chip_index) {
        const Coord chip(chip_index / mesh_cols, chip_index % mesh_cols);
        for (uint32_t ring = 0; ring < plan.num_rings; ++ring) {
            for (uint32_t direction = 0; direction < 2; ++direction) {
                const auto downstream = plan.chip_on_ring[chip_index][ring].neighbour(direction);
                if (!downstream) {
                    continue;
                }
                const auto hop = std::pair{chip_index, plan.linear_chip_index(*downstream)};
                auto& users = users_of_hop[hop];
                users.push_back(ring * 2 + direction);
                max_users_per_hop = std::max<uint32_t>(max_users_per_hop, users.size());
                auto [it, inserted] = links_of_hop.try_emplace(hop);
                if (!inserted) {
                    continue;
                }
                auto& links = it->second;
                links = tt::tt_fabric::get_forwarding_link_indices(
                    *mesh_device, mesh_device->get_fabric_node_id(chip), mesh_device->get_fabric_node_id(*downstream));
                TT_FATAL(
                    !links.empty(),
                    "high_bw_all_gather: {} -> {} is not a direct fabric neighbour (fabric config {})",
                    chip,
                    *downstream,
                    args.fabric_config);
                if (args.num_links.has_value()) {
                    TT_FATAL(
                        links.size() >= *args.num_links,
                        "high_bw_all_gather: {} -> {} has {} usable link(s), {} requested",
                        chip,
                        *downstream,
                        links.size(),
                        *args.num_links);
                } else {
                    links_per_hop = std::min<uint32_t>(links_per_hop, links.size());
                }
            }
        }
    }
    TT_FATAL(
        links_per_hop >= max_users_per_hop,
        "high_bw_all_gather: {} rings share a hop and need a link each, but only {} link(s) per hop are used",
        max_users_per_hop,
        links_per_hop);
    const uint32_t L = links_per_hop / max_users_per_hop;
    plan.num_links = L;
    plan.links_per_hop = L * max_users_per_hop;
    std::vector<std::vector<std::vector<uint32_t>>> forwarding_links(
        num_chips, std::vector<std::vector<uint32_t>>(plan.num_rings * 2));
    for (const auto& [hop, users] : users_of_hop) {
        const auto& links = links_of_hop.at(hop);
        for (uint32_t user = 0; user < users.size(); ++user) {
            forwarding_links[hop.first][users[user]].assign(links.begin() + user * L, links.begin() + (user + 1) * L);
        }
    }

    // Placement: every link worker with a fabric connection as close as the grid allows (NoC1 hops, the sender's NoC)
    // to its link's Ethernet core, then the receive-only link workers and the local copy cores on free cores.
    const auto allowed_cores = cores_in_row_major_order(allowed_core_ranges);
    // local copy cores: from the middle row of the grid downward (away from the Ethernet row), then upward
    std::vector<tt::tt_metal::CoreCoord> local_copy_core_order(allowed_cores.begin(), allowed_cores.end());
    const auto middle_y = local_copy_core_order[local_copy_core_order.size() / 2].y;
    std::stable_partition(local_copy_core_order.begin(), local_copy_core_order.end(), [&](const auto& core) {
        return core.y >= middle_y;
    });
    // one per link of a hop; a non-interleaved input takes two per link, since they also convert the own shard page by
    // page for the link workers (QuietBox, GLM KV cache, 6 KiB: 4 is fastest; 6 wins only at 14 KiB)
    const uint32_t num_local_copy_cores = input_interleaved ? plan.links_per_hop : 2 * plan.links_per_hop;
    for (uint32_t chip_index = 0; chip_index < num_chips; ++chip_index) {
        const Coord chip(chip_index / mesh_cols, chip_index % mesh_cols);
        auto* device = tt::tt_metal::internal::get_device(*mesh_device, chip);
        auto& fabric_link_workers = plan.fabric_link_workers[chip_index];
        fabric_link_workers.assign(plan.num_rings * 2 * L, {});
        std::set<std::pair<size_t, size_t>> taken;
        auto take = [&](const tt::tt_metal::CoreCoord& core) { taken.insert({core.x, core.y}); };
        auto is_free = [&](const tt::tt_metal::CoreCoord& core) { return !taken.contains({core.x, core.y}); };
        for (uint32_t ring = 0; ring < plan.num_rings; ++ring) {
            for (uint32_t direction = 0; direction < 2; ++direction) {
                const auto& on_ring = plan.chip_on_ring[chip_index][ring];
                const auto downstream = on_ring.neighbour(direction);
                const bool has_fabric_connection =
                    downstream.has_value() && (!on_ring.outgoing_shards(direction).empty() ||
                                               signals_started_to_downstream(plan, chip_index, ring, direction));
                if (!has_fabric_connection) {
                    continue;
                }
                const auto& links = forwarding_links[chip_index][ring * 2 + direction];
                for (uint32_t link = 0; link < L; ++link) {
                    const auto eth_core = tt::tt_fabric::get_forwarding_eth_core(
                        *mesh_device,
                        mesh_device->get_fabric_node_id(chip),
                        mesh_device->get_fabric_node_id(*downstream),
                        links[link]);
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
                        best.has_value(),
                        "high_bw_all_gather: the core grid {} has too few cores",
                        allowed_core_ranges);
                    take(*best);
                    fabric_link_workers[fabric_link_worker_index(ring, direction, link, L)] =
                        FabricLinkWorker{*best, true, links[link]};
                    log_debug(
                        tt::LogOp,
                        "high_bw_all_gather: chip {} ring {} direction {} link {} worker {} eth {}",
                        chip,
                        ring,
                        direction,
                        links[link],
                        *best,
                        eth_core);
                }
            }
        }
        for (auto& fabric_link_worker : fabric_link_workers) {
            if (fabric_link_worker.has_fabric_connection) {
                continue;
            }
            auto it = std::find_if(allowed_cores.begin(), allowed_cores.end(), is_free);
            TT_FATAL(
                it != allowed_cores.end(),
                "high_bw_all_gather: the core grid {} has too few cores",
                allowed_core_ranges);
            take(*it);
            fabric_link_worker.core = *it;
        }
        for (uint32_t link = 0; link < num_local_copy_cores; ++link) {
            auto it = std::find_if(local_copy_core_order.begin(), local_copy_core_order.end(), is_free);
            TT_FATAL(
                it != local_copy_core_order.end(),
                "high_bw_all_gather: the core grid {} has too few cores",
                allowed_core_ranges);
            take(*it);
            plan.local_copy_cores[chip_index].push_back(*it);
            log_debug(tt::LogOp, "high_bw_all_gather: chip {} local copy core {} at {}", chip, link, *it);
        }
    }
    return plan;
}

// Outer slices and pages per outer slice of the chip shard (see kernels/fabric/fabric_all_gather_chunk_walk.hpp).
ChipShardPageGeometry derive_shard_page_geometry(
    const HighBwAllGatherParams& args, const HighBwAllGatherInputs& tensor_args, bool selected_batch) {
    const auto& input = tensor_args.input_tensor;
    const auto& shape = input.padded_shape();
    const int32_t tensor_rank = static_cast<int32_t>(shape.rank());
    const int32_t dim = args.dim;
    const bool tile = input.layout() == Layout::TILE;
    const auto tile_spec = tile ? input.tensor_spec().tile() : tt::tt_metal::Tile();
    // the length of dim i, counted in pages
    auto length_in_pages = [&](int32_t i) -> uint32_t {
        const uint32_t dim_length = (i == 0 && selected_batch) ? 1u : shape[i];
        if (tile && i == tensor_rank - 1) {
            return dim_length / tile_spec.get_width();
        }
        if (tile && i == tensor_rank - 2) {
            return dim_length / tile_spec.get_height();
        }
        if (!tile && i == tensor_rank - 1) {
            return 1;
        }
        return dim_length;
    };
    ChipShardPageGeometry geometry;
    geometry.num_outer_slices = 1;
    for (int32_t i = 0; i < dim; ++i) {
        geometry.num_outer_slices *= length_in_pages(i);
    }
    geometry.pages_per_outer_slice = 1;
    for (int32_t i = dim; i < tensor_rank; ++i) {
        geometry.pages_per_outer_slice *= length_in_pages(i);
    }
    const uint32_t num_pages = input.mesh_buffer().num_pages();
    geometry.pages_per_cache_slot = selected_batch ? num_pages / shape[0] : num_pages;
    TT_FATAL(
        geometry.num_outer_slices * geometry.pages_per_outer_slice == geometry.pages_per_cache_slot,
        "high_bw_all_gather: page geometry {} x {} does not cover the {} pages of a cache slot (shape {}, dim {})",
        geometry.num_outer_slices,
        geometry.pages_per_outer_slice,
        geometry.pages_per_cache_slot,
        shape,
        dim);
    geometry.local_gather_dim_length = shape[dim];
    geometry.gather_dim_elements_per_page = tile && dim == tensor_rank - 1
                                                ? tile_spec.get_width()
                                                : (tile && dim == tensor_rank - 2 ? tile_spec.get_height() : 1);
    geometry.num_ranks = args.num_devices;
    if (tensor_args.has_gathered_prefix_metadata()) {
        const uint32_t full_gathered_length = shape[dim] * args.num_devices;
        TT_FATAL(
            full_gathered_length % args.gathered_slab_global == 0,
            "high_bw_all_gather: gathered_slab_global {} must divide the full gathered length {}",
            args.gathered_slab_global,
            full_gathered_length);
        TT_FATAL(
            (args.gathered_slab_global / args.num_devices) % geometry.gather_dim_elements_per_page == 0,
            "high_bw_all_gather: the per-chip KV slab {} must be a multiple of the {} elements a page spans along dim "
            "{}",
            args.gathered_slab_global / args.num_devices,
            geometry.gather_dim_elements_per_page,
            dim);
        const uint32_t num_kv_slabs = full_gathered_length / args.gathered_slab_global;
        TT_FATAL(
            geometry.pages_per_outer_slice % num_kv_slabs == 0,
            "high_bw_all_gather: {} pages per outer slice do not split into {} KV slabs",
            geometry.pages_per_outer_slice,
            num_kv_slabs);
        geometry.pages_per_kv_slab = geometry.pages_per_outer_slice / num_kv_slabs;
    }
    return geometry;
}

uint32_t host_valid_pages_per_outer_slice(const HighBwAllGatherParams& args, const ChipShardPageGeometry& geometry) {
    if (!args.gathered_dim_size.has_value()) {
        return geometry.pages_per_outer_slice;
    }
    const uint64_t valid_local_length = *args.gathered_dim_size / args.num_devices;
    TT_FATAL(
        valid_local_length % geometry.gather_dim_elements_per_page == 0,
        "high_bw_all_gather: the valid length {} per chip must be a multiple of the {} elements a page spans (tile "
        "alignment)",
        valid_local_length,
        geometry.gather_dim_elements_per_page);
    const uint64_t pages = valid_local_length * geometry.pages_per_outer_slice;
    TT_FATAL(
        pages % geometry.local_gather_dim_length == 0,
        "high_bw_all_gather: gathered_dim_size {} does not cover whole pages",
        *args.gathered_dim_size);
    return static_cast<uint32_t>(pages / geometry.local_gather_dim_length);
}

uint32_t host_cache_slot_first_page(const HighBwAllGatherParams& args, const ChipShardPageGeometry& geometry) {
    return args.input_batch_index.value_or(0) * geometry.pages_per_cache_slot;
}

// Common args of the readers and local copy writers (first) and of the senders (second); the same for every program
// of the mesh workload. Each kernel's own args come first, then the chip shard geometry block (read by
// read_chip_shard_geometry in kernels/fabric/fabric_all_gather_common.hpp): reader [0] input [1] output [2]
// shards-arrived counter [3] cache slot tensor [4] layers per user [5] layer [6] cache slot first page (host) [7] pages
// per cache slot, geometry at 8; sender [0] output [1] shards-arrived counter [2] downstream-started counter, geometry
// at 3. The chip shard geometry: [0] prefix tensor address [1] valid pages per outer slice (host) [2] outer slices [3]
// pages per outer slice [4] ranks [5] KV slab [6] full gathered length [7] pages per KV slab.
constexpr uint32_t kReaderInputAddressArg = 0;
constexpr uint32_t kReaderOutputAddressArg = 1;
constexpr uint32_t kReaderShardsArrivedCounterArg = 2;
constexpr uint32_t kReaderBatchIndexAddressArg = 3;
constexpr uint32_t kReaderLayersPerUserArg = 4;
constexpr uint32_t kReaderLayerArg = 5;
constexpr uint32_t kReaderCacheSlotFirstPageArg = 6;
constexpr uint32_t kReaderGeometryArg = 8;
constexpr uint32_t kSenderOutputAddressArg = 0;
constexpr uint32_t kSenderShardsArrivedCounterArg = 1;
constexpr uint32_t kSenderDownstreamStartedCounterArg = 2;
constexpr uint32_t kSenderGeometryArg = 3;
constexpr uint32_t kGeometryPrefixAddressArg = 0;
constexpr uint32_t kGeometryValidPagesPerOuterSliceArg = 1;

// Kernel indices: the order create_workload_descriptor pushes the kernels (a descriptor kernel's index is its handle).
constexpr uint32_t kFabricLinkWorkerReaderKernel = 0;
constexpr uint32_t kFabricLinkWorkerSenderKernel = 1;
constexpr uint32_t kLocalCopyCoreReaderKernel = 2;
constexpr uint32_t kLocalCopyCoreWriterKernel = 3;

// Positions of the two counters in WorkloadDescriptor::semaphores.
constexpr size_t kDownstreamStartedCounterSlot = 0;
constexpr size_t kShardsArrivedCounterSlot = 1;

// The common args derived from hash-excluded attributes (input_batch_index, gathered_dim_size, batch_slot_num_layers,
// batch_slot_layer_idx). The cache-miss build and the cache-hit patch both take them from here.
struct HashExcludedCommonArgs {
    uint32_t layers_per_user = 1;
    uint32_t layer = 0;
    uint32_t cache_slot_first_page = 0;
    uint32_t valid_pages_per_outer_slice = 0;
};

HashExcludedCommonArgs hash_excluded_common_args(
    const HighBwAllGatherParams& args, const ChipShardPageGeometry& geometry) {
    return HashExcludedCommonArgs{
        .layers_per_user = args.batch_slot_num_layers,
        .layer = args.batch_slot_layer_idx,
        .cache_slot_first_page = host_cache_slot_first_page(args, geometry),
        .valid_pages_per_outer_slice = host_valid_pages_per_outer_slice(args, geometry)};
}

// Tensor addresses are Buffer* bindings, patched on every cache hit; an absent metadata tensor is a null Buffer* (0).
std::pair<tt::tt_metal::KernelDescriptor::RTArgList, tt::tt_metal::KernelDescriptor::RTArgList> common_args(
    const HighBwAllGatherParams& args,
    const HighBwAllGatherInputs& tensor_args,
    const Tensor& output,
    const ChipShardPageGeometry& geometry,
    uint32_t shards_arrived_counter_address,
    uint32_t downstream_started_counter_address) {
    const auto hash_excluded = hash_excluded_common_args(args, geometry);
    tt::tt_metal::Buffer* batch_index_buffer =
        tensor_args.has_batch_index_metadata() ? tensor_args.input_batch_index_tensor->buffer() : nullptr;
    tt::tt_metal::Buffer* prefix_buffer =
        tensor_args.has_gathered_prefix_metadata() ? tensor_args.gathered_prefix_tensor->buffer() : nullptr;

    tt::tt_metal::KernelDescriptor::RTArgList reader_common_args;
    reader_common_args.push_back(tensor_args.input_tensor.buffer());
    reader_common_args.push_back(output.buffer());
    // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor); a caller-supplied one is
    // re-applied on every hit by HighBwAllGatherMeshWorkloadFactory::override_runtime_arguments
    reader_common_args.push_back(shards_arrived_counter_address);
    reader_common_args.push_back(batch_index_buffer);
    reader_common_args.push_back(hash_excluded.layers_per_user);
    reader_common_args.push_back(hash_excluded.layer);
    reader_common_args.push_back(hash_excluded.cache_slot_first_page);
    reader_common_args.push_back(geometry.pages_per_cache_slot);

    tt::tt_metal::KernelDescriptor::RTArgList sender_common_args;
    sender_common_args.push_back(output.buffer());
    // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor); a caller-supplied one is
    // re-applied on every hit by HighBwAllGatherMeshWorkloadFactory::override_runtime_arguments
    sender_common_args.push_back(shards_arrived_counter_address);
    // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor); a caller-supplied one is
    // re-applied on every hit by HighBwAllGatherMeshWorkloadFactory::override_runtime_arguments
    sender_common_args.push_back(downstream_started_counter_address);

    // the chip shard geometry block, appended to both
    for (auto* common : {&reader_common_args, &sender_common_args}) {
        common->push_back(prefix_buffer);
        common->push_back(hash_excluded.valid_pages_per_outer_slice);
        common->push_back(geometry.num_outer_slices);
        common->push_back(geometry.pages_per_outer_slice);
        common->push_back(geometry.num_ranks);
        common->push_back(args.gathered_slab_global);
        common->push_back(static_cast<uint32_t>(tensor_args.input_tensor.padded_shape()[args.dim] * args.num_devices));
        common->push_back(geometry.pages_per_kv_slab);
    }
    return {std::move(reader_common_args), std::move(sender_common_args)};
}

CoreRangeSet resolve_fabric_worker_cores(MeshDevice* mesh_device, const HighBwAllGatherParams& args) {
    TT_FATAL(
        mesh_device->get_active_sub_device_manager_id() == args.subdevice_manager_id,
        "high_bw_all_gather active subdevice manager changed from {} to {} during operation dispatch",
        args.subdevice_manager_id,
        mesh_device->get_active_sub_device_manager_id());
    const auto subdevice_id = args.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto subdevice_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    TT_FATAL(
        subdevice_cores.contains(args.resolved_worker_core_grid),
        "high_bw_all_gather resolved worker grid {} must be fully contained in TENSIX subdevice {} cores {}",
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
        "high_bw_all_gather {} cores {} must cover all of the op's cores {}",
        name,
        cores,
        required_cores);
}

// The fabric link worker cores of a cached program (the cores that use the two global counters): the cores its
// fabric-link-worker reader kernel holds runtime args on, since create_workload_descriptor gives every link worker
// reader args and places that kernel on exactly those cores. Only read when a hit swaps the caller's semaphores.
CoreRangeSet fabric_link_worker_cores_of(const tt::tt_metal::Program& program) {
    std::vector<tt::tt_metal::CoreCoord> cores;
    const auto& runtime_args_by_core = tt::tt_metal::GetRuntimeArgs(program, kFabricLinkWorkerReaderKernel);
    for (size_t x = 0; x < runtime_args_by_core.size(); ++x) {
        for (size_t y = 0; y < runtime_args_by_core[x].size(); ++y) {
            if (runtime_args_by_core[x][y].size() > 0) {
                cores.emplace_back(x, y);
            }
        }
    }
    return CoreRangeSet(cores);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

using namespace CMAKE_UNIQUE_NAMESPACE;

tt::tt_metal::WorkloadDescriptor HighBwAllGatherFabricFactory::create_workload_descriptor(
    const HighBwAllGatherParams& args,
    const HighBwAllGatherInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const auto& input = tensor_args.input_tensor;
    auto* mesh_device = input.device();
    const auto subdevice_id = args.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = resolve_fabric_worker_cores(mesh_device, args);

    const bool input_interleaved = input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
    const GatherPlan plan = build_gather_plan(args, input, mesh_device, available_cores, input_interleaved);
    const bool selected_batch = args.input_batch_index.has_value() || tensor_args.has_batch_index_metadata();
    const auto chip_shard_geometry = derive_shard_page_geometry(args, tensor_args, selected_batch);

    // Shards-arrived counter (data_valid_semaphore) and downstream-started counter (ready_semaphore, the fence), at one
    // address on every core of every chip.
    const auto counter_buffer_type = ttnn::ccl::prefer_l1_small_buffer_type(*mesh_device);
    const bool external_counters = args.ready_semaphore.has_value();
    if (!external_counters) {
        ttnn::ccl::warn_if_semaphores_fall_back_to_l1(counter_buffer_type);
    }
    auto downstream_started_counter = external_counters ? *args.ready_semaphore
                                                        : ttnn::global_semaphore::create_global_semaphore(
                                                              mesh_device, available_cores, 0, counter_buffer_type);
    auto shards_arrived_counter = external_counters ? *args.data_valid_semaphore
                                                    : ttnn::global_semaphore::create_global_semaphore(
                                                          mesh_device, available_cores, 0, counter_buffer_type);
    if (external_counters) {
        for (const auto& chip_fabric_link_workers : plan.fabric_link_workers) {
            std::vector<tt::tt_metal::CoreCoord> cores;
            cores.reserve(chip_fabric_link_workers.size());
            for (const auto& fabric_link_worker : chip_fabric_link_workers) {
                cores.push_back(fabric_link_worker.core);
            }
            const CoreRangeSet core_set(cores);
            validate_semaphore_core_coverage(downstream_started_counter, core_set, "ready_semaphore");
            validate_semaphore_core_coverage(shards_arrived_counter, core_set, "data_valid_semaphore");
        }
    } else {
        // every chip's counters must be zero before any chip's first call can increment a neighbour's
        ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, subdevices);
    }
    // Park both counters on the descriptor so they live as long as the cached workload: the op's own, or copies of the
    // caller's (a cache hit that brings other caller semaphores replaces these, see override_runtime_arguments).
    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    static_assert(kDownstreamStartedCounterSlot == 0 && kShardsArrivedCounterSlot == 1);
    workload_descriptor.semaphores.push_back(downstream_started_counter);
    workload_descriptor.semaphores.push_back(shards_arrived_counter);

    const uint32_t page_bytes = input.mesh_buffer().aligned_page_size();
    const uint32_t fabric_payload_bytes = static_cast<uint32_t>(args.packet_size);
    // a page larger than the payload is a chunk of its own, sent as several packets
    const uint32_t pages_per_fabric_chunk = std::max<uint32_t>(1, fabric_payload_bytes / page_bytes);
    const uint32_t fabric_chunk_bytes = pages_per_fabric_chunk * page_bytes;
    const uint32_t chunks_per_cb_batch = std::max<uint32_t>(
        1, std::min<uint32_t>(kMaxChunksPerCbBatch, kFabricChunkCbBudgetBytes / (2 * fabric_chunk_bytes)));
    const uint32_t num_dram_banks = mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const uint32_t L = plan.num_links;
    // the link workers of one (ring, link) own chunk numbers ring * L + link, + num_rings * L, ... in each direction
    const uint32_t owned_chunk_stride = plan.num_rings * L;
    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const bool cache_slot_from_metadata = tensor_args.has_batch_index_metadata();
    const bool valid_prefix_from_metadata = tensor_args.has_gathered_prefix_metadata();

    // Compile-time args, see the kernels. Absent metadata tensors pass the output's accessor as a placeholder.
    const Tensor& batch_index_tensor = cache_slot_from_metadata ? *tensor_args.input_batch_index_tensor : output_tensor;
    const Tensor& prefix_tensor = valid_prefix_from_metadata ? *tensor_args.gathered_prefix_tensor : output_tensor;
    auto append_accessor = [](std::vector<uint32_t>& compile_args, const Tensor& tensor) {
        tt::tt_metal::TensorAccessorArgs(*tensor.buffer()).append_to(compile_args);
    };
    // blocks_converted_semaphore (per program, one per local copy core): conversion blocks of a non-interleaved input's
    // own shard that copy core has written into the output.
    const uint32_t num_local_copy_cores = plan.local_copy_cores[0].size();
    auto reader_compile_args = [&](bool local_copy_core, uint32_t blocks_converted_semaphore) {
        std::vector<uint32_t> compile_args{
            kFabricChunkCbIndex,
            kReaderMetadataCbIndex,
            page_bytes,
            pages_per_fabric_chunk,
            num_dram_banks,
            chunks_per_cb_batch,
            cache_slot_from_metadata ? 1u : 0u,
            valid_prefix_from_metadata ? 1u : 0u,
            input_interleaved ? 1u : 0u,
            local_copy_core ? 1u : 0u,
            blocks_converted_semaphore,
            num_local_copy_cores,
            kInputRunCbIndex};
        append_accessor(compile_args, input);
        append_accessor(compile_args, output_tensor);
        append_accessor(compile_args, batch_index_tensor);
        append_accessor(compile_args, prefix_tensor);
        return compile_args;
    };
    const auto reader_config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_0_default};
    std::vector<uint32_t> sender_compile_args{
        kFabricChunkCbIndex,
        kWriterMetadataCbIndex,
        page_bytes,
        pages_per_fabric_chunk,
        num_dram_banks,
        chunks_per_cb_batch,
        valid_prefix_from_metadata ? 1u : 0u,
        fabric_payload_bytes};
    append_accessor(sender_compile_args, output_tensor);
    append_accessor(sender_compile_args, prefix_tensor);
    // the sender and the local copy writer share this configuration
    const auto writer_config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_1_default};
    auto local_copy_writer_compile_args = [&](uint32_t blocks_converted_semaphore) {
        std::vector<uint32_t> compile_args{
            kFabricChunkCbIndex,
            kWriterMetadataCbIndex,
            page_bytes,
            pages_per_fabric_chunk,
            num_dram_banks,
            chunks_per_cb_batch,
            valid_prefix_from_metadata ? 1u : 0u,
            input_interleaved ? 1u : 0u,
            blocks_converted_semaphore,
            kInputRunCbIndex};
        append_accessor(compile_args, output_tensor);
        append_accessor(compile_args, prefix_tensor);
        return compile_args;
    };

    // the same for every program: one geometry and one pair of counters (global semaphores) for the whole mesh
    const auto [reader_common_args, sender_common_args] = common_args(
        args,
        tensor_args,
        output_tensor,
        chip_shard_geometry,
        shards_arrived_counter.address(),
        downstream_started_counter.address());
    // 1D fabric routes by hop count (always one), 2D by destination chip
    const bool is_2d = tt::tt_fabric::is_2d_fabric_config(args.fabric_config);

    workload_descriptor.programs.reserve(tensor_coords.coords().size());
    for (const auto& chip : tensor_coords.coords()) {
        const uint32_t chip_index = plan.linear_chip_index(chip);
        tt::tt_metal::ProgramDescriptor desc;
        const auto& fabric_link_workers = plan.fabric_link_workers[chip_index];
        const auto& local_copy_cores = plan.local_copy_cores[chip_index];

        std::vector<tt::tt_metal::CoreCoord> fabric_link_worker_cores;
        for (const auto& fabric_link_worker : fabric_link_workers) {
            fabric_link_worker_cores.push_back(fabric_link_worker.core);
        }
        std::vector<tt::tt_metal::CoreCoord> all_cores = fabric_link_worker_cores;
        all_cores.insert(all_cores.end(), local_copy_cores.begin(), local_copy_cores.end());
        const auto fabric_link_worker_core_set = CoreRangeSet(fabric_link_worker_cores);
        const auto local_copy_core_set = CoreRangeSet(local_copy_cores);
        const auto all_core_set = CoreRangeSet(all_cores);

        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = 2 * chunks_per_cb_batch * fabric_chunk_bytes,
            .core_ranges = all_core_set,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(kFabricChunkCbIndex),
                .data_format = data_format,
                .page_size = page_bytes}}}});
        for (const uint32_t metadata_cb : {kReaderMetadataCbIndex, kWriterMetadataCbIndex}) {
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = kMetadataCbBytes,
                .core_ranges = all_core_set,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(metadata_cb),
                    .data_format = tt::DataFormat::UInt32,
                    .page_size = kMetadataCbBytes}}}});
        }

        if (!input_interleaved) {
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = 2 * chunks_per_cb_batch * kInputRunCbPageBytes,
                .core_ranges = local_copy_core_set,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(kInputRunCbIndex),
                    .data_format = tt::DataFormat::UInt32,
                    .page_size = kInputRunCbPageBytes}}}});
        }
        // one per local copy core, consecutive ids: conversion blocks of the own shard that copy core has done
        // (non-interleaved input). They are this program's first semaphores, so they take ids 0, 1, ...; the fabric
        // connection semaphores appended per link worker below take the next free ids on that core.
        const uint32_t blocks_converted_semaphore = 0;
        for (uint32_t c = 0; c < num_local_copy_cores; ++c) {
            desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
                .id = blocks_converted_semaphore + c, .core_ranges = fabric_link_worker_core_set, .initial_value = 0});
        }
        const std::string dir(kKernelDir);
        auto make_kernel = [&](const char* source,
                               const CoreRangeSet& cores,
                               std::vector<uint32_t> compile_time_args,
                               const tt::tt_metal::DataMovementConfigDescriptor& config) {
            tt::tt_metal::KernelDescriptor kernel;
            kernel.kernel_source = dir + source;
            kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
            kernel.core_ranges = cores;
            kernel.compile_time_args = std::move(compile_time_args);
            kernel.config = config;
            return kernel;
        };
        // pushed in kernel-index order (kFabricLinkWorkerReaderKernel, ...); the override relies on it
        desc.kernels.push_back(make_kernel(
            "fabric_all_gather_reader.cpp",
            fabric_link_worker_core_set,
            reader_compile_args(false, blocks_converted_semaphore),
            reader_config));
        desc.kernels.push_back(make_kernel(
            "fabric_all_gather_sender.cpp", fabric_link_worker_core_set, sender_compile_args, writer_config));
        desc.kernels.push_back(make_kernel(
            "fabric_all_gather_reader.cpp",
            local_copy_core_set,
            reader_compile_args(true, blocks_converted_semaphore),
            reader_config));
        desc.kernels.push_back(make_kernel(
            "fabric_all_gather_copy_writer.cpp",
            local_copy_core_set,
            local_copy_writer_compile_args(blocks_converted_semaphore),
            writer_config));
        auto& fabric_link_worker_reader = desc.kernels[kFabricLinkWorkerReaderKernel];
        auto& fabric_link_worker_sender = desc.kernels[kFabricLinkWorkerSenderKernel];
        auto& local_copy_core_reader = desc.kernels[kLocalCopyCoreReaderKernel];
        auto& local_copy_core_writer = desc.kernels[kLocalCopyCoreWriterKernel];

        fabric_link_worker_reader.emplace_common_runtime_args(reader_common_args);
        fabric_link_worker_sender.emplace_common_runtime_args(sender_common_args);
        local_copy_core_reader.emplace_common_runtime_args(reader_common_args);
        local_copy_core_writer.emplace_common_runtime_args(reader_common_args);

        for (uint32_t ring = 0; ring < plan.num_rings; ++ring) {
            const auto& on_ring = plan.chip_on_ring[chip_index][ring];
            for (uint32_t direction = 0; direction < 2; ++direction) {
                const auto downstream = on_ring.neighbour(direction);
                const auto upstream = on_ring.upstream(direction);
                const auto& outgoing_shards = on_ring.outgoing_shards(direction);
                const uint32_t shards_expected_from_upstream =
                    upstream
                        ? plan.chip_on_ring[plan.linear_chip_index(*upstream)][ring].outgoing_shards(direction).size()
                        : 0;
                const bool signal_started_to_downstream =
                    signals_started_to_downstream(plan, chip_index, ring, direction);
                for (uint32_t link = 0; link < L; ++link) {
                    const auto& fabric_link_worker =
                        fabric_link_workers[fabric_link_worker_index(ring, direction, link, L)];
                    const uint32_t first_owned_chunk = ring * L + link;

                    std::vector<uint32_t> reader_args{
                        first_owned_chunk, owned_chunk_stride, static_cast<uint32_t>(outgoing_shards.size())};
                    reader_args.insert(reader_args.end(), outgoing_shards.begin(), outgoing_shards.end());
                    fabric_link_worker_reader.runtime_args.emplace_back(
                        fabric_link_worker.core, std::move(reader_args));

                    // the started signal goes to the downstream's link worker that sends back into this chip; data and
                    // shards-arrived increments go to the downstream's link worker of the same direction
                    tt::tt_metal::CoreCoord started_signal_target{0, 0}, downstream_worker{0, 0};
                    tt::tt_fabric::FabricNodeId downstream_node = mesh_device->get_fabric_node_id(chip);
                    if (downstream) {
                        auto* downstream_device = tt::tt_metal::internal::get_device(*mesh_device, *downstream);
                        const auto& downstream_workers = plan.fabric_link_workers[plan.linear_chip_index(*downstream)];
                        started_signal_target = downstream_device->worker_core_from_logical_core(
                            downstream_workers[fabric_link_worker_index(ring, 1 - direction, link, L)].core);
                        downstream_worker = downstream_device->worker_core_from_logical_core(
                            downstream_workers[fabric_link_worker_index(ring, direction, link, L)].core);
                        downstream_node = mesh_device->get_fabric_node_id(*downstream);
                    }
                    std::vector<uint32_t> sender_args{
                        first_owned_chunk,
                        owned_chunk_stride,
                        signal_started_to_downstream ? 1u : 0u,
                        static_cast<uint32_t>(started_signal_target.x),
                        static_cast<uint32_t>(started_signal_target.y),
                        static_cast<uint32_t>(downstream_worker.x),
                        static_cast<uint32_t>(downstream_worker.y),
                        static_cast<uint32_t>(*downstream_node.mesh_id),
                        is_2d ? downstream_node.chip_id : 1u,
                        shards_expected_from_upstream,
                        static_cast<uint32_t>(outgoing_shards.size())};
                    sender_args.insert(sender_args.end(), outgoing_shards.begin(), outgoing_shards.end());
                    if (fabric_link_worker.has_fabric_connection) {
                        tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                            *mesh_device,
                            mesh_device->get_fabric_node_id(chip),
                            mesh_device->get_fabric_node_id(*downstream),
                            fabric_link_worker.fabric_link_index,
                            desc,
                            fabric_link_worker.core,
                            sender_args);
                    }
                    fabric_link_worker_sender.runtime_args.emplace_back(
                        fabric_link_worker.core, std::move(sender_args));
                }
            }
        }
        // A cache hit that swaps in caller semaphores reads the link worker cores back from which cores have reader
        // args (fabric_link_worker_cores_of), so every link worker must get a non-empty reader arg list.
        TT_FATAL(
            fabric_link_worker_reader.runtime_args.size() == fabric_link_workers.size() &&
                std::all_of(
                    fabric_link_worker_reader.runtime_args.begin(),
                    fabric_link_worker_reader.runtime_args.end(),
                    [](const auto& core_args) { return !core_args.second.empty(); }),
            "high_bw_all_gather: every fabric link worker needs reader runtime args ({} workers, {} arg lists)",
            fabric_link_workers.size(),
            fabric_link_worker_reader.runtime_args.size());
        // Local copy core c of n owns chunk numbers c, c + n, ... of this chip's shard (interleaved input), or
        // conversion blocks c, c + n, ... of it (non-interleaved input, for_each_contiguous_input_run); the latter
        // signals every fabric link worker of this chip per block.
        const uint32_t rank = plan.rank_of_chip[chip_index];
        auto* device = tt::tt_metal::internal::get_device(*mesh_device, chip);
        std::vector<uint32_t> local_copy_writer_args{
            0, num_local_copy_cores, rank, static_cast<uint32_t>(fabric_link_workers.size())};
        for (const auto& fabric_link_worker : fabric_link_workers) {
            const auto noc = device->worker_core_from_logical_core(fabric_link_worker.core);
            local_copy_writer_args.push_back(noc.x);
            local_copy_writer_args.push_back(noc.y);
        }
        for (uint32_t c = 0; c < num_local_copy_cores; ++c) {
            local_copy_core_reader.runtime_args.emplace_back(
                local_copy_cores[c],
                tt::tt_metal::KernelDescriptor::CoreRuntimeArgs{
                    c, num_local_copy_cores, 1, chunk_walk::make_outgoing_shard(rank, chunk_walk::kWholeChipShard)});
            local_copy_writer_args[0] = c;
            local_copy_core_writer.runtime_args.emplace_back(local_copy_cores[c], local_copy_writer_args);
        }

        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(chip), std::move(desc)});
    }
    return workload_descriptor;
}

HighBwAllGatherMeshWorkloadFactory::cached_mesh_workload_t HighBwAllGatherMeshWorkloadFactory::create_mesh_workload(
    const HighBwAllGatherParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const HighBwAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    return descriptor_adapter_t::create_mesh_workload(args, tensor_coords, tensor_args, output_tensor);
}

void HighBwAllGatherMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const HighBwAllGatherParams& args,
    const HighBwAllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    // Between calls only the common args change (addresses, cache slot, valid length, and caller semaphores), and they
    // are the same for every program. The tensor addresses are Buffer* bindings; the rest is patched here.
    const bool external_counters = args.ready_semaphore.has_value();
    const auto swaps_counters = [&](const descriptor_adapter_t::shared_variables_t& shared) {
        const auto& parked = shared.workload_descriptor.semaphores;
        return external_counters &&
               (parked.at(kDownstreamStartedCounterSlot).address() != args.ready_semaphore->address() ||
                parked.at(kShardsArrivedCounterSlot).address() != args.data_valid_semaphore->address());
    };
    // The hash-excluded values, the same for every program. Derived (and their bounds checked) before anything is
    // patched, as is the semaphore check below: a failed check then leaves the workload as it was.
    const bool selected_batch = args.input_batch_index.has_value() || tensor_args.has_batch_index_metadata();
    const auto hash_excluded =
        hash_excluded_common_args(args, derive_shard_page_geometry(args, tensor_args, selected_batch));
    // Validate every program before patching any.
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        if (swaps_counters(cached_workload.shared_variables.at(coordinate_range))) {
            const auto fabric_link_worker_cores = fabric_link_worker_cores_of(program);
            validate_semaphore_core_coverage(*args.ready_semaphore, fabric_link_worker_cores, "ready_semaphore");
            validate_semaphore_core_coverage(
                *args.data_valid_semaphore, fabric_link_worker_cores, "data_valid_semaphore");
        }
    }

    // Default adapter behaviour: patch the Buffer* bindings (input, output, cache slot and prefix tensors). When the
    // cache-filling call passed one buffer in two tensor slots, the adapter could not tell which slot holds which
    // buffer, recorded no bindings, and would refuse this hit; the addresses are then written directly below instead,
    // as the legacy factory always did.
    const bool bindings_resolved = std::all_of(
        cached_workload.shared_variables.begin(), cached_workload.shared_variables.end(), [](const auto& entry) {
            return !entry.second.resolved_bindings.empty();
        });
    if (bindings_resolved) {
        descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output_tensor);
    }
    const auto address_of = [](const std::optional<Tensor>& tensor) -> uint32_t {
        return tensor.has_value() ? static_cast<uint32_t>(tensor->mesh_buffer().address()) : 0u;
    };
    const uint32_t input_address = static_cast<uint32_t>(tensor_args.input_tensor.mesh_buffer().address());
    const uint32_t output_address = static_cast<uint32_t>(output_tensor.mesh_buffer().address());
    const uint32_t batch_index_address =
        tensor_args.has_batch_index_metadata() ? address_of(tensor_args.input_batch_index_tensor) : 0u;
    const uint32_t prefix_address =
        tensor_args.has_gathered_prefix_metadata() ? address_of(tensor_args.gathered_prefix_tensor) : 0u;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared = cached_workload.shared_variables.at(coordinate_range);
        if (swaps_counters(shared)) {
            // keep the caller's current semaphores alive with the cached workload, as the op's own are
            shared.workload_descriptor.semaphores.at(kDownstreamStartedCounterSlot) = *args.ready_semaphore;
            shared.workload_descriptor.semaphores.at(kShardsArrivedCounterSlot) = *args.data_valid_semaphore;
        }
        // The parked counters are the ones the programs must use: unchanged for the op's own (workload-scoped, so
        // rewriting them is a no-op), the call's for caller-supplied ones.
        const uint32_t shards_arrived_counter_address =
            static_cast<uint32_t>(shared.workload_descriptor.semaphores.at(kShardsArrivedCounterSlot).address());
        const uint32_t downstream_started_counter_address =
            static_cast<uint32_t>(shared.workload_descriptor.semaphores.at(kDownstreamStartedCounterSlot).address());
        auto patch_reader_common = [&](uint32_t kernel) {
            auto& common = GetCommonRuntimeArgs(program, kernel);
            if (!bindings_resolved) {
                common[kReaderInputAddressArg] = input_address;
                common[kReaderOutputAddressArg] = output_address;
                common[kReaderBatchIndexAddressArg] = batch_index_address;
                common[kReaderGeometryArg + kGeometryPrefixAddressArg] = prefix_address;
            }
            common[kReaderShardsArrivedCounterArg] = shards_arrived_counter_address;
            common[kReaderLayersPerUserArg] = hash_excluded.layers_per_user;
            common[kReaderLayerArg] = hash_excluded.layer;
            common[kReaderCacheSlotFirstPageArg] = hash_excluded.cache_slot_first_page;
            common[kReaderGeometryArg + kGeometryValidPagesPerOuterSliceArg] =
                hash_excluded.valid_pages_per_outer_slice;
        };
        patch_reader_common(kFabricLinkWorkerReaderKernel);
        patch_reader_common(kLocalCopyCoreReaderKernel);
        patch_reader_common(kLocalCopyCoreWriterKernel);  // the local copy writer takes the reader's common args
        auto& sender_common = GetCommonRuntimeArgs(program, kFabricLinkWorkerSenderKernel);
        if (!bindings_resolved) {
            sender_common[kSenderOutputAddressArg] = output_address;
            sender_common[kSenderGeometryArg + kGeometryPrefixAddressArg] = prefix_address;
        }
        sender_common[kSenderShardsArrivedCounterArg] = shards_arrived_counter_address;
        sender_common[kSenderDownstreamStartedCounterArg] = downstream_started_counter_address;
        sender_common[kSenderGeometryArg + kGeometryValidPagesPerOuterSliceArg] =
            hash_excluded.valid_pages_per_outer_slice;
    }
}

}  // namespace ttnn::operations::experimental::quasar::high_bw_all_gather
