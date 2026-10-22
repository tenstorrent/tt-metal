// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_program_factory.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/indestructible.hpp>

#include "kernels/matmul_reduce_scatter_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast_host.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

using namespace tt::tt_metal;
using namespace ::matmul_reduce_scatter;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;

namespace {
namespace factory_impl {

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/matmul_reduce_scatter/device/kernels/";

constexpr uint32_t kMailboxStride = 16;  // bytes per matmul core in a port's mailbox
constexpr uint32_t kPortGroup = 4;       // chunks per port batch (cb PORT_CHUNKS holds 2 batches)
constexpr uint32_t kPortChunkBatches = 2;
constexpr uint32_t kPacketHeaderPool = 8;  // port sender: kPortGroup data headers + 1 credit header
static_assert(kPortGroup + 1 <= kPacketHeaderPool, "port sender headers: data headers + 1 credit header (pool of 8)");
constexpr uint32_t kPrefetchSteps = 1;  // K-block windows a sender reads ahead of its multicast
constexpr uint32_t kNumWaves = 1;       // one wave per block: a wave is the whole per-core block
constexpr uint32_t kIncEveryWave = 1;   // every unit's last chunk per core carries the arrival increment
constexpr uint32_t kFwd = 0;            // direction index: fwd
constexpr uint32_t kBwd = 1;            // direction index: bwd

// Per-regime fill levers (measured on the resident focus case and the stream MiMo case).
struct RegimeLevers {
    uint32_t lookahead_after_barrier;
    uint32_t a_read_noc;
    uint32_t w_read_noc;
    uint32_t senders_per_line;
    bool disjoint_senders;  // diagonal_disjoint (else diagonal)
};
constexpr RegimeLevers kResidentLevers{
    .lookahead_after_barrier = 1, .a_read_noc = 0, .w_read_noc = 1, .senders_per_line = 2, .disjoint_senders = true};
constexpr RegimeLevers kStreamLevers{
    .lookahead_after_barrier = 0, .a_read_noc = 0, .w_read_noc = 0, .senders_per_line = 1, .disjoint_senders = false};

constexpr std::array<uint32_t, 4> kRoleOfMask{role::COPY, role::ADD_FWD, role::ADD_BWD, role::ADD_BOTH};

std::string kernel_path(const char* name) { return std::string(kKernelDir) + name; }

uint32_t stage_tile_bytes() { return tt::tile_size(tt::DataFormat::Float16_b); }

// Never destroyed: its tensors and semaphores would otherwise be freed by a static destructor at process exit,
// after the runtime they belong to has been torn down.
std::map<int, MeshResources>& resource_registry() {
    static ttsl::Indestructible<std::map<int, MeshResources>> registry;
    return registry.get();
}

// Byte offsets of the forward / backward arrival zones inside a core's staging slice.
std::array<uint32_t, 2> zone_offsets(const Plan& plan) {
    const uint32_t slot_bytes = plan.P * stage_tile_bytes();
    const uint32_t fwd = kDepthHandoff * slot_bytes;
    return {fwd, fwd + (plan.zone_sets * plan.slots_fwd * slot_bytes)};
}

uint32_t global_sem_address(const MeshResources& res, GlobalSem which) {
    return static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(
        *res.semaphores[static_cast<uint32_t>(which)]));  // smuggled-rta-ok: persistent GlobalSemaphore
}

// The matmul grid sits beside the transport strip: row strip -> from logical row 1, column strip -> from (0, 0).
CoreCoord mm_origin(const GridPlan& grid) { return grid.row_strip ? CoreCoord{0, 1} : CoreCoord{0, 0}; }

CoreCoord mm_core(const GridPlan& grid, uint32_t gm, uint32_t gn) {
    const CoreCoord o = mm_origin(grid);
    return grid.m_on_x ? CoreCoord{o.x + gm, o.y + gn} : CoreCoord{o.x + gn, o.y + gm};
}

// k (the row-major matmul core index) of a matmul core.
uint32_t mm_index(const GridPlan& grid, const CoreCoord& core) {
    const CoreCoord o = mm_origin(grid);
    const auto dx = static_cast<uint32_t>(core.x - o.x);
    const auto dy = static_cast<uint32_t>(core.y - o.y);
    return grid.m_on_x ? (dx * grid.grid_n) + dy : (dy * grid.grid_n) + dx;
}

MeshCoordinate coord_at(const MeshCoordinate& coord, uint32_t axis, uint32_t pos) {
    std::array<uint32_t, 2> v{coord[0], coord[1]};
    v[axis] = pos;
    return MeshCoordinate(v[0], v[1]);
}

using Peers = std::array<std::optional<MeshCoordinate>, 2>;  // [fwd, bwd]

Peers peers_of(const MeshCoordinate& coord, uint32_t axis, uint32_t group_size, bool ring) {
    const uint32_t p = coord[axis];
    Peers peers;
    if (ring || p + 1 < group_size) {
        peers[kFwd] = coord_at(coord, axis, (p + 1) % group_size);
    }
    if (ring || p > 0) {
        peers[kBwd] = coord_at(coord, axis, (p + group_size - 1) % group_size);
    }
    return peers;
}

using PortKey = std::pair<uint32_t, uint32_t>;  // (direction, link)
using PortMap = std::map<PortKey, CoreCoord>;

// Ports in the transport strip, evenly spread (the fallback placement).
PortMap place_ports_in_strip(bool row_strip, uint32_t gx, uint32_t num_links) {
    PortMap cores;
    const uint32_t n = 2 * num_links;
    for (uint32_t i = 0; i < n; ++i) {
        const PortKey key{i % 2, i / 2};
        if (row_strip) {
            cores[key] = CoreCoord{std::min(gx - 1, (((2 * i) + 1) * gx) / (2 * n)), 0};
        } else {
            cores[key] = CoreCoord{gx - 1, i};
        }
    }
    return cores;
}

// Each linked port at the strip cell with the fewest NoC1 hops (-Y then -X, wrapping, in physical coordinates) to its
// link's Ethernet core, ties to the lower row then the lower column; receive-only ends take the first free strip cell.
// The hops are counted to the worker nearest that Ethernet core instead: no worker sits in the rows / columns between
// the two, so every cell's count differs from its count to the Ethernet core by the same amount and the ranking is the
// same.
PortMap place_ports(
    MeshDevice* mesh,
    const MeshCoordinate& coord,
    const GridPlan& grid,
    uint32_t gx,
    uint32_t gy,
    uint32_t num_links,
    const Peers& peers) {
    std::vector<std::pair<uint32_t, uint32_t>> allowed;
    if (grid.row_strip) {
        for (uint32_t x = 0; x < gx; ++x) {
            allowed.emplace_back(x, 0);
        }
    } else {
        for (uint32_t y = 0; y < gy; ++y) {
            allowed.emplace_back(gx - 1, y);
        }
    }
    IDevice* device = mesh->get_device(coord);
    const auto src = mesh->get_fabric_node_id(coord);
    PortMap cores;
    std::set<std::pair<uint32_t, uint32_t>> taken;
    for (uint32_t l = 0; l < num_links; ++l) {
        for (uint32_t d : {kFwd, kBwd}) {
            if (!peers[d].has_value()) {
                continue;
            }
            const auto dst = mesh->get_fabric_node_id(*peers[d]);
            const auto links = tt::tt_fabric::get_forwarding_link_indices(src, dst);
            TT_FATAL(l < links.size(), "matmul_reduce_scatter: hop has {} link(s), link {} requested", links.size(), l);
            const CoreCoord nearest =
                tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(
                    *device, tt::tt_fabric::get_forwarding_eth_core(src, dst, links[l]), NOC::NOC_1)
                    .logical_coord;
            std::optional<std::tuple<uint32_t, size_t, size_t>> best_score;
            std::optional<std::pair<uint32_t, uint32_t>> best;
            for (const auto& cell : allowed) {
                if (taken.contains(cell)) {
                    continue;
                }
                const CoreCoord logical{cell.first, cell.second};
                const CoreCoord v = device->worker_core_from_logical_core(logical);
                const std::tuple<uint32_t, size_t, size_t> score{
                    tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                        mesh, coord, logical, nearest, NOC::NOC_1),
                    v.y,
                    v.x};
                if (!best_score.has_value() || score < *best_score) {
                    best_score = score;
                    best = cell;
                }
            }
            if (!best.has_value()) {
                return place_ports_in_strip(grid.row_strip, gx, num_links);
            }
            cores[{d, l}] = CoreCoord{best->first, best->second};
            taken.insert(*best);
        }
    }
    for (const auto& [key, unused] : place_ports_in_strip(grid.row_strip, gx, num_links)) {
        if (cores.contains(key)) {
            continue;
        }
        const auto free =
            std::find_if(allowed.begin(), allowed.end(), [&](const auto& c) { return !taken.contains(c); });
        TT_FATAL(free != allowed.end(), "matmul_reduce_scatter: the transport strip has no free cell for a port");
        cores[key] = CoreCoord{free->first, free->second};
        taken.insert(*free);
    }
    return cores;
}

RoleKind role_of_direction(uint32_t d) { return d == kFwd ? RoleKind::FWD : RoleKind::BWD; }

uint32_t position_in(const std::vector<uint32_t>& list, uint32_t value) {
    const auto it = std::find(list.begin(), list.end(), value);
    TT_FATAL(it != list.end(), "matmul_reduce_scatter: block {} is not an arrival of the downstream chip", value);
    return static_cast<uint32_t>(it - list.begin());
}

struct UnitWalk {
    uint32_t entry;
    std::array<uint32_t, 2> blocks;
    uint32_t mask;
    uint32_t h1_first;
};

// Port (d, l)'s units on a chip: one per entry index whose halves (of the link) send in direction d.
std::vector<UnitWalk> unit_walks(const ChipSchedule& sc, uint32_t d, uint32_t link_halves) {
    std::vector<UnitWalk> out;
    for (uint32_t e = 0; e < sc.entries[0].size(); ++e) {
        const std::array<uint32_t, 2> blk{sc.entries[0][e], sc.entries[1][e]};
        uint32_t mask = 0;
        for (uint32_t h = 0; h < 2; ++h) {
            if (((link_halves >> h) & 1u) != 0 && sc.roles[blk[h]][h].kind == role_of_direction(d)) {
                mask |= 1u << h;
            }
        }
        if (mask == 0) {
            continue;
        }
        // walk a chain-start half (ready right after its matmul) before a relay half (waits for its arrival)
        const bool h1_first = mask == 3 && sc.roles[blk[1]][1].is_start && !sc.roles[blk[0]][0].is_start;
        out.push_back(UnitWalk{e, blk, mask, h1_first ? 1u : 0u});
    }
    return out;
}

struct McastFamilyPlan {
    const char* prefix;
    ttnn::kernel_lib::host::Mcast1DShape shape;
    std::optional<uint32_t> start;  // absent family when empty
};

// Senders per line, diagonal: A lines (fixed gm, length grid_n) send from gn = (start + gm) % grid_n, W lines
// (fixed gn, length grid_m) from gm = (start + gn) % grid_m; part 1 starts half a line further. Disjoint: each W
// family's start overlaps the senders already placed the least.
std::array<McastFamilyPlan, 4> plan_mcast_senders(
    bool disjoint,
    uint32_t grid_m,
    uint32_t grid_n,
    bool a_two_parts,
    bool w_two_parts,
    ttnn::kernel_lib::host::Mcast1DShape a_shape,
    ttnn::kernel_lib::host::Mcast1DShape w_shape) {
    auto cores = [&](bool is_a, uint32_t start) {
        std::set<std::pair<uint32_t, uint32_t>> out;
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
    std::set<std::pair<uint32_t, uint32_t>> taken;
    std::array<McastFamilyPlan, 4> plan{
        McastFamilyPlan{"a", a_shape, std::nullopt},
        McastFamilyPlan{"a2", a_shape, std::nullopt},
        McastFamilyPlan{"b", w_shape, std::nullopt},
        McastFamilyPlan{"b2", w_shape, std::nullopt}};
    const std::array<bool, 4> active{true, a_two_parts, true, w_two_parts};
    for (uint32_t i = 0; i < plan.size(); ++i) {
        if (!active[i]) {
            continue;
        }
        const bool is_a = i < 2;
        const uint32_t line_len = is_a ? grid_n : grid_m;
        uint32_t start = (i % 2 == 0) ? 0 : line_len / 2;
        if (disjoint && !is_a) {
            const uint32_t nominal = start;
            std::optional<std::pair<size_t, uint32_t>> best;
            for (uint32_t s0 = 0; s0 < line_len; ++s0) {
                size_t overlap = 0;
                for (const auto& c : cores(false, s0)) {
                    overlap += taken.contains(c) ? 1 : 0;
                }
                const std::pair<size_t, uint32_t> key{overlap, (s0 + line_len - nominal) % line_len};
                if (!best.has_value() || key < *best) {
                    best = key;
                    start = s0;
                }
            }
        }
        const auto placed = cores(is_a, start);
        taken.insert(placed.begin(), placed.end());
        plan[i].start = start;
    }
    return plan;
}

CBDescriptor plain_cb(
    uint32_t index, uint32_t pages, uint32_t page_bytes, tt::DataFormat format, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = pages * page_bytes,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index), .data_format = format, .page_size = page_bytes}},
    };
}

CoreRangeSet merged(const std::vector<CoreRange>& ranges) { return CoreRangeSet(ranges).merge_ranges(); }

}  // namespace factory_impl
}  // namespace

using namespace factory_impl;

MeshResources& acquire_mesh_resources(MeshDevice* mesh, uint32_t staging_tiles) {
    MeshResources& res = resource_registry()[mesh->id()];
    const CoreCoord grid_size = mesh->compute_with_storage_grid_size();
    const CoreRangeSet grid(CoreRange(CoreCoord{0, 0}, CoreCoord{grid_size.x - 1, grid_size.y - 1}));
    const auto num_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
    if (res.mesh != mesh) {
        res = MeshResources{};
        res.mesh = mesh;
        for (auto& semaphore : res.semaphores) {
            semaphore = ttnn::global_semaphore::create_global_semaphore(mesh, grid, 0);
        }
        const uint32_t words = num_cores * kMailboxStride / sizeof(uint32_t);
        const MemoryConfig mailbox_config(
            TensorMemoryLayout::HEIGHT_SHARDED,
            BufferType::L1,
            ShardSpec(grid, {1, words}, ShardOrientation::ROW_MAJOR));
        res.mailbox = ttnn::zeros(
            ttnn::Shape({num_cores, words}), DataType::UINT32, Layout::ROW_MAJOR, std::ref(*mesh), mailbox_config);
        // every counter / mailbox is zero before any chip's first call
        tt::tt_metal::distributed::Synchronize(*mesh, std::nullopt, {});
    }
    if (res.staging_tiles != staging_tiles || !res.staging.has_value()) {
        if (res.staging.has_value()) {
            res.staging->deallocate(/*force=*/true);
            res.staging.reset();
        }
        const uint32_t tile = tt::constants::TILE_HEIGHT;
        const MemoryConfig staging_config(
            TensorMemoryLayout::HEIGHT_SHARDED,
            BufferType::L1,
            ShardSpec(grid, {staging_tiles * tile, tile}, ShardOrientation::ROW_MAJOR));
        const TensorSpec spec(
            ttnn::Shape({num_cores * staging_tiles * tile, tile}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), staging_config));
        res.staging = ttnn::create_device_tensor(spec, mesh);
        res.staging_tiles = staging_tiles;
    }
    return res;
}

const MeshResources& resources_of(const MeshDevice* mesh) {
    const auto& registry = resource_registry();
    const auto it = registry.find(mesh->id());
    TT_FATAL(
        it != registry.end() && it->second.mesh == mesh,
        "matmul_reduce_scatter: the mesh's resources were not created (acquire_mesh_resources)");
    return it->second;
}

ProgramDescriptor MatmulReduceScatterProgramFactory::create_descriptor(
    const MatmulReduceScatterParams& attrs,
    const MatmulReduceScatterInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "matmul_reduce_scatter: one program per mesh coordinate");
    TT_FATAL(
        cb::ZERO < hal::get_num_dataflow_buffers(),
        "matmul_reduce_scatter: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        cb::ZERO,
        hal::get_num_dataflow_buffers());
    const MeshCoordinate& coord = *mesh_dispatch_coordinate;
    const Tensor& a = tensor_args.input;
    const Tensor& w = tensor_args.weight;
    const Tensor& staging = tensor_args.staging;
    MeshDevice* mesh = a.device();
    const MeshResources& res = resources_of(mesh);
    const Plan& plan = attrs.plan;
    const GridPlan& gd = plan.grid;

    // ---- geometry ----
    const uint32_t tile = tt::constants::TILE_HEIGHT;
    const uint32_t axis = attrs.cluster_axis;
    const uint32_t G = mesh->shape()[axis];
    const uint32_t p = coord[axis];
    const bool ring = attrs.kind != ScheduleKind::LINE;
    const uint32_t L = attrs.num_links;
    const uint32_t Mt = a.logical_shape()[-2] / tile;
    const uint32_t Kt = a.logical_shape()[-1] / tile;
    const uint32_t Nt = w.logical_shape()[-1] / tile;
    const bool scatter_h = attrs.scatter_dim == -2;
    const uint32_t block_m = scatter_h ? Mt / G : Mt;
    const uint32_t block_n = scatter_h ? Nt : Nt / G;
    const CoreCoord grid_size = mesh->compute_with_storage_grid_size();
    const auto gx = static_cast<uint32_t>(grid_size.x);
    const auto gy = static_cast<uint32_t>(grid_size.y);
    const uint32_t core_m = gd.core_m;
    const uint32_t core_n = gd.core_n;
    const uint32_t grid_m = gd.grid_m;
    const uint32_t grid_n = gd.grid_n;
    const uint32_t P = plan.P;
    const uint32_t kb = plan.kb;
    const uint32_t win_m = core_m;  // one wave: the streamed windows are the whole per-core block
    const uint32_t win_n = core_n;
    const uint32_t num_entries = G * kNumWaves;
    const uint32_t stage_bytes = stage_tile_bytes();
    const uint32_t hold_bytes = tt::tile_size(tt::DataFormat::Float32);
    TT_FATAL(kb > 0 && Kt % kb == 0, "matmul_reduce_scatter: K-block {} does not divide Kt {}", kb, Kt);
    const uint32_t nkb = Kt / kb;

    const uint32_t wave_word = (core_m << 16) | (core_n << 24);  // row_off 0 | col_off 0 | rows | cols
    const uint32_t wave_mm_word = core_m | (core_n << 8) | (gd.sb_h << 16) | (gd.sb_w << 24);
    const uint32_t wave_tiles = core_m * core_n;

    const uint32_t n_cores = grid_m * grid_n;
    const auto half_of = [&](uint32_t k) -> uint32_t {
        const uint32_t gm = k / grid_n;
        const uint32_t gn = k % grid_n;
        switch (plan.half_split) {
            case HalfSplit::N: return gn >= tt::div_up(grid_n, 2) ? 1 : 0;
            case HalfSplit::M: return gm >= tt::div_up(grid_m, 2) ? 1 : 0;
            case HalfSplit::K: break;
        }
        return k < tt::div_up(n_cores, 2) ? 0 : 1;
    };
    std::vector<uint32_t> link_halves(L, 0);  // bit h: some core of link l is in half h
    for (uint32_t k = 0; k < n_cores; ++k) {
        link_halves[k % L] |= 1u << half_of(k);
    }
    const CoreRange mm_range(mm_core(gd, 0, 0), mm_core(gd, grid_m - 1, grid_n - 1));
    const CoreRangeSet mm_crs(mm_range);

    const Peers peers = peers_of(coord, axis, G, ring);
    const PortMap ports = place_ports(mesh, coord, gd, gx, gy, L, peers);
    const auto virt = [&](const CoreCoord& c) { return mesh->worker_core_from_logical_core(c); };
    const auto node = [&](const MeshCoordinate& c) { return mesh->get_fabric_node_id(c); };

    const RegimeLevers& levers = plan.resident ? kResidentLevers : kStreamLevers;
    const uint32_t a_split = (levers.senders_per_line >= 2 && win_m >= 2) ? tt::div_up(win_m, 2) : win_m;
    const uint32_t w_split = (levers.senders_per_line >= 2 && kb >= 2) ? tt::div_up(kb, 2) : kb;
    const bool rA = plan.resident && !scatter_h;
    const bool rW = plan.resident && scatter_h;
    const uint32_t a_pages = rA ? core_m * Kt : plan.depth_a * win_m * kb;
    const uint32_t w_pages = rW ? Kt * core_n : plan.depth_w * kb * win_n;
    const auto tA = static_cast<uint32_t>(a.buffer()->aligned_page_size());
    const auto tW = static_cast<uint32_t>(w.buffer()->aligned_page_size());

    const uint32_t num_chunks = plan.n_m * plan.n_n;
    const bool m_outer = !scatter_h;
    const uint32_t n_inner = m_outer ? plan.n_n : plan.n_m;
    const uint32_t credit_chunks = num_chunks > plan.zone_sets ? num_chunks - plan.zone_sets : 0;
    const std::array<uint32_t, 2> ring_slots{plan.zone_sets * plan.slots_fwd, plan.zone_sets * plan.slots_bwd};
    const std::array<uint32_t, 2> zone_off = zone_offsets(plan);
    const std::array<uint32_t, 2> arr_addr{
        global_sem_address(res, GlobalSem::ARR_FWD), global_sem_address(res, GlobalSem::ARR_BWD)};
    const uint32_t ready_addr = global_sem_address(res, GlobalSem::READY);
    const uint32_t credit_addr = global_sem_address(res, GlobalSem::CREDIT);
    const uint32_t staging_addr = staging.buffer()->address();  // smuggled-rta-ok: per-mesh staging; override rewrites
    const uint32_t mailbox_addr =
        tensor_args.mailbox.buffer()->address();  // smuggled-rta-ok: persistent per-mesh mailbox
    const auto chunk_tiles = static_cast<uint32_t>(
        std::max<size_t>(1, std::min<size_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / stage_bytes, P)));

    const TransportPlan transport = plan_transport(G, attrs.kind, plan.half_split != HalfSplit::K);
    const ChipSchedule& sc = transport.chips[p];

    ProgramDescriptor desc;

    // ---- semaphores: consumed_fwd, consumed_bwd, freed on every matmul and port core, before any other ----
    std::vector<CoreRange> all_ranges{mm_range};
    for (const auto& [key, core] : ports) {
        all_ranges.emplace_back(core, core);
    }
    const CoreRangeSet all_crs = merged(all_ranges);
    for (uint32_t expected : {sem::CONSUMED_FWD, sem::CONSUMED_BWD, sem::FREED}) {
        const auto id = desc.find_available_semaphore_id(mm_range.start_coord, tt::CoreType::WORKER);
        TT_FATAL(
            id.has_value() && *id == expected,
            "matmul_reduce_scatter: semaphore {} must take id {}",
            expected,
            expected);
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = *id, .core_type = tt::CoreType::WORKER, .core_ranges = all_crs, .initial_value = 0});
    }

    // per (direction of the arrival, link): matmul cores whose half receives that direction's arrivals
    std::map<PortKey, uint32_t> receivers;
    for (uint32_t k = 0; k < n_cores; ++k) {
        for (uint32_t d : {kFwd, kBwd}) {
            receivers[{d, k % L}] += sc.arrivals[half_of(k)][d].empty() ? 0 : 1;
        }
    }

    // ---- matmul kernels ----
    KernelDescriptor reader;
    reader.kernel_source = kernel_path("matmul_reduce_scatter_reader.cpp");
    reader.core_ranges = mm_crs;
    reader.compile_time_args.assign(reader_ct::COUNT, 0);
    {
        auto& ct = reader.compile_time_args;
        ct[reader_ct::CB_ACT_K_WINDOW] = cb::ACT_K_WINDOW;
        ct[reader_ct::CB_WEIGHT_K_WINDOW] = cb::WEIGHT_K_WINDOW;
        ct[reader_ct::CB_ARRIVALS_FWD] = cb::ARRIVALS_FWD;
        ct[reader_ct::CB_ARRIVALS_BWD] = cb::ARRIVALS_BWD;
        ct[reader_ct::CORE_M] = win_m;
        ct[reader_ct::CORE_N] = win_n;
        ct[reader_ct::KB] = kb;
        ct[reader_ct::NUM_K_BLOCKS] = nkb;
        ct[reader_ct::KT] = Kt;
        ct[reader_ct::NT] = Nt;
        ct[reader_ct::A_ROW_PER_BLOCK] = scatter_h ? block_m : 0;
        ct[reader_ct::W_COL_PER_BLOCK] = scatter_h ? 0 : block_n;
        ct[reader_ct::RESIDENT_A] = rA ? 1 : 0;
        ct[reader_ct::RESIDENT_W] = rW ? 1 : 0;
        ct[reader_ct::P] = P;
        ct[reader_ct::NUM_ENTRIES] = num_entries;
        ct[reader_ct::A_TILE_BYTES] = tA;
        ct[reader_ct::W_TILE_BYTES] = tW;
        ct[reader_ct::A_CAP_WINDOWS] = a_pages / (win_m * kb);
        ct[reader_ct::W_CAP_WINDOWS] = w_pages / (kb * win_n);
        ct[reader_ct::A_SPLIT] = a_split;
        ct[reader_ct::W_SPLIT] = w_split;
        ct[reader_ct::PREFETCH_STEPS] = kPrefetchSteps;
        ct[reader_ct::LOOKAHEAD_AFTER_BARRIER] = levers.lookahead_after_barrier;
        ct[reader_ct::A_READ_NOC] = levers.a_read_noc;
        ct[reader_ct::W_READ_NOC] = levers.w_read_noc;
        ct[reader_ct::NUM_WAVES] = kNumWaves;
        ct[reader_ct::NUM_CHUNKS] = num_chunks;
        ct[reader_ct::N_INNER] = n_inner;
        ct[reader_ct::M_OUTER] = m_outer ? 1 : 0;
        ct[reader_ct::CHUNK_M] = plan.chunk_m;
        ct[reader_ct::CHUNK_N] = plan.chunk_n;
        ct[reader_ct::BLOCK_M] = block_m;
        ct[reader_ct::BLOCK_N] = block_n;
    }
    TensorAccessorArgs(a.buffer()).append_to(reader.compile_time_args);
    TensorAccessorArgs(w.buffer()).append_to(reader.compile_time_args);
    {
        KernelDescriptor::RTArgList common;
        common.push_back(a.buffer());
        common.push_back(w.buffer());
        common.push_back(arr_addr[kFwd]);
        common.push_back(arr_addr[kBwd]);
        common.push_back(credit_chunks);
        common.push_back(wave_word);
        reader.emplace_common_runtime_args(common);
    }
    TT_FATAL(
        reader.common_runtime_args.size() == reader_common::WAVES + kNumWaves,
        "matmul_reduce_scatter: reader common args out of sync with matmul_reduce_scatter_args.hpp");
    reader.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_0_default};

    KernelDescriptor compute;
    compute.kernel_source = kernel_path("matmul_reduce_scatter_compute.cpp");
    compute.core_ranges = mm_crs;
    compute.compile_time_args.assign(compute_ct::COUNT, 0);
    {
        auto& ct = compute.compile_time_args;
        ct[compute_ct::CB_ACT_K_WINDOW] = cb::ACT_K_WINDOW;
        ct[compute_ct::CB_WEIGHT_K_WINDOW] = cb::WEIGHT_K_WINDOW;
        ct[compute_ct::CB_PARTIAL_HOLD] = cb::PARTIAL_HOLD;
        ct[compute_ct::CB_FINISHED] = cb::FINISHED;
        ct[compute_ct::CB_ARRIVALS_FWD] = cb::ARRIVALS_FWD;
        ct[compute_ct::CB_ARRIVALS_BWD] = cb::ARRIVALS_BWD;
        ct[compute_ct::WIN_N] = win_n;
        ct[compute_ct::KB] = kb;
        ct[compute_ct::NUM_K_BLOCKS] = nkb;
        ct[compute_ct::P] = P;
        ct[compute_ct::NUM_ENTRIES] = num_entries;
        ct[compute_ct::NUM_WAVES] = kNumWaves;
        ct[compute_ct::SUPER_BLOCK] = plan.super_block;
        ct[compute_ct::CB_ZERO] = cb::ZERO;
        ct[compute_ct::NUM_CHUNKS] = num_chunks;
        ct[compute_ct::N_INNER] = n_inner;
        ct[compute_ct::RESIDENT] = plan.resident ? 1 : 0;
    }
    compute.emplace_common_runtime_args({wave_mm_word});
    TT_FATAL(
        compute.common_runtime_args.size() == compute_common::WAVES + kNumWaves,
        "matmul_reduce_scatter: compute common args out of sync with matmul_reduce_scatter_args.hpp");
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = attrs.math_fidelity,
        .fp32_dest_acc_en = true,  // the cross-device adds accumulate in fp32 DEST
        .dst_full_sync_en = true,  // 8-tile fp32 subblocks
        .math_approx_mode = attrs.math_approx_mode,
    };

    KernelDescriptor writer;
    writer.kernel_source = kernel_path("matmul_reduce_scatter_writer.cpp");
    writer.core_ranges = mm_crs;
    writer.compile_time_args.assign(writer_ct::COUNT, 0);
    {
        auto& ct = writer.compile_time_args;
        ct[writer_ct::CB_FINISHED] = cb::FINISHED;
        ct[writer_ct::P] = P;
        ct[writer_ct::NUM_WAVES] = kNumWaves;
        ct[writer_ct::OUT_N_TILES] = block_n;
        ct[writer_ct::DEPTH_HANDOFF] = kDepthHandoff;
        ct[writer_ct::TILE_BYTES] = stage_bytes;
        ct[writer_ct::NUM_ENTRIES] = num_entries;
        ct[writer_ct::CB_ZERO] = cb::ZERO;
        ct[writer_ct::NUM_CHUNKS] = num_chunks;
        ct[writer_ct::N_INNER] = n_inner;
        ct[writer_ct::M_OUTER] = m_outer ? 1 : 0;
        ct[writer_ct::CHUNK_M] = plan.chunk_m;
        ct[writer_ct::CHUNK_N] = plan.chunk_n;
        ct[writer_ct::BLOCK_M] = block_m;
        ct[writer_ct::BLOCK_N] = block_n;
    }
    TensorAccessorArgs(output.buffer()).append_to(writer.compile_time_args);
    {
        KernelDescriptor::RTArgList common;
        common.push_back(output.buffer());
        common.push_back(wave_word);
        writer.emplace_common_runtime_args(common);
    }
    TT_FATAL(
        writer.common_runtime_args.size() == writer_common::WAVES + kNumWaves,
        "matmul_reduce_scatter: writer common args out of sync with matmul_reduce_scatter_args.hpp");
    writer.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_1_default};

    // ---- matmul cores: per-core runtime args ----
    for (uint32_t gm = 0; gm < grid_m; ++gm) {
        for (uint32_t gn = 0; gn < grid_n; ++gn) {
            const uint32_t k = (gm * grid_n) + gn;
            const CoreCoord core = mm_core(gd, gm, gn);
            const uint32_t h = half_of(k);
            const uint32_t l = k % L;
            const std::vector<EntryFinish> finish = entry_finish_plan(sc, h);
            const std::array<std::optional<CoreCoord>, 2> port_v{
                peers[kFwd].has_value() ? std::optional<CoreCoord>(virt(ports.at({kFwd, l}))) : std::nullopt,
                peers[kBwd].has_value() ? std::optional<CoreCoord>(virt(ports.at({kBwd, l}))) : std::nullopt};
            const std::array<uint32_t, 2> n_arr{
                static_cast<uint32_t>(sc.arrivals[h][kFwd].size()), static_cast<uint32_t>(sc.arrivals[h][kBwd].size())};
            // a consumed chunk of d arrivals is reported to the port sending back upstream (opposite direction)
            const std::array<std::optional<CoreCoord>, 2> rep{port_v[kBwd], port_v[kFwd]};
            for (uint32_t d : {kFwd, kBwd}) {
                TT_FATAL(n_arr[d] == 0 || rep[d].has_value(), "matmul_reduce_scatter: arrivals without a port");
            }
            const auto xy16 = [](const std::optional<CoreCoord>& c) {
                return c.has_value() ? static_cast<uint32_t>(c->x | (c->y << 16)) : 0u;
            };

            std::vector<uint32_t> r_args(reader_rt::ENTRIES, 0);
            r_args[reader_rt::ROW0] = gm * core_m;
            r_args[reader_rt::COL0] = gn * core_n;
            r_args[reader_rt::TOTAL_FWD] = n_arr[kFwd];  // one increment per arrival (one wave per block)
            r_args[reader_rt::TOTAL_BWD] = n_arr[kBwd];
            r_args[reader_rt::SLOTS_FWD] = n_arr[kFwd];
            r_args[reader_rt::SLOTS_BWD] = n_arr[kBwd];
            r_args[reader_rt::REP_FWD_XY] = n_arr[kFwd] != 0 ? xy16(rep[kFwd]) : 0;
            r_args[reader_rt::REP_BWD_XY] = n_arr[kBwd] != 0 ? xy16(rep[kBwd]) : 0;
            std::vector<uint32_t> c_args;
            std::vector<uint32_t> w_args(writer_rt::KINDS, 0);
            w_args[writer_rt::ROW0] = gm * core_m;
            w_args[writer_rt::COL0] = gn * core_n;
            w_args[writer_rt::FWD_PORT_XY] = xy16(port_v[kFwd]);
            w_args[writer_rt::BWD_PORT_XY] = xy16(port_v[kBwd]);
            w_args[writer_rt::MAILBOX_SLOT_ADDR] = mailbox_addr + (k * kMailboxStride);
            const auto& entries = sc.entries[h];
            for (uint32_t i = 0; i < entries.size(); ++i) {
                const uint32_t b = entries[i];
                const uint32_t mask = finish[i].mask;
                const uint32_t lag = finish[i].lag ? 1 : 0;
                const auto threshold = [&](uint32_t d) -> uint32_t {
                    if ((mask & (1u << d)) == 0) {
                        return 0;
                    }
                    const auto& thr = sc.thresholds[h][d];
                    const auto it = thr.find(b);
                    return it != thr.end() ? it->second : 0;
                };
                r_args.push_back(b | (mask << 16) | (lag << 20));
                r_args.push_back(threshold(kFwd) | (threshold(kBwd) << 16));
                c_args.push_back(kRoleOfMask[mask] | (lag << 8));
                const RoleKind kind_b = sc.roles[b][h].kind;
                w_args.push_back(
                    kind_b == RoleKind::FIN ? kind::FIN : (kind_b == RoleKind::FWD ? kind::FWD : kind::BWD));
            }
            TT_FATAL(
                r_args.size() == reader_rt::ENTRIES + (reader_rt::ENTRY_WORDS * entries.size()),
                "matmul_reduce_scatter: reader entry words out of sync with matmul_reduce_scatter_args.hpp");
            KernelDescriptor::RTArgList r_list;
            r_list.append(r_args);
            reader.emplace_runtime_args(core, r_list);
            KernelDescriptor::RTArgList c_list;
            c_list.append(c_args);
            compute.emplace_runtime_args(core, c_list);
            KernelDescriptor::RTArgList w_list;
            w_list.append(w_args);
            writer.emplace_runtime_args(core, w_list);
        }
    }

    // ---- multicasts of A ("a", "a2") and W ("b", "b2"), attached to the reader after its own args ----
    {
        using ttnn::kernel_lib::host::Mcast1DShape;
        const Mcast1DShape a_shape = gd.m_on_x ? Mcast1DShape::PerColumn : Mcast1DShape::PerRow;
        const Mcast1DShape w_shape = gd.m_on_x ? Mcast1DShape::PerRow : Mcast1DShape::PerColumn;
        const auto sender_plan = plan_mcast_senders(
            levers.disjoint_senders, grid_m, grid_n, a_split < win_m, w_split < kb, a_shape, w_shape);
        const ttnn::kernel_lib::host::McastConfig mc_cfg{.noc = NOC::NOC_0, .handshake = true};
        for (const auto& family : sender_plan) {
            if (!family.start.has_value()) {
                ttnn::kernel_lib::host::attach_absent(reader, family.prefix);
                continue;
            }
            const ttnn::kernel_lib::host::Mcast1D mcast(
                mesh,
                mm_crs,
                family.shape,
                ttnn::kernel_lib::host::Mcast1DFixedSenderConfig{
                    .starting_sender_index = *family.start,
                    .sender_placement = ttnn::kernel_lib::host::Mcast1DSenderPlacement::Diagonal},
                mc_cfg);
            mcast.attach(desc, family.prefix, reader);
        }
    }

    // ---- port cores ----
    std::vector<uint32_t> port_ct(port_ct::COUNT, 0);
    port_ct[port_ct::CB_PORT_CHUNKS] = cb::PORT_CHUNKS;
    port_ct[port_ct::P] = P;
    port_ct[port_ct::CHUNK_TILES] = chunk_tiles;
    port_ct[port_ct::GROUP] = kPortGroup;
    port_ct[port_ct::TILE_BYTES] = stage_bytes;
    port_ct[port_ct::DEPTH_HANDOFF] = kDepthHandoff;
    port_ct[port_ct::MAILBOX_STRIDE] = kMailboxStride;
    port_ct[port_ct::MAX_CORES] = tt::div_up(n_cores, L);
    port_ct[port_ct::NUM_WAVES] = kNumWaves;
    port_ct[port_ct::INC_EVERY_WAVE] = kIncEveryWave;
    port_ct[port_ct::NUM_CHUNKS] = num_chunks;
    port_ct[port_ct::ENTRIES_PER_CHUNK] = num_entries;
    port_ct[port_ct::ZONE_SETS] = plan.zone_sets;

    KernelDescriptor port_reader;
    port_reader.kernel_source = kernel_path("matmul_reduce_scatter_port_reader.cpp");
    port_reader.compile_time_args = port_ct;
    {
        KernelDescriptor::RTArgList common;
        common.push_back(staging.buffer());
        common.push_back(tensor_args.mailbox.buffer());
        common.push_back(wave_tiles);
        port_reader.emplace_common_runtime_args(common);
    }
    TT_FATAL(
        port_reader.common_runtime_args.size() == port_reader_common::WAVE_TILES + kNumWaves,
        "matmul_reduce_scatter: port reader common args out of sync with matmul_reduce_scatter_args.hpp");
    port_reader.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_0_default};

    KernelDescriptor port_sender;
    port_sender.kernel_source = kernel_path("matmul_reduce_scatter_port_sender.cpp");
    port_sender.compile_time_args = port_ct;
    port_sender.emplace_common_runtime_args({ready_addr, credit_addr, wave_tiles});
    TT_FATAL(
        port_sender.common_runtime_args.size() == port_sender_common::WAVE_TILES + kNumWaves,
        "matmul_reduce_scatter: port sender common args out of sync with matmul_reduce_scatter_args.hpp");
    port_sender.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_1_default};

    // ports in (link, direction name) order: bwd before fwd on each link
    std::vector<PortKey> port_order;
    for (uint32_t l = 0; l < L; ++l) {
        for (uint32_t d : {kBwd, kFwd}) {
            if (ports.contains({d, l})) {
                port_order.emplace_back(d, l);
            }
        }
    }
    std::vector<CoreRange> used_ranges;
    const auto src_node = node(coord);
    for (const auto& [d, l] : port_order) {
        if (!peers[d].has_value()) {
            continue;
        }
        const CoreCoord pcore = ports.at({d, l});
        used_ranges.emplace_back(pcore, pcore);
        const MeshCoordinate& peer = *peers[d];
        const uint32_t q = d == kFwd ? (p + 1) % G : (p + G - 1) % G;
        const ChipSchedule& down = transport.chips[q];
        std::vector<uint32_t> core_words;
        for (uint32_t k = l; k < n_cores; k += L) {
            const CoreCoord vc = virt(mm_core(gd, k / grid_n, k % grid_n));
            core_words.push_back(static_cast<uint32_t>(vc.x | (vc.y << 8)) | (half_of(k) << 16) | (k << 20));
        }
        std::vector<uint32_t> units;
        for (const UnitWalk& u : unit_walks(sc, d, link_halves[l])) {
            const uint32_t s0 = (u.mask & 1u) != 0 ? position_in(down.arrivals[0][d], u.blocks[0]) : 0;
            const uint32_t s1 = (u.mask & 2u) != 0 ? position_in(down.arrivals[1][d], u.blocks[1]) : 0;
            units.push_back(u.entry | (u.mask << 8) | (u.h1_first << 10) | (s0 << 16) | (s1 << 24));
        }

        std::vector<uint32_t> pr_args(port_reader_rt::CORES, 0);
        pr_args[port_reader_rt::CONSUMED_ID] = d == kFwd ? sem::CONSUMED_FWD : sem::CONSUMED_BWD;
        pr_args[port_reader_rt::NUM_CORES] = static_cast<uint32_t>(core_words.size());
        pr_args[port_reader_rt::NUM_UNITS] = static_cast<uint32_t>(units.size());
        pr_args.insert(pr_args.end(), core_words.begin(), core_words.end());
        pr_args.insert(pr_args.end(), units.begin(), units.end());
        KernelDescriptor::RTArgList pr_list;
        pr_list.append(pr_args);
        port_reader.emplace_runtime_args(pcore, pr_list);

        // ready fence: the peer's opposite port sends into this chip; tell it this chip has started
        const uint32_t opp = d == kFwd ? kBwd : kFwd;
        bool send_ready = false;
        for (uint32_t b = 0; b < num_entries; ++b) {
            for (uint32_t h = 0; h < 2; ++h) {
                send_ready = send_ready ||
                             (((link_halves[l] >> h) & 1u) != 0 && down.roles[b][h].kind == role_of_direction(opp));
            }
        }
        const PortMap peer_ports = place_ports(mesh, peer, gd, gx, gy, L, peers_of(peer, axis, G, ring));
        const CoreCoord rv = virt(peer_ports.at({opp, l}));  // the peer chip's port that writes into this chip
        const auto pn = node(peer);
        const uint32_t expect = receivers[{opp, l}];
        const uint32_t fwd_total = expect > 0 ? credit_chunks : 0;
        if (credit_chunks != 0) {
            TT_FATAL(
                unit_walks(down, opp, link_halves[l]).empty() == (expect == 0),
                "matmul_reduce_scatter: credit forwarding out of sync (chip {}, link {})",
                p,
                l);
        }
        std::vector<uint32_t> ps_args(port_sender_rt::CORES, 0);
        ps_args[port_sender_rt::ZONE_BASE] = staging_addr + zone_off[d];
        ps_args[port_sender_rt::ARRIVAL_ADDR] = arr_addr[d];
        ps_args[port_sender_rt::SEND_READY] = send_ready ? 1 : 0;
        ps_args[port_sender_rt::READY_X] = static_cast<uint32_t>(rv.x);
        ps_args[port_sender_rt::READY_Y] = static_cast<uint32_t>(rv.y);
        ps_args[port_sender_rt::DST_MESH_ID] = static_cast<uint32_t>(*pn.mesh_id);
        ps_args[port_sender_rt::DST_CHIP_ID] = static_cast<uint32_t>(pn.chip_id);
        ps_args[port_sender_rt::NUM_CORES] = static_cast<uint32_t>(core_words.size());
        ps_args[port_sender_rt::NUM_UNITS] = static_cast<uint32_t>(units.size());
        ps_args[port_sender_rt::WAIT_CREDITS] = units.empty() ? 0 : credit_chunks;
        ps_args[port_sender_rt::EXPECT] = expect;
        ps_args[port_sender_rt::FWD_TOTAL] = fwd_total;
        ps_args[port_sender_rt::RING_SLOTS] = ring_slots[d];
        ps_args[port_sender_rt::DOWN_PER_CHUNK0] = static_cast<uint32_t>(down.arrivals[0][d].size());
        ps_args[port_sender_rt::DOWN_PER_CHUNK1] = static_cast<uint32_t>(down.arrivals[1][d].size());
        ps_args.insert(ps_args.end(), core_words.begin(), core_words.end());
        ps_args.insert(ps_args.end(), units.begin(), units.end());
        const auto links = tt::tt_fabric::get_forwarding_link_indices(src_node, pn);
        TT_FATAL(l < links.size(), "matmul_reduce_scatter: hop has {} link(s), link {} requested", links.size(), l);
        tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
            src_node, pn, links[l], desc, pcore, ps_args, tt::CoreType::WORKER);
        KernelDescriptor::RTArgList ps_list;
        ps_list.append(ps_args);
        port_sender.emplace_runtime_args(pcore, ps_list);
    }
    const CoreRangeSet used_port_crs = merged(used_ranges);
    port_reader.core_ranges = used_port_crs;
    port_sender.core_ranges = used_port_crs;

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(port_reader));
    desc.kernels.push_back(std::move(port_sender));
    TT_FATAL(desc.kernels.size() == kernel_index::COUNT, "matmul_reduce_scatter: kernel order out of sync");

    // ---- circular buffers ----
    const tt::DataFormat stage_format = tt::DataFormat::Float16_b;
    desc.cbs.push_back(plain_cb(cb::ACT_K_WINDOW, a_pages, tA, datatype_to_dataformat_converter(a.dtype()), mm_crs));
    desc.cbs.push_back(plain_cb(cb::WEIGHT_K_WINDOW, w_pages, tW, datatype_to_dataformat_converter(w.dtype()), mm_crs));
    desc.cbs.push_back(plain_cb(cb::PARTIAL_HOLD, plan.depth_hold * P, hold_bytes, tt::DataFormat::Float32, mm_crs));
    desc.cbs.push_back(cb_descriptor_from_sharded_tensor(
        static_cast<uint8_t>(cb::FINISHED), staging, 0, kDepthHandoff * P * stage_bytes, mm_crs));
    desc.cbs.push_back(cb_descriptor_from_sharded_tensor(
        static_cast<uint8_t>(cb::ARRIVALS_FWD), staging, zone_off[kFwd], ring_slots[kFwd] * P * stage_bytes, mm_crs));
    desc.cbs.push_back(cb_descriptor_from_sharded_tensor(
        static_cast<uint8_t>(cb::ARRIVALS_BWD), staging, zone_off[kBwd], ring_slots[kBwd] * P * stage_bytes, mm_crs));
    desc.cbs.push_back(plain_cb(
        cb::PORT_CHUNKS, kPortChunkBatches * kPortGroup * chunk_tiles, stage_bytes, stage_format, used_port_crs));
    desc.cbs.push_back(plain_cb(cb::ZERO, 1, stage_bytes, stage_format, mm_crs));
    return desc;
}

void MatmulReduceScatterProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MatmulReduceScatterParams& attrs,
    const MatmulReduceScatterInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The key (attrs, plan included, and the tensor specs) pins every compile-time arg, the core split and every
    // schedule word; a hit rewrites the addresses: A / W / output, the staging tensor (CBs FINISHED / ARRIVALS_*,
    // the ports' zone bases and finished base), the mailbox and the global semaphores.
    const MeshResources& res = resources_of(tensor_args.input.device());
    const uint32_t staging_addr = tensor_args.staging.buffer()->address();
    const uint32_t mailbox_addr = tensor_args.mailbox.buffer()->address();
    const std::array<uint32_t, 2> arr_addr{
        global_sem_address(res, GlobalSem::ARR_FWD), global_sem_address(res, GlobalSem::ARR_BWD)};
    const std::array<uint32_t, 2> zone_off = zone_offsets(attrs.plan);

    auto& reader_args = GetCommonRuntimeArgs(program, kernel_index::READER);
    reader_args[reader_common::A_ADDR] = tensor_args.input.buffer()->address();
    reader_args[reader_common::W_ADDR] = tensor_args.weight.buffer()->address();
    reader_args[reader_common::ARR_FWD_ADDR] = arr_addr[kFwd];
    reader_args[reader_common::ARR_BWD_ADDR] = arr_addr[kBwd];
    GetCommonRuntimeArgs(program, kernel_index::WRITER)[writer_common::OUT_ADDR] = output.buffer()->address();
    auto& port_reader_args = GetCommonRuntimeArgs(program, kernel_index::PORT_READER);
    port_reader_args[port_reader_common::FINISHED_BASE] = staging_addr;
    port_reader_args[port_reader_common::MAILBOX_BASE] = mailbox_addr;
    auto& port_sender_args = GetCommonRuntimeArgs(program, kernel_index::PORT_SENDER);
    port_sender_args[port_sender_common::READY_ADDR] = global_sem_address(res, GlobalSem::READY);
    port_sender_args[port_sender_common::CREDIT_ADDR] = global_sem_address(res, GlobalSem::CREDIT);

    auto& writer_grid = GetRuntimeArgs(program, kernel_index::WRITER);
    for (size_t x = 0; x < writer_grid.size(); ++x) {
        for (size_t y = 0; y < writer_grid[x].size(); ++y) {
            auto& args = writer_grid[x][y];
            if (args.size() > 0) {
                args[writer_rt::MAILBOX_SLOT_ADDR] =
                    mailbox_addr + (mm_index(attrs.plan.grid, CoreCoord{x, y}) * kMailboxStride);
            }
        }
    }
    // a port's direction is its port reader's consumed semaphore (same core)
    auto& port_reader_grid = GetRuntimeArgs(program, kernel_index::PORT_READER);
    auto& port_sender_grid = GetRuntimeArgs(program, kernel_index::PORT_SENDER);
    for (size_t x = 0; x < port_sender_grid.size(); ++x) {
        for (size_t y = 0; y < port_sender_grid[x].size(); ++y) {
            auto& args = port_sender_grid[x][y];
            if (args.size() == 0) {
                continue;
            }
            const uint32_t d = port_reader_grid[x][y][port_reader_rt::CONSUMED_ID] == sem::CONSUMED_FWD ? kFwd : kBwd;
            args[port_sender_rt::ZONE_BASE] = staging_addr + zone_off[d];
            args[port_sender_rt::ARRIVAL_ADDR] = arr_addr[d];
        }
    }

    const Buffer& staging = *tensor_args.staging.buffer();
    for (const auto& circular_buffer : program.circular_buffers()) {
        if (!circular_buffer->globally_allocated()) {
            continue;
        }
        const auto& indices = circular_buffer->buffer_indices();
        if (indices.contains(cb::FINISHED)) {
            UpdateDynamicCircularBufferAddress(program, circular_buffer->id(), staging, 0);
        } else if (indices.contains(cb::ARRIVALS_FWD)) {
            UpdateDynamicCircularBufferAddress(program, circular_buffer->id(), staging, zone_off[kFwd]);
        } else if (indices.contains(cb::ARRIVALS_BWD)) {
            UpdateDynamicCircularBufferAddress(program, circular_buffer->id(), staging, zone_off[kBwd]);
        }
    }
}

}  // namespace ttnn::operations::matmul_reduce_scatter
