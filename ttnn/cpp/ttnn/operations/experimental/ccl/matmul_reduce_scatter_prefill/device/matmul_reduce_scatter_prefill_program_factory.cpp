// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_prefill_program_factory.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <internal/mesh_device.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/indestructible.hpp>

#include "kernels/matmul_reduce_scatter_prefill_args.hpp"
#include "ttnn/kernel_lib/mcast/host/mcast.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill {

using namespace tt::tt_metal;
namespace args = ::matmul_reduce_scatter_prefill;
using detail::Direction;
using ttnn::kernel_lib::host::Mcast;
using ttnn::kernel_lib::host::McastConfig;
using ttnn::kernel_lib::host::McastCoreOrder;
using ttnn::kernel_lib::host::McastFixedSenderConfig;
using ttnn::kernel_lib::host::McastSenderPlacement;

namespace {

// The op's kernels, next to the args header they share with this factory (installed at the same path).
constexpr const char* kMrsKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/kernels/";

// Direction indices of the planner's [Fwd/Bwd] arrays.
constexpr uint32_t kFwd = 0;
constexpr uint32_t kBwd = 1;

std::string mrs_kernel_path(const char* name) { return std::string(kMrsKernelDir) + name; }

// bf16 tile: finished sub-blocks, arrival zones, port chunks, the zero tile and the output.
uint32_t mrs_stage_tile_bytes() { return tt::tile_size(tt::DataFormat::Float16_b); }

// Offset of a direction's arrival zone inside a core's staging slice (cb::FINISHED's slots come first).
uint32_t mrs_zone_offset(const detail::Plan& plan, uint32_t dir) {
    const uint32_t slot_bytes = plan.P * mrs_stage_tile_bytes();
    return (detail::kDepthHandoff + (dir == kBwd ? plan.zone_sets * plan.sched.slots[kFwd] : 0u)) * slot_bytes;
}

using SlotValue = std::variant<uint32_t, Buffer*>;

// A fixed slot list filled by name, as the descriptor's runtime-arg list (Buffer* entries stay bindings).
KernelDescriptor::RTArgList mrs_rt_list(const std::vector<SlotValue>& slots) {
    KernelDescriptor::RTArgList list;
    list.reserve(slots.size());
    for (const auto& slot : slots) {
        std::visit([&list](auto value) { list.push_back(value); }, slot);
    }
    return list;
}

void mrs_append(std::vector<uint32_t>& dst, const std::vector<uint32_t>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

uint32_t mrs_dir_index(Direction dir) { return dir == Direction::Fwd ? kFwd : kBwd; }

using PortKey = std::pair<uint32_t, uint32_t>;  // (direction, link)

// _place_ports: evenly along the strip (row strip) or down the last column (column strip).
std::map<PortKey, CoreCoord> mrs_spread_ports(detail::Strip strip, uint32_t gx, uint32_t num_links) {
    std::map<PortKey, CoreCoord> cores;
    const uint32_t n = 2 * num_links;
    for (uint32_t i = 0; i < n; ++i) {
        const PortKey key{i % 2, i / 2};
        if (strip == detail::Strip::Col) {
            cores[key] = CoreCoord(gx - 1, i);
        } else {
            cores[key] = CoreCoord(std::min(gx - 1, (2 * i + 1) * gx / (2 * n)), 0);
        }
    }
    return cores;
}

// _place_ports_near_eth: each linked port at the free strip cell with the fewest NoC1 hops to its link's Ethernet core
// (ties to the lower row, then the lower column); receive-only ends take the lowest free strip cell. Falls back to the
// spread placement under emulation or when the Ethernet core cannot be resolved.
std::map<PortKey, CoreCoord> mrs_place_ports(
    distributed::MeshDevice* mesh,
    const MeshCoordinate& coord,
    detail::Strip strip,
    uint32_t gx,
    uint32_t gy,
    uint32_t num_links,
    const std::array<std::optional<MeshCoordinate>, 2>& peers) {
    auto fallback = mrs_spread_ports(strip, gx, num_links);
    if (std::getenv("TT_METAL_EMULE_MODE") != nullptr) {
        return fallback;
    }
    std::vector<CoreCoord> allowed;  // sorted by (x, y)
    if (strip == detail::Strip::Row) {
        for (uint32_t x = 0; x < gx; ++x) {
            allowed.emplace_back(x, 0);
        }
    } else {
        for (uint32_t y = 0; y < gy; ++y) {
            allowed.emplace_back(gx - 1, y);
        }
    }
    std::set<std::pair<size_t, size_t>> taken;
    std::map<PortKey, CoreCoord> cores;
    try {
        IDevice* device = tt::tt_metal::internal::get_device(*mesh, coord);
        const auto src = mesh->get_fabric_node_id(coord);
        for (uint32_t l = 0; l < num_links; ++l) {
            for (uint32_t d : {kFwd, kBwd}) {
                if (!peers[d].has_value()) {
                    continue;
                }
                const auto dst = mesh->get_fabric_node_id(*peers[d]);
                const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, src, dst);
                if (l >= links.size()) {
                    return fallback;
                }
                const CoreCoord eth = tt::tt_fabric::get_forwarding_eth_core(*mesh, src, dst, links[l]);
                // NoC1 hops are directional (-y, then -x, wrapping). The worker grid is a product of physical columns
                // and rows, so the closest worker minimizes each axis' hop term on its own, and a cell's hops to the
                // Ethernet core are its hops to that worker plus the worker's own.
                const auto closest =
                    tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(*device, eth, NOC::NOC_1);
                std::optional<CoreCoord> best;
                std::tuple<uint32_t, size_t, size_t> best_key{};
                for (const CoreCoord& c : allowed) {
                    if (taken.contains({c.x, c.y})) {
                        continue;
                    }
                    const uint32_t hops =
                        closest.distance_in_noc_hops + tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                                                           device, c, closest.logical_coord, NOC::NOC_1);
                    const std::tuple<uint32_t, size_t, size_t> key{hops, c.y, c.x};
                    if (!best.has_value() || key < best_key) {
                        best = c;
                        best_key = key;
                    }
                }
                if (!best.has_value()) {
                    return fallback;
                }
                cores[{d, l}] = *best;
                taken.insert({best->x, best->y});
            }
        }
    } catch (const std::exception&) {
        return fallback;
    }
    for (uint32_t i = 0; i < 2 * num_links; ++i) {
        const PortKey key{i % 2, i / 2};
        if (cores.contains(key)) {
            continue;
        }
        for (const CoreCoord& c : allowed) {
            if (!taken.contains({c.x, c.y})) {
                cores[key] = c;
                taken.insert({c.x, c.y});
                break;
            }
        }
        TT_FATAL(cores.contains(key), "matmul_reduce_scatter_prefill: no free strip cell for a port");
    }
    return cores;
}

// plan_mcast_senders: families a, a2, b, b2; absent when the window is not split. Both regimes place the senders
// diagonally; the resident regime ("diagonal_disjoint") shifts each W family's start to overlap the senders already
// placed the least.
// A family multicasts along each matmul-grid row (PerRow) or column (PerColumn) independently.
enum class McastLine { PerRow, PerColumn };

struct McastFamilyPlan {
    const char* prefix;
    McastLine shape;
    std::optional<uint32_t> start;
};

std::vector<McastFamilyPlan> mrs_plan_mcast_senders(
    bool disjoint,
    uint32_t grid_m,
    uint32_t grid_n,
    bool a_two_parts,
    bool w_two_parts,
    McastLine a_shape_1d,
    McastLine w_shape_1d) {
    auto cores = [&](bool is_a, uint32_t start) {
        std::set<std::pair<uint32_t, uint32_t>> out;
        if (is_a) {
            for (uint32_t gm = 0; gm < grid_m; ++gm) {
                out.insert({gm, (start + gm) % grid_n});
            }
        } else {
            for (uint32_t gn = 0; gn < grid_n; ++gn) {
                out.insert({(start + gn) % grid_m, gn});
            }
        }
        return out;
    };
    struct Family {
        const char* prefix;
        bool is_a;
        bool second;
        McastLine shape;
        bool active;
        uint32_t line_len;
    };
    const std::array<Family, 4> families{{
        {"a", true, false, a_shape_1d, true, grid_n},
        {"a2", true, true, a_shape_1d, a_two_parts, grid_n},
        {"b", false, false, w_shape_1d, true, grid_m},
        {"b2", false, true, w_shape_1d, w_two_parts, grid_m},
    }};
    std::set<std::pair<uint32_t, uint32_t>> taken;
    std::vector<McastFamilyPlan> plan;
    for (const Family& f : families) {
        if (!f.active) {
            plan.push_back({f.prefix, f.shape, std::nullopt});
            continue;
        }
        uint32_t start = f.second ? f.line_len / 2 : 0;
        if (disjoint && !f.is_a) {
            const uint32_t nominal = start;
            std::optional<std::pair<size_t, uint32_t>> best;
            for (uint32_t s0 = 0; s0 < f.line_len; ++s0) {
                size_t overlap = 0;
                for (const auto& c : cores(false, s0)) {
                    overlap += taken.contains(c) ? 1 : 0;
                }
                const std::pair<size_t, uint32_t> key{overlap, (s0 + f.line_len - nominal) % f.line_len};
                if (!best.has_value() || key < *best) {
                    best = key;
                    start = s0;
                }
            }
        }
        for (const auto& c : cores(f.is_a, start)) {
            taken.insert(c);
        }
        plan.push_back({f.prefix, f.shape, start});
    }
    return plan;
}

// unit_walks: port (d, l)'s units on a chip: one per entry index where some half of the link holds a block whose
// role there is d.
struct PortUnit {
    uint32_t entry;
    std::array<uint32_t, 2> block;
    uint32_t mask;
    uint32_t h1_first;
};

std::vector<PortUnit> mrs_unit_walks(
    const detail::ChipSchedule& sc, uint32_t dir, const std::array<bool, 2>& link_halves) {
    std::vector<PortUnit> out;
    for (uint32_t e = 0; e < sc.entries[0].size(); ++e) {
        const std::array<uint32_t, 2> blk{sc.entries[0][e], sc.entries[1][e]};
        uint32_t mask = 0;
        for (uint32_t h = 0; h < 2; ++h) {
            const Direction role_dir = sc.roles[blk[h]][h].dir;
            if (link_halves[h] && role_dir != Direction::Fin && mrs_dir_index(role_dir) == dir) {
                mask |= 1u << h;
            }
        }
        if (mask == 0) {
            continue;
        }
        // walk a chain-start half (ready right after its matmul) before a relay half (waits for its arrival)
        const uint32_t h1_first =
            (mask == 3 && sc.roles[blk[1]][1].is_start && !sc.roles[blk[0]][0].is_start) ? 1u : 0u;
        out.push_back({e, blk, mask, h1_first});
    }
    return out;
}

uint32_t mrs_index_of(const std::vector<uint32_t>& list, uint32_t value) {
    const auto it = std::find(list.begin(), list.end(), value);
    TT_FATAL(it != list.end(), "matmul_reduce_scatter_prefill: block {} missing from the downstream arrivals", value);
    return static_cast<uint32_t>(it - list.begin());
}

CBDescriptor mrs_plain_cb(
    uint32_t buffer_index, uint32_t pages, uint32_t page_bytes, tt::DataFormat format, const CoreRangeSet& cores) {
    return CBDescriptor{
        .total_size = pages * page_bytes,
        .core_ranges = cores,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(buffer_index), .data_format = format, .page_size = page_bytes}},
    };
}

uint32_t mrs_xy16(const CoreCoord& c) { return static_cast<uint32_t>(c.x | (c.y << 16)); }

using MeshKey = decltype(std::declval<distributed::MeshDevice&>().id());

std::map<MeshKey, MeshResources>& mrs_resource_map() {
    static ttsl::Indestructible<std::map<MeshKey, MeshResources>> resources;
    return resources.get();
}

// guards mrs_resource_map(); held for every lookup and creation
std::mutex& mrs_resource_mutex() {
    static ttsl::Indestructible<std::mutex> mutex;
    return mutex.get();
}

std::optional<MatmulReduceScatterPrefillParams>& mrs_last_plan_slot() {
    static ttsl::Indestructible<std::optional<MatmulReduceScatterPrefillParams>> plan;
    return plan.get();
}

}  // namespace

ProgramDescriptor MatmulReduceScatterPrefillProgramFactory::create_descriptor(
    const MatmulReduceScatterPrefillParams& attrs,
    const MatmulReduceScatterPrefillInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "matmul_reduce_scatter_prefill: one program per mesh coordinate");
    TT_FATAL(
        args::cb::ZERO < hal::get_num_dataflow_buffers(),
        "matmul_reduce_scatter_prefill: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        args::cb::ZERO,
        hal::get_num_dataflow_buffers());
    TT_FATAL(detail::kPortGroup + 1 <= 8, "port sender headers: PORT_GROUP data headers + 1 credit header (pool of 8)");

    const MeshCoordinate& coord = *mesh_dispatch_coordinate;
    const Tensor& a = tensor_args.input_tensor;
    const Tensor& w = tensor_args.weight;
    distributed::MeshDevice* mesh = a.device();
    MeshResources& res = mesh_resources(mesh);
    const Tensor& staging = tensor_args.staging;

    const detail::Plan& plan = attrs.plan();
    const detail::GridChoice& gd = plan.grid;
    const detail::WavePlan& waves = plan.waves;
    const uint32_t L = attrs.num_links;
    const uint32_t axis = attrs.cluster_axis;
    const uint32_t G = mesh->shape()[axis];
    const bool ring = attrs.kind != detail::ScheduleKind::Line;
    const uint32_t p = coord[axis];
    TT_FATAL(
        plan.sched.chips.size() == G,
        "matmul_reduce_scatter_prefill: the plan's schedule has {} chips, not {}",
        plan.sched.chips.size(),
        G);

    auto coord_at = [&](uint32_t pos) {
        return axis == 0 ? MeshCoordinate(pos, coord[1]) : MeshCoordinate(coord[0], pos);
    };
    auto peers_of = [&](uint32_t pos) {
        std::array<std::optional<MeshCoordinate>, 2> peers;
        if (ring || pos + 1 < G) {
            peers[kFwd] = coord_at((pos + 1) % G);
        }
        if (ring || pos > 0) {
            peers[kBwd] = coord_at((pos + G - 1) % G);
        }
        return peers;
    };
    const auto peers = peers_of(p);

    // ---- geometry ----
    const auto& a_shape = a.logical_shape();
    const uint32_t Mt = a_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t Kt = a_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Nt = w.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const int32_t scatter_dim = attrs.scatter_dim;
    const uint32_t block_m = scatter_dim == -2 ? Mt / G : Mt;
    const uint32_t block_n = scatter_dim == -2 ? Nt : Nt / G;
    const CoreCoord grid = mesh->compute_with_storage_grid_size();
    const auto gx = static_cast<uint32_t>(grid.x);
    const auto gy = static_cast<uint32_t>(grid.y);
    const uint32_t W = waves.n;
    const uint32_t win_m = waves.win_m;
    const uint32_t win_n = waves.win_n;
    const uint32_t num_entries = G * W;
    const uint32_t P = plan.P;
    const uint32_t core_m = gd.core_m;
    const uint32_t core_n = gd.core_n;
    const uint32_t kb = plan.kb;
    TT_FATAL(kb > 0 && Kt % kb == 0, "matmul_reduce_scatter_prefill: K-block {} does not divide Kt {}", kb, Kt);
    TT_FATAL(win_m > 0 && win_n > 0 && P > 0, "matmul_reduce_scatter_prefill: empty wave");

    std::vector<uint32_t> wave_words;
    std::vector<uint32_t> wave_mm_words;
    std::vector<uint32_t> wave_tiles;
    for (const detail::WaveDesc& wd : waves.descs) {
        wave_words.push_back(wd.row_off | (wd.col_off << 8) | (wd.rows << 16) | (wd.cols << 24));
        const auto sb = detail::choose_subblock(wd.rows, wd.cols);
        wave_mm_words.push_back(wd.rows | (wd.cols << 8) | (sb[0] << 16) | (sb[1] << 24));
        wave_tiles.push_back(wd.rows * wd.cols);
    }

    // ---- matmul grid ----
    const uint32_t x0 = 0;
    const uint32_t y0 = gd.strip == detail::Strip::Row ? 1 : 0;
    const bool m_on_x = gd.orient == detail::Orient::MOnX;
    auto mm_core = [&](uint32_t gm, uint32_t gn) {
        return m_on_x ? CoreCoord(x0 + gm, y0 + gn) : CoreCoord(x0 + gn, y0 + gm);
    };
    struct MmCore {
        uint32_t gm;
        uint32_t gn;
        CoreCoord core;
    };
    std::vector<MmCore> mm_cores;  // k = row-major index
    for (uint32_t gm = 0; gm < gd.grid_m; ++gm) {
        for (uint32_t gn = 0; gn < gd.grid_n; ++gn) {
            mm_cores.push_back({gm, gn, mm_core(gm, gn)});
        }
    }
    const auto n_cores = static_cast<uint32_t>(mm_cores.size());
    auto half = [&](uint32_t k) -> uint32_t {
        switch (plan.half_split) {
            case detail::HalfSplit::N: return mm_cores[k].gn >= tt::div_up(gd.grid_n, 2u) ? 1 : 0;
            case detail::HalfSplit::M: return mm_cores[k].gm >= tt::div_up(gd.grid_m, 2u) ? 1 : 0;
            case detail::HalfSplit::K: break;
        }
        return k < tt::div_up(n_cores, 2u) ? 0 : 1;
    };
    auto link_of = [&](uint32_t k) { return k % L; };
    const CoreRangeSet mm_crs(CoreRange(mm_core(0, 0), mm_core(gd.grid_m - 1, gd.grid_n - 1)));

    std::map<uint32_t, std::map<PortKey, CoreCoord>> ports_by_pos;
    auto ports_at = [&](uint32_t pos) -> const std::map<PortKey, CoreCoord>& {
        auto it = ports_by_pos.find(pos);
        if (it == ports_by_pos.end()) {
            it = ports_by_pos.emplace(pos, mrs_place_ports(mesh, coord_at(pos), gd.strip, gx, gy, L, peers_of(pos)))
                     .first;
        }
        return it->second;
    };
    const std::map<PortKey, CoreCoord> ports = ports_at(p);
    auto port_vcore = [&](uint32_t pos, uint32_t d, uint32_t l) {
        return mesh->worker_core_from_logical_core(ports_at(pos).at({d, l}));
    };
    std::vector<std::array<bool, 2>> link_halves(L, {false, false});
    for (uint32_t k = 0; k < n_cores; ++k) {
        link_halves[link_of(k)][half(k)] = true;
    }
    const McastLine a_shape_1d = m_on_x ? McastLine::PerColumn : McastLine::PerRow;
    const McastLine w_shape_1d = m_on_x ? McastLine::PerRow : McastLine::PerColumn;

    const uint32_t nkb = Kt / kb;
    const bool resident = plan.resident;
    // Per-regime fill levers: resident 2 senders per line, W reads on NoC1, look-ahead after the read
    // barrier, disjoint diagonal senders; stream 1 sender, both reads on NoC0, look-ahead before, diagonal senders.
    const uint32_t senders = resident ? 2 : 1;
    const uint32_t a_split = senders >= 2 && win_m >= 2 ? tt::div_up(win_m, 2u) : win_m;
    const uint32_t w_split = senders >= 2 && kb >= 2 ? tt::div_up(kb, 2u) : kb;
    const uint32_t rA = (resident && scatter_dim == -1) ? 1 : 0;
    const uint32_t rW = (resident && scatter_dim == -2) ? 1 : 0;
    const uint32_t a_pages = rA != 0 ? core_m * Kt : plan.depths[0] * win_m * kb;
    const uint32_t w_pages = rW != 0 ? Kt * core_n : plan.depths[1] * kb * win_n;
    const auto tA = static_cast<uint32_t>(a.buffer()->aligned_page_size());
    const auto tW = static_cast<uint32_t>(w.buffer()->aligned_page_size());
    const uint32_t a_row_per_block = scatter_dim == -2 ? block_m : 0;
    const uint32_t w_col_per_block = scatter_dim == -1 ? block_n : 0;

    const uint32_t num_chunks = plan.n_m * plan.n_n;
    const uint32_t m_outer = scatter_dim == -1 ? 1 : 0;
    const uint32_t n_inner = m_outer != 0 ? plan.n_n : plan.n_m;
    const uint32_t zone_sets = plan.zone_sets;
    const uint32_t credit_chunks = num_chunks > zone_sets ? num_chunks - zone_sets : 0;
    const std::array<uint32_t, 2> ring_slots{zone_sets * plan.sched.slots[kFwd], zone_sets * plan.sched.slots[kBwd]};

    const uint32_t tile_bytes = mrs_stage_tile_bytes();
    const std::array<uint32_t, 2> zone_off{mrs_zone_offset(plan, kFwd), mrs_zone_offset(plan, kBwd)};
    // Persistent global semaphores (the mesh's lifetime): stable for every cache entry.
    const auto sem_addr = [&](MeshSem s) {
        return static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(res.sems[s]));
    };
    const std::array<uint32_t, 2> arr_addr{sem_addr(MeshSem::ARR_FWD), sem_addr(MeshSem::ARR_BWD)};
    const uint32_t ready_addr = sem_addr(MeshSem::READY);
    const uint32_t credit_addr = sem_addr(MeshSem::CREDIT);
    const auto max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    const uint32_t chunk_tiles = std::max(1u, std::min(max_payload / tile_bytes, P));

    // ---- compile-time args ----
    std::vector<uint32_t> reader_ct(args::reader_ct::COUNT);
    reader_ct[args::reader_ct::CB_ACT_K_WINDOW] = args::cb::ACT_K_WINDOW;
    reader_ct[args::reader_ct::CB_WEIGHT_K_WINDOW] = args::cb::WEIGHT_K_WINDOW;
    reader_ct[args::reader_ct::CB_ARRIVALS_FWD] = args::cb::ARRIVALS_FWD;
    reader_ct[args::reader_ct::CB_ARRIVALS_BWD] = args::cb::ARRIVALS_BWD;
    reader_ct[args::reader_ct::CORE_M] = win_m;
    reader_ct[args::reader_ct::CORE_N] = win_n;
    reader_ct[args::reader_ct::KB] = kb;
    reader_ct[args::reader_ct::NUM_K_BLOCKS] = nkb;
    reader_ct[args::reader_ct::KT] = Kt;
    reader_ct[args::reader_ct::NT] = Nt;
    reader_ct[args::reader_ct::A_ROW_PER_BLOCK] = a_row_per_block;
    reader_ct[args::reader_ct::W_COL_PER_BLOCK] = w_col_per_block;
    reader_ct[args::reader_ct::RESIDENT_A] = rA;
    reader_ct[args::reader_ct::RESIDENT_W] = rW;
    reader_ct[args::reader_ct::P] = P;
    reader_ct[args::reader_ct::NUM_ENTRIES] = num_entries;
    reader_ct[args::reader_ct::A_TILE_BYTES] = tA;
    reader_ct[args::reader_ct::W_TILE_BYTES] = tW;
    reader_ct[args::reader_ct::A_CAP_WINDOWS] = a_pages / (win_m * kb);
    reader_ct[args::reader_ct::W_CAP_WINDOWS] = w_pages / (kb * win_n);
    reader_ct[args::reader_ct::A_SPLIT] = a_split;
    reader_ct[args::reader_ct::W_SPLIT] = w_split;
    reader_ct[args::reader_ct::PREFETCH_STEPS] = detail::kPrefetchSteps;
    reader_ct[args::reader_ct::LOOKAHEAD_AFTER_BARRIER] = resident ? 1 : 0;
    reader_ct[args::reader_ct::A_READ_NOC] = 0;
    reader_ct[args::reader_ct::W_READ_NOC] = resident ? 1 : 0;
    reader_ct[args::reader_ct::NUM_WAVES] = W;
    reader_ct[args::reader_ct::NUM_CHUNKS] = num_chunks;
    reader_ct[args::reader_ct::N_INNER] = n_inner;
    reader_ct[args::reader_ct::M_OUTER] = m_outer;
    reader_ct[args::reader_ct::CHUNK_M] = plan.chunk_m;
    reader_ct[args::reader_ct::CHUNK_N] = plan.chunk_n;
    reader_ct[args::reader_ct::BLOCK_M] = block_m;
    reader_ct[args::reader_ct::BLOCK_N] = block_n;
    TensorAccessorArgs(*a.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*w.buffer()).append_to(reader_ct);

    std::vector<uint32_t> compute_ct(args::compute_ct::COUNT);
    compute_ct[args::compute_ct::CB_ACT_K_WINDOW] = args::cb::ACT_K_WINDOW;
    compute_ct[args::compute_ct::CB_WEIGHT_K_WINDOW] = args::cb::WEIGHT_K_WINDOW;
    compute_ct[args::compute_ct::CB_PARTIAL_HOLD] = args::cb::PARTIAL_HOLD;
    compute_ct[args::compute_ct::CB_FINISHED] = args::cb::FINISHED;
    compute_ct[args::compute_ct::CB_ARRIVALS_FWD] = args::cb::ARRIVALS_FWD;
    compute_ct[args::compute_ct::CB_ARRIVALS_BWD] = args::cb::ARRIVALS_BWD;
    compute_ct[args::compute_ct::WIN_N] = win_n;
    compute_ct[args::compute_ct::KB] = kb;
    compute_ct[args::compute_ct::NUM_K_BLOCKS] = nkb;
    compute_ct[args::compute_ct::P] = P;
    compute_ct[args::compute_ct::NUM_ENTRIES] = num_entries;
    compute_ct[args::compute_ct::NUM_WAVES] = W;
    compute_ct[args::compute_ct::SUPER_BLOCK] = plan.super_block;
    compute_ct[args::compute_ct::CB_ZERO] = args::cb::ZERO;
    compute_ct[args::compute_ct::NUM_CHUNKS] = num_chunks;
    compute_ct[args::compute_ct::N_INNER] = n_inner;
    compute_ct[args::compute_ct::RESIDENT] = resident ? 1 : 0;

    std::vector<uint32_t> writer_ct(args::writer_ct::COUNT);
    writer_ct[args::writer_ct::CB_FINISHED] = args::cb::FINISHED;
    writer_ct[args::writer_ct::P] = P;
    writer_ct[args::writer_ct::NUM_WAVES] = W;
    writer_ct[args::writer_ct::OUT_N_TILES] = block_n;
    writer_ct[args::writer_ct::DEPTH_HANDOFF] = detail::kDepthHandoff;
    writer_ct[args::writer_ct::TILE_BYTES] = tile_bytes;
    writer_ct[args::writer_ct::NUM_ENTRIES] = num_entries;
    writer_ct[args::writer_ct::CB_ZERO] = args::cb::ZERO;
    writer_ct[args::writer_ct::NUM_CHUNKS] = num_chunks;
    writer_ct[args::writer_ct::N_INNER] = n_inner;
    writer_ct[args::writer_ct::M_OUTER] = m_outer;
    writer_ct[args::writer_ct::CHUNK_M] = plan.chunk_m;
    writer_ct[args::writer_ct::CHUNK_N] = plan.chunk_n;
    writer_ct[args::writer_ct::BLOCK_M] = block_m;
    writer_ct[args::writer_ct::BLOCK_N] = block_n;
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);

    std::vector<uint32_t> port_ct(args::port_ct::COUNT);
    port_ct[args::port_ct::CB_PORT_CHUNKS] = args::cb::PORT_CHUNKS;
    port_ct[args::port_ct::P] = P;
    port_ct[args::port_ct::CHUNK_TILES] = chunk_tiles;
    port_ct[args::port_ct::GROUP] = detail::kPortGroup;
    port_ct[args::port_ct::TILE_BYTES] = tile_bytes;
    port_ct[args::port_ct::DEPTH_HANDOFF] = detail::kDepthHandoff;
    port_ct[args::port_ct::MAILBOX_STRIDE] = detail::kMailboxStride;
    port_ct[args::port_ct::MAX_CORES] = tt::div_up(n_cores, L);
    port_ct[args::port_ct::NUM_WAVES] = W;
    port_ct[args::port_ct::INC_EVERY_WAVE] = 1;  // ARRIVAL_INC_PER_WAVE: every wave's unit carries its increment
    port_ct[args::port_ct::NUM_CHUNKS] = num_chunks;
    port_ct[args::port_ct::ENTRIES_PER_CHUNK] = num_entries;
    port_ct[args::port_ct::ZONE_SETS] = zone_sets;

    // ---- program semaphores (before any multicast attaches its own) ----
    ProgramDescriptor desc;
    std::vector<CoreRange> all_ranges{CoreRange(mm_core(0, 0), mm_core(gd.grid_m - 1, gd.grid_n - 1))};
    for (const auto& [key, c] : ports) {
        all_ranges.emplace_back(c, c);
    }
    const CoreRangeSet all_crs = CoreRangeSet(all_ranges).merge_ranges();
    for (uint32_t expected : {args::sem::CONSUMED_FWD, args::sem::CONSUMED_BWD, args::sem::FREED}) {
        for (const CoreRange& range : all_crs.ranges()) {
            const auto id = desc.find_available_semaphore_id(range.start_coord, tt::CoreType::WORKER);
            TT_FATAL(
                id.has_value() && *id == expected,
                "matmul_reduce_scatter_prefill: semaphore id {} is not free on core {}",
                expected,
                range.start_coord);
        }
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = expected, .core_type = tt::CoreType::WORKER, .core_ranges = all_crs, .initial_value = 0});
    }

    // matmul cores per (arrival direction, link) whose half receives that direction's arrivals
    const detail::ChipSchedule& sc = plan.sched.chips[p];
    std::array<std::vector<uint32_t>, 2> receivers{std::vector<uint32_t>(L, 0), std::vector<uint32_t>(L, 0)};
    for (uint32_t k = 0; k < n_cores; ++k) {
        for (uint32_t d : {kFwd, kBwd}) {
            receivers[d][link_of(k)] += sc.arrivals[half(k)][d].empty() ? 0 : 1;
        }
    }

    // ---- matmul cores: per-half entry words (the same on every core of a half) ----
    std::array<std::vector<uint32_t>, 2> ent_words;
    std::array<std::vector<uint32_t>, 2> roles_words;
    std::array<std::vector<uint32_t>, 2> kind_words;
    for (uint32_t h = 0; h < 2; ++h) {
        const auto finish = detail::entry_finish_plan(sc, h);
        TT_FATAL(
            finish.size() == sc.entries[h].size(), "matmul_reduce_scatter_prefill: finish plan / entry order mismatch");
        for (size_t i = 0; i < finish.size(); ++i) {
            const uint32_t b = sc.entries[h][i];
            const auto [mask, lag] = finish[i];
            const auto threshold = [&](uint32_t d) -> uint32_t {
                if ((mask & (1u << d)) == 0) {
                    return 0;
                }
                const auto& thr = sc.thresholds[h][d];
                const auto it = thr.find(b);
                return it == thr.end() ? 0 : it->second;
            };
            // b is the virtual block (b * W + wave): the reader addresses the real block's operand slice
            ent_words[h].push_back((b / W) | (mask << 16) | (lag << 20));
            ent_words[h].push_back(threshold(kFwd) | (threshold(kBwd) << 16));
            roles_words[h].push_back(mask | (lag << args::role::LAG_SHIFT));
            const Direction dir = sc.roles[b][h].dir;
            kind_words[h].push_back(
                dir == Direction::Fin ? args::kind::FIN : (dir == Direction::Fwd ? args::kind::FWD : args::kind::BWD));
        }
        TT_FATAL(
            ent_words[h].size() == args::reader_rt::ENTRY_WORDS * finish.size(),
            "matmul_reduce_scatter_prefill: reader entry words are out of sync with "
            "matmul_reduce_scatter_prefill_args.hpp");
    }

    KernelDescriptor reader;
    reader.kernel_source = mrs_kernel_path("matmul_reduce_scatter_prefill_reader.cpp");
    reader.core_ranges = mm_crs;
    reader.compile_time_args = std::move(reader_ct);
    reader.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_0_default};

    KernelDescriptor compute;
    compute.kernel_source = mrs_kernel_path("matmul_reduce_scatter_prefill_compute.cpp");
    compute.core_ranges = mm_crs;
    compute.compile_time_args = std::move(compute_ct);
    // the cross-device adds accumulate in fp32 DEST; full sync restores 8-tile fp32 subblocks
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = attrs.math_fidelity,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .math_approx_mode = attrs.math_approx_mode,
    };

    KernelDescriptor writer;
    writer.kernel_source = mrs_kernel_path("matmul_reduce_scatter_prefill_writer.cpp");
    writer.core_ranges = mm_crs;
    writer.compile_time_args = std::move(writer_ct);
    writer.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_1_default};

    for (uint32_t k = 0; k < n_cores; ++k) {
        const auto& [gm, gn, core] = mm_cores[k];
        const uint32_t h = half(k);
        const uint32_t l = link_of(k);
        std::optional<CoreCoord> pf;
        std::optional<CoreCoord> pb;
        if (peers[kFwd].has_value()) {
            pf = port_vcore(p, kFwd, l);
        }
        if (peers[kBwd].has_value()) {
            pb = port_vcore(p, kBwd, l);
        }
        const std::array<uint32_t, 2> n_arr{
            static_cast<uint32_t>(sc.arrivals[h][kFwd].size()), static_cast<uint32_t>(sc.arrivals[h][kBwd].size())};
        // a consumed chunk of d arrivals is reported to the port sending back upstream (opposite direction)
        const std::array<std::optional<CoreCoord>, 2> rep{pb, pf};
        for (uint32_t d : {kFwd, kBwd}) {
            TT_FATAL(
                n_arr[d] == 0 || rep[d].has_value(),
                "matmul_reduce_scatter_prefill: core {} receives arrivals with no port to report to",
                k);
        }

        std::vector<uint32_t> r_rt(args::reader_rt::COUNT);
        r_rt[args::reader_rt::ROW0] = gm * core_m;
        r_rt[args::reader_rt::COL0] = gn * core_n;
        // ARRIVAL_INC_PER_WAVE: every arrival carries its own increment
        r_rt[args::reader_rt::TOTAL_FWD] = n_arr[kFwd];
        r_rt[args::reader_rt::TOTAL_BWD] = n_arr[kBwd];
        r_rt[args::reader_rt::SLOTS_FWD] = n_arr[kFwd];
        r_rt[args::reader_rt::SLOTS_BWD] = n_arr[kBwd];
        r_rt[args::reader_rt::REP_FWD_XY] = n_arr[kFwd] != 0 ? mrs_xy16(*rep[kFwd]) : 0;
        r_rt[args::reader_rt::REP_BWD_XY] = n_arr[kBwd] != 0 ? mrs_xy16(*rep[kBwd]) : 0;
        mrs_append(r_rt, wave_words);
        mrs_append(r_rt, ent_words[h]);
        reader.runtime_args.emplace_back(core, std::move(r_rt));

        std::vector<uint32_t> c_rt = wave_mm_words;
        mrs_append(c_rt, roles_words[h]);
        compute.runtime_args.emplace_back(core, std::move(c_rt));

        std::vector<uint32_t> w_rt(args::writer_rt::COUNT);
        w_rt[args::writer_rt::ROW0] = gm * core_m;
        w_rt[args::writer_rt::COL0] = gn * core_n;
        w_rt[args::writer_rt::FWD_PORT_XY] = pf.has_value() ? mrs_xy16(*pf) : 0;
        w_rt[args::writer_rt::BWD_PORT_XY] = pb.has_value() ? mrs_xy16(*pb) : 0;
        w_rt[args::writer_rt::MAILBOX_SLOT_OFFSET] = k * detail::kMailboxStride;
        mrs_append(w_rt, wave_words);
        mrs_append(w_rt, kind_words[h]);
        writer.runtime_args.emplace_back(core, std::move(w_rt));
    }

    std::vector<SlotValue> reader_common(args::reader_common::COUNT, 0u);
    reader_common[args::reader_common::A_ADDR] = a.buffer();
    reader_common[args::reader_common::W_ADDR] = w.buffer();
    reader_common[args::reader_common::ARR_FWD_ADDR] = arr_addr[kFwd];
    reader_common[args::reader_common::ARR_BWD_ADDR] = arr_addr[kBwd];
    reader_common[args::reader_common::REPORT_CHUNKS] = credit_chunks;
    reader.emplace_common_runtime_args(mrs_rt_list(reader_common));

    std::vector<SlotValue> writer_common(args::writer_common::COUNT, 0u);
    writer_common[args::writer_common::OUT_ADDR] = output.buffer();
    writer_common[args::writer_common::MAILBOX_ADDR] = res.mailbox.buffer();
    writer.emplace_common_runtime_args(mrs_rt_list(writer_common));

    // ---- multicast families (after the reader's own args) ----
    const McastConfig mc_cfg{.noc = NOC::NOC_0, .handshake = true};
    const CoreRange mm_box = mm_crs.bounding_box();
    for (const McastFamilyPlan& fam : mrs_plan_mcast_senders(
             resident, gd.grid_m, gd.grid_n, a_split < win_m, w_split < kb, a_shape_1d, w_shape_1d)) {
        if (!fam.start.has_value()) {
            ttnn::kernel_lib::host::attach_absent_mcast(reader, fam.prefix);
            continue;
        }
        // One group per grid line, ordered along the line; Staggered advances the sender by one per line.
        const bool per_row = fam.shape == McastLine::PerRow;
        const uint32_t line_len = per_row ? mm_box.grid_size().x : mm_box.grid_size().y;
        const McastFixedSenderConfig sender{.sender_index = *fam.start, .placement = McastSenderPlacement::Staggered};
        const std::array kernels{std::ref(reader)};
        Mcast(*mesh, mc_cfg, mm_crs, line_len, sender, per_row ? McastCoreOrder::RowMajor : McastCoreOrder::ColumnMajor)
            .attach(desc, fam.prefix, kernels, static_cast<uint32_t>(desc.semaphores.size()));
    }

    // ---- port cores ----
    KernelDescriptor port_reader;
    port_reader.kernel_source = mrs_kernel_path("matmul_reduce_scatter_prefill_port_reader.cpp");
    port_reader.compile_time_args = port_ct;
    port_reader.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_0_default};
    KernelDescriptor port_sender;
    port_sender.kernel_source = mrs_kernel_path("matmul_reduce_scatter_prefill_port_sender.cpp");
    port_sender.compile_time_args = std::move(port_ct);
    port_sender.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_1_default};

    const auto src_node = mesh->get_fabric_node_id(coord);
    std::vector<CoreRange> port_ranges;
    for (uint32_t l = 0; l < L; ++l) {
        for (uint32_t d : {kBwd, kFwd}) {  // (link, "bwd" < "fwd") order
            if (!peers[d].has_value()) {
                continue;
            }
            const MeshCoordinate& peer = *peers[d];
            const CoreCoord pcore = ports.at({d, l});
            port_ranges.emplace_back(pcore, pcore);
            const uint32_t q = d == kFwd ? (p + 1) % G : (p + G - 1) % G;
            const detail::ChipSchedule& down = plan.sched.chips[q];
            const uint32_t opp = 1 - d;

            std::vector<uint32_t> core_words;
            for (uint32_t k = 0; k < n_cores; ++k) {
                if (link_of(k) != l) {
                    continue;
                }
                const CoreCoord vc = mesh->worker_core_from_logical_core(mm_cores[k].core);
                core_words.push_back(static_cast<uint32_t>(vc.x | (vc.y << 8)) | (half(k) << 16) | (k << 20));
            }
            std::vector<uint32_t> units;
            for (const PortUnit& u : mrs_unit_walks(sc, d, link_halves[l])) {
                const uint32_t s0 = (u.mask & 1) != 0 ? mrs_index_of(down.arrivals[0][d], u.block[0]) : 0;
                const uint32_t s1 = (u.mask & 2) != 0 ? mrs_index_of(down.arrivals[1][d], u.block[1]) : 0;
                units.push_back(u.entry | (u.mask << 8) | (u.h1_first << 10) | (s0 << 16) | (s1 << 24));
            }

            std::vector<uint32_t> pr_rt(args::port_reader_rt::COUNT);
            pr_rt[args::port_reader_rt::CONSUMED_ID] = d == kFwd ? args::sem::CONSUMED_FWD : args::sem::CONSUMED_BWD;
            pr_rt[args::port_reader_rt::NUM_CORES] = static_cast<uint32_t>(core_words.size());
            pr_rt[args::port_reader_rt::NUM_UNITS] = static_cast<uint32_t>(units.size());
            mrs_append(pr_rt, wave_tiles);
            mrs_append(pr_rt, core_words);
            mrs_append(pr_rt, units);
            port_reader.runtime_args.emplace_back(pcore, std::move(pr_rt));

            // ready fence: the peer's opposite port sends into this chip; tell it this chip has started
            bool send_ready = false;
            for (uint32_t b = 0; b < num_entries && !send_ready; ++b) {
                for (uint32_t h = 0; h < 2; ++h) {
                    const Direction dir = down.roles[b][h].dir;
                    send_ready =
                        send_ready || (link_halves[l][h] && dir != Direction::Fin && mrs_dir_index(dir) == opp);
                }
            }
            const CoreCoord rv = port_vcore(q, opp, l);  // the peer chip's port that writes into this chip
            const auto peer_node = mesh->get_fabric_node_id(peer);
            // chunk credits: wait for one per chunk beyond the zone sets, forward one per chunk this chip's cores
            // consumed of the peer's (opp) arrivals
            const uint32_t expect = receivers[opp][l];
            const uint32_t fwd_total = expect > 0 ? credit_chunks : 0;
            if (credit_chunks != 0) {
                TT_FATAL(
                    mrs_unit_walks(down, opp, link_halves[l]).empty() == (expect == 0),
                    "matmul_reduce_scatter_prefill: credit forwarding mismatch at chip {} link {}",
                    p,
                    l);
            }

            std::vector<uint32_t> ps_rt(args::port_sender_rt::COUNT);
            ps_rt[args::port_sender_rt::ZONE_OFFSET] = zone_off[d];
            ps_rt[args::port_sender_rt::ARRIVAL_ADDR] = arr_addr[d];
            ps_rt[args::port_sender_rt::SEND_READY] = send_ready ? 1 : 0;
            ps_rt[args::port_sender_rt::READY_X] = static_cast<uint32_t>(rv.x);
            ps_rt[args::port_sender_rt::READY_Y] = static_cast<uint32_t>(rv.y);
            ps_rt[args::port_sender_rt::DST_MESH_ID] = *peer_node.mesh_id;
            ps_rt[args::port_sender_rt::DST_CHIP_ID] = peer_node.chip_id;
            ps_rt[args::port_sender_rt::NUM_CORES] = static_cast<uint32_t>(core_words.size());
            ps_rt[args::port_sender_rt::NUM_UNITS] = static_cast<uint32_t>(units.size());
            ps_rt[args::port_sender_rt::WAIT_CREDITS] = units.empty() ? 0 : credit_chunks;
            ps_rt[args::port_sender_rt::EXPECT] = expect;
            ps_rt[args::port_sender_rt::FWD_TOTAL] = fwd_total;
            ps_rt[args::port_sender_rt::RING_SLOTS] = ring_slots[d];
            ps_rt[args::port_sender_rt::DOWN_PER_CHUNK0] = static_cast<uint32_t>(down.arrivals[0][d].size());
            ps_rt[args::port_sender_rt::DOWN_PER_CHUNK1] = static_cast<uint32_t>(down.arrivals[1][d].size());
            mrs_append(ps_rt, wave_tiles);
            mrs_append(ps_rt, core_words);
            mrs_append(ps_rt, units);
            const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, src_node, peer_node);
            TT_FATAL(l < links.size(), "matmul_reduce_scatter_prefill: link {} of chip {} is not usable", l, p);
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                src_node, peer_node, links[l], desc, pcore, ps_rt, tt::CoreType::WORKER);
            port_sender.runtime_args.emplace_back(pcore, std::move(ps_rt));
        }
    }
    TT_FATAL(!port_ranges.empty(), "matmul_reduce_scatter_prefill: chip {} has no peer", p);
    const CoreRangeSet used_port_crs = CoreRangeSet(port_ranges).merge_ranges();
    port_reader.core_ranges = used_port_crs;
    port_sender.core_ranges = used_port_crs;

    std::vector<SlotValue> port_reader_common(args::port_reader_common::COUNT, 0u);
    port_reader_common[args::port_reader_common::FINISHED_BASE] = staging.buffer();
    port_reader_common[args::port_reader_common::MAILBOX_BASE] = res.mailbox.buffer();
    port_reader.emplace_common_runtime_args(mrs_rt_list(port_reader_common));

    std::vector<SlotValue> port_sender_common(args::port_sender_common::COUNT, 0u);
    port_sender_common[args::port_sender_common::STAGING_ADDR] = staging.buffer();
    port_sender_common[args::port_sender_common::READY_ADDR] = ready_addr;
    port_sender_common[args::port_sender_common::CREDIT_ADDR] = credit_addr;
    port_sender.emplace_common_runtime_args(mrs_rt_list(port_sender_common));

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(compute));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(port_reader));
    desc.kernels.push_back(std::move(port_sender));
    TT_FATAL(
        desc.kernels.size() == args::kernel_index::COUNT,
        "matmul_reduce_scatter_prefill: kernel order is out of sync with matmul_reduce_scatter_prefill_args.hpp");

    // ---- circular buffers ----
    const uint32_t finished_bytes = detail::kDepthHandoff * P * tile_bytes;
    desc.cbs.push_back(
        mrs_plain_cb(args::cb::ACT_K_WINDOW, a_pages, tA, datatype_to_dataformat_converter(a.dtype()), mm_crs));
    desc.cbs.push_back(
        mrs_plain_cb(args::cb::WEIGHT_K_WINDOW, w_pages, tW, datatype_to_dataformat_converter(w.dtype()), mm_crs));
    desc.cbs.push_back(mrs_plain_cb(
        args::cb::PARTIAL_HOLD,
        plan.depth_hold * P,
        tt::tile_size(tt::DataFormat::Float32),
        tt::DataFormat::Float32,
        mm_crs));
    desc.cbs.push_back(cb_descriptor_from_sharded_tensor(args::cb::FINISHED, staging, 0, finished_bytes, mm_crs));
    desc.cbs.push_back(cb_descriptor_from_sharded_tensor(
        args::cb::ARRIVALS_FWD, staging, zone_off[kFwd], ring_slots[kFwd] * P * tile_bytes, mm_crs));
    desc.cbs.push_back(cb_descriptor_from_sharded_tensor(
        args::cb::ARRIVALS_BWD, staging, zone_off[kBwd], ring_slots[kBwd] * P * tile_bytes, mm_crs));
    desc.cbs.push_back(mrs_plain_cb(
        args::cb::PORT_CHUNKS,
        2 * detail::kPortGroup * chunk_tiles,
        tile_bytes,
        tt::DataFormat::Float16_b,
        used_port_crs));
    desc.cbs.push_back(mrs_plain_cb(args::cb::ZERO, 1, tile_bytes, tt::DataFormat::Float16_b, mm_crs));
    return desc;
}

MatmulReduceScatterPrefillMeshWorkloadFactory::cached_mesh_workload_t
MatmulReduceScatterPrefillMeshWorkloadFactory::create_mesh_workload(
    const MatmulReduceScatterPrefillParams& attrs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MatmulReduceScatterPrefillInputs& tensor_args,
    Tensor& output) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared;
    const detail::Plan& plan = attrs.plan();
    for (const auto& coord : tensor_coords.coords()) {
        Program program{MatmulReduceScatterPrefillProgramFactory::create_descriptor(attrs, tensor_args, output, coord)};
        shared_variables_t sv{
            .reader = GetCommonRuntimeArgs(program, args::kernel_index::READER),
            .writer = GetCommonRuntimeArgs(program, args::kernel_index::WRITER),
            .port_reader = GetCommonRuntimeArgs(program, args::kernel_index::PORT_READER),
            .port_sender = GetCommonRuntimeArgs(program, args::kernel_index::PORT_SENDER),
        };
        uint32_t found = 0;
        for (const auto& circular_buffer : program.circular_buffers()) {
            const auto& indices = circular_buffer->buffer_indices();
            const uint32_t slot = indices.contains(args::cb::FINISHED)       ? 0
                                  : indices.contains(args::cb::ARRIVALS_FWD) ? 1
                                  : indices.contains(args::cb::ARRIVALS_BWD) ? 2
                                                                             : 3;
            if (slot == 3) {
                continue;
            }
            sv.staging_cbs[slot] = circular_buffer->id();
            sv.staging_offsets[slot] = slot == 0 ? 0 : mrs_zone_offset(plan, slot == 1 ? kFwd : kBwd);
            ++found;
        }
        TT_FATAL(
            found == 3, "matmul_reduce_scatter_prefill: expected 3 staging-backed circular buffers, found {}", found);
        const ttnn::MeshCoordinateRange range(coord);
        workload.add_program(range, std::move(program));
        shared.emplace(range, sv);
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared)};
}

void MatmulReduceScatterPrefillMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached,
    const MatmulReduceScatterPrefillParams& /*attrs*/,
    const MatmulReduceScatterPrefillInputs& tensor_args,
    Tensor& output) {
    // The key (attributes, plan inputs included, and the tensor specs) pins every compile-time arg and every per-core
    // slot, so a hit computes the call's values once and writes only what can move between calls into each program:
    // - A, W and the output: every call;
    // - the staging tensor and the three CBs it backs: only when it moved (every call allocates its own staging, which
    //   usually lands at the same address). Re-pointing a CB marks its dispatch config changed, so an unchanged CB
    //   keeps its cached config;
    // - the global semaphores and the mailbox: never. They are created once per mesh (ensure_mesh_resources) and live
    //   as long as the mesh and its program cache, so create_descriptor's values stay right.
    //   port_sender_rt::ARRIVAL_ADDR (per core) is the same kind of address.
    const auto a_addr = static_cast<uint32_t>(tensor_args.input_tensor.buffer()->address());
    const auto w_addr = static_cast<uint32_t>(tensor_args.weight.buffer()->address());
    const auto out_addr = static_cast<uint32_t>(output.buffer()->address());
    const Buffer& staging = *tensor_args.staging.buffer();
    const auto staging_addr = static_cast<uint32_t>(staging.address());
    for (const auto& [range, sv] : cached.shared_variables) {  // a walk, no lookup per program
        sv.reader.get()[args::reader_common::A_ADDR] = a_addr;
        sv.reader.get()[args::reader_common::W_ADDR] = w_addr;
        sv.writer.get()[args::writer_common::OUT_ADDR] = out_addr;
        if (sv.port_reader.get()[args::port_reader_common::FINISHED_BASE] == staging_addr) {
            continue;
        }
        sv.port_reader.get()[args::port_reader_common::FINISHED_BASE] = staging_addr;
        sv.port_sender.get()[args::port_sender_common::STAGING_ADDR] = staging_addr;
        auto& program = cached.workload.get_programs().at(range);  // only when the staging tensor moved
        for (size_t i = 0; i < sv.staging_cbs.size(); ++i) {
            UpdateDynamicCircularBufferAddress(program, sv.staging_cbs[i], staging, sv.staging_offsets[i]);
        }
    }
}

MeshResources& ensure_mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh) {
    const std::lock_guard<std::mutex> lock(mrs_resource_mutex());
    auto& resources = mrs_resource_map();
    const CoreCoord g = mesh->compute_with_storage_grid_size();
    const CoreRangeSet grid(CoreRange(CoreCoord(0, 0), CoreCoord(g.x - 1, g.y - 1)));
    const auto n = static_cast<uint32_t>(g.x * g.y);
    auto it = resources.find(mesh->id());
    if (it == resources.end()) {
        // creating them writes the mailbox and synchronizes the mesh, which a trace capture cannot record
        for (uint8_t cq = 0; cq < mesh->num_hw_cqs(); ++cq) {
            TT_FATAL(
                !mesh->mesh_command_queue(cq).trace_id().has_value(),
                "matmul_reduce_scatter_prefill: the first call on a mesh creates its semaphores and mailbox and cannot "
                "be traced; call the op once before begin_trace_capture");
        }
        std::array<GlobalSemaphore, 4> sems{
            ttnn::global_semaphore::create_global_semaphore(mesh, grid, 0),
            ttnn::global_semaphore::create_global_semaphore(mesh, grid, 0),
            ttnn::global_semaphore::create_global_semaphore(mesh, grid, 0),
            ttnn::global_semaphore::create_global_semaphore(mesh, grid, 0)};
        const uint32_t words = n * detail::kMailboxStride / static_cast<uint32_t>(sizeof(uint32_t));
        const MemoryConfig mbox_cfg(
            TensorMemoryLayout::HEIGHT_SHARDED,
            BufferType::L1,
            ShardSpec(grid, {1, words}, ShardOrientation::ROW_MAJOR));
        const TensorSpec mbox_spec(
            ttnn::Shape({n, words}), TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), mbox_cfg));
        Tensor mailbox =
            Tensor::from_vector(std::vector<uint32_t>(static_cast<size_t>(n) * words, 0u), mbox_spec, mesh);
        tt::tt_metal::distributed::Synchronize(
            mesh, std::nullopt);  // every counter / mailbox zero before any chip's first call
        it = resources.emplace(mesh->id(), MeshResources{.sems = std::move(sems), .mailbox = std::move(mailbox)}).first;
    }
    return it->second;
}

Tensor allocate_staging(tt::tt_metal::distributed::MeshDevice* mesh, uint32_t staging_tiles) {
    // The staging layout of a size is the same on every call: build it once (the sharded page mapping over the full
    // grid is most of create_device_tensor's cost) and allocate only the buffer per call. Mesh ids are never reused,
    // so a mesh's grid and shape are fixed under its key.
    struct StagingLayout {
        TensorSpec spec;
        tt::tt_metal::distributed::DeviceLocalBufferConfig device_local;
        tt::tt_metal::distributed::ReplicatedBufferConfig replicated;
        tt::tt_metal::TensorTopology topology;
    };
    static ttsl::Indestructible<std::map<std::pair<MeshKey, uint32_t>, StagingLayout>> layouts;
    static std::mutex layouts_mutex;
    const auto key = std::make_pair(mesh->id(), staging_tiles);
    const StagingLayout* layout = nullptr;
    {
        const std::lock_guard<std::mutex> lock(layouts_mutex);
        if (const auto it = layouts.get().find(key); it != layouts.get().end()) {
            layout = &it->second;  // std::map entries stay put, and none is ever erased
        }
    }
    if (layout != nullptr) {
        // What create_device_tensor does around the allocation, so trace allocation tracking and graph capture see
        // the same as on the first call.
        auto guard = tt::tt_metal::make_allocation_context_guard("ttnn.allocate_tensor_on_device");
        tt::tt_metal::GraphTracker::instance().track_function_start(
            "tt::tt_metal::create_device_tensor",
            layout->spec.logical_shape(),
            layout->spec.tensor_layout().get_data_type(),
            layout->spec.tensor_layout().get_layout(),
            mesh,
            layout->spec.tensor_layout().get_memory_config());
        auto buffer = tt::tt_metal::distributed::MeshBuffer::create(layout->replicated, layout->device_local, mesh);
        Tensor staging = set_tensor_id(Tensor(
            tt::tt_metal::mesh_tensor_from_buffer_with_topology(std::move(*buffer), layout->spec, layout->topology)));
        tt::tt_metal::GraphTracker::instance().track_function_end(staging);
        return staging;
    }
    const CoreCoord g = mesh->compute_with_storage_grid_size();
    const CoreRangeSet grid(CoreRange(CoreCoord(0, 0), CoreCoord(g.x - 1, g.y - 1)));
    const auto n = static_cast<uint32_t>(g.x * g.y);
    const MemoryConfig cfg(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(
            grid,
            {staging_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR));
    const TensorSpec spec(
        ttnn::Shape({n * staging_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), cfg));
    Tensor first = create_device_tensor(spec, mesh);
    const std::lock_guard<std::mutex> lock(layouts_mutex);
    layouts.get().try_emplace(
        key,
        StagingLayout{
            .spec = spec,
            .device_local =
                {.page_size = spec.compute_page_size_bytes(),
                 .buffer_type = BufferType::L1,
                 .sharding_args = spec.compute_buffer_sharding_args()},
            .replicated = {.size = spec.compute_packed_buffer_size_bytes()},
            .topology = first.tensor_topology()});
    return first;
}

MeshResources& mesh_resources(tt::tt_metal::distributed::MeshDevice* mesh) {
    const std::lock_guard<std::mutex> lock(mrs_resource_mutex());
    auto& resources = mrs_resource_map();
    const auto it = resources.find(mesh->id());
    TT_FATAL(
        it != resources.end(), "matmul_reduce_scatter_prefill: the mesh's resources were not created by the entry");
    return it->second;
}

void record_last_plan(const MatmulReduceScatterPrefillParams& attrs) { mrs_last_plan_slot() = attrs; }

const std::optional<MatmulReduceScatterPrefillParams>& last_plan() { return mrs_last_plan_slot(); }

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill
