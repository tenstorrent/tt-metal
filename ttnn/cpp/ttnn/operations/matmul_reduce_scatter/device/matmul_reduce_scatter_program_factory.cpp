// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_program_factory.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
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
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

using namespace tt::tt_metal;
namespace args = ::matmul_reduce_scatter;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
namespace noc_geometry = tt::tt_metal::experimental::Device;

namespace {

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/matmul_reduce_scatter/device/kernels/";
constexpr uint32_t kNumDirs = 2;         // 0 fwd, 1 bwd
constexpr uint32_t kHeaderPoolSize = 8;  // port sender: kPortGroup data headers + 1 credit header

std::string kernel_path(const char* name) { return std::string(kKernelDir) + name; }

uint32_t ceil_half(uint32_t v) { return tt::div_up(v, 2u); }

uint32_t xy16(const CoreCoord& c) { return static_cast<uint32_t>(c.x) | (static_cast<uint32_t>(c.y) << 16); }

// The chip's neighbours along the group: [0] fwd (next), [1] bwd (previous); none at a line's ends.
std::array<std::optional<MeshCoordinate>, kNumDirs> group_peers(
    const std::vector<MeshCoordinate>& group, uint32_t p, bool ring) {
    const auto G = static_cast<uint32_t>(group.size());
    std::array<std::optional<MeshCoordinate>, kNumDirs> peers;
    if (ring || p + 1 < G) {
        peers[0] = group[(p + 1) % G];
    }
    if (ring || p > 0) {
        peers[1] = group[(p + G - 1) % G];
    }
    return peers;
}

// Port cores of one chip, indexed l * 2 + d: evenly along the strip (the Python _place_ports).
std::vector<CoreCoord> spread_ports(bool row_strip, uint32_t gx, uint32_t L) {
    const uint32_t n = 2 * L;
    std::vector<CoreCoord> cores(n);
    for (uint32_t i = 0; i < n; ++i) {
        cores[i] = row_strip ? CoreCoord(std::min(gx - 1, ((2 * i) + 1) * gx / (2 * n)), 0) : CoreCoord(gx - 1, i);
    }
    return cores;
}

// Port cores of one chip, indexed l * 2 + d: each linked port at the strip cell with the fewest NoC1 hops (the
// sender's NoC: -y then -x, wrapping) to its link's Ethernet core, ties to the lower row, then the lower column; the
// receive-only ends on the first free cells (the Python _place_ports_near_eth). On emulation: spread_ports.
// The hops are measured on the chip's physical grid through the worker closest to the Ethernet core: every strip
// cell's directional distance to the Ethernet core is its distance to that worker plus a constant, so both rank the
// cells alike.
std::vector<CoreCoord> place_ports(
    MeshDevice* mesh,
    const MeshCoordinate& coord,
    const std::array<std::optional<MeshCoordinate>, kNumDirs>& peers,
    bool row_strip,
    uint32_t gx,
    uint32_t gy,
    uint32_t L) {
    if (std::getenv("TT_METAL_EMULE_MODE") != nullptr) {
        return spread_ports(row_strip, gx, L);
    }
    std::set<std::pair<uint32_t, uint32_t>> allowed;
    if (row_strip) {
        for (uint32_t x = 0; x < gx; ++x) {
            allowed.insert({x, 0});
        }
    } else {
        for (uint32_t y = 0; y < gy; ++y) {
            allowed.insert({gx - 1, y});
        }
    }
    std::vector<std::optional<CoreCoord>> cores(2 * L);
    std::set<std::pair<uint32_t, uint32_t>> taken;
    const auto src = mesh->get_fabric_node_id(coord);
    const IDevice& chip = *mesh->get_device(coord);
    for (uint32_t l = 0; l < L; ++l) {
        for (uint32_t d = 0; d < kNumDirs; ++d) {
            if (!peers[d].has_value()) {
                continue;
            }
            const auto dst = mesh->get_fabric_node_id(*peers[d]);
            const auto link = tt::tt_fabric::get_forwarding_link_indices(src, dst).at(l);
            const CoreCoord eth = tt::tt_fabric::get_forwarding_eth_core(src, dst, link);
            const CoreCoord nearest = noc_geometry::get_closest_worker_to_eth_core(chip, eth, NOC::NOC_1).logical_coord;
            std::optional<std::pair<std::array<uint32_t, 3>, std::pair<uint32_t, uint32_t>>> best;
            for (const auto& cell : allowed) {
                if (taken.contains(cell)) {
                    continue;
                }
                const uint32_t hops = noc_geometry::get_worker_noc_hop_distance(
                    mesh, coord, CoreCoord(cell.first, cell.second), nearest, NOC::NOC_1);
                const std::array<uint32_t, 3> score{hops, cell.second, cell.first};
                if (!best.has_value() || score < best->first) {
                    best = {score, cell};
                }
            }
            if (!best.has_value()) {
                return spread_ports(row_strip, gx, L);
            }
            cores[(l * kNumDirs) + d] = CoreCoord(best->second.first, best->second.second);
            taken.insert(best->second);
        }
    }
    std::vector<CoreCoord> out(2 * L);
    for (uint32_t i = 0; i < 2 * L; ++i) {
        if (!cores[i].has_value()) {
            const auto it =
                std::find_if(allowed.begin(), allowed.end(), [&](const auto& cell) { return !taken.contains(cell); });
            TT_FATAL(it != allowed.end(), "matmul_reduce_scatter: no free strip cell for port {}", i);
            cores[i] = CoreCoord(it->first, it->second);
            taken.insert(*it);
        }
        out[i] = *cores[i];
    }
    return out;
}

struct Unit {
    uint32_t entry = 0;
    std::array<uint32_t, 2> block{};
    uint32_t mask = 0;
    uint32_t h1_first = 0;
};

// Port (d, l)'s units on a chip: one per entry index where a half of link l has role d.
std::vector<Unit> unit_walks(const ChipSchedule& sc, uint32_t d, uint32_t link_halves) {
    std::vector<Unit> out;
    const auto dir = static_cast<RoleKind>(d);
    for (uint32_t e = 0; e < sc.entries[0].size(); ++e) {
        Unit u{.entry = e, .block = {sc.entries[0][e], sc.entries[1][e]}};
        for (uint32_t h = 0; h < 2; ++h) {
            if (((link_halves >> h) & 1u) != 0 && sc.roles[u.block[h]][h].kind == dir) {
                u.mask |= 1u << h;
            }
        }
        if (u.mask == 0) {
            continue;
        }
        // a chain-start half (ready right after its matmul) walks before a relay half (waits for its arrival)
        u.h1_first =
            static_cast<uint32_t>(u.mask == 3 && sc.roles[u.block[1]][1].is_start && !sc.roles[u.block[0]][0].is_start);
        out.push_back(u);
    }
    return out;
}

uint32_t index_in(const std::vector<uint32_t>& list, uint32_t b) {
    const auto it = std::find(list.begin(), list.end(), b);
    TT_FATAL(it != list.end(), "matmul_reduce_scatter: block {} missing from the downstream arrivals", b);
    return static_cast<uint32_t>(it - list.begin());
}

uint32_t threshold_of(const std::map<uint32_t, uint32_t>& thresholds, uint32_t b) {
    const auto it = thresholds.find(b);
    return it == thresholds.end() ? 0u : it->second;
}

uint32_t take_semaphore_id(ProgramDescriptor& desc, const CoreRangeSet& cores, const CoreCoord& probe) {
    const auto id = desc.find_available_semaphore_id(probe, tt::CoreType::WORKER);
    TT_FATAL(id.has_value(), "matmul_reduce_scatter: no free semaphore id");
    desc.semaphores.push_back(SemaphoreDescriptor{.id = *id, .core_ranges = cores, .initial_value = 0});
    return *id;
}

CBDescriptor plain_cb(
    uint32_t index, uint32_t pages, uint32_t page_bytes, tt::DataFormat fmt, const CoreRangeSet& crs) {
    return CBDescriptor{
        .total_size = pages * page_bytes,
        .core_ranges = crs,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index), .data_format = fmt, .page_size = page_bytes}},
    };
}

}  // namespace

ProgramDescriptor MatmulReduceScatterProgramFactory::create_descriptor(
    const MatmulReduceScatterParams& attrs,
    const MatmulReduceScatterInputs& tensors,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        args::cb::ZERO < hal::get_num_dataflow_buffers(),
        "matmul_reduce_scatter: circular-buffer id {} exceeds the {} circular buffers of this architecture",
        args::cb::ZERO,
        hal::get_num_dataflow_buffers());
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "matmul_reduce_scatter: one program per mesh coordinate");
    TT_FATAL(
        kPortGroup + 1 <= kHeaderPoolSize,
        "port sender headers: PORT_GROUP data headers + 1 credit header (pool of {})",
        kHeaderPoolSize);

    const Tensor& a = tensors.input_tensor;
    const Tensor& w = tensors.weight;
    const Tensor& staging = tensors.staging;
    MeshDevice* mesh = a.device();
    const ChunkPlan& plan = attrs.plan;
    const GridPlan& gd = plan.grid;
    const uint32_t G = attrs.group_size;
    const uint32_t L = attrs.num_links;
    const bool ring = attrs.kind != ScheduleKind::Line;
    const MeshCoordinate& coord = *mesh_dispatch_coordinate;

    // ---- this chip's group and position ----
    const auto groups = build_groups(mesh->shape(), attrs.cluster_axis);
    const std::vector<MeshCoordinate>* group = nullptr;
    uint32_t p = 0;
    for (const auto& grp : groups) {
        const auto it = std::find(grp.begin(), grp.end(), coord);
        if (it != grp.end()) {
            group = &grp;
            p = static_cast<uint32_t>(it - grp.begin());
        }
    }
    TT_FATAL(group != nullptr, "matmul_reduce_scatter: coordinate {} is in no group", coord);
    const TransportPlan transport = plan_transport(G, attrs.kind, plan.resident && plan.half_split != HalfSplit::K);
    const ChipSchedule& sc = transport.chips.at(p);
    const auto peers = group_peers(*group, p, ring);

    // ---- geometry ----
    const uint32_t Kt = a.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Nt = w.logical_shape()[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Mt = a.logical_shape()[-2] / tt::constants::TILE_HEIGHT;
    const bool scatter_m = attrs.scatter_dim == -2;
    const uint32_t block_m = scatter_m ? Mt / G : Mt;
    const uint32_t block_n = scatter_m ? Nt : Nt / G;
    const uint32_t core_m = gd.core_m;
    const uint32_t core_n = gd.core_n;
    const uint32_t grid_m = gd.grid_m;
    const uint32_t grid_n = gd.grid_n;
    const uint32_t kb = plan.kb;
    const uint32_t P = plan.p;
    // one wave per block: the streamed windows are the whole per-core block
    constexpr uint32_t num_waves = 1;
    const uint32_t win_m = core_m;
    const uint32_t win_n = core_n;
    const uint32_t num_entries = G * num_waves;
    const uint32_t wave_word = (core_m << 16) | (core_n << 24);
    const uint32_t wave_mm_word = core_m | (core_n << 8) | (gd.subblock.h << 16) | (gd.subblock.w << 24);
    const uint32_t wave_tiles = core_m * core_n;
    const CoreCoord grid = mesh->compute_with_storage_grid_size();
    const auto gx = static_cast<uint32_t>(grid.x);
    const auto gy = static_cast<uint32_t>(grid.y);

    // ---- matmul grid ----
    const uint32_t y0 = gd.row_strip ? 1 : 0;
    auto mm_core = [&](uint32_t gm, uint32_t gn) {
        return gd.m_on_x ? CoreCoord(gm, y0 + gn) : CoreCoord(gn, y0 + gm);
    };
    struct MmCore {
        uint32_t gm;
        uint32_t gn;
        CoreCoord core;
        uint32_t half;
        uint32_t link;
    };
    const uint32_t n_cores = grid_m * grid_n;
    std::vector<MmCore> mm_cores;
    mm_cores.reserve(n_cores);
    for (uint32_t gm = 0; gm < grid_m; ++gm) {
        for (uint32_t gn = 0; gn < grid_n; ++gn) {
            const auto k = static_cast<uint32_t>(mm_cores.size());
            uint32_t half = k < ceil_half(n_cores) ? 0 : 1;
            if (plan.half_split == HalfSplit::N) {
                half = gn >= ceil_half(grid_n) ? 1 : 0;
            } else if (plan.half_split == HalfSplit::M) {
                half = gm >= ceil_half(grid_m) ? 1 : 0;
            }
            mm_cores.push_back({gm, gn, mm_core(gm, gn), half, k % L});
        }
    }
    const CoreRangeSet mm_crs(CoreRange(mm_core(0, 0), mm_core(grid_m - 1, grid_n - 1)));
    std::vector<uint32_t> link_halves(L, 0);
    for (const auto& mc : mm_cores) {
        link_halves[mc.link] |= 1u << mc.half;
    }

    // ---- ports: this chip's and its peers' (a port writes into the peer's opposite port) ----
    const auto ports = place_ports(mesh, coord, peers, gd.row_strip, gx, gy, L);
    std::array<std::vector<CoreCoord>, kNumDirs> peer_ports;
    for (uint32_t d = 0; d < kNumDirs; ++d) {
        if (peers[d].has_value()) {
            const uint32_t q = d == 0 ? (p + 1) % G : (p + G - 1) % G;
            peer_ports[d] = place_ports(mesh, *peers[d], group_peers(*group, q, ring), gd.row_strip, gx, gy, L);
        }
    }
    auto port_vcore = [&](uint32_t d, uint32_t l) {
        return mesh->worker_core_from_logical_core(ports[(l * kNumDirs) + d]);
    };

    // ---- regime ----
    const uint32_t nkb = Kt / kb;
    const uint32_t senders = plan.resident ? 2 : 1;
    const uint32_t lookahead_after_barrier = plan.resident ? 1 : 0;
    const uint32_t a_read_noc = 0;
    const uint32_t w_read_noc = plan.resident ? 1 : 0;
    const uint32_t a_split = senders >= 2 && win_m >= 2 ? ceil_half(win_m) : win_m;
    const uint32_t w_split = senders >= 2 && kb >= 2 ? ceil_half(kb) : kb;
    const bool rA = plan.resident && attrs.scatter_dim == -1;
    const bool rW = plan.resident && scatter_m;
    const uint32_t a_pages = rA ? core_m * Kt : plan.depth_a * win_m * kb;
    const uint32_t w_pages = rW ? Kt * core_n : plan.depth_w * kb * win_n;
    const auto tA = static_cast<uint32_t>(a.buffer()->aligned_page_size());
    const auto tW = static_cast<uint32_t>(w.buffer()->aligned_page_size());

    const uint32_t num_chunks = plan.n_m * plan.n_n;
    const bool m_outer = attrs.scatter_dim == -1;
    const uint32_t n_inner = m_outer ? plan.n_n : plan.n_m;
    const uint32_t credit_chunks = num_chunks > plan.zone_sets ? num_chunks - plan.zone_sets : 0;
    const std::array<uint32_t, kNumDirs> ring_slots = {plan.zone_sets * plan.slots[0], plan.zone_sets * plan.slots[1]};

    const uint32_t stage_tile_bytes = staging.tensor_spec().tile().get_tile_size(tt::DataFormat::Float16_b);
    const uint32_t hold_tile_bytes = tt::tile_size(tt::DataFormat::Float32);
    const std::array<uint32_t, kNumDirs> zone_off = {
        staging_zone_offset(plan, 0, stage_tile_bytes), staging_zone_offset(plan, 1, stage_tile_bytes)};

    MeshResources& res = mesh_resources(mesh, staging_tiles(plan));
    const std::array<uint32_t, kNumDirs> arrival_addr = {
        static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(res.arrivals_fwd)),
        static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(res.arrivals_bwd))};
    const auto ready_addr = static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(res.ready));
    const auto credit_addr = static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(res.credit));
    const auto mailbox_addr =
        static_cast<uint32_t>(res.mailbox.buffer()->address());  // smuggled-rta-ok: persistent per-mesh resource
    const uint32_t chunk_tiles = std::max<uint32_t>(
        1,
        std::min<uint32_t>(
            static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / stage_tile_bytes), P));

    ProgramDescriptor desc;

    // ---- semaphores: program-local ids on every matmul and port core, first, in the header's order ----
    std::vector<CoreRange> all_ranges{CoreRange(mm_core(0, 0), mm_core(grid_m - 1, grid_n - 1))};
    for (const auto& c : ports) {
        all_ranges.emplace_back(c, c);
    }
    const CoreRangeSet all_crs = CoreRangeSet(all_ranges).merge_ranges();
    for (const uint32_t expected : {args::sem::CONSUMED_FWD, args::sem::CONSUMED_BWD, args::sem::FREED}) {
        const uint32_t id = take_semaphore_id(desc, all_crs, mm_core(0, 0));
        TT_FATAL(id == expected, "matmul_reduce_scatter: semaphore id {} where the kernels expect {}", id, expected);
    }

    // ---- matmul reader ----
    KernelDescriptor reader;
    reader.kernel_source = kernel_path("matmul_reduce_scatter_reader.cpp");
    reader.core_ranges = mm_crs;
    reader.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_0_default};
    {
        auto& ct = reader.compile_time_args;
        ct.assign(args::reader_ct::COUNT, 0u);
        ct[args::reader_ct::CB_ACT_K_WINDOW] = args::cb::ACT_K_WINDOW;
        ct[args::reader_ct::CB_WEIGHT_K_WINDOW] = args::cb::WEIGHT_K_WINDOW;
        ct[args::reader_ct::CB_ARRIVALS_FWD] = args::cb::ARRIVALS_FWD;
        ct[args::reader_ct::CB_ARRIVALS_BWD] = args::cb::ARRIVALS_BWD;
        ct[args::reader_ct::CORE_M] = win_m;
        ct[args::reader_ct::CORE_N] = win_n;
        ct[args::reader_ct::KB] = kb;
        ct[args::reader_ct::NUM_K_BLOCKS] = nkb;
        ct[args::reader_ct::KT] = Kt;
        ct[args::reader_ct::NT] = Nt;
        ct[args::reader_ct::A_ROW_PER_BLOCK] = scatter_m ? block_m : 0;
        ct[args::reader_ct::W_COL_PER_BLOCK] = scatter_m ? 0 : block_n;
        ct[args::reader_ct::RESIDENT_A] = static_cast<uint32_t>(rA);
        ct[args::reader_ct::RESIDENT_W] = static_cast<uint32_t>(rW);
        ct[args::reader_ct::P] = P;
        ct[args::reader_ct::NUM_ENTRIES] = num_entries;
        ct[args::reader_ct::A_TILE_BYTES] = tA;
        ct[args::reader_ct::W_TILE_BYTES] = tW;
        ct[args::reader_ct::A_CAP_WINDOWS] = a_pages / (win_m * kb);
        ct[args::reader_ct::W_CAP_WINDOWS] = w_pages / (kb * win_n);
        ct[args::reader_ct::A_SPLIT] = a_split;
        ct[args::reader_ct::W_SPLIT] = w_split;
        ct[args::reader_ct::PREFETCH_STEPS] = kPrefetchSteps;
        ct[args::reader_ct::LOOKAHEAD_AFTER_BARRIER] = lookahead_after_barrier;
        ct[args::reader_ct::A_READ_NOC] = a_read_noc;
        ct[args::reader_ct::W_READ_NOC] = w_read_noc;
        ct[args::reader_ct::NUM_WAVES] = num_waves;
        ct[args::reader_ct::NUM_CHUNKS] = num_chunks;
        ct[args::reader_ct::N_INNER] = n_inner;
        ct[args::reader_ct::M_OUTER] = static_cast<uint32_t>(m_outer);
        ct[args::reader_ct::CHUNK_M] = plan.chunk_m;
        ct[args::reader_ct::CHUNK_N] = plan.chunk_n;
        ct[args::reader_ct::BLOCK_M] = block_m;
        ct[args::reader_ct::BLOCK_N] = block_n;
        TensorAccessorArgs(*a.buffer()).append_to(ct);
        TensorAccessorArgs(*w.buffer()).append_to(ct);

        KernelDescriptor::RTArgList common;
        common.push_back(a.buffer());
        common.push_back(w.buffer());
        common.push_back(arrival_addr[0]);  // persistent GlobalSemaphore
        common.push_back(arrival_addr[1]);
        common.push_back(credit_chunks);  // chunks whose consumption is reported (the rest needs no credit)
        common.append(std::vector<uint32_t>(num_waves, wave_word));
        reader.emplace_common_runtime_args(common);
    }

    // ---- matmul compute ----
    KernelDescriptor compute;
    compute.kernel_source = kernel_path("matmul_reduce_scatter_compute.cpp");
    compute.core_ranges = mm_crs;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = attrs.math_fidelity,
        .fp32_dest_acc_en = true,  // the cross-device adds accumulate in fp32 DEST
        .dst_full_sync_en = true,  // 8-tile fp32 subblocks
        .math_approx_mode = attrs.math_approx_mode,
    };
    {
        auto& ct = compute.compile_time_args;
        ct.assign(args::compute_ct::COUNT, 0u);
        ct[args::compute_ct::CB_ACT_K_WINDOW] = args::cb::ACT_K_WINDOW;
        ct[args::compute_ct::CB_WEIGHT_K_WINDOW] = args::cb::WEIGHT_K_WINDOW;
        ct[args::compute_ct::CB_PARTIAL_HOLD] = args::cb::PARTIAL_HOLD;
        ct[args::compute_ct::CB_FINISHED] = args::cb::FINISHED;
        ct[args::compute_ct::CB_ARRIVALS_FWD] = args::cb::ARRIVALS_FWD;
        ct[args::compute_ct::CB_ARRIVALS_BWD] = args::cb::ARRIVALS_BWD;
        ct[args::compute_ct::WIN_N] = win_n;
        ct[args::compute_ct::KB] = kb;
        ct[args::compute_ct::NUM_K_BLOCKS] = nkb;
        ct[args::compute_ct::P] = P;
        ct[args::compute_ct::NUM_ENTRIES] = num_entries;
        ct[args::compute_ct::NUM_WAVES] = num_waves;
        ct[args::compute_ct::SUPER_BLOCK] = plan.super_block;
        ct[args::compute_ct::CB_ZERO] = args::cb::ZERO;
        ct[args::compute_ct::NUM_CHUNKS] = num_chunks;
        ct[args::compute_ct::N_INNER] = n_inner;
        ct[args::compute_ct::RESIDENT] = static_cast<uint32_t>(plan.resident);
        compute.common_runtime_args.assign(args::compute_common::COUNT + num_waves, wave_mm_word);
    }

    // ---- matmul writer ----
    KernelDescriptor writer;
    writer.kernel_source = kernel_path("matmul_reduce_scatter_writer.cpp");
    writer.core_ranges = mm_crs;
    writer.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_1_default};
    {
        auto& ct = writer.compile_time_args;
        ct.assign(args::writer_ct::COUNT, 0u);
        ct[args::writer_ct::CB_FINISHED] = args::cb::FINISHED;
        ct[args::writer_ct::P] = P;
        ct[args::writer_ct::NUM_WAVES] = num_waves;
        ct[args::writer_ct::OUT_N_TILES] = block_n;
        ct[args::writer_ct::DEPTH_HANDOFF] = kDepthHandoff;
        ct[args::writer_ct::TILE_BYTES] = stage_tile_bytes;
        ct[args::writer_ct::NUM_ENTRIES] = num_entries;
        ct[args::writer_ct::CB_ZERO] = args::cb::ZERO;
        ct[args::writer_ct::NUM_CHUNKS] = num_chunks;
        ct[args::writer_ct::N_INNER] = n_inner;
        ct[args::writer_ct::M_OUTER] = static_cast<uint32_t>(m_outer);
        ct[args::writer_ct::CHUNK_M] = plan.chunk_m;
        ct[args::writer_ct::CHUNK_N] = plan.chunk_n;
        ct[args::writer_ct::BLOCK_M] = block_m;
        ct[args::writer_ct::BLOCK_N] = block_n;
        TensorAccessorArgs(*output.buffer()).append_to(ct);

        KernelDescriptor::RTArgList common;
        common.push_back(output.buffer());
        common.append(std::vector<uint32_t>(num_waves, wave_word));
        writer.emplace_common_runtime_args(common);
    }

    // ---- matmul cores: per-core runtime args ----
    std::array<std::vector<uint32_t>, kNumDirs> receivers;  // [d][l]: cores of link l receiving d arrivals
    receivers[0].assign(L, 0);
    receivers[1].assign(L, 0);
    for (const auto& mc : mm_cores) {
        for (uint32_t d = 0; d < kNumDirs; ++d) {
            receivers[d][mc.link] += sc.arrivals[mc.half][d].empty() ? 0 : 1;
        }
    }
    constexpr std::array<uint32_t, 4> kRoleOfMask = {
        args::role::COPY, args::role::ADD_FWD, args::role::ADD_BWD, args::role::ADD_BOTH};
    std::array<std::vector<EntryFinish>, 2> finish = {entry_finish_plan(sc, 0), entry_finish_plan(sc, 1)};
    reader.runtime_args.reserve(n_cores);
    compute.runtime_args.reserve(n_cores);
    writer.runtime_args.reserve(n_cores);
    for (uint32_t k = 0; k < n_cores; ++k) {
        const MmCore& mc = mm_cores[k];
        const uint32_t h = mc.half;
        std::vector<uint32_t> ent_words;
        std::vector<uint32_t> roles_k;
        std::vector<uint32_t> kinds_k;
        ent_words.reserve(args::reader_rt::ENTRY_WORDS * num_entries);
        for (uint32_t e = 0; e < sc.entries[h].size(); ++e) {
            const uint32_t b = sc.entries[h][e];
            const uint32_t mask = finish[h][e].mask;
            const uint32_t lag = finish[h][e].lag ? 1 : 0;
            const uint32_t thr_f = (mask & 1u) != 0 ? threshold_of(sc.thresholds[h][0], b) : 0;
            const uint32_t thr_b = (mask & 2u) != 0 ? threshold_of(sc.thresholds[h][1], b) : 0;
            ent_words.push_back((b / num_waves) | (mask << 16) | (lag << 20));
            ent_words.push_back(thr_f | (thr_b << 16));
            roles_k.push_back(kRoleOfMask.at(mask) | (lag << 8));
            const RoleKind rk = sc.roles[b][h].kind;
            kinds_k.push_back(
                rk == RoleKind::Fin ? args::kind::FIN : (rk == RoleKind::Fwd ? args::kind::FWD : args::kind::BWD));
        }
        std::array<std::optional<CoreCoord>, kNumDirs> port_of;  // this core's fwd / bwd port (virtual)
        for (uint32_t d = 0; d < kNumDirs; ++d) {
            if (peers[d].has_value()) {
                port_of[d] = port_vcore(d, mc.link);
            }
        }
        const std::array<uint32_t, kNumDirs> n_arr = {
            static_cast<uint32_t>(sc.arrivals[h][0].size()), static_cast<uint32_t>(sc.arrivals[h][1].size())};
        // a consumed chunk of d arrivals is reported to the port sending back upstream (the opposite direction)
        const std::array<std::optional<CoreCoord>, kNumDirs> rep = {port_of[1], port_of[0]};
        for (uint32_t d = 0; d < kNumDirs; ++d) {
            TT_FATAL(n_arr[d] == 0 || rep[d].has_value(), "matmul_reduce_scatter: arrivals without a reporting port");
        }

        std::vector<uint32_t> r_args(args::reader_rt::COUNT, 0u);
        r_args[args::reader_rt::ROW0] = mc.gm * core_m;
        r_args[args::reader_rt::COL0] = mc.gn * core_n;
        r_args[args::reader_rt::TOTAL_FWD] = n_arr[0];  // one increment per arrival (an increment every wave)
        r_args[args::reader_rt::TOTAL_BWD] = n_arr[1];
        r_args[args::reader_rt::SLOTS_FWD] = n_arr[0];
        r_args[args::reader_rt::SLOTS_BWD] = n_arr[1];
        r_args[args::reader_rt::REP_FWD_XY] = n_arr[0] != 0 ? xy16(*rep[0]) : 0;
        r_args[args::reader_rt::REP_BWD_XY] = n_arr[1] != 0 ? xy16(*rep[1]) : 0;
        r_args.insert(r_args.end(), ent_words.begin(), ent_words.end());
        reader.runtime_args.emplace_back(mc.core, std::move(r_args));

        std::vector<uint32_t> c_args(args::compute_rt::COUNT, 0u);
        c_args.insert(c_args.end(), roles_k.begin(), roles_k.end());
        compute.runtime_args.emplace_back(mc.core, std::move(c_args));

        std::vector<uint32_t> w_args(args::writer_rt::COUNT, 0u);
        w_args[args::writer_rt::ROW0] = mc.gm * core_m;
        w_args[args::writer_rt::COL0] = mc.gn * core_n;
        w_args[args::writer_rt::FWD_PORT_XY] = port_of[0].has_value() ? xy16(*port_of[0]) : 0;
        w_args[args::writer_rt::BWD_PORT_XY] = port_of[1].has_value() ? xy16(*port_of[1]) : 0;
        w_args[args::writer_rt::MAILBOX_SLOT_ADDR] = mailbox_addr + (k * kMailboxStride);
        w_args.insert(w_args.end(), kinds_k.begin(), kinds_k.end());
        writer.runtime_args.emplace_back(mc.core, std::move(w_args));
    }

    // ---- multicast families (the reader's own args are in place) ----
    const auto a_shape_1d =
        gd.m_on_x ? ttnn::kernel_lib::host::Mcast1DShape::PerColumn : ttnn::kernel_lib::host::Mcast1DShape::PerRow;
    const auto w_shape_1d =
        gd.m_on_x ? ttnn::kernel_lib::host::Mcast1DShape::PerRow : ttnn::kernel_lib::host::Mcast1DShape::PerColumn;
    const auto sender_plan = plan_mcast_senders(
        plan.resident ? SenderPlacement::DiagonalDisjoint : SenderPlacement::Diagonal,
        grid_m,
        grid_n,
        a_split < win_m,
        w_split < kb,
        a_shape_1d,
        w_shape_1d);
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
                .starting_sender_index = *family.start, .sender_placement = family.placement},
            mc_cfg);
        mcast.attach(desc, family.prefix, reader);
    }

    // ---- port kernels ----
    KernelDescriptor port_reader;
    port_reader.kernel_source = kernel_path("matmul_reduce_scatter_port_reader.cpp");
    port_reader.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_0_default};
    KernelDescriptor port_sender;
    port_sender.kernel_source = kernel_path("matmul_reduce_scatter_port_sender.cpp");
    port_sender.config =
        DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_1_default};
    {
        std::vector<uint32_t> ct(args::port_ct::COUNT, 0u);
        ct[args::port_ct::CB_PORT_CHUNKS] = args::cb::PORT_CHUNKS;
        ct[args::port_ct::P] = P;
        ct[args::port_ct::CHUNK_TILES] = chunk_tiles;
        ct[args::port_ct::GROUP] = kPortGroup;
        ct[args::port_ct::TILE_BYTES] = stage_tile_bytes;
        ct[args::port_ct::DEPTH_HANDOFF] = kDepthHandoff;
        ct[args::port_ct::MAILBOX_STRIDE] = kMailboxStride;
        ct[args::port_ct::MAX_CORES] = tt::div_up(n_cores, L);
        ct[args::port_ct::NUM_WAVES] = num_waves;
        ct[args::port_ct::INC_EVERY_WAVE] = 1;  // every wave's unit carries the arrival increment
        ct[args::port_ct::NUM_CHUNKS] = num_chunks;
        ct[args::port_ct::ENTRIES_PER_CHUNK] = num_entries;
        ct[args::port_ct::ZONE_SETS] = plan.zone_sets;
        port_reader.compile_time_args = ct;
        port_sender.compile_time_args = std::move(ct);

        KernelDescriptor::RTArgList reader_common;
        reader_common.push_back(staging.buffer());  // cb_finished slot 0 on every core
        reader_common.push_back(mailbox_addr);
        reader_common.append(std::vector<uint32_t>(num_waves, wave_tiles));
        port_reader.emplace_common_runtime_args(reader_common);

        KernelDescriptor::RTArgList sender_common;
        sender_common.push_back(staging.buffer());
        sender_common.push_back(ready_addr);  // persistent GlobalSemaphores
        sender_common.push_back(credit_addr);
        sender_common.append(std::vector<uint32_t>(num_waves, wave_tiles));
        port_sender.emplace_common_runtime_args(sender_common);
    }
    TT_FATAL(
        reader.common_runtime_args.size() == args::reader_common::COUNT + num_waves &&
            writer.common_runtime_args.size() == args::writer_common::COUNT + num_waves &&
            port_reader.common_runtime_args.size() == args::port_reader_common::COUNT + num_waves &&
            port_sender.common_runtime_args.size() == args::port_sender_common::COUNT + num_waves,
        "matmul_reduce_scatter: common runtime argument layout is out of sync with matmul_reduce_scatter_args.hpp");
    const auto src_node = mesh->get_fabric_node_id(coord);
    std::vector<CoreRange> used_ranges;
    for (uint32_t l = 0; l < L; ++l) {
        for (uint32_t d = 0; d < kNumDirs; ++d) {
            if (!peers[d].has_value()) {
                continue;
            }
            const CoreCoord pcore = ports[(l * kNumDirs) + d];
            used_ranges.emplace_back(pcore, pcore);
            const uint32_t q = d == 0 ? (p + 1) % G : (p + G - 1) % G;
            const ChipSchedule& down = transport.chips.at(q);
            const uint32_t opp = 1 - d;

            std::vector<uint32_t> core_words;
            for (uint32_t k = 0; k < n_cores; ++k) {
                if (mm_cores[k].link != l) {
                    continue;
                }
                const CoreCoord vc = mesh->worker_core_from_logical_core(mm_cores[k].core);
                core_words.push_back(
                    static_cast<uint32_t>(vc.x) | (static_cast<uint32_t>(vc.y) << 8) | (mm_cores[k].half << 16) |
                    (k << 20));
            }
            std::vector<uint32_t> units;
            for (const Unit& u : unit_walks(sc, d, link_halves[l])) {
                const uint32_t s0 = (u.mask & 1u) != 0 ? index_in(down.arrivals[0][d], u.block[0]) : 0;
                const uint32_t s1 = (u.mask & 2u) != 0 ? index_in(down.arrivals[1][d], u.block[1]) : 0;
                units.push_back(u.entry | (u.mask << 8) | (u.h1_first << 10) | (s0 << 16) | (s1 << 24));
            }
            const auto num_core_words = static_cast<uint32_t>(core_words.size());
            const auto num_units = static_cast<uint32_t>(units.size());

            std::vector<uint32_t> pr_args(args::port_reader_rt::COUNT, 0u);
            pr_args[args::port_reader_rt::CONSUMED_ID] = d == 0 ? args::sem::CONSUMED_FWD : args::sem::CONSUMED_BWD;
            pr_args[args::port_reader_rt::NUM_CORES] = num_core_words;
            pr_args[args::port_reader_rt::NUM_UNITS] = num_units;
            pr_args.insert(pr_args.end(), core_words.begin(), core_words.end());
            pr_args.insert(pr_args.end(), units.begin(), units.end());
            port_reader.runtime_args.emplace_back(pcore, std::move(pr_args));

            // ready fence: the peer's opposite port sends into this chip; tell it this chip has started
            bool send_ready = false;
            for (uint32_t b = 0; b < num_entries; ++b) {
                for (uint32_t h = 0; h < 2; ++h) {
                    send_ready = send_ready || (((link_halves[l] >> h) & 1u) != 0 &&
                                                down.roles[b][h].kind == static_cast<RoleKind>(opp));
                }
            }
            const CoreCoord rv = mesh->worker_core_from_logical_core(peer_ports[d][(l * kNumDirs) + opp]);
            const auto dst_node = mesh->get_fabric_node_id(*peers[d]);
            // chunk credits: wait for one per chunk beyond the zone sets, forward one per chunk this chip's cores
            // consumed of the peer's (opposite-direction) arrivals
            const uint32_t expect = receivers[opp][l];
            const uint32_t fwd_total = expect > 0 ? credit_chunks : 0;
            if (credit_chunks != 0) {
                TT_FATAL(
                    unit_walks(down, opp, link_halves[l]).empty() == (expect == 0),
                    "matmul_reduce_scatter: credit forwarding does not match the peer's sends");
            }
            std::vector<uint32_t> ps_args(args::port_sender_rt::COUNT, 0u);
            ps_args[args::port_sender_rt::ZONE_OFFSET] = zone_off[d];
            ps_args[args::port_sender_rt::ARRIVAL_ADDR] = arrival_addr[d];  // persistent GlobalSemaphore
            ps_args[args::port_sender_rt::SEND_READY] = send_ready ? 1 : 0;
            ps_args[args::port_sender_rt::READY_X] = static_cast<uint32_t>(rv.x);
            ps_args[args::port_sender_rt::READY_Y] = static_cast<uint32_t>(rv.y);
            ps_args[args::port_sender_rt::DST_MESH_ID] = static_cast<uint32_t>(*dst_node.mesh_id);
            ps_args[args::port_sender_rt::DST_CHIP_ID] = static_cast<uint32_t>(dst_node.chip_id);
            ps_args[args::port_sender_rt::NUM_CORES] = num_core_words;
            ps_args[args::port_sender_rt::NUM_UNITS] = num_units;
            ps_args[args::port_sender_rt::WAIT_CREDITS] = units.empty() ? 0 : credit_chunks;
            ps_args[args::port_sender_rt::EXPECT] = expect;
            ps_args[args::port_sender_rt::FWD_TOTAL] = fwd_total;
            ps_args[args::port_sender_rt::RING_SLOTS] = ring_slots[d];
            ps_args[args::port_sender_rt::DOWN_PER_CHUNK0] = static_cast<uint32_t>(down.arrivals[0][d].size());
            ps_args[args::port_sender_rt::DOWN_PER_CHUNK1] = static_cast<uint32_t>(down.arrivals[1][d].size());
            ps_args.insert(ps_args.end(), core_words.begin(), core_words.end());
            ps_args.insert(ps_args.end(), units.begin(), units.end());
            const uint32_t link = tt::tt_fabric::get_forwarding_link_indices(src_node, dst_node).at(l);
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                src_node, dst_node, link, desc, pcore, ps_args, tt::CoreType::WORKER);
            port_sender.runtime_args.emplace_back(pcore, std::move(ps_args));
        }
    }
    const CoreRangeSet used_port_crs = CoreRangeSet(used_ranges).merge_ranges();
    port_reader.core_ranges = used_port_crs;
    port_sender.core_ranges = used_port_crs;

    desc.kernels.resize(kernel_index::COUNT);
    desc.kernels[kernel_index::READER] = std::move(reader);
    desc.kernels[kernel_index::COMPUTE] = std::move(compute);
    desc.kernels[kernel_index::WRITER] = std::move(writer);
    desc.kernels[kernel_index::PORT_READER] = std::move(port_reader);
    desc.kernels[kernel_index::PORT_SENDER] = std::move(port_sender);

    // ---- circular buffers ----
    desc.cbs.push_back(plain_cb(args::cb::ACT_K_WINDOW, a_pages, tA, cb_dataformat_for(a.dtype()), mm_crs));
    desc.cbs.push_back(plain_cb(args::cb::WEIGHT_K_WINDOW, w_pages, tW, cb_dataformat_for(w.dtype()), mm_crs));
    desc.cbs.push_back(
        plain_cb(args::cb::PARTIAL_HOLD, plan.depth_hold * P, hold_tile_bytes, tt::DataFormat::Float32, mm_crs));
    desc.cbs.push_back(ttnn::cb_descriptor_from_sharded_tensor(
        static_cast<uint8_t>(args::cb::FINISHED), staging, 0, kDepthHandoff * P * stage_tile_bytes, mm_crs));
    desc.cbs.push_back(ttnn::cb_descriptor_from_sharded_tensor(
        static_cast<uint8_t>(args::cb::ARRIVALS_FWD),
        staging,
        zone_off[0],
        ring_slots[0] * P * stage_tile_bytes,
        mm_crs));
    desc.cbs.push_back(ttnn::cb_descriptor_from_sharded_tensor(
        static_cast<uint8_t>(args::cb::ARRIVALS_BWD),
        staging,
        zone_off[1],
        ring_slots[1] * P * stage_tile_bytes,
        mm_crs));
    desc.cbs.push_back(plain_cb(
        args::cb::PORT_CHUNKS,
        2 * kPortGroup * chunk_tiles,
        stage_tile_bytes,
        tt::DataFormat::Float16_b,
        used_port_crs));
    desc.cbs.push_back(plain_cb(args::cb::ZERO, 1, stage_tile_bytes, tt::DataFormat::Float16_b, mm_crs));
    return desc;
}

void MatmulReduceScatterProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MatmulReduceScatterParams& attrs,
    const MatmulReduceScatterInputs& tensors,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // The key (attributes, plan included, and tensor specs) pins every compile-time arg, every per-core arg and the
    // persistent-resource addresses; what moves between calls is the tensors' buffers.
    const uint32_t a_addr = static_cast<uint32_t>(tensors.input_tensor.buffer()->address());
    const uint32_t w_addr = static_cast<uint32_t>(tensors.weight.buffer()->address());
    const uint32_t staging_addr = static_cast<uint32_t>(tensors.staging.buffer()->address());
    auto& reader = GetCommonRuntimeArgs(program, kernel_index::READER);
    reader[args::reader_common::A_ADDR] = a_addr;
    reader[args::reader_common::W_ADDR] = w_addr;
    GetCommonRuntimeArgs(program, kernel_index::WRITER)[args::writer_common::OUT_ADDR] =
        static_cast<uint32_t>(output.buffer()->address());
    GetCommonRuntimeArgs(program, kernel_index::PORT_READER)[args::port_reader_common::FINISHED_BASE] = staging_addr;
    GetCommonRuntimeArgs(program, kernel_index::PORT_SENDER)[args::port_sender_common::STAGING_ADDR] = staging_addr;

    // The staging-backed CBs, by CB id, at their offsets in the staging tensor.
    const ChunkPlan& plan = attrs.plan;
    const uint32_t stage_tile_bytes = tensors.staging.tensor_spec().tile().get_tile_size(tt::DataFormat::Float16_b);
    const Buffer& staging = *tensors.staging.buffer();
    for (const auto& circular_buffer : program.circular_buffers()) {
        if (!circular_buffer->globally_allocated()) {
            continue;
        }
        const auto& indices = circular_buffer->buffer_indices();
        uint32_t offset = 0;
        if (indices.contains(args::cb::ARRIVALS_FWD)) {
            offset = staging_zone_offset(plan, 0, stage_tile_bytes);
        } else if (indices.contains(args::cb::ARRIVALS_BWD)) {
            offset = staging_zone_offset(plan, 1, stage_tile_bytes);
        }
        UpdateDynamicCircularBufferAddress(program, circular_buffer->id(), staging, offset);
    }
}

// ---- shared with the entry / device operation -----------------------------------------------------------------

MeshResources& mesh_resources(MeshDevice* mesh, uint32_t staging_tiles) {
    struct Entry {
        int mesh_id;
        MeshResources res;
    };
    // Never destroyed: the tensors and semaphores it holds must not be released by a static destructor at process
    // exit, after the runtime singletons their deallocation goes through are gone.
    static ttsl::Indestructible<std::unordered_map<MeshDevice*, Entry>> resources;
    auto& cache = resources.get();
    const CoreCoord g = mesh->compute_with_storage_grid_size();
    const auto gx = static_cast<uint32_t>(g.x);
    const auto gy = static_cast<uint32_t>(g.y);
    const uint32_t n = gx * gy;
    const CoreRangeSet grid(CoreRange(CoreCoord(0, 0), CoreCoord(gx - 1, gy - 1)));

    auto it = cache.find(mesh);
    if (it == cache.end() || it->second.mesh_id != mesh->id()) {
        if (it != cache.end()) {
            cache.erase(it);
        }
        const uint32_t row_words = n * kMailboxStride / sizeof(uint32_t);
        const MemoryConfig mbox_cfg(
            TensorMemoryLayout::HEIGHT_SHARDED,
            BufferType::L1,
            ShardSpec(grid, {1, row_words}, ShardOrientation::ROW_MAJOR));
        const TensorSpec mbox_spec(
            ttnn::Shape({n, row_words}), TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), mbox_cfg));
        Tensor mailbox =
            Tensor::from_vector(std::vector<uint32_t>(static_cast<size_t>(n) * row_words, 0u), mbox_spec, mesh);
        MeshResources res{
            .arrivals_fwd = ttnn::global_semaphore::create_global_semaphore(mesh, grid, 0),
            .arrivals_bwd = ttnn::global_semaphore::create_global_semaphore(mesh, grid, 0),
            .ready = ttnn::global_semaphore::create_global_semaphore(mesh, grid, 0),
            .credit = ttnn::global_semaphore::create_global_semaphore(mesh, grid, 0),
            .mailbox = std::move(mailbox),
            .staging = std::nullopt,
            .staging_tiles = 0,
        };
        tt::tt_metal::distributed::Synchronize(mesh, std::nullopt);  // every counter / mailbox zero before any call
        it = cache.emplace(mesh, Entry{.mesh_id = mesh->id(), .res = std::move(res)}).first;
    }
    MeshResources& res = it->second.res;
    if (res.staging_tiles != staging_tiles) {
        res.staging.reset();
        const uint32_t tile_h = tt::constants::TILE_HEIGHT;
        const uint32_t tile_w = tt::constants::TILE_WIDTH;
        const MemoryConfig cfg(
            TensorMemoryLayout::HEIGHT_SHARDED,
            BufferType::L1,
            ShardSpec(grid, {staging_tiles * tile_h, tile_w}, ShardOrientation::ROW_MAJOR));
        const TensorSpec spec(
            ttnn::Shape({n * staging_tiles * tile_h, tile_w}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), cfg));
        res.staging = create_device_tensor(spec, mesh);
        res.staging_tiles = staging_tiles;
    }
    return res;
}

std::vector<std::vector<ttnn::MeshCoordinate>> build_groups(
    const tt::tt_metal::distributed::MeshShape& mesh_shape, uint32_t cluster_axis) {
    const uint32_t rows = mesh_shape[0];
    const uint32_t cols = mesh_shape[1];
    std::vector<std::vector<ttnn::MeshCoordinate>> groups;
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

bool fabric_config_wraps(tt::tt_fabric::FabricConfig fabric_config, uint32_t cluster_axis) {
    using tt::tt_fabric::FabricConfig;
    if (fabric_config == FabricConfig::FABRIC_2D_TORUS_XY || fabric_config == FabricConfig::FABRIC_1D_RING) {
        return true;
    }
    if (fabric_config == FabricConfig::FABRIC_2D_TORUS_X) {
        return cluster_axis == 1;
    }
    if (fabric_config == FabricConfig::FABRIC_2D_TORUS_Y) {
        return cluster_axis == 0;
    }
    return false;
}

uint32_t available_links(
    const tt::tt_metal::distributed::MeshDevice& mesh,
    const std::vector<std::vector<ttnn::MeshCoordinate>>& groups,
    bool ring) {
    std::optional<uint32_t> fewest;
    auto hop = [&](const ttnn::MeshCoordinate& s, const ttnn::MeshCoordinate& d) {
        const auto n = static_cast<uint32_t>(
            tt::tt_fabric::get_forwarding_link_indices(mesh.get_fabric_node_id(s), mesh.get_fabric_node_id(d)).size());
        fewest = fewest.has_value() ? std::min(*fewest, n) : n;
    };
    for (const auto& grp : groups) {
        for (size_t i = 0; i + 1 < grp.size(); ++i) {
            hop(grp[i], grp[i + 1]);
            hop(grp[i + 1], grp[i]);
        }
        if (ring && grp.size() >= 2) {
            hop(grp.back(), grp.front());
            hop(grp.front(), grp.back());
        }
    }
    return fewest.value_or(0);
}

}  // namespace ttnn::operations::matmul_reduce_scatter
