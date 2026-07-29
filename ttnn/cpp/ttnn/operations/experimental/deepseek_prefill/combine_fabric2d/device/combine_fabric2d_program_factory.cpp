// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_program_factory.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

// ---------------------------------------------------------------------------------------------
// L1 layout — identical on every tensix core of every device in the mesh.
//
// Uniformity is load-bearing: a producer on chip A addresses the drain sink of the peer worker on chip B
// without knowing anything about B's harvesting or eth positions. All offsets are relative to the L1
// unreserved base (uniform for a given arch/config).
// ---------------------------------------------------------------------------------------------
constexpr uint32_t PKT_HDR_DRAIN_OFF = 0x0000;  // scratch header for the drain's filler packets
// Target of the drain's value-0 atomic increments. A remote atomic inc needs a real L1 address on the
// far chip; nothing ever reads this word, which is exactly the point.
constexpr uint32_t DRAIN_SINK_OFF = 0x0400;
// 1 kB of per-worker telemetry, in the gap before the source ring, so it costs the payload nothing.
// Sized for expansion; the current record is 17 words.
constexpr uint32_t TELEMETRY_OFF = 0x0800;
constexpr uint32_t TELEMETRY_SIZE = 0x0400;
constexpr uint32_t PROD_BUF_OFF = 0x1000;  // the producer's L1 source ring
constexpr uint32_t L1_SLACK = 0x1000;      // keep clear of whatever sits at the very top of L1
static_assert(PKT_HDR_DRAIN_OFF < DRAIN_SINK_OFF, "drain sink overlaps the drain packet header");
static_assert(DRAIN_SINK_OFF < TELEMETRY_OFF, "telemetry region overlaps the drain sink");
static_assert(TELEMETRY_OFF + TELEMETRY_SIZE <= PROD_BUF_OFF, "telemetry region overlaps the source ring");

// Telemetry record word layout, shared with the producer kernel. Kept explicit (rather than a struct)
// because the kernel writes it by index and the host reads it by index.
enum TelemetryWord : uint32_t {
    TELEM_MAGIC = 0,  // written LAST by the kernel; zeroed at kernel entry
    TELEM_TOKENS_SENT = 1,
    TELEM_TOKEN_SIZE = 2,
    TELEM_NUM_IN_TOKENS = 3,
    TELEM_T_FIRST_SEND_LO = 4,
    TELEM_T_FIRST_SEND_HI = 5,
    TELEM_T_LAST_SEND_LO = 6,
    TELEM_T_LAST_SEND_HI = 7,
    TELEM_T_DRAINED_LO = 8,  // when the EDM drain proved every packet reached the far chip
    TELEM_T_DRAINED_HI = 9,
    TELEM_EDM_SLOTS = 10,      // EDM sender-channel buffer slots (the send pipeline's depth)
    TELEM_DRAIN_PACKETS = 11,  // header-only fillers the drain pushed (= edm_slots - 1)
    TELEM_OUT_BASE_PAGE = 12,  // first page written in the PEER chip's output region
    // Stall attribution: where the producer's cycles actually go. Two disjoint buckets measured with the
    // same wall clock as the timestamps above, so they are directly comparable to the send window.
    TELEM_WAIT_SLOT_CY_LO = 13,  // blocked in wait_for_empty_write_slot => the eth side is the limiter
    TELEM_WAIT_SLOT_CY_HI = 14,
    TELEM_ISSUE_CY_LO = 15,  // issuing the payload: header stamp + 2 NoC writes
    TELEM_ISSUE_CY_HI = 16,
    TELEM_NUM_WORDS = 17,
};
// Bumped whenever the record layout changes, so a stale record from an older kernel reads as invalid
// instead of being misparsed.
constexpr uint32_t TELEMETRY_MAGIC = 0xCF2D0004u;

// ---------------------------------------------------------------------------------------------
// Physical-column geometry
//
// A worker and the eth core it drives must share a PHYSICAL (noc0) x. That is the only coordinate
// space in which cores of different types are comparable: logical coords are per-core-type dense
// indices, and virtual/translated coords put eth and tensix in disjoint ranges (on Blackhole an eth
// core's translated coord is literally (20 + eth_channel, 25)). Match on physical x; convert back to
// logical only for kernel placement and to virtual only for NoC addressing.
// ---------------------------------------------------------------------------------------------
struct DeviceGeometry {
    // physical worker x -> (physical worker y -> logical worker core), restricted to cores this op may
    // program (inside compute_with_storage_grid_size — the dispatch column is NOT in here).
    std::map<uint32_t, std::map<uint32_t, CoreCoord>> columns;
};

DeviceGeometry build_device_geometry(tt::tt_metal::IDevice* dev, const CoreCoord& compute_grid) {
    DeviceGeometry geom;
    for (uint32_t ly = 0; ly < compute_grid.y; ly++) {
        for (uint32_t lx = 0; lx < compute_grid.x; lx++) {
            const CoreCoord logical{lx, ly};
            const auto phys = tt::tt_metal::experimental::Device::get_physical_core_from_logical_core(
                dev, logical, tt::CoreType::WORKER);
            geom.columns[static_cast<uint32_t>(phys.x)][static_cast<uint32_t>(phys.y)] = logical;
        }
    }
    return geom;
}

// The worker in `phys_x` adjacent to the eth row (smallest physical y — eth sits at the low-y edge of
// the grid). nullopt if that column has no programmable worker at all: it may be harvested (differs
// per chip within one mesh) or host the dispatch cores.
std::optional<CoreCoord> adjacent_worker_in_column(const DeviceGeometry& geom, uint32_t phys_x) {
    auto it = geom.columns.find(phys_x);
    if (it == geom.columns.end() || it->second.empty()) {
        return std::nullopt;
    }
    return it->second.begin()->second;
}

struct WorkerPlacement {
    CoreCoord eth_logical;    // the fabric router this worker drives
    uint32_t eth_phys_x = 0;  // its physical column
    uint32_t link_idx = 0;    // link index (routing plane) to open the connection on
    tt::tt_fabric::FabricNodeId peer_node{tt::tt_fabric::MeshId{0}, 0};  // chip across the cable
    CoreCoord worker_logical;                                            // where the producer goes
    CoreCoord worker_physical;  // noc0 coords; x == eth_phys_x unless relocated
    CoreCoord worker_virtual;   // what a remote producer must address
    bool in_eth_column = true;  // false => relocated, no longer adjacent to its router
};

// Every worker this device will host, keyed by the eth core it serves.
struct DevicePlacement {
    std::map<CoreCoord, WorkerPlacement> by_eth_logical;
};

// Decide where this device puts a worker for each of its fabric eth cores.
//
// Depends ONLY on this device's own eth set and its own harvesting, so the answer is stable no matter
// who asks — which is what lets a neighbor rely on it. Deliberately decides for the WHOLE device at
// once: a producer's args need the peer worker's coords on the neighbor chip, and those are NOT simply
// (neighbor_eth.x, y_min) because the neighbor's eth column may itself be tensix-unfriendly.
DevicePlacement decide_placement(
    ttnn::MeshDevice* mesh,
    const ttnn::MeshCoordinate& coord,
    uint32_t axis,
    uint32_t num_links,
    const CoreCoord& compute_grid) {
    auto* dev = mesh->get_device(coord);
    const auto self_node = mesh->get_fabric_node_id(coord);
    const auto geom = build_device_geometry(dev, compute_grid);
    const auto mesh_shape = mesh->shape();

    // This device's fabric eth cores: num_links toward the forward axis neighbor and num_links toward
    // the backward one. Every one is full duplex, so every one gets a producer.
    struct EthEntry {
        CoreCoord eth_logical;
        uint32_t eth_phys_x;
        uint32_t link_idx;
        tt::tt_fabric::FabricNodeId peer_node;
    };
    std::vector<EthEntry> eths;
    for (int delta : {1, -1}) {
        const auto nbr =
            coord.get_neighbor(mesh_shape, delta, static_cast<int32_t>(axis), ttnn::MeshCoordinate::BoundaryMode::WRAP);
        TT_FATAL(nbr.has_value(), "combine_fabric2d: no axis-{} neighbor of {} at delta {}", axis, coord, delta);
        if (*nbr == coord) {
            continue;  // degenerate axis; nothing to talk to
        }
        const auto nbr_node = mesh->get_fabric_node_id(*nbr);
        const auto links = tt::tt_fabric::get_forwarding_link_indices(self_node, nbr_node);
        const uint32_t n = std::min<uint32_t>(num_links, links.size());
        TT_FATAL(n > 0, "combine_fabric2d: no forwarding links from {} to {}", self_node, nbr_node);
        for (uint32_t k = 0; k < n; k++) {
            const CoreCoord eth_logical =
                tt::tt_fabric::get_forwarding_link_logical_eth_core(self_node, nbr_node, links[k]);
            const auto eth_phys = tt::tt_metal::experimental::Device::get_physical_core_from_logical_core(
                dev, eth_logical, tt::CoreType::ETH);
            eths.push_back(EthEntry{eth_logical, static_cast<uint32_t>(eth_phys.x), links[k], nbr_node});
        }
        if (n < num_links) {
            log_warning(
                tt::LogOp,
                "combine_fabric2d {}: only {} of {} requested links toward {}",
                self_node,
                n,
                num_links,
                nbr_node);
        }
    }
    // Deterministic order so the relocation fallback is reproducible.
    std::sort(
        eths.begin(), eths.end(), [](const EthEntry& a, const EthEntry& b) { return a.eth_phys_x < b.eth_phys_x; });

    // Columns that must stay free for a co-located worker: any column holding one of OUR eth cores.
    // Taking one for a relocated worker could displace the worker that belongs there.
    std::set<uint32_t> eth_columns;
    for (const auto& e : eths) {
        eth_columns.insert(e.eth_phys_x);
    }
    std::set<uint32_t> used_columns;

    auto make_placement = [&](const EthEntry& e, const CoreCoord& worker, bool in_eth_column) {
        WorkerPlacement wp;
        wp.eth_logical = e.eth_logical;
        wp.eth_phys_x = e.eth_phys_x;
        wp.link_idx = e.link_idx;
        wp.peer_node = e.peer_node;
        wp.worker_logical = worker;
        wp.worker_physical =
            tt::tt_metal::experimental::Device::get_physical_core_from_logical_core(dev, worker, tt::CoreType::WORKER);
        wp.worker_virtual = dev->virtual_core_from_logical_core(worker, tt::CoreType::WORKER);
        wp.in_eth_column = in_eth_column;
        return wp;
    };

    DevicePlacement placement;
    // Pass 1: everyone who can sit in their own eth column does.
    for (const auto& e : eths) {
        const auto w = adjacent_worker_in_column(geom, e.eth_phys_x);
        if (!w.has_value()) {
            continue;
        }
        auto wp = make_placement(e, *w, /*in_eth_column=*/true);
        TT_FATAL(
            wp.worker_physical.x == e.eth_phys_x,
            "combine_fabric2d: worker/eth column mismatch ({} vs {})",
            wp.worker_physical.x,
            e.eth_phys_x);
        used_columns.insert(e.eth_phys_x);
        placement.by_eth_logical.emplace(e.eth_logical, wp);
    }
    // Pass 2: relocate the rest to the leftmost column that is neither one of our eth columns nor
    // already taken. Runs after pass 1 so a relocated worker can never steal a co-located worker's
    // column, whatever order the eth cores come in.
    for (const auto& e : eths) {
        if (placement.by_eth_logical.count(e.eth_logical)) {
            continue;
        }
        std::optional<CoreCoord> chosen;
        uint32_t chosen_x = 0;
        for (const auto& [phys_x, rows] : geom.columns) {  // std::map => ascending x, i.e. leftmost first
            if (eth_columns.count(phys_x) || used_columns.count(phys_x) || rows.empty()) {
                continue;
            }
            chosen = rows.begin()->second;
            chosen_x = phys_x;
            break;
        }
        TT_FATAL(
            chosen.has_value(),
            "combine_fabric2d {}: eth core ({},{}) is in physical column x={} with no programmable worker, and no "
            "unoccupied column is left to relocate its worker to.",
            self_node,
            e.eth_logical.x,
            e.eth_logical.y,
            e.eth_phys_x);
        used_columns.insert(chosen_x);
        placement.by_eth_logical.emplace(e.eth_logical, make_placement(e, *chosen, /*in_eth_column=*/false));
        log_warning(
            tt::LogOp,
            "combine_fabric2d {}: eth core ({},{}) physical column x={} has no programmable worker (harvested or "
            "dispatch); relocated its worker to column x={}, adding {} NoC hop(s) to its router.",
            self_node,
            e.eth_logical.x,
            e.eth_logical.y,
            e.eth_phys_x,
            chosen_x,
            chosen_x > e.eth_phys_x ? chosen_x - e.eth_phys_x : e.eth_phys_x - chosen_x);
    }
    return placement;
}

// Lazy device -> placement cache. A device's placement must be decided before that device can serve as
// anyone's neighbor, so building the op for D also forces placement for D's cable neighbors — but NOT
// their programs, which are built on their own create call.
class PlacementCache {
public:
    PlacementCache(ttnn::MeshDevice* mesh, uint32_t axis, uint32_t num_links, const CoreCoord& compute_grid) :
        mesh_(mesh), axis_(axis), num_links_(num_links), compute_grid_(compute_grid) {}

    const DevicePlacement& get(const ttnn::MeshCoordinate& coord) {
        auto it = cache_.find(coord);
        if (it == cache_.end()) {
            it = cache_.emplace(coord, decide_placement(mesh_, coord, axis_, num_links_, compute_grid_)).first;
        }
        return it->second;
    }

private:
    ttnn::MeshDevice* mesh_;
    uint32_t axis_;
    uint32_t num_links_;
    CoreCoord compute_grid_;
    std::map<ttnn::MeshCoordinate, DevicePlacement> cache_;
};

struct L1Layout {
    uint32_t pkt_hdr_drain;
    uint32_t drain_sink;
    uint32_t telemetry;
    uint32_t prod_buf;
    uint32_t pkt_hdr_ring;  // num_in_tokens prebuilt payload headers
};

// The telemetry address depends only on the L1 base, so the readback path can resolve it without any of
// the run's sizing.
uint32_t telemetry_addr(ttnn::MeshDevice* mesh) {
    return static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1)) +
           TELEMETRY_OFF;
}

L1Layout compute_l1_layout(ttnn::MeshDevice* mesh, uint32_t num_in_tokens, uint32_t token_size_bytes) {
    const uint32_t base =
        static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    L1Layout l;
    l.pkt_hdr_drain = base + PKT_HDR_DRAIN_OFF;
    l.drain_sink = base + DRAIN_SINK_OFF;
    l.telemetry = base + TELEMETRY_OFF;
    l.prod_buf = base + PROD_BUF_OFF;
    // One prebuilt header per ring slot, past the ring itself.
    l.pkt_hdr_ring = l.prod_buf + num_in_tokens * token_size_bytes;
    const uint32_t hdr_ring_bytes =
        num_in_tokens * static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    const uint32_t end = l.pkt_hdr_ring + hdr_ring_bytes;
    const uint32_t l1_size = static_cast<uint32_t>(mesh->get_devices().front()->l1_size_per_core());
    TT_FATAL(
        end + L1_SLACK <= l1_size,
        "combine_fabric2d: L1 layout needs {} B (ends at 0x{:x}) but the core has only {} B of L1. "
        "Reduce num_in_tokens ({}) or token_size_bytes ({}).",
        end - base,
        end,
        l1_size,
        num_in_tokens,
        token_size_bytes);
    return l;
}

bool same_coord(const std::vector<uint32_t>& a, const ttnn::MeshCoordinate& b) {
    const auto bc = b.coords();
    return a.size() == bc.size() && std::equal(a.begin(), a.end(), bc.begin());
}

// Per-coordinate program: one producer kernel per fabric eth core of this device, on a worker core in
// that eth core's physical column, each owning that eth channel's single fabric connection and executing
// exactly one of the caller's movement descriptors.
//
// This is where the ONLY coupling between the caller's movement list and the op's internals lives: a
// movement whose `dst` is chip D must go to a producer whose cable reaches D. With num_links cables per
// neighbour there are num_links equally valid producers for each such movement, and which one gets which
// is arbitrary — we take them in the deterministic order both maps iterate in. Nothing outside this
// function depends on that choice.
tt::tt_metal::ProgramDescriptor build_program_for_coord(
    const CombineFabric2dParams& args,
    const ttnn::MeshCoordinate& coord,
    PlacementCache& placements,
    const std::map<uint32_t, ttnn::MeshCoordinate>& chip_to_coord,
    const L1Layout& l1,
    tt::tt_metal::Buffer* dram_out_buf,
    tt::tt_metal::Buffer* dram_in_buf) {
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh = args.device;
    auto* dev = mesh->get_device(coord);
    const auto self_node = mesh->get_fabric_node_id(coord);
    const uint32_t dram_out_addr = static_cast<uint32_t>(dram_out_buf->address());
    const uint32_t dram_in_addr = static_cast<uint32_t>(dram_in_buf->address());

    const auto& self_placement = placements.get(coord);

    // This device's movements, bucketed by destination coordinate. Each bucket is consumed in order as
    // the matching producers are walked below.
    std::map<ttnn::MeshCoordinate, std::vector<const CombineFabric2dMovement*>> pending_by_dst;
    uint32_t mine = 0;
    for (const auto& m : args.movements) {
        if (!same_coord(m.src, coord)) {
            continue;
        }
        mine++;
        bool placed = false;
        for (const auto& c : ttnn::MeshCoordinateRange(mesh->shape())) {
            if (same_coord(m.dst, c)) {
                pending_by_dst[c].push_back(&m);
                placed = true;
                break;
            }
        }
        TT_FATAL(
            placed,
            "combine_fabric2d {}: movement src {} names destination {}, which is not a coordinate of this {} mesh",
            self_node,
            coord,
            movement_coord_str(m.dst),
            mesh->shape());
    }
    TT_FATAL(
        mine == self_placement.by_eth_logical.size(),
        "combine_fabric2d {} {}: got {} movement(s) for this device but it has {} fabric cable(s). Every cable "
        "needs exactly one movement (2 directions x num_links {}).",
        coord,
        self_node,
        mine,
        self_placement.by_eth_logical.size(),
        args.num_links);

    std::string summary;
    for (const auto& [eth_logical, wp] : self_placement.by_eth_logical) {
        // Peer = the worker serving the eth core at the far end of THIS eth core's cable. Cable truth,
        // not plane-index arithmetic: our producer writes into this eth core's EDM, so for a
        // single-hop destination the packet physically emerges at the far end of this cable.
        const auto far = dev->get_connected_ethernet_core(eth_logical);
        const uint32_t far_chip = static_cast<uint32_t>(std::get<0>(far));
        const CoreCoord far_eth_logical = std::get<1>(far);
        auto cit = chip_to_coord.find(far_chip);
        TT_FATAL(
            cit != chip_to_coord.end(),
            "combine_fabric2d {}: eth core ({},{}) cables to chip {}, which is not in this mesh",
            self_node,
            eth_logical.x,
            eth_logical.y,
            far_chip);
        // Forces the neighbor's placement if not yet decided. We only READ it; the neighbor's programs
        // are built on its own create call.
        const auto& peer_placement = placements.get(cit->second);
        auto pit = peer_placement.by_eth_logical.find(far_eth_logical);
        TT_FATAL(
            pit != peer_placement.by_eth_logical.end(),
            "combine_fabric2d {}: our eth core ({},{}) cables to eth core ({},{}) on {}, but that core is not in the "
            "neighbor's fabric eth set, so it has no worker. Link selection disagrees across the cable.",
            self_node,
            eth_logical.x,
            eth_logical.y,
            far_eth_logical.x,
            far_eth_logical.y,
            mesh->get_fabric_node_id(cit->second));
        const CoreCoord peer_virtual = pit->second.worker_virtual;

        // Claim a movement bound for the chip this cable actually reaches. Cable truth again: the packets
        // this producer sends can only emerge at the far end of its own cable, so a movement is only
        // assignable here if its `dst` is that far chip.
        const ttnn::MeshCoordinate& far_coord = cit->second;
        auto bucket = pending_by_dst.find(far_coord);
        TT_FATAL(
            bucket != pending_by_dst.end() && !bucket->second.empty(),
            "combine_fabric2d {} {}: eth core ({},{}) cables to {}, but no (remaining) movement asks to send there. "
            "Movements must name a destination for every cable, {} per neighbour.",
            coord,
            self_node,
            eth_logical.x,
            eth_logical.y,
            far_coord,
            args.num_links);
        const CombineFabric2dMovement& mv = *bucket->second.back();
        bucket->second.pop_back();

        tt::tt_metal::KernelDescriptor prod;
        prod.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
            "producer_combine_fabric2d.cpp";
        prod.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        prod.core_ranges = CoreRangeSet(CoreRange(wp.worker_logical));
        prod.compile_time_args = {
            args.output_tokens_per_movement,
            args.input_tokens_per_movement,
            args.token_size_bytes,
            static_cast<uint32_t>(wp.peer_node.chip_id),
            *wp.peer_node.mesh_id,
            static_cast<uint32_t>(peer_virtual.x),
            static_cast<uint32_t>(peer_virtual.y),
            l1.prod_buf,
            l1.pkt_hdr_ring,
            l1.pkt_hdr_drain,
            l1.drain_sink,
            l1.telemetry,
            args.stall_telemetry,
            mv.in_base_token,
            mv.out_base_token,
            dram_out_addr,
            dram_in_addr,
        };
        // 17+: TensorAccessorArgs for the interleaved output buffer, then the input buffer (both
        // compile-time config).
        tt::tt_metal::TensorAccessorArgs(dram_out_buf).append_to(prod.compile_time_args);
        tt::tt_metal::TensorAccessorArgs(dram_in_buf).append_to(prod.compile_time_args);
        prod.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            // NOC_1 routes -Y first, so worker (eth row + 1) -> eth core is a single hop.
            .noc = tt::tt_metal::NOC::NOC_1,
        };
        auto prod_id = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
        desc.kernels.push_back(std::move(prod));
        {
            std::vector<uint32_t> rt_raw;
            rt_raw.push_back(1u);  // num_connections
            const std::vector<tt::tt_fabric::FabricNodeId> dst_nodes = {wp.peer_node};
            const std::vector<uint32_t> conn_links = {wp.link_idx};
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                self_node, dst_nodes, conn_links, desc, prod_id, wp.worker_logical, rt_raw);
            tt::tt_metal::KernelDescriptor::RTArgList rt;
            rt.append(rt_raw);
            desc.kernels[prod_id].emplace_runtime_args(wp.worker_logical, rt);
        }

        summary += fmt::format(
            "{}[eth({},{}) phys_x {} link {} -> worker logical ({},{}) phys ({},{}){} peer {} virt ({},{}) "
            "in[{},{}) -> {} out[{},{})]",
            summary.empty() ? "" : " ",
            eth_logical.x,
            eth_logical.y,
            wp.eth_phys_x,
            wp.link_idx,
            wp.worker_logical.x,
            wp.worker_logical.y,
            wp.worker_physical.x,
            wp.worker_physical.y,
            wp.in_eth_column ? "" : " RELOCATED",
            wp.peer_node,
            peer_virtual.x,
            peer_virtual.y,
            mv.in_base_token,
            mv.in_base_token + args.input_tokens_per_movement,
            far_coord,
            mv.out_base_token,
            mv.out_base_token + args.output_tokens_per_movement);
    }

    log_info(
        tt::LogOp,
        "combine_fabric2d {} {}: {} producer(s): {}",
        coord,
        self_node,
        self_placement.by_eth_logical.size(),
        summary);
    return desc;
}

}  // namespace

CombineFabric2dTelemetry read_telemetry(ttnn::MeshDevice* mesh_device, uint32_t num_links, uint32_t axis) {
    TT_FATAL(mesh_device != nullptr, "combine_fabric2d read_telemetry: mesh_device is null");
    const auto mesh_shape = mesh_device->shape();
    TT_FATAL(axis < mesh_shape.dims(), "combine_fabric2d: axis {} out of range for mesh shape {}", axis, mesh_shape);

    const auto grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t addr = telemetry_addr(mesh_device);

    CombineFabric2dTelemetry out;
    out.clock_mhz = static_cast<uint32_t>(mesh_device->get_devices().front()->get_clock_rate_mhz());

    // Physical chip id -> mesh coordinate, so a worker's record can name the chip its tokens landed on in
    // the same coordinate space the caller uses to index per-device tensor shards.
    std::map<uint32_t, ttnn::MeshCoordinate> chip_to_coord;
    for (const auto& c : ttnn::MeshCoordinateRange(mesh_shape)) {
        chip_to_coord.emplace(static_cast<uint32_t>(mesh_device->get_device(c)->id()), c);
    }

    for (const auto& coord : ttnn::MeshCoordinateRange(mesh_shape)) {
        auto* dev = mesh_device->get_device(coord);
        // Placement is a pure function of (device, axis, num_links), so recomputing it here reproduces
        // exactly the cores the op programmed — no state has to survive from the run.
        const auto placement = decide_placement(mesh_device, coord, axis, num_links, grid);
        for (const auto& [eth_logical, wp] : placement.by_eth_logical) {
            CombineFabric2dWorkerTelemetry w;
            w.device_id = static_cast<uint32_t>(dev->id());
            w.mesh_coord.assign(coord.coords().begin(), coord.coords().end());
            w.worker_logical = wp.worker_logical;
            w.worker_physical = wp.worker_physical;
            w.eth_logical = eth_logical;
            w.eth_phys_x = wp.eth_phys_x;
            w.link_idx = wp.link_idx;
            w.relocated = !wp.in_eth_column;
            w.peer_mesh_id = *wp.peer_node.mesh_id;
            w.peer_chip_id = static_cast<uint32_t>(wp.peer_node.chip_id);
            // The chip this worker's tokens actually land on is the far end of ITS cable — the same
            // resolution the program factory used to pick the destination page range.
            const auto far = dev->get_connected_ethernet_core(eth_logical);
            const auto fit = chip_to_coord.find(static_cast<uint32_t>(std::get<0>(far)));
            if (fit != chip_to_coord.end()) {
                w.peer_coord.assign(fit->second.coords().begin(), fit->second.coords().end());
            }

            std::vector<uint32_t> words;
            const bool ok = tt::tt_metal::detail::ReadFromDeviceL1(
                dev, wp.worker_logical, addr, TELEM_NUM_WORDS * sizeof(uint32_t), words);
            if (ok && words.size() >= TELEM_NUM_WORDS && words[TELEM_MAGIC] == TELEMETRY_MAGIC) {
                w.valid = true;
                w.tokens_sent = words[TELEM_TOKENS_SENT];
                w.token_size_bytes = words[TELEM_TOKEN_SIZE];
                w.num_in_tokens = words[TELEM_NUM_IN_TOKENS];
                w.t_first_send = static_cast<uint64_t>(words[TELEM_T_FIRST_SEND_LO]) |
                                 (static_cast<uint64_t>(words[TELEM_T_FIRST_SEND_HI]) << 32);
                w.t_last_send = static_cast<uint64_t>(words[TELEM_T_LAST_SEND_LO]) |
                                (static_cast<uint64_t>(words[TELEM_T_LAST_SEND_HI]) << 32);
                w.t_drained = static_cast<uint64_t>(words[TELEM_T_DRAINED_LO]) |
                              (static_cast<uint64_t>(words[TELEM_T_DRAINED_HI]) << 32);
                w.edm_slots = words[TELEM_EDM_SLOTS];
                w.drain_packets = words[TELEM_DRAIN_PACKETS];
                w.out_base_page = words[TELEM_OUT_BASE_PAGE];
                w.wait_slot_cycles = static_cast<uint64_t>(words[TELEM_WAIT_SLOT_CY_LO]) |
                                     (static_cast<uint64_t>(words[TELEM_WAIT_SLOT_CY_HI]) << 32);
                w.issue_cycles = static_cast<uint64_t>(words[TELEM_ISSUE_CY_LO]) |
                                 (static_cast<uint64_t>(words[TELEM_ISSUE_CY_HI]) << 32);
            }
            out.workers.push_back(std::move(w));
        }
    }
    return out;
}

tt::tt_metal::WorkloadDescriptor CombineFabric2dProgramFactory::create_workload_descriptor(
    const CombineFabric2dParams& operation_attributes,
    const CombineFabric2dInputs& tensor_args,
    ttnn::Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    auto* mesh_device = operation_attributes.device;
    const auto mesh_shape = mesh_device->shape();
    const uint32_t axis = operation_attributes.axis;
    TT_FATAL(axis < mesh_shape.dims(), "combine_fabric2d: axis {} out of range for mesh shape {}", axis, mesh_shape);
    TT_FATAL(
        mesh_shape[axis] > 1,
        "combine_fabric2d: mesh axis {} has extent {}; need at least 2 chips to send anywhere",
        axis,
        mesh_shape[axis]);

    const uint32_t fabric_max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    TT_FATAL(
        operation_attributes.token_size_bytes <= fabric_max_payload,
        "combine_fabric2d: token_size_bytes {} exceeds the fabric max payload {} (one packet per token)",
        operation_attributes.token_size_bytes,
        fabric_max_payload);

    const auto grid = mesh_device->compute_with_storage_grid_size();
    const auto l1 = compute_l1_layout(
        mesh_device, operation_attributes.input_tokens_per_movement, operation_attributes.token_size_bytes);
    log_info(
        tt::LogOp,
        "combine_fabric2d L1: prod_buf 0x{:x} hdr_ring 0x{:x} ({} input tokens x {} B), drain_sink 0x{:x}",
        l1.prod_buf,
        l1.pkt_hdr_ring,
        operation_attributes.input_tokens_per_movement,
        operation_attributes.token_size_bytes,
        l1.drain_sink);

    // Physical chip id -> mesh coordinate, to turn a cable's far chip into a placement lookup.
    std::map<uint32_t, ttnn::MeshCoordinate> chip_to_coord;
    for (const auto& c : ttnn::MeshCoordinateRange(mesh_shape)) {
        chip_to_coord.emplace(static_cast<uint32_t>(mesh_device->get_device(c)->id()), c);
    }

    PlacementCache placements(mesh_device, axis, operation_attributes.num_links, grid);

    // Both regions are caller-owned interleaved DRAM buffers whose base address is uniform across the
    // mesh, so a producer can address the same buffer on any chip by page index.
    auto* dram_in_buf = tensor_args.input.buffer();
    auto* dram_out_buf = tensor_return_value.buffer();
    TT_FATAL(dram_in_buf != nullptr, "combine_fabric2d: input tensor has no device buffer");
    TT_FATAL(dram_out_buf != nullptr, "combine_fabric2d: output tensor has no device buffer");
    TT_FATAL(
        dram_in_buf->aligned_page_size() == operation_attributes.token_size_bytes,
        "combine_fabric2d: input page size {} must equal token_size_bytes {} (one page = one token)",
        dram_in_buf->aligned_page_size(),
        operation_attributes.token_size_bytes);
    TT_FATAL(
        dram_out_buf->aligned_page_size() == operation_attributes.token_size_bytes,
        "combine_fabric2d: output page size {} must equal token_size_bytes {} (one page = one token)",
        dram_out_buf->aligned_page_size(),
        operation_attributes.token_size_bytes);
    // Every movement's region must fit in the buffer it indexes. Checked here rather than in validate()
    // because only the buffers know the real per-device page count.
    const uint32_t in_pages = static_cast<uint32_t>(dram_in_buf->num_pages());
    const uint32_t out_pages = static_cast<uint32_t>(dram_out_buf->num_pages());
    for (const auto& m : operation_attributes.movements) {
        TT_FATAL(
            m.in_base_token + operation_attributes.input_tokens_per_movement <= in_pages,
            "combine_fabric2d: movement src {} reads input tokens [{}, {}) but the input buffer holds {} per device",
            movement_coord_str(m.src),
            m.in_base_token,
            m.in_base_token + operation_attributes.input_tokens_per_movement,
            in_pages);
        TT_FATAL(
            m.out_base_token + operation_attributes.output_tokens_per_movement <= out_pages,
            "combine_fabric2d: movement src {} -> dst {} writes output tokens [{}, {}) but the output buffer holds "
            "{} per device",
            movement_coord_str(m.src),
            movement_coord_str(m.dst),
            m.out_base_token,
            m.out_base_token + operation_attributes.output_tokens_per_movement,
            out_pages);
    }

    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_program_for_coord(
            operation_attributes, coord, placements, chip_to_coord, l1, dram_out_buf, dram_in_buf);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
