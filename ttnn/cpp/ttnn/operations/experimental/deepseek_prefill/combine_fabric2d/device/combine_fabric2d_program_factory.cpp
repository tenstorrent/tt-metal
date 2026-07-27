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
#include <tt-metalium/device.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt_stl/assert.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

// ---------------------------------------------------------------------------------------------
// L1 layout — identical on every tensix core of every device in the mesh.
//
// Uniformity is load-bearing: a producer on chip A addresses slot k of the receiver ring on chip B by
// computing recv_buf + k * chunk, with no knowledge of B's harvesting or eth positions. All offsets
// are relative to the L1 unreserved base (uniform for a given arch/config).
// ---------------------------------------------------------------------------------------------
constexpr uint32_t PKT_HDR_PAYLOAD_OFF = 0x0000;  // packet header for the payload sends
constexpr uint32_t PKT_HDR_CREDIT_OFF = 0x0400;   // separate header for credit-return sends
constexpr uint32_t PROD_BUF_OFF = 0x1000;         // producer's rotating source data
constexpr uint32_t L1_SLACK = 0x1000;             // keep clear of the global-semaphore region at the L1 top

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
    CoreCoord worker_logical;                                            // where both kernels go
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
    // the backward one. Every one is full duplex, so every one gets a worker.
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
    uint32_t pkt_hdr_payload;
    uint32_t pkt_hdr_credit;
    uint32_t prod_buf;
    uint32_t recv_buf;
};

L1Layout compute_l1_layout(ttnn::MeshDevice* mesh, uint32_t num_slots, uint32_t chunk_size_bytes, uint32_t sem_floor) {
    const uint32_t base =
        static_cast<uint32_t>(mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    L1Layout l;
    l.pkt_hdr_payload = base + PKT_HDR_PAYLOAD_OFF;
    l.pkt_hdr_credit = base + PKT_HDR_CREDIT_OFF;
    l.prod_buf = base + PROD_BUF_OFF;
    l.recv_buf = l.prod_buf + num_slots * chunk_size_bytes;
    const uint32_t end = l.recv_buf + num_slots * chunk_size_bytes;
    TT_FATAL(
        end + L1_SLACK <= sem_floor,
        "combine_fabric2d: L1 layout needs {} B (ends at 0x{:x}) but the global-semaphore region starts at 0x{:x}. "
        "Reduce num_slots ({}) or chunk_size_bytes ({}).",
        end - base,
        end,
        sem_floor,
        num_slots,
        chunk_size_bytes);
    return l;
}

// Per-coordinate program: for each of this device's fabric eth cores, ONE worker core running a
// producer (writer RISC — owns that eth channel's single fabric connection) and a receiver (reader
// RISC — no connection, since being a fabric destination requires none).
tt::tt_metal::ProgramDescriptor build_program_for_coord(
    const CombineFabric2dParams& args,
    const ttnn::MeshCoordinate& coord,
    PlacementCache& placements,
    const std::map<uint32_t, ttnn::MeshCoordinate>& chip_to_coord,
    const L1Layout& l1,
    uint32_t write_up_to_addr,
    uint32_t data_ready_addr,
    uint32_t credits_addr) {
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh = args.device;
    auto* dev = mesh->get_device(coord);
    const auto self_node = mesh->get_fabric_node_id(coord);

    const auto& self_placement = placements.get(coord);
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

        // ---- Producer (writer RISC). Owns the eth channel's single fabric connection: sends payload
        // ---- tokens AND forwards the co-located receiver's credit returns.
        tt::tt_metal::KernelDescriptor prod;
        prod.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
            "producer_combine_fabric2d.cpp";
        prod.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        prod.core_ranges = CoreRangeSet(CoreRange(wp.worker_logical));
        prod.compile_time_args = {
            args.num_tokens,
            args.num_slots,
            args.chunk_size_bytes,
            static_cast<uint32_t>(wp.peer_node.chip_id),
            *wp.peer_node.mesh_id,
            static_cast<uint32_t>(peer_virtual.x),
            static_cast<uint32_t>(peer_virtual.y),
            l1.prod_buf,
            l1.recv_buf,
            l1.pkt_hdr_payload,
            l1.pkt_hdr_credit,
            write_up_to_addr,
            data_ready_addr,
            credits_addr,
        };
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

        // ---- Receiver (reader RISC). No fabric connection: it polls its own L1 and hands credits to
        // ---- the producer sharing its core.
        tt::tt_metal::KernelDescriptor recv;
        recv.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
            "receiver_combine_fabric2d.cpp";
        recv.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        recv.core_ranges = CoreRangeSet(CoreRange(wp.worker_logical));
        recv.compile_time_args = {
            args.num_tokens,
            args.num_slots,
            args.chunk_size_bytes,
            l1.recv_buf,
            data_ready_addr,
            credits_addr,
            static_cast<uint32_t>(wp.worker_virtual.x),
            static_cast<uint32_t>(wp.worker_virtual.y),
        };
        recv.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            // Only used for a self-targeted atomic (cross-RISC visibility), so direction is moot.
            .noc = tt::tt_metal::NOC::NOC_0,
        };
        desc.kernels.push_back(std::move(recv));  // no fabric connection => no rt args

        summary += fmt::format(
            "{}[eth({},{}) phys_x {} link {} -> worker logical ({},{}) phys ({},{}){} peer {} virt ({},{})]",
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
            peer_virtual.y);
    }

    log_info(
        tt::LogOp,
        "combine_fabric2d {} {}: {} worker core(s), each producer+receiver: {}",
        coord,
        self_node,
        self_placement.by_eth_logical.size(),
        summary);
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor CombineFabric2dProgramFactory::create_workload_descriptor(
    const CombineFabric2dParams& operation_attributes,
    const CombineFabric2dInputs& /*tensor_args*/,
    ttnn::Tensor& /*tensor_return_value*/,
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
        operation_attributes.chunk_size_bytes <= fabric_max_payload,
        "combine_fabric2d: chunk_size_bytes {} exceeds the fabric max payload {} (one packet per token)",
        operation_attributes.chunk_size_bytes,
        fabric_max_payload);

    // Three single-writer monotonic counters (see the plan): each has exactly one writer, so there is
    // no cross-RISC read-modify-write to race and no negative NoC atomic. Readers keep their own local
    // consumed count and work on the difference. `write_up_to` starts at num_slots so a producer can
    // fill the ring once before any credit returns. All live on the full worker grid so their L1
    // addresses are uniform across the mesh (worker cores are eth-derived and differ per device).
    const auto grid = mesh_device->compute_with_storage_grid_size();
    const CoreRangeSet all_workers(CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}));
    auto write_up_to_sem = ttnn::global_semaphore::create_global_semaphore(
        mesh_device, all_workers, operation_attributes.num_slots, tt::tt_metal::BufferType::L1);
    auto data_ready_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, all_workers, 0, tt::tt_metal::BufferType::L1);
    auto credits_to_return_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, all_workers, 0, tt::tt_metal::BufferType::L1);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    const uint32_t write_up_to_addr = static_cast<uint32_t>(write_up_to_sem.address());
    const uint32_t data_ready_addr = static_cast<uint32_t>(data_ready_sem.address());
    const uint32_t credits_addr = static_cast<uint32_t>(credits_to_return_sem.address());
    const uint32_t sem_floor = std::min({write_up_to_addr, data_ready_addr, credits_addr});
    const auto l1 = compute_l1_layout(
        mesh_device, operation_attributes.num_slots, operation_attributes.chunk_size_bytes, sem_floor);
    log_info(
        tt::LogOp,
        "combine_fabric2d L1: prod_buf 0x{:x} recv_buf 0x{:x} ({} slots x {} B), sems write_up_to 0x{:x} (init {}) "
        "data_ready 0x{:x} credits_to_return 0x{:x}",
        l1.prod_buf,
        l1.recv_buf,
        operation_attributes.num_slots,
        operation_attributes.chunk_size_bytes,
        write_up_to_addr,
        operation_attributes.num_slots,
        data_ready_addr,
        credits_addr);

    // Physical chip id -> mesh coordinate, to turn a cable's far chip into a placement lookup.
    std::map<uint32_t, ttnn::MeshCoordinate> chip_to_coord;
    for (const auto& c : ttnn::MeshCoordinateRange(mesh_device->shape())) {
        chip_to_coord.emplace(static_cast<uint32_t>(mesh_device->get_device(c)->id()), c);
    }

    PlacementCache placements(mesh_device, axis, operation_attributes.num_links, grid);

    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    workload_descriptor.semaphores.push_back(write_up_to_sem);
    workload_descriptor.semaphores.push_back(data_ready_sem);
    workload_descriptor.semaphores.push_back(credits_to_return_sem);
    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_program_for_coord(
            operation_attributes,
            coord,
            placements,
            chip_to_coord,
            l1,
            write_up_to_addr,
            data_ready_addr,
            credits_addr);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
