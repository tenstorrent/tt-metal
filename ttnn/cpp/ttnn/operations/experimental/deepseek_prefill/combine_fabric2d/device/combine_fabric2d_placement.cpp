// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_placement.hpp"

#include <algorithm>
#include <optional>
#include <set>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

// A worker and the eth core it drives must share a PHYSICAL (noc0) x. That is the only coordinate space in
// which cores of different types are comparable: logical coords are per-core-type dense indices, and
// virtual/translated coords put eth and tensix in disjoint ranges (on Blackhole an eth core's translated
// coord is literally (20 + eth_channel, 25)).
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

// The worker in `phys_x` adjacent to the eth row (smallest physical y — eth sits at the low-y edge of the
// grid). nullopt if that column has no programmable worker at all: it may be harvested (differs per chip
// within one mesh) or host the dispatch cores.
std::optional<CoreCoord> adjacent_worker_in_column(const DeviceGeometry& geom, uint32_t phys_x) {
    auto it = geom.columns.find(phys_x);
    if (it == geom.columns.end() || it->second.empty()) {
        return std::nullopt;
    }
    return it->second.begin()->second;
}

struct EthEntry {
    StreamId stream;
    CoreCoord eth_logical;
    uint32_t eth_phys_x;
    ttnn::MeshCoordinate downstream_coord;
    tt::tt_fabric::FabricNodeId downstream_node;
};

std::vector<EthEntry> fabric_eth_cores(
    ttnn::MeshDevice* mesh,
    tt::tt_metal::IDevice* dev,
    const ttnn::MeshCoordinate& coord,
    uint32_t axis,
    uint32_t num_links) {
    const auto self_node = mesh->get_fabric_node_id(coord);
    std::vector<EthEntry> eths;
    for (int delta : {1, -1}) {
        const auto nbr = coord.get_neighbor(
            mesh->shape(), delta, static_cast<int32_t>(axis), ttnn::MeshCoordinate::BoundaryMode::WRAP);
        TT_FATAL(nbr.has_value(), "combine_fabric2d: no axis-{} neighbor of {} at delta {}", axis, coord, delta);
        TT_FATAL(
            *nbr != coord,
            "combine_fabric2d: axis {} wraps {} onto itself, so there is no neighbour to send to",
            axis,
            coord);
        const auto nbr_node = mesh->get_fabric_node_id(*nbr);
        const auto links = tt::tt_fabric::get_forwarding_link_indices(self_node, nbr_node);
        TT_FATAL(
            links.size() >= num_links,
            "combine_fabric2d {}: {} forwarding links toward {}, {} requested",
            self_node,
            links.size(),
            nbr_node,
            num_links);
        for (uint32_t k = 0; k < num_links; k++) {
            const CoreCoord eth_logical =
                tt::tt_fabric::get_forwarding_link_logical_eth_core(self_node, nbr_node, links[k]);
            const auto eth_phys = tt::tt_metal::experimental::Device::get_physical_core_from_logical_core(
                dev, eth_logical, tt::CoreType::ETH);
            eths.push_back(EthEntry{
                make_stream_id(k, delta == 1), eth_logical, static_cast<uint32_t>(eth_phys.x), *nbr, nbr_node});
        }
    }
    // Deterministic order so the relocation fallback is reproducible.
    std::sort(
        eths.begin(), eths.end(), [](const EthEntry& a, const EthEntry& b) { return a.eth_phys_x < b.eth_phys_x; });
    return eths;
}

StreamPlacements decide_device_placement(
    ttnn::MeshDevice* mesh,
    const ttnn::MeshCoordinate& coord,
    uint32_t axis,
    uint32_t num_links,
    const CoreCoord& compute_grid) {
    auto* dev = mesh->get_device(coord);
    const auto self_node = mesh->get_fabric_node_id(coord);
    const auto geom = build_device_geometry(dev, compute_grid);
    const auto eths = fabric_eth_cores(mesh, dev, coord, axis, num_links);

    // Columns that must stay free for a co-located worker: any column holding one of OUR eth cores. Taking
    // one for a relocated worker could displace the worker that belongs there.
    std::set<uint32_t> eth_columns;
    for (const auto& e : eths) {
        eth_columns.insert(e.eth_phys_x);
    }
    std::set<uint32_t> used_columns;

    auto make_placement = [&](const EthEntry& e, const CoreCoord& worker) {
        return StreamPlacement{
            worker,
            dev->virtual_core_from_logical_core(worker, tt::CoreType::WORKER),
            e.eth_logical,
            e.downstream_coord,
            e.downstream_node};
    };

    StreamPlacements placements;
    // Pass 1: everyone who can sit in their own eth column does.
    for (const auto& e : eths) {
        const auto w = adjacent_worker_in_column(geom, e.eth_phys_x);
        if (!w.has_value()) {
            continue;
        }
        used_columns.insert(e.eth_phys_x);
        placements.emplace(e.stream, make_placement(e, *w));
    }
    // Pass 2: relocate the rest to the leftmost column that is neither one of our eth columns nor already
    // taken. Runs after pass 1 so a relocated worker can never steal a co-located worker's column, whatever
    // order the eth cores come in.
    for (const auto& e : eths) {
        if (placements.count(e.stream)) {
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
        placements.emplace(e.stream, make_placement(e, *chosen));
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
    return placements;
}

// Every stream must reach the same stream on the chip its cable physically lands on. A mismatch means link
// selection disagreed with the cabling, which would send tokens to a core that is not draining them.
void validate_cable_continuity(ttnn::MeshDevice* mesh, const MeshPlacement& placement) {
    // get_connected_ethernet_core answers in PHYSICAL chip ids, which are a different namespace from the
    // fabric chip ids in FabricNodeId, so the comparison has to happen in mesh coordinates.
    std::map<uint32_t, ttnn::MeshCoordinate> coord_by_physical_chip;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
        coord_by_physical_chip.emplace(static_cast<uint32_t>(mesh->get_device(coord)->id()), coord);
    }

    for (const auto& [coord, streams] : placement) {
        auto* dev = mesh->get_device(coord);
        for (const auto& [stream, sp] : streams) {
            const auto cabled = static_cast<uint32_t>(std::get<0>(dev->get_connected_ethernet_core(sp.eth_logical)));
            const auto it = coord_by_physical_chip.find(cabled);
            TT_FATAL(
                it != coord_by_physical_chip.end(),
                "combine_fabric2d {}: stream {} is cabled to chip {}, which is not in this mesh",
                coord,
                stream,
                cabled);
            TT_FATAL(
                it->second == sp.downstream_coord,
                "combine_fabric2d {}: stream {} routes toward {} but its eth core ({},{}) is cabled to {}",
                coord,
                stream,
                sp.downstream_coord,
                sp.eth_logical.x,
                sp.eth_logical.y,
                it->second);
            const auto downstream = placement.find(sp.downstream_coord);
            TT_FATAL(
                downstream != placement.end() && downstream->second.count(stream) > 0,
                "combine_fabric2d {}: stream {} continues on {}, which has no worker for that stream",
                coord,
                stream,
                sp.downstream_coord);
        }
    }
}

}  // namespace

MeshPlacement decide_placement(ttnn::MeshDevice* mesh, uint32_t axis, uint32_t num_links) {
    TT_FATAL(mesh != nullptr, "combine_fabric2d: mesh device is null");
    const auto compute_grid = mesh->compute_with_storage_grid_size();
    MeshPlacement placement;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
        placement.emplace(coord, decide_device_placement(mesh, coord, axis, num_links, compute_grid));
    }
    validate_cable_continuity(mesh, placement);
    return placement;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
