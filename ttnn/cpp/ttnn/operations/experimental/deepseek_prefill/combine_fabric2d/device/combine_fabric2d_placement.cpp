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

}  // namespace

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

    auto make_placement = [&](const EthEntry& e, const CoreCoord& worker) {
        WorkerPlacement wp;
        wp.eth_logical = e.eth_logical;
        wp.link_idx = e.link_idx;
        wp.peer_node = e.peer_node;
        wp.worker_logical = worker;
        wp.worker_virtual = dev->virtual_core_from_logical_core(worker, tt::CoreType::WORKER);
        return wp;
    };

    DevicePlacement placement;
    // Pass 1: everyone who can sit in their own eth column does.
    for (const auto& e : eths) {
        const auto w = adjacent_worker_in_column(geom, e.eth_phys_x);
        if (!w.has_value()) {
            continue;
        }
        used_columns.insert(e.eth_phys_x);
        placement.by_eth_logical.emplace(e.eth_logical, make_placement(e, *w));
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
        placement.by_eth_logical.emplace(e.eth_logical, make_placement(e, *chosen));
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

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
