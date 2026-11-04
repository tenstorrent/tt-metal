// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "port_placement.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>

#include <internal/mesh_device.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block {

namespace placement_detail {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;

bool row_major_less(const CoreCoord& a, const CoreCoord& b) { return std::tie(a.y, a.x) < std::tie(b.y, b.x); }

// NoC hops from a worker cell to a port's Ethernet core on one chip.
class EthDistance {
public:
    EthDistance(MeshDevice& mesh_device, ChipCoord chip, tt::tt_metal::NOC noc) :
        mesh_device_(mesh_device),
        coord_(chip.r, chip.c),
        device_(tt::tt_metal::internal::get_device(mesh_device, coord_)),
        noc_(noc) {}

    // The closest worker to the Ethernet core `link` of chip -> peer forwards through, and its hops.
    tt::tt_metal::experimental::Device::CoreAtNocHops closest(ChipCoord chip, ChipCoord peer, uint32_t link) const {
        const auto eth = tt::tt_fabric::get_forwarding_eth_core(
            mesh_device_,
            mesh_device_.get_fabric_node_id(MeshCoordinate(chip.r, chip.c)),
            mesh_device_.get_fabric_node_id(MeshCoordinate(peer.r, peer.c)),
            link);
        return tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(*device_, eth, noc_);
    }

    uint32_t hops(const CoreCoord& cell, const tt::tt_metal::experimental::Device::CoreAtNocHops& closest) const {
        const uint32_t to_closest =
            cell == closest.logical_coord
                ? 0
                : tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                      device_, cell, closest.logical_coord, noc_);
        return to_closest + closest.distance_in_noc_hops;
    }

private:
    MeshDevice& mesh_device_;
    MeshCoordinate coord_;
    tt::tt_metal::IDevice* device_;
    tt::tt_metal::NOC noc_;
};

PortCores place_nearest(
    MeshDevice& mesh_device, const MeshPlan& plan, tt::tt_metal::NOC noc, const std::vector<CoreCoord>& allowed) {
    PortCores out;
    for (const auto& [chip, ports] : plan.chips) {
        const EthDistance dist(mesh_device, chip, noc);
        std::set<std::pair<size_t, size_t>> taken;
        for (Dir d : kDirs) {
            const auto& port = ports[static_cast<size_t>(d)];
            if (!port) {
                continue;
            }
            for (uint32_t li = 0; li < port->links.size(); ++li) {
                const auto closest = dist.closest(chip, port->peer, port->links[li]);
                std::optional<CoreCoord> best;
                uint32_t best_hops = std::numeric_limits<uint32_t>::max();
                for (const auto& cell : allowed) {  // (row-major: a tie keeps the lower row, then column)
                    if (taken.contains({cell.x, cell.y})) {
                        continue;
                    }
                    const uint32_t h = dist.hops(cell, closest);
                    if (h < best_hops) {
                        best_hops = h;
                        best = cell;
                    }
                }
                TT_FATAL(
                    best.has_value(), "moe_block: too few allowed cells for the ports of chip ({}, {})", chip.r, chip.c);
                taken.insert({best->x, best->y});
                out[PortKey{chip, d, li}] = *best;
            }
        }
    }
    return out;
}

}  // namespace placement_detail

std::vector<tt::tt_metal::CoreCoord> all_worker_cores(const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    std::vector<tt::tt_metal::CoreCoord> cells;
    cells.reserve(grid.x * grid.y);
    for (size_t y = 0; y < grid.y; ++y) {
        for (size_t x = 0; x < grid.x; ++x) {
            cells.emplace_back(x, y);
        }
    }
    return cells;
}

PortCores place_ports(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const MeshPlan& plan,
    const std::optional<std::vector<tt::tt_metal::CoreCoord>>& allowed_in) {
    auto allowed = allowed_in.value_or(all_worker_cores(mesh_device));
    std::sort(allowed.begin(), allowed.end(), placement_detail::row_major_less);
    allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
    return placement_detail::place_nearest(mesh_device, plan, tt::tt_metal::NOC::NOC_1, allowed);
}

}  // namespace ttnn::operations::experimental::moe_block
