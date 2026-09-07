// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_placement.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

namespace {

// The sender reaches its eth core over NOC_1, so that is the NOC distances are minimised on.
constexpr tt::tt_metal::NOC SENDER_NOC = tt::tt_metal::NOC::NOC_1;

struct WorkerCandidate {
    tt::tt_metal::CoreCoord worker;
    uint32_t noc_hops = 0;
    ttnn::MeshCoordinate downstream_coord{0, 0};
    tt::tt_fabric::FabricNodeId downstream_node{tt::tt_fabric::MeshId{0}, 0};
    uint32_t link_idx = 0;
};

StreamPlacements decide_device_placement(
    ttnn::MeshDevice* mesh, const ttnn::MeshCoordinate& coord, uint32_t axis, uint32_t num_links) {
    auto* dev = mesh->get_device(coord);
    const auto self_node = mesh->get_fabric_node_id(coord);

    std::map<StreamId, WorkerCandidate> candidates;
    std::map<StreamId, tt::tt_metal::CoreCoord> eth_core_of;
    for (int delta : {1, -1}) {
        const auto nbr = coord.get_neighbor(
            mesh->shape(), delta, static_cast<int32_t>(axis), ttnn::MeshCoordinate::BoundaryMode::WRAP);
        TT_FATAL(nbr.has_value(), "dispatch_fabric2d: no axis-{} neighbor of {} at delta {}", axis, coord, delta);
        TT_FATAL(
            *nbr != coord,
            "dispatch_fabric2d: axis {} wraps {} onto itself, so there is no neighbour to send to",
            axis,
            coord);
        const auto nbr_node = mesh->get_fabric_node_id(*nbr);
        const auto links = tt::tt_fabric::get_forwarding_link_indices(self_node, nbr_node);
        TT_FATAL(
            links.size() >= num_links,
            "dispatch_fabric2d {}: {} forwarding links toward {}, {} requested",
            self_node,
            links.size(),
            nbr_node,
            num_links);
        for (uint32_t k = 0; k < num_links; k++) {
            // links[k], not k: the returned indices are the forwarding-capable subset of the
            // direction's channels, so an ordinal is not a link index.
            const uint32_t link_idx = links[k];
            const auto eth_core = tt::tt_fabric::get_forwarding_eth_core(self_node, nbr_node, link_idx);
            uint32_t noc_hops = 0;
            const tt::tt_metal::CoreCoord worker =
                tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(dev, eth_core, SENDER_NOC, noc_hops);
            const StreamId stream = make_stream_id(k, delta == 1);
            eth_core_of[stream] = eth_core;
            candidates.emplace(stream, WorkerCandidate{worker, noc_hops, *nbr, nbr_node, link_idx});
        }
    }

    // Every send is a single hop, so the two directions must leave by different cables. On an axis that
    // is not wrap-wired the "neighbour" one way round is the far end of the line, and its route's first
    // hop leaves by the SAME eth core as the other direction -- so both streams would open a connection
    // on one EDM channel, which stores a single worker_xy and deadlocks both of them permanently at open.
    // Catching it here turns a 32-chip hang into a message.
    for (uint32_t k = 0; k < num_links; k++) {
        const StreamId cw = make_stream_id(k, true);
        const StreamId ccw = make_stream_id(k, false);
        TT_FATAL(
            eth_core_of.at(cw) != eth_core_of.at(ccw),
            "dispatch_fabric2d {}: link {} leaves by eth core {} in both directions, so axis {} is not "
            "wrap-wired. This op sends single hops around a ring; run it on a topology that wraps that "
            "axis (e.g. FABRIC_2D_TORUS_Y or _TORUS_XY), not a line or mesh.",
            self_node,
            k,
            eth_core_of.at(cw),
            axis);
    }

    StreamPlacements placements;
    std::set<tt::tt_metal::CoreCoord> taken;
    auto assign = [&](StreamId stream, const WorkerCandidate& candidate, const tt::tt_metal::CoreCoord& worker) {
        TT_FATAL(
            taken.insert(worker).second,
            "dispatch_fabric2d {}: stream {} was placed on {}, which another stream already owns",
            self_node,
            stream,
            worker);
        placements.emplace(
            stream,
            StreamPlacement{
                worker,
                dev->virtual_core_from_logical_core(worker, tt::CoreType::WORKER),
                candidate.downstream_coord,
                candidate.downstream_node,
                candidate.link_idx});
    };

    // Nearest-first, so a stream already as close as it can get keeps that core and any displacement
    // falls on a stream that was further out anyway. Eth cores own a row no worker sits in, so the
    // NOC_1 -y leg always costs a hop and sharing the eth core's column is what removes the -x leg --
    // which means the floor is one hop and several streams can sit on it. Two of a chip's eth cores
    // can share a column, so even a nearest stream may find its core taken and has to walk.
    std::vector<StreamId> order;
    order.reserve(candidates.size());
    for (const auto& [stream, candidate] : candidates) {
        order.push_back(stream);
    }
    std::stable_sort(order.begin(), order.end(), [&](StreamId a, StreamId b) {
        return candidates.at(a).noc_hops < candidates.at(b).noc_hops;
    });

    const uint32_t grid_width = mesh->compute_with_storage_grid_size().x;
    for (const StreamId stream : order) {
        const auto& candidate = candidates.at(stream);
        tt::tt_metal::CoreCoord worker = candidate.worker;
        for (uint32_t tried = 0; taken.contains(worker); tried++) {
            TT_FATAL(
                tried < grid_width,
                "dispatch_fabric2d {}: no free worker on row {} for stream {}",
                self_node,
                worker.y,
                stream);
            worker.x = (worker.x + 1) % grid_width;
        }
        assign(stream, candidate, worker);
    }
    return placements;
}

}  // namespace

MeshPlacement decide_placement(ttnn::MeshDevice* mesh, uint32_t axis, uint32_t num_links) {
    TT_FATAL(mesh != nullptr, "dispatch_fabric2d: mesh device is null");
    MeshPlacement placement;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
        placement.emplace(coord, decide_device_placement(mesh, coord, axis, num_links));
    }
    return placement;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
