// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_fabric2d_placement.hpp"

#include <algorithm>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

namespace {

// The sender reaches its eth core over NOC_1, so that is the NOC whose hop distance is minimised.
constexpr tt::tt_metal::NOC SENDER_NOC = tt::tt_metal::NOC::NOC_1;

struct WorkerCandidate {
    tt::tt_metal::CoreCoord worker;
    uint32_t noc_hops = 0;
    ttnn::MeshCoordinate downstream_coord{0, 0};
    tt::tt_fabric::FabricNodeId downstream_node{tt::tt_fabric::MeshId{0}, 0};
    uint32_t link_idx = 0;
};

// Allowed cores by logical row, each row in ascending x.
using CoreRows = std::map<std::size_t, std::vector<tt::tt_metal::CoreCoord>>;

// Sorted explicitly: corerange_to_cores walks each range column-major by default, and even with
// row_wise it only orders within a range, concatenating ranges in set order.
std::vector<tt::tt_metal::CoreCoord> row_major_cores(const tt::tt_metal::CoreRangeSet& crs) {
    std::vector<tt::tt_metal::CoreCoord> cores = corerange_to_cores(crs);
    std::sort(cores.begin(), cores.end(), [](const auto& a, const auto& b) {
        return std::tie(a.y, a.x) < std::tie(b.y, b.x);
    });
    return cores;
}

bool contains(const CoreRows& rows, const tt::tt_metal::CoreCoord& core) {
    const auto row = rows.find(core.y);
    return row != rows.end() &&
           std::binary_search(row->second.begin(), row->second.end(), core, [](auto a, auto b) { return a.x < b.x; });
}

// Where a stream goes when its nearest core is taken. The sender reaches its eth core over NOC_1, -y then
// -x, so in the nearest core's row each column to its right costs one more hop, while a column to its left
// wraps the whole row, least from column 0. The stream's own row is searched first: a stream a row lower
// moves the untilizer pool a row lower too (decide_untilizer_cores), out of a two-row sub-device. Later rows
// follow, each by the same rule, then the rows above, which wrap on -y.
std::optional<tt::tt_metal::CoreCoord> free_core_near(
    const CoreRows& rows, const tt::tt_metal::CoreCoord& nearest, const std::set<tt::tt_metal::CoreCoord>& taken) {
    const auto first = static_cast<std::size_t>(std::distance(rows.begin(), rows.find(nearest.y)));
    for (std::size_t i = 0; i < rows.size(); i++) {
        const auto& row = std::next(rows.begin(), (first + i) % rows.size())->second;
        const auto split = std::find_if(row.begin(), row.end(), [&](const auto& c) { return c.x >= nearest.x; });
        for (auto it = split; it != row.end(); ++it) {
            if (!taken.contains(*it)) {
                return *it;
            }
        }
        for (auto it = row.begin(); it != split; ++it) {
            if (!taken.contains(*it)) {
                return *it;
            }
        }
    }
    return std::nullopt;
}

StreamPlacements decide_device_placement(
    ttnn::MeshDevice* mesh,
    const ttnn::MeshCoordinate& coord,
    uint32_t axis,
    uint32_t num_links,
    const CoreRows& allowed_rows,
    std::size_t num_allowed) {
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
            const auto closest =
                tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(*dev, eth_core, SENDER_NOC);
            const StreamId stream = make_stream_id(k, delta == 1);
            eth_core_of[stream] = eth_core;
            candidates.emplace(
                stream, WorkerCandidate{closest.logical_coord, closest.distance_in_noc_hops, *nbr, nbr_node, link_idx});
        }
    }

    // Every send is a single hop, so the two directions must leave by different eth cores. On an axis that
    // is not wrap-wired, the neighbour one way round is the far end of the line, and the first hop toward
    // it leaves by the same eth core as the other direction. Both streams would then open a connection on
    // one EDM channel, which stores a single worker_xy, and both would deadlock at open.
    for (uint32_t k = 0; k < num_links; k++) {
        const StreamId cw = make_stream_id(k, true);
        const StreamId ccw = make_stream_id(k, false);
        TT_FATAL(
            eth_core_of.at(cw) != eth_core_of.at(ccw),
            "dispatch_fabric2d {}: link {} leaves by eth core {} in both directions, so axis {} is not "
            "wrap-wired. This op sends single hops around a ring; run it on a topology that wraps that "
            "axis (e.g. {} or FABRIC_2D_TORUS_XY), not a line or mesh.",
            self_node,
            k,
            eth_core_of.at(cw),
            axis,
            torus_for_axis(axis));
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

    // Nearest first, so a stream that already has its closest core keeps it, and any stream that has to
    // move was further away anyway. Eth cores sit in a core row with no workers, so the NOC_1 -y leg always
    // costs a hop, and a worker in the eth core's column avoids the -x leg: the minimum is one hop, and
    // several streams can have it. Whether two eth cores share a nearest worker, and which two, depends on
    // the chip's harvested columns, so even a nearest stream may find its core taken and have to move.
    std::vector<StreamId> order;
    order.reserve(candidates.size());
    for (const auto& [stream, candidate] : candidates) {
        order.push_back(stream);
    }
    std::stable_sort(order.begin(), order.end(), [&](StreamId a, StreamId b) {
        return candidates.at(a).noc_hops < candidates.at(b).noc_hops;
    });

    for (const StreamId stream : order) {
        const auto& candidate = candidates.at(stream);
        // Refused: cores outside allowed_cores belong to other work on the chip, so a nearest core
        // outside it means the caller's core set is wrong.
        TT_FATAL(
            contains(allowed_rows, candidate.worker),
            "dispatch_fabric2d {}: the worker nearest stream {}'s eth core is {}, which is outside the "
            "{} cores this op was given. Widen the subdevice_id's core set to include it.",
            self_node,
            stream,
            candidate.worker,
            num_allowed);
        if (!taken.contains(candidate.worker)) {
            assign(stream, candidate, candidate.worker);
            continue;
        }
        const auto moved = free_core_near(allowed_rows, candidate.worker, taken);
        TT_FATAL(
            moved.has_value(),
            "dispatch_fabric2d {}: every one of the {} cores this op was given is taken; stream {} has nowhere "
            "to go",
            self_node,
            num_allowed,
            stream);
        assign(stream, candidate, *moved);
    }
    return placements;
}

}  // namespace

MeshPlacement decide_placement(
    ttnn::MeshDevice* mesh, uint32_t axis, uint32_t num_links, const tt::tt_metal::CoreRangeSet& allowed_cores) {
    TT_FATAL(mesh != nullptr, "dispatch_fabric2d: mesh device is null");
    const std::vector<tt::tt_metal::CoreCoord> cores = row_major_cores(allowed_cores);
    TT_FATAL(
        cores.size() >= stream_count(num_links),
        "dispatch_fabric2d: {} worker cores for {} streams",
        cores.size(),
        stream_count(num_links));
    CoreRows rows;
    for (const auto& core : cores) {
        rows[core.y].push_back(core);
    }
    MeshPlacement placement;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
        placement.emplace(coord, decide_device_placement(mesh, coord, axis, num_links, rows, cores.size()));
    }
    return placement;
}

std::vector<tt::tt_metal::CoreCoord> spare_cores(
    const tt::tt_metal::CoreRangeSet& allowed_cores, const StreamPlacements& streams) {
    std::set<tt::tt_metal::CoreCoord> taken;
    for (const auto& [stream, placement] : streams) {
        taken.insert(placement.worker_logical);
    }
    std::vector<tt::tt_metal::CoreCoord> spare;
    for (const auto& core : row_major_cores(allowed_cores)) {
        if (!taken.contains(core)) {
            spare.push_back(core);
        }
    }
    return spare;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
