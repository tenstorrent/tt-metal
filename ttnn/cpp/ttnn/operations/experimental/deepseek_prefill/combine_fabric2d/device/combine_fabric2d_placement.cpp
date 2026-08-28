// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_placement.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <set>

#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

// The sender reaches its eth core over NOC_1, so that is the NOC distances are minimised on.
constexpr tt::tt_metal::NOC SENDER_NOC = tt::tt_metal::NOC::NOC_1;

// Eth cores own a row no worker sits in, so the NOC_1 -y leg always costs one hop and a shared column
// is exactly what removes the -x leg. One hop therefore means "same column as the eth core".
constexpr uint32_t SENDER_NOC_MIN_ETH_HOPS = 1;

struct WorkerCandidate {
    CoreCoord worker;
    uint32_t noc_hops = 0;
    ttnn::MeshCoordinate downstream_coord{0, 0};
    tt::tt_fabric::FabricNodeId downstream_node{tt::tt_fabric::MeshId{0}, 0};
};

// Both groups go in the one row adjacent to the senders' row, each in a contiguous block of columns,
// blocks ordered by their group's leftmost sender so a group sits over the senders it feeds.
UntilizerGroups decide_untilizers(
    const StreamPlacements& streams,
    std::set<CoreCoord>& taken,
    tt::tt_metal::IDevice* dev,
    const CoreCoord& grid,
    const tt::tt_fabric::FabricNodeId& who) {
    std::array<std::size_t, UNTILIZER_GROUPS> leftmost_sender;
    leftmost_sender.fill(std::numeric_limits<std::size_t>::max());
    std::size_t sender_row = streams.begin()->second.worker_logical.y;
    for (const auto& [stream, placement] : streams) {
        leftmost_sender[untilizer_group_of(stream)] =
            std::min(leftmost_sender[untilizer_group_of(stream)], placement.worker_logical.x);
        TT_FATAL(
            placement.worker_logical.y == sender_row,
            "combine_fabric2d {}: sender of stream {} is on row {}, not the single sender row {}",
            who,
            stream,
            placement.worker_logical.y,
            sender_row);
    }
    TT_FATAL(
        sender_row + 1 < grid.y,
        "combine_fabric2d {}: senders occupy row {}, leaving no adjacent row in a {}-row grid",
        who,
        sender_row,
        grid.y);
    TT_FATAL(
        UNTILIZER_GROUPS * untilizers_per_group() <= grid.x,
        "combine_fabric2d {}: {} groups of {} untilizers do not fit a {}-column row",
        who,
        UNTILIZER_GROUPS,
        untilizers_per_group(),
        grid.x);

    std::array<uint32_t, UNTILIZER_GROUPS> order;
    std::iota(order.begin(), order.end(), 0u);
    std::sort(
        order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return leftmost_sender[a] < leftmost_sender[b]; });

    UntilizerGroups groups;
    std::size_t column = 0;
    for (uint32_t g : order) {
        TT_FATAL(
            leftmost_sender[g] != std::numeric_limits<std::size_t>::max(),
            "combine_fabric2d {}: untilizer group {} has no senders to serve",
            who,
            g);
        for (uint32_t i = 0; i < untilizers_per_group(); i++) {
            const CoreCoord core{column++, sender_row + 1};
            TT_FATAL(
                taken.insert(core).second,
                "combine_fabric2d {}: untilizer {} of group {} lands on {}, which is already owned",
                who,
                i,
                g,
                core);
            groups[g].push_back(
                UntilizerPlacement{core, dev->virtual_core_from_logical_core(core, tt::CoreType::WORKER)});
        }
    }
    return groups;
}

DevicePlacement decide_device_placement(
    ttnn::MeshDevice* mesh, const ttnn::MeshCoordinate& coord, uint32_t axis, uint32_t num_links) {
    auto* dev = mesh->get_device(coord);
    const auto self_node = mesh->get_fabric_node_id(coord);

    std::map<StreamId, WorkerCandidate> candidates;
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
            uint32_t noc_hops = 0;
            const CoreCoord worker = tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(
                dev, tt::tt_fabric::get_forwarding_eth_core(self_node, nbr_node, k), SENDER_NOC, noc_hops);
            candidates.emplace(make_stream_id(k, delta == 1), WorkerCandidate{worker, noc_hops, *nbr, nbr_node});
        }
    }

    StreamPlacements placements;
    std::set<CoreCoord> taken;
    auto assign = [&](StreamId stream, const WorkerCandidate& candidate, const CoreCoord& worker) {
        TT_FATAL(
            taken.insert(worker).second,
            "combine_fabric2d {}: stream {} was placed on {}, which another stream already owns",
            self_node,
            stream,
            worker);
        placements.emplace(
            stream,
            StreamPlacement{
                worker,
                dev->virtual_core_from_logical_core(worker, tt::CoreType::WORKER),
                candidate.downstream_coord,
                candidate.downstream_node});
    };

    for (const auto& [stream, candidate] : candidates) {
        if (candidate.noc_hops == SENDER_NOC_MIN_ETH_HOPS) {
            assign(stream, candidate, candidate.worker);
        }
    }
    const uint32_t grid_width = mesh->compute_with_storage_grid_size().x;
    for (const auto& [stream, candidate] : candidates) {
        if (candidate.noc_hops == SENDER_NOC_MIN_ETH_HOPS) {
            continue;
        }
        CoreCoord worker = candidate.worker;
        for (uint32_t tried = 0; taken.contains(worker); tried++) {
            TT_FATAL(
                tried < grid_width,
                "combine_fabric2d {}: no free worker on row {} for stream {}",
                self_node,
                worker.y,
                stream);
            worker.x = (worker.x + 1) % grid_width;
        }
        assign(stream, candidate, worker);
    }
    return DevicePlacement{
        placements, decide_untilizers(placements, taken, dev, mesh->compute_with_storage_grid_size(), self_node)};
}

}  // namespace

uint32_t untilizers_per_group() {
    static const uint32_t n = [] {
        const char* env = std::getenv("CMBF2D_UNTILIZERS_PER_GROUP");
        const uint32_t v = env != nullptr ? static_cast<uint32_t>(std::atoi(env)) : MAX_UNTILIZERS_PER_GROUP;
        TT_FATAL(
            v >= 1 && v <= MAX_UNTILIZERS_PER_GROUP,
            "CMBF2D_UNTILIZERS_PER_GROUP is {}, outside 1..{}",
            v,
            MAX_UNTILIZERS_PER_GROUP);
        return v;
    }();
    return n;
}

MeshPlacement decide_placement(ttnn::MeshDevice* mesh, uint32_t axis, uint32_t num_links) {
    TT_FATAL(mesh != nullptr, "combine_fabric2d: mesh device is null");

    MeshPlacement placement;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
        placement.emplace(coord, decide_device_placement(mesh, coord, axis, num_links));
    }
    return placement;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
