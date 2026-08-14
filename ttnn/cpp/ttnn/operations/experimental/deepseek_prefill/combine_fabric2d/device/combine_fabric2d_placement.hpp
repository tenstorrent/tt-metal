// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// One reader+producer pair and the fabric cable it drives.
struct WorkerPlacement {
    CoreCoord eth_logical;
    uint32_t link_idx = 0;  // routing plane to open the fabric connection on
    tt::tt_fabric::FabricNodeId peer_node{tt::tt_fabric::MeshId{0}, 0};  // chip across the cable
    CoreCoord worker_logical;                                            // where to place the kernels
    CoreCoord worker_virtual;                                            // what a producer on another chip must address
};

// Every worker one device hosts, keyed by the eth core it serves.
struct DevicePlacement {
    std::map<CoreCoord, WorkerPlacement> by_eth_logical;
};

// Where `coord` puts a worker for each of its fabric eth cores. Depends only on that device's own eth
// set and its own harvesting, so the answer is stable no matter who asks — which is what lets a neighbour
// rely on it. Decides for the whole device at once, because a producer's args need the peer worker's
// coordinates on the neighbour chip.
DevicePlacement decide_placement(
    ttnn::MeshDevice* mesh,
    const ttnn::MeshCoordinate& coord,
    uint32_t axis,
    uint32_t num_links,
    const CoreCoord& compute_grid);

// Lazy device -> placement cache. Building the op for D forces placement for D's cable neighbours, since
// D's producers address workers on them.
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

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
