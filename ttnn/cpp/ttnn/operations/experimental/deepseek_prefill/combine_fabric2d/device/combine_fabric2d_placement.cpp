// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_placement.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

namespace {

// One packed column per stream, row 0. Of the fixed layouts swept on a Blackhole galaxy these were the
// steadiest run to run and within 2% of the fastest; every layout spread further across the grid was at
// least 9% slower.
constexpr uint32_t worker_column(StreamId stream) { return stream; }

StreamPlacements decide_device_placement(
    ttnn::MeshDevice* mesh, const ttnn::MeshCoordinate& coord, uint32_t axis, uint32_t num_links) {
    auto* dev = mesh->get_device(coord);
    const auto self_node = mesh->get_fabric_node_id(coord);

    StreamPlacements placements;
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
            const auto stream = make_stream_id(k, delta == 1);
            const CoreCoord worker{worker_column(stream), 0};
            placements.emplace(
                stream,
                StreamPlacement{
                    worker, dev->virtual_core_from_logical_core(worker, tt::CoreType::WORKER), *nbr, nbr_node});
        }
    }
    return placements;
}

}  // namespace

MeshPlacement decide_placement(ttnn::MeshDevice* mesh, uint32_t axis, uint32_t num_links) {
    TT_FATAL(mesh != nullptr, "combine_fabric2d: mesh device is null");
    TT_FATAL(
        worker_column(stream_count(num_links) - 1) < mesh->compute_with_storage_grid_size().x,
        "combine_fabric2d: {} streams need worker columns up to {}, but the compute grid is only {} wide",
        stream_count(num_links),
        worker_column(stream_count(num_links) - 1),
        mesh->compute_with_storage_grid_size().x);

    MeshPlacement placement;
    for (const auto& coord : ttnn::MeshCoordinateRange(mesh->shape())) {
        placement.emplace(coord, decide_device_placement(mesh, coord, axis, num_links));
    }
    return placement;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
