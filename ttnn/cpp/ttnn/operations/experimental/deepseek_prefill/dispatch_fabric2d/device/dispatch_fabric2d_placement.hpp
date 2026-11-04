// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// The single-axis torus fabric config that wraps `axis`, for error messages.
constexpr const char* torus_for_axis(uint32_t axis) { return axis == 0 ? "FABRIC_2D_TORUS_Y" : "FABRIC_2D_TORUS_X"; }

// pos: a chip's position along the dispatch axis, 0 to extent - 1 (a mesh row or column depending on
// cluster_axis).
//
// A stream is one fabric link in one direction along the dispatch axis: a reader and a sender on one core.
// A stream keeps its id across chips: the stream with the same id on the next chip continues in the same
// direction on the same link.
//
// Each stream needs its own EDM channel. The EDM stores a single worker_xy per channel, so two senders on
// one channel deadlock.
using StreamId = uint32_t;

constexpr StreamId make_stream_id(uint32_t link_idx, bool is_cw) { return link_idx * 2 + (is_cw ? 0u : 1u); }
// The stream on a chip's downstream neighbour that sends back to that chip: the same link ordinal, the other
// direction. The pairing is one to one, which is all the launch credit needs.
constexpr StreamId reverse_stream(StreamId stream) { return stream ^ 1u; }
constexpr uint32_t stream_count(uint32_t num_links) { return num_links * 2; }

struct StreamPlacement {
    tt::tt_metal::CoreCoord worker_logical;       // where this stream's kernels go
    tt::tt_metal::CoreCoord worker_virtual;       // what a sender on another chip addresses
    ttnn::MeshCoordinate downstream_coord{0, 0};  // chip across the cable
    tt::tt_fabric::FabricNodeId downstream_node{tt::tt_fabric::MeshId{0}, 0};
    uint32_t link_idx = 0;  // element of get_forwarding_link_indices toward downstream_node
};

using StreamPlacements = std::map<StreamId, StreamPlacement>;
using MeshPlacement = std::map<ttnn::MeshCoordinate, StreamPlacements>;

// Placement for every chip and every stream on the mesh. Decided for the whole mesh at once because a
// sender's arguments name the worker serving the same stream on the downstream chip.
//
// `allowed_cores` is every core the op may use: the caller's sub-device, or the whole compute grid when no
// sub-device manager is loaded. A stream goes on the worker nearest its eth core, and the op refuses if
// that core is not in allowed_cores. If another stream already has it, the stream moves to a free core in the
// same row: first the nearest to its right, then from column 0; only a full row sends it to the next row.
// Cores outside allowed_cores belong to other work on the chip.
MeshPlacement decide_placement(
    ttnn::MeshDevice* mesh, uint32_t axis, uint32_t num_links, const tt::tt_metal::CoreRangeSet& allowed_cores);

// Cores in allowed_cores that no stream took, row-major. Nothing else runs on them.
std::vector<tt::tt_metal::CoreCoord> spare_cores(
    const tt::tt_metal::CoreRangeSet& allowed_cores, const StreamPlacements& streams);

// Spare cores for the untilizer pool: `per_link` per link, capped at `num_tile_rows` because an extra core
// would do no work. Taken from the core row directly under the streams and dealt round robin over the
// streams' columns, so every stream gets one untilizer before any gets a second and the pool's DRAM traffic
// stays near the columns the streams already use.
//
// A sub-device without that core row is refused: on the streams' own row the pool would share the NoC row
// the streams already fill. `*spilled` is set when that row holds too few spare cores and the rest are taken
// from elsewhere, which the caller reports.
std::vector<tt::tt_metal::CoreCoord> decide_untilizer_pool(
    const tt::tt_metal::CoreRangeSet& allowed_cores,
    const StreamPlacements& streams,
    uint32_t num_tile_rows,
    uint32_t per_link,
    bool* spilled);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
