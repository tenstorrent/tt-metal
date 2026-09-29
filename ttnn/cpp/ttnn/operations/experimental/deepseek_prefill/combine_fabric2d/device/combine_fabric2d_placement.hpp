// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// A stream is one routing plane travelled in one direction along the ring axis. Every chip runs one
// reader+sender pair per stream, and a stream keeps its identity across chips: the pair on the next chip
// with the same id continues in the same direction on the same plane.
using StreamId = uint32_t;

constexpr StreamId make_stream_id(uint32_t link_idx, bool is_cw) { return link_idx * 2 + (is_cw ? 0u : 1u); }
constexpr uint32_t stream_count(uint32_t num_links) { return num_links * 2; }

struct StreamPlacement {
    tt::tt_metal::CoreCoord worker_logical;       // where this stream's kernels go
    tt::tt_metal::CoreCoord worker_virtual;       // what a sender on another chip addresses
    ttnn::MeshCoordinate downstream_coord{0, 0};  // chip across the cable
    tt::tt_fabric::FabricNodeId downstream_node{tt::tt_fabric::MeshId{0}, 0};
};

using StreamPlacements = std::map<StreamId, StreamPlacement>;
using MeshPlacement = std::map<ttnn::MeshCoordinate, StreamPlacements>;

// Placement for every chip and every stream on the mesh. Decided for the whole mesh at once because a
// sender's arguments name the worker serving the same stream on the downstream chip.
MeshPlacement decide_placement(ttnn::MeshDevice* mesh, uint32_t axis, uint32_t num_links);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
