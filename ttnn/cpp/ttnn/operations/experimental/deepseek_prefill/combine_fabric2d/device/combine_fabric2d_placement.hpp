// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// A stream is one routing plane travelled in one direction along the ring axis. Every chip runs one
// reader+sender pair per stream, and a stream keeps its identity across chips: the pair on the next chip
// with the same id continues in the same direction on the same plane.
using StreamId = uint32_t;

constexpr StreamId make_stream_id(uint32_t link_idx, bool is_cw) { return link_idx * 2 + (is_cw ? 0u : 1u); }
constexpr bool stream_is_cw(StreamId stream) { return stream % 2 == 0; }
constexpr uint32_t stream_count(uint32_t num_links) { return num_links * 2; }

struct StreamPlacement {
    CoreCoord worker_logical;                     // where this stream's kernels go
    CoreCoord worker_virtual;                     // what a sender on another chip addresses
    ttnn::MeshCoordinate downstream_coord{0, 0};  // chip across the cable
    tt::tt_fabric::FabricNodeId downstream_node{tt::tt_fabric::MeshId{0}, 0};
};

using StreamPlacements = std::map<StreamId, StreamPlacement>;

// Untilizers are grouped by the ring direction whose senders they feed, because a direction's senders walk
// the token index monotonically and in the same direction: one group can serve both of them in order.
constexpr uint32_t UNTILIZER_GROUPS = 2;
constexpr uint32_t untilizer_group_of(StreamId stream) { return stream % UNTILIZER_GROUPS; }

// Cores per group, from CMBF2D_UNTILIZERS_PER_GROUP. Spreading a group's staging over more cores trades
// worker cores for L1 read ports; which way that goes is a measurement, so it is a knob. Both groups share
// one row, so the grid width bounds UNTILIZER_GROUPS * untilizers_per_group() - six per group on a
// harvested Blackhole; five until a measurement asks for the sixth.
uint32_t untilizers_per_group();
constexpr uint32_t MAX_UNTILIZERS_PER_GROUP = 5;

struct UntilizerPlacement {
    CoreCoord logical;
    CoreCoord worker_virtual;  // what a reader on this chip addresses
};

using UntilizerGroups = std::array<std::vector<UntilizerPlacement>, UNTILIZER_GROUPS>;

struct DevicePlacement {
    StreamPlacements streams;
    UntilizerGroups untilizers;  // indexed by untilizer_group_of(stream)
};

using MeshPlacement = std::map<ttnn::MeshCoordinate, DevicePlacement>;

// Placement for every chip on the mesh. Decided for the whole mesh at once because a sender's arguments
// name the worker serving the same stream on the downstream chip.
MeshPlacement decide_placement(ttnn::MeshDevice* mesh, uint32_t axis, uint32_t num_links);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
