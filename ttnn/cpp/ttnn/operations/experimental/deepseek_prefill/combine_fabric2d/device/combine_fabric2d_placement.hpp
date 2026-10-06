// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <map>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/distributed/types.hpp"
#include "kernels/dataflow/combine_fabric2d_stream.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

using cmbf2d::make_stream_id;
using cmbf2d::stream_count;
using cmbf2d::stream_is_cw;
using cmbf2d::StreamId;

struct StreamPlacement {
    tt::tt_metal::CoreCoord worker_logical;       // where this stream's kernels go
    tt::tt_metal::CoreCoord worker_virtual;       // what a sender on another chip addresses
    ttnn::MeshCoordinate downstream_coord{0, 0};  // chip across the cable
    tt::tt_fabric::FabricNodeId downstream_node{tt::tt_fabric::MeshId{0}, 0};
};

using StreamPlacements = std::map<StreamId, StreamPlacement>;

// Untilizers are grouped by the ring direction whose senders they feed, because a direction's senders walk
// the token index monotonically and in the same direction: one group can serve both of them in order.
constexpr uint32_t UNTILIZER_GROUPS = 2;
constexpr uint32_t untilizer_group_of(StreamId stream) { return stream % UNTILIZER_GROUPS; }

// Cores per group, from CMBF2D_UNTILIZERS_PER_GROUP. Spreading a group's staging over more cores trades
// worker cores for L1 read ports; which way that goes is a measurement, so it is a knob. The ceiling here
// only bounds the knob; whether a value fits is decided by placement, which knows the grid and the senders.
uint32_t untilizers_per_group();
constexpr uint32_t DEFAULT_UNTILIZERS_PER_GROUP = 5;
constexpr uint32_t MAX_UNTILIZERS_PER_GROUP = 10;

struct UntilizerPlacement {
    tt::tt_metal::CoreCoord logical;
    tt::tt_metal::CoreCoord worker_virtual;  // what a reader on this chip addresses
};

using UntilizerGroups = std::array<std::vector<UntilizerPlacement>, UNTILIZER_GROUPS>;

// The core that folds the routed expert's per-writer reports into one `ready` count. Placed LAST, in any cell of
// the two combine rows the senders and untilizers left free: decide_untilizers claims whole columns and
// refuses a cell that is already taken, so reserving one for the collector first could break it.
struct CollectorPlacement {
    tt::tt_metal::CoreCoord logical;
    tt::tt_metal::CoreCoord worker_virtual;  // what a routed-expert writer addresses
};

struct DevicePlacement {
    StreamPlacements streams;
    UntilizerGroups untilizers;  // indexed by untilizer_group_of(stream)
    // Set only when the caller asked for a collector; the standalone op has nothing to collect.
    std::optional<CollectorPlacement> collector;
};

using MeshPlacement = std::map<ttnn::MeshCoordinate, DevicePlacement>;

// Placement for every chip on the mesh. Decided for the whole mesh at once because a sender's arguments
// name the worker serving the same stream on the downstream chip. `untilizers_per_group` of zero reserves no
// untilizer cores at all, which is what a caller with nothing to untilize asks for.
// `with_collector` reserves one more cell of the two combine rows for the core that folds the routed
// expert's completion reports. Only the overlapped caller needs it, and it costs a worker core.
MeshPlacement decide_placement(
    ttnn::MeshDevice* mesh,
    uint32_t axis,
    uint32_t num_links,
    uint32_t untilizers_per_group,
    bool with_collector = false);

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
