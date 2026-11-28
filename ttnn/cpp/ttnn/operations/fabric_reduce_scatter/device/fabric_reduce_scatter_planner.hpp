// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/distributed/types.hpp"
#include "fabric_reduce_scatter_device_operation_types.hpp"

// The group / ring / placement plan of fabric_all_gather_py.plan + fabric_reduce_scatter._with_final_cores,
// and the reduce-scatter schedule (_schedule). Derived only from the mesh and the fabric: solved once per
// (mesh id, mesh shape, fabric node ids, PlanInputs) and returned by const&.
namespace ttnn::operations::fabric_reduce_scatter::plan {

enum Direction : uint32_t { FWD = 0, BWD = 1 };

// The plan's inputs: the attributes it depends on.
struct PlanInputs {
    std::optional<uint32_t> cluster_axis;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear;
    uint32_t num_links = 1;
    Scheme scheme = Scheme::RING;
    Placement placement = Placement::AUTO;
    bool operator==(const PlanInputs&) const = default;
};

// One chip's place on one ring (Python chips[coord]["rings"][j]). Chips are row-major mesh indices.
struct RingPosition {
    uint32_t p = 0;                // position along the ring
    std::optional<uint32_t> next;  // chip index, none at a line end
    std::optional<uint32_t> prev;
    bool ag_sends[2] = {false, false};  // the all-gather schedule toward next / prev is non-empty (rg[d])
    std::vector<uint32_t> links[2];     // forwarding link indices used toward next / prev (rg["fwd_links"])
};

struct ChipPlan {
    uint32_t out = 0;  // output slot: row-major rank in the group
    std::vector<RingPosition> rings;
    // Port cores, Python dict order: index port_index(j, d, l) = (j * 2 + d) * num_links + l.
    std::vector<tt::tt_metal::CoreCoord> ports;
    std::vector<tt::tt_metal::CoreCoord> copy;    // one per link
    std::vector<tt::tt_metal::CoreCoord> finals;  // [lane 0 half 0, lane 0 half 1, lane 1 half 0, ...]
};

struct Plan {
    uint32_t group_size = 0;  // G
    uint32_t n_rings = 1;     // 2 for dual_cycles
    uint32_t num_links = 1;
    bool ring = false;                      // Ring topology with G > 2
    std::vector<ChipPlan> chips;            // indexed by row-major mesh index
    std::vector<uint32_t> fabric_mesh_ids;  // per chip: FabricNodeId mesh id
    std::vector<uint32_t> fabric_chip_ids;  // per chip: FabricNodeId chip id
};

inline uint32_t port_index(uint32_t ring, uint32_t dir, uint32_t link, uint32_t num_links) {
    return ((ring * 2) + dir) * num_links + link;
}

// Memoized per mesh (id, shape) and inputs; solves on the first call. Throws std::invalid_argument where the
// Python raised ValueError (no direct link, fewer usable links than num_links, too few cores, dual_cycles
// preconditions) and TT_FATALs on the Python's asserts (G >= 2).
const Plan& get_plan(tt::tt_metal::distributed::MeshDevice& mesh_device, const PlanInputs& inputs);

// Python _schedule(chips, coord, j, d, ring): the blocks port (ring j, direction d) of `chip` sends,
// farthest first, as (block, slot at the receiver), and how many lead (start their chain here).
struct PortSchedule {
    std::vector<std::pair<uint32_t, uint32_t>> blocks;
    uint32_t lead = 0;
};
PortSchedule rs_schedule(const Plan& plan, uint32_t chip, uint32_t ring, Direction dir);

// Python _needs_ready(chips, coord, j, d).
bool needs_ready(const Plan& plan, uint32_t chip, uint32_t ring, Direction dir);

// Python _chunks_per_shard (part 0 = whole shard, 1 / 2 = first / second half of the bank set).
uint32_t chunks_per_shard(
    uint32_t shard_pages,
    uint32_t first_bank,
    uint32_t stride,
    uint32_t num_banks,
    uint32_t run_pages,
    uint32_t part = 0);

// Python _incs: ceil(chunks / inc_every), 0 for no chunks.
inline uint32_t increments(uint32_t chunks, uint32_t inc_every) {
    return chunks == 0 ? 0 : (chunks + inc_every - 1) / inc_every;
}

}  // namespace ttnn::operations::fabric_reduce_scatter::plan
