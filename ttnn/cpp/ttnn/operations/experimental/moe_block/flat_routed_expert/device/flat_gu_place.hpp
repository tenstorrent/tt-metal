// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gate/up weight placement by NoC link load (FlatGuSource auto_tensix): which bank holds each gate/up core's share and
// on which NoC the core reads it (NOC0 / NOC1), chosen per chip from that chip's own geometry (harvested worker columns
// move the cores' physical NoC positions; the DRAM tiles' positions come from its soc descriptor). A static model: every flow (weights; the x relays' NOC1 reads and the down
// cores' NOC0 reads as background) adds its bytes to each NoC link on its dimension-order route (NOC0 east then south,
// NOC1 north then west, both on the 17 x 12 torus); the search minimises the busiest link (then the sum of squares).
// Each bank keeps 2 x RG cores (its two weight slots): the weight layout is the same for every placement, so chips
// with different placements share the same packed weights.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

struct FlatRoutedExpertPlan;

// One chip's geometry as the placement sees it (NOC0 grid positions)
struct FlatGuGeom {
    std::vector<tt::tt_metal::CoreCoord> cells;                   // the gate/up cells, in flat_gu_cells order
    std::vector<std::array<tt::tt_metal::CoreCoord, 2>> banks;    // per bank: NOC0 endpoint, NOC1 endpoint
    std::vector<tt::tt_metal::CoreCoord> helpers, down;           // the x helper relays, the down cores
    // every position, for deduplication: chips with equal keys get the same placement
    std::vector<uint32_t> key() const;
    bool operator==(const FlatGuGeom&) const = default;
};

// The gate/up cells of a plan (both rectangles, row-major per rectangle): the placement's cell order
std::vector<tt::tt_metal::CoreCoord> flat_gu_cells(const FlatRoutedExpertPlan& p);

FlatGuGeom flat_gu_geom(
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const FlatRoutedExpertPlan& p,
    const tt::tt_metal::distributed::MeshCoordinate& coord);

// Bytes a call (any common scale): one gate/up core's share, all x reads, all down reads
struct FlatGuTraffic {
    double gu_core = 1.0, x = 0.0, down = 0.0;
};

// A placement: per cell (flat_gu_cells order) its bank and how (FlatGuPlace::How)
struct FlatGuAssign {
    std::vector<uint32_t> bank, how;
};

// The busiest link's bytes (and the time at 64 B / cycle, 1.35 GHz, us) of a placement on a chip
struct FlatGuLoad {
    double max_link = 0.0, sumsq = 0.0;
};
FlatGuLoad flat_gu_load(const FlatGuGeom& g, const FlatGuAssign& a, const FlatGuTraffic& t);

// The search: minimises the worst chip's busiest link over `geoms` (one geometry: a per-chip placement; several: one
// compromise placement for all of them). Every core reads its own share. Deterministic (fixed seed).
// restarts of the local search (seeds): per chip, and for the mesh's compromise placement
inline constexpr uint32_t kFlatGuRestarts = 6, kFlatGuCompromiseRestarts = 2;
FlatGuAssign flat_gu_search(
    const std::vector<const FlatGuGeom*>& geoms,
    const FlatGuTraffic& t,
    uint32_t rg,
    uint32_t iters,
    uint32_t restarts = kFlatGuRestarts);


// As FlatLayoutOptions.gu_place words (FlatGuPlace::word) of the plan's cells
std::vector<uint32_t> flat_gu_words(const FlatRoutedExpertPlan& p, const FlatGuAssign& a);
FlatGuAssign flat_gu_from_words(const FlatRoutedExpertPlan& p, const std::vector<uint32_t>& words);

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
