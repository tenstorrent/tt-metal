// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <umd/device/types/arch.hpp>
#include "ttnn/tensor/tensor_spec.hpp"

namespace ttnn::operations::ccl {
struct AllGatherParams;
}

// Host plan for all_gather: geometry and legality, algorithm choice, slicing, the unicast schedule and the multicast
// route sets. Pure index math over plain numbers; only make_plan_in reads specs and the HAL.
namespace ttnn::operations::ccl::all_gather_plan {

enum class AgAlgo : uint8_t { Composite, Unicast, Multicast };

// Logical neighbor directions, numbered as eth_chan_directions: E/W run along axis 1 (E forward), N/S along axis 0
// (S forward).
enum Dir : uint32_t { kEast = 0, kWest = 1, kNorth = 2, kSouth = 3 };

// Page content c*, bank pitch a* (aligned page size), read alignment r* (DRAM or L1), all in bytes. P is the fabric's
// max payload. pages_per_stripe_row: input pages this device contributes to one output row.
struct GeoIn {
    uint32_t ci, co, ai, ao, ri, ro, in_pages, pages_per_stripe_row, N, P;
};

// A: the cut quantum in bytes; a unit is A bytes. Ui/Uo: input/output page footprint in units. S: units per
// stripe-row, R: rows. E: units per walk element (Uo when a stripe-row holds whole output pages, else 1). P: payload
// cap rounded down to A. dst_cross/src_cross: runs may continue across output/input pages.
struct Geometry {
    uint32_t A, Ui, Uo, S, R, E, elements, P;
    bool whole_page, dst_cross, src_cross;
};

struct PlanIn {
    tt::ARCH arch;
    GeoIn geo;
    bool shape_ok, ring, true_2d, forced_unicast;
    uint32_t N, links, avail_cores;
    uint64_t output_packed_bytes;
};

// A contiguous range of walk elements.
struct Slice {
    uint32_t e0, ne;
};

// links is the effective link count of the chosen algorithm; slices has one entry per link. ring is false for N == 2.
// composite_reason says which rule ruled out each native algorithm (empty unless algo is Composite).
struct Plan {
    AgAlgo algo;
    Geometry geo;
    bool ring;
    uint32_t N, links;
    std::vector<Slice> slices;
    std::string composite_reason;
};

// One unicast core (link, direction). Counts are in units unless named otherwise.
struct UnicastDir {
    uint32_t I, stripe_step, last_skip, last_take, iter_units, last_units, recv_units;
    bool local_copy;
};

// dir: the logical neighbor (a Dir). Range bytes sit at physical slots: range = e | w << 8 | n << 16 | s << 24.
struct McastConn {
    uint32_t dir;
    uint32_t range_v0, range_v1;
};
struct McastRoutes {
    std::vector<McastConn> conns;
    bool alternates;
};

// Precondition for any native algorithm that can be checked before the output spec exists: no padding on the gather
// dim. make_plan_in adds the output page-grid check.
bool gather_shape_ok(const tt::tt_metal::TensorSpec& in, int32_t dim_from_end, uint32_t N);

// nullopt: illegal for this algorithm. reads_output: unicast, whose relay reads its own output. Only checks P >= A;
// P <= NOC_MAX_BURST_SIZE is a kernel static_assert, so a fabric config breaking it fails at JIT time.
std::optional<Geometry> make_geometry(const GeoIn&, bool reads_output);

PlanIn make_plan_in(
    const AllGatherParams&,
    const tt::tt_metal::TensorSpec& in,
    const tt::tt_metal::TensorSpec& out,
    uint32_t avail_cores);
// Same, with the HAL values passed in (host tests).
PlanIn make_plan_in(
    const AllGatherParams&,
    const tt::tt_metal::TensorSpec& in,
    const tt::tt_metal::TensorSpec& out,
    uint32_t avail_cores,
    tt::ARCH arch,
    uint32_t dram_alignment,
    uint32_t l1_alignment);

Plan make_plan(const PlanIn&);

// links: unicast's effective links. P: the packet payload. stripe_row_bytes: what one device contributes to one
// output row (input page bytes x pages per stripe row).
bool prefer_unicast(
    tt::ARCH,
    bool ring,
    uint32_t N,
    uint32_t links,
    uint32_t P,
    uint64_t busiest_link_bytes,
    uint64_t stripe_row_bytes);

// Near-equal contiguous element ranges, never empty (needs 1 <= links <= elements).
std::vector<Slice> make_slices(uint32_t elements, uint32_t links);

UnicastDir unicast_schedule(const Plan&, uint32_t dev, uint32_t link, uint32_t dir /*0 fwd, 1 bwd*/);

// Multicast route sets. mesh_shape: devices per axis; axis selects the active axis of a 1D collective (nullopt: every
// axis with more than one device). axis_ring: the axis wraps (false for 2 devices: both directions would reach the
// same neighbor). phys_dir maps each logical direction to its physical eth_chan_directions slot (from the fabric's
// forwarding direction to that neighbor); identity when the axes run E/W and N/S.
McastRoutes mcast_routes(
    const Plan&,
    std::array<uint32_t, 2> mesh_shape,
    std::array<uint32_t, 2> my_coord,
    std::optional<uint32_t> axis,
    std::array<bool, 2> axis_ring,
    std::array<uint32_t, 4> phys_dir = {kEast, kWest, kNorth, kSouth});

}  // namespace ttnn::operations::ccl::all_gather_plan
